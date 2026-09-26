"""Generate the submission files for the test split.

Writes <out-dir>/candidate_pairs.tsv (the exact candidate set scored by the
model) and <out-dir>/matching_results.tsv (candidates whose match probability
is >= the threshold chosen in train.py), one row per Source-1 entity, then runs
utils/validate_submission.py when it is available.

    python src/predict.py [--split test] [--threads 8]
"""

import argparse
import json
import os
import subprocess
import sys
import time

import lightgbm as lgb
import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ber import data, pipeline, translit  # noqa: E402


def main():
    d = data.default_dirs()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=d["data"])
    ap.add_argument("--work-dir", default=d["work"])
    ap.add_argument("--out-dir", default=d["out"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--chunk", type=int, default=20_000, help="Source-1 entities per feature chunk")
    ap.add_argument("--threshold", type=float, default=None, help="override the trained threshold")
    ap.add_argument("--limit", type=int, default=None, help="only the first N Source-1 entities (smoke test)")
    args = ap.parse_args()
    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)  # noqa: E731

    data.prepare_split(args.data_dir, args.work_dir, args.split, log=log)
    with open(os.path.join(args.work_dir, "model_config.json")) as f:
        config = json.load(f)
    booster = lgb.Booster(model_file=os.path.join(args.work_dir, "model.txt"))
    thr = args.threshold if args.threshold is not None else config["threshold"]
    mapper = translit.Mapper(translit.load(os.path.join(args.work_dir, "translit.json")))
    log(f"model: {len(config['features'])} features, threshold {thr:.2f}")

    all_ids = pq.read_table(data.prepared_path(args.work_dir, args.split, 1), columns=["id"]).column("id").to_numpy()
    wanted = all_ids[:args.limit] if args.limit else None
    os.makedirs(args.out_dir, exist_ok=True)
    cand_path = os.path.join(args.out_dir, "candidate_pairs.tsv")
    match_path = os.path.join(args.out_dir, "matching_results.tsv")
    written = set()
    n_pairs = n_matches = 0
    with open(cand_path + ".tmp", "w", encoding="utf-8") as fc, open(match_path + ".tmp", "w", encoding="utf-8") as fm:
        fc.write("source1_entity_id\tcandidate_entity_ids\n")
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        for country in pipeline.countries(args.work_dir, args.split):
            log(f"country {country}")
            qtab, ctab, csrc = pipeline.load_country(args.work_dir, args.split, country, wanted)
            if qtab.num_rows == 0:
                continue
            Q, C = pipeline.arrays(qtab, mapper), pipeline.arrays(ctab, mapper)
            q_ids = qtab.column("id").to_numpy()
            c_num = ctab.column("id").to_numpy()

            def on_chunk(P, X, names):
                nonlocal n_pairs, n_matches
                assert names == config["features"], "feature mismatch between train and predict"
                prob = booster.predict(X, num_threads=args.threads)
                keep = prob >= thr
                q, c = P["q"], P["c"]
                starts = np.flatnonzero(np.r_[True, q[1:] != q[:-1]])
                ends = np.r_[starts[1:], len(q)]
                labels = [f"S{a}-{b}" for a, b in zip(csrc[c].tolist(), c_num[c].tolist())]
                for s, e in zip(starts.tolist(), ends.tolist()):
                    sid = f"S1-{q_ids[q[s]]}"
                    cands = labels[s:e]
                    matched = [x for x, k in zip(cands, keep[s:e].tolist()) if k]
                    fc.write(f"{sid}\t{','.join(cands)}\n")
                    fm.write(f"{sid}\t{','.join(matched)}\n")
                    written.add(q_ids[q[s]])
                    n_pairs += e - s
                    n_matches += len(matched)

            pipeline.run_country(Q, C, config["retrieval"], args.threads, args.chunk, on_chunk, log)
            del Q, C, qtab, ctab, c_num
        # Source-1 entities without any candidate still need a (empty) row
        ids = wanted if wanted is not None else all_ids
        missing = [i for i in ids.tolist() if i not in written]
        for i in missing:
            fc.write(f"S1-{i}\t\n")
            fm.write(f"S1-{i}\t\n")
    os.replace(cand_path + ".tmp", cand_path)
    os.replace(match_path + ".tmp", match_path)
    n = len(ids)
    log(f"wrote {match_path} and {cand_path}: {n:,} entities, {n_pairs / max(n, 1):.1f} candidates and "
        f"{n_matches / max(n, 1):.2f} matches per entity, {len(missing):,} without candidates")

    validator = os.path.join(data.REPO_DIR, "utils", "validate_submission.py")
    if os.path.exists(validator) and args.split == "test" and not args.limit:
        subprocess.run([sys.executable, validator, "--matching", match_path, "--candidate", cand_path,
                        "--test-dir", os.path.join(args.data_dir, "test")], check=False)


if __name__ == "__main__":
    sys.exit(main())
