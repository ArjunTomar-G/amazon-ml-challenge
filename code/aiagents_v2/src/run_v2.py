"""v2 end-to-end, from a fresh start: raw TSVs -> test-like training data -> full training -> outputs.

    python src/run_v2.py --data-dir <student_resource/dataset> --work-dir <NEW empty dir> --out-dir <out>
                         [--selftrain]

Fresh start: the work dir must be new or empty.  Nothing from an earlier run
(parquet caches, dictionaries, candidates, features, models) is reused, and
every module comes from this folder.  An interrupted run can be resumed with
--from <step> on the same work dir (it carries this run's RUN_V2.json marker).

Steps (each in its own process, as in v1):
    augment       synthetic sibling groups at test density -> <work>/data (train files; test copied)
    prep ... predict-test   the v1 method, trained on the augmented training data
    output        exclusivity + the decision rule chosen on the (test-like) validation fold
                  -> <out>/matching_results.tsv, <out>/candidate_pairs.tsv
    selftrain     (--selftrain) cross-fitted self-training for countries without labels (France)
    variants      extra files for leaderboard comparison -> <out>/variants/<name>/matching_results.tsv
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent
MARKER = "RUN_V2.json"


def steps(a):
    s = [("augment", ["augment.py", "--data-dir", a.data_dir, "--out-dir", str(Path(a.work_dir) / "data"),
                      "--target-unmatched", str(a.target_unmatched), "--min-shift", str(a.min_shift),
                      "--group-sizes", *map(str, a.group_sizes), "--seed", str(a.seed)]),
         ("prep", ["prep.py"]),
         ("pruner", ["candidates.py", "pruner"]),
         ("cand-train", ["candidates.py", "train"]),
         ("cand-test", ["candidates.py", "test"]),
         ("feats-train", ["model.py", "feats", "train"]),
         ("feats-test", ["model.py", "feats", "test"]),
         ("stage1", ["model.py", "stage1"]),
         ("ctx-train", ["stage2.py", "ctx", "train"]),
         ("ctx-test", ["stage2.py", "ctx", "test"]),
         ("stage2", ["stage2.py", "train"]),
         ("predict-test", ["stage2.py", "predict", "test"]),
         ("output", ["output.py", "test"])]
    variants = ["v1", "unseen_thr0.8", "unseen_thr0.9", "guard0.3"]
    if a.selftrain:
        s.append(("selftrain", ["selftrain.py", "--work-dir", a.work_dir]))
        variants += ["st", "st+unseen_thr0.9"]
    s.append(("variants", ["variants.py", "--work-dir", a.work_dir, "--out-dir", a.out_dir, "--variants", *variants]))
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, help="original dataset folder with train/ and test/")
    ap.add_argument("--work-dir", required=True, help="NEW or empty scratch folder (~50 GB)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--selftrain", action="store_true", help="also self-train on unlabelled test countries")
    ap.add_argument("--target-unmatched", type=float, default=0.37)
    ap.add_argument("--group-sizes", type=float, nargs=3, default=[0.35, 0.55, 0.10])
    ap.add_argument("--min-shift", type=int, default=1)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--from", dest="start", default=None, help="resume THIS run from a step")
    a = ap.parse_args()
    a.data_dir, a.work_dir, a.out_dir = (str(Path(p).resolve()) for p in (a.data_dir, a.work_dir, a.out_dir))
    work = Path(a.work_dir)
    marker = work / MARKER
    if a.start is None:
        if work.exists() and any(work.iterdir()):
            raise SystemExit(f"{work} is not empty. v2 always trains from a fresh start: pass a new "
                             f"--work-dir (or resume this run with --from <step>).")
        work.mkdir(parents=True, exist_ok=True)
        json.dump({k: v for k, v in vars(a).items()}, open(marker, "w"), indent=1)
    elif not marker.exists():
        raise SystemExit(f"--from needs a work dir created by run_v2.py (no {MARKER} in {work})")

    env = dict(os.environ, ER_DATA_DIR=str(work / "data"), ER_WORK_DIR=a.work_dir, ER_OUT_DIR=a.out_dir,
               PYTHONIOENCODING="utf-8")
    todo = steps(a)
    names = [n for n, _ in todo]
    if a.start:
        todo = todo[names.index(a.start):]
    for name, cmd in todo:
        t = time.time()
        print(f"=== step {name}: {' '.join(cmd[:3])}", flush=True)
        r = subprocess.run([sys.executable, "-W", "ignore", *cmd], cwd=SRC, env=env)
        if r.returncode != 0:
            raise SystemExit(f"step {name} failed with exit code {r.returncode}; fix and resume with --from {name}")
        print(f"=== step {name} done in {time.time() - t:.0f}s", flush=True)
    dec = json.load(open(work / "models" / "decision.json"))
    print(f"validation (test-like fold) macro F0.5 = {dec['f05']:.5f} with rule {dec['rule']} {dec['param']}")
    print(f"submission: {a.out_dir}/matching_results.tsv  (variants in {a.out_dir}/variants/)")


if __name__ == "__main__":
    main()
