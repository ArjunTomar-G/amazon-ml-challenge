"""End-to-end pipeline: raw TSVs -> normalisation -> blocking -> matching -> outputs.

    python src/run_pipeline.py --data-dir ../../student_resource/dataset \
                               --work-dir ./work --out-dir ./output

Every stage runs in its own Python process (memory is returned to the OS
between stages) and writes its results to --work-dir, so the pipeline can be
resumed from any stage with --from <step> or run partially with --only.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent

STEPS = [
    ("prep", ["prep.py"]),                          # TSV->parquet, translit dict, normalisation
    ("pruner", ["candidates.py", "pruner"]),        # learn the candidate pruner (train only)
    ("cand-train", ["candidates.py", "train"]),     # blocking + pruning on train
    ("cand-test", ["candidates.py", "test"]),       # blocking + pruning on test
    ("feats-train", ["model.py", "feats", "train"]),  # pairwise features, every train pair
    ("feats-test", ["model.py", "feats", "test"]),    # pairwise features, every test pair
    ("stage1", ["model.py", "stage1"]),             # cross-fitted stage-1 LightGBM + p1
    ("ctx-train", ["stage2.py", "ctx", "train"]),   # context features
    ("ctx-test", ["stage2.py", "ctx", "test"]),
    ("stage2", ["stage2.py", "train"]),             # stage-2 model + validation + decision rule
    ("predict-test", ["stage2.py", "predict", "test"]),
    ("output", ["output.py", "test"]),              # matching_results.tsv + candidate_pairs.tsv
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="folder with train/ and test/ TSV folders")
    ap.add_argument("--work-dir", required=True, help="scratch folder (needs ~40 GB)")
    ap.add_argument("--out-dir", required=True, help="where the two output TSVs are written")
    ap.add_argument("--from", dest="start", default=None, help="resume from this step")
    ap.add_argument("--only", nargs="*", default=None, help="run only these steps")
    args = ap.parse_args()

    env = dict(os.environ)
    env["ER_DATA_DIR"] = str(Path(args.data_dir).resolve())
    env["ER_WORK_DIR"] = str(Path(args.work_dir).resolve())
    env["ER_OUT_DIR"] = str(Path(args.out_dir).resolve())
    env["PYTHONIOENCODING"] = "utf-8"

    names = [s for s, _ in STEPS]
    todo = names
    if args.start:
        todo = names[names.index(args.start):]
    if args.only:
        todo = [n for n in names if n in args.only]
    for name, cmd in STEPS:
        if name not in todo:
            continue
        t = time.time()
        print(f"=== step {name}: {' '.join(cmd)}", flush=True)
        r = subprocess.run([sys.executable, "-W", "ignore", *cmd], cwd=SRC, env=env)
        if r.returncode != 0:
            raise SystemExit(f"step {name} failed with exit code {r.returncode}")
        print(f"=== step {name} done in {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
