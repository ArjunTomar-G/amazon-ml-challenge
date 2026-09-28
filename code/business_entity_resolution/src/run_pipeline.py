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
# v13: larger cross-encoders (crossenc.py variants) whose logits are extra stacker / rescue features
# "b" = multilingual-e5-base, "l" = multilingual-e5-large
STACK_EXTRA = "b,l"                              # stacker features
RESCUE_EXTRA = "b,l"                             # rescue-model features (both passes)
EXTRA_CE = ",".join(sorted(set(STACK_EXTRA.split(",")) | set(RESCUE_EXTRA.split(","))))   # models to train
BIG = dict(ER_CE_FIT="fitb", ER_CE_GC="0", ER_CE_INFER_TOKENS="32768")   # reference-style training set, 24 GB GPU

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
    # stage 3: transformer cross-encoder on the uncertain band (US / India) + stacker
    ("fetch-model", ["fetch_model.py"]),            # multilingual-e5-small (MIT) -> <work>/hf
    ("ce-select", ["crossenc.py", "select"]),
    ("ce-tokenize", ["crossenc.py", "tokenize"]),
    ("ce-train-A", ["crossenc.py", "train", "A"]),
    ("ce-train-B", ["crossenc.py", "train", "B"]),
    ("ce-score", ["crossenc.py", "score"]),
    ("ce-train-A-r", ["crossenc.py", "train", "A"]),   # ensemble member: record text first, other seed
    ("ce-train-B-r", ["crossenc.py", "train", "B"]),
    ("ce-score-r", ["crossenc.py", "score"]),
    ("stack-cv", ["stack.py", "cv"], dict(ER_CE_BASEFEAT="0")),    # stacker on the validation fold (5-fold CV)
    ("stack-fit", ["stack.py", "fit"], dict(ER_CE_BASEFEAT="0")),  # stacked test probabilities (v12)
    # rescue retrieval for records blocking left unlinked (US / India)
    ("rescue-retriever", ["retriever.py"]),         # contrastively fine-tuned retriever (models/bienc)
    ("rescue-embed-train", ["rescue.py", "embed", "train"]),
    ("rescue-embed-test", ["rescue.py", "embed", "test"]),
    ("rescue-retrieve-train", ["rescue.py", "retrieve", "train"]),
    ("rescue-retrieve-test", ["rescue.py", "retrieve", "test"]),
    ("rescue-feats-train", ["rescue.py", "feats", "train"]),
    ("rescue-feats-test", ["rescue.py", "feats", "test"]),
    ("rescue-cheap", ["rescue.py", "cheap"]),
    ("rescue-ce-train", ["rescue.py", "ce", "train"]),
    ("rescue-ce-test", ["rescue.py", "ce", "test"]),
    ("rescue-model", ["rescue.py", "model"]),
    # v11: rescue v2 for records WITHOUT an address (retriever trained on records with and without one)
    ("rescue2-retriever", ["retriever2.py", "small"]),  # models/bienc2
    ("rescue2-embed-train", ["rescue2.py", "embed", "train"]),
    ("rescue2-embed-test", ["rescue2.py", "embed", "test"]),
    ("rescue2-retrieve-train", ["rescue2.py", "retrieve", "train"]),
    ("rescue2-retrieve-test", ["rescue2.py", "retrieve", "test"]),
    ("rescue2-feats-train", ["rescue2.py", "feats", "train"]),
    ("rescue2-feats-test", ["rescue2.py", "feats", "test"]),
    ("rescue2-cheap", ["rescue2.py", "cheap"]),
    ("rescue2-ce-train", ["rescue2.py", "ce", "train"]),
    ("rescue2-ce-test", ["rescue2.py", "ce", "test"]),
    ("rescue2-model", ["rescue2.py", "model"]),
    # v10c decisions (France rules A,A0,B,C,D,T2 + v10c rescue) -> <work>/out_v10c, then the v11 changes on
    # top of them (France rule T3, rescue v2 for the records still unlinked) -> --out-dir (build_v11.py)
    ("final-v10c", ["final.py", "--prob", "stack", "--thr", "0.8", "--rescue", "rescue", "--france", "A,A0,B,C,D,T2"]),
    ("final-v12", ["build_v11.py", "--thr", "0.8", "--rank-t1", "0.65"]),
    # v13: multilingual-e5-base (and -large) cross-encoders trained on a larger set (the uncertain band 0.002-0.998
    # plus 600 k confident pairs, lr 5e-5 / 3e-5), their logits as extra stacker and rescue features; the changed
    # US / India decisions are applied to the v12 file (build_v13.py)
    ("fetch-base", ["fetch_model.py", "base"]),
    ("ce-select-big", ["crossenc.py", "select_fit"], dict(BIG, ER_CE_VARIANT="b")),
    ("ce-tokenize-b", ["crossenc.py", "tokenize"], dict(BIG, ER_CE_VARIANT="b")),
    ("ce-train-A-b", ["crossenc.py", "train", "A"], dict(BIG, ER_CE_VARIANT="b", ER_CE_LR="5e-5")),
    ("ce-train-B-b", ["crossenc.py", "train", "B"], dict(BIG, ER_CE_VARIANT="b", ER_CE_LR="5e-5")),
    ("ce-score-b", ["crossenc.py", "score"], dict(BIG, ER_CE_VARIANT="b")),
    ("fetch-large", ["fetch_model.py", "large"], dict(ONLY_IF="l")),
    ("ce-tokenize-l", ["crossenc.py", "tokenize"], dict(BIG, ER_CE_VARIANT="l", ONLY_IF="l")),
    ("ce-train-A-l", ["crossenc.py", "train", "A"], dict(BIG, ER_CE_VARIANT="l", ER_CE_LR="3e-5", ONLY_IF="l")),
    ("ce-train-B-l", ["crossenc.py", "train", "B"], dict(BIG, ER_CE_VARIANT="l", ER_CE_LR="3e-5", ONLY_IF="l")),
    ("ce-score-l", ["crossenc.py", "score"], dict(BIG, ER_CE_VARIANT="l", ONLY_IF="l")),
    ("stack-cv-big", ["stack.py", "cv"], dict(ER_CE_EXTRA=STACK_EXTRA)),
    ("stack-fit-big", ["stack.py", "fit"], dict(ER_CE_EXTRA=STACK_EXTRA, ER_STACK_OUT="test_pfinal_big.npy")),
    *[(f"rescue-ce-{sp}-{v}", ["rescue.py", "ce", sp, v]) for v in RESCUE_EXTRA.split(",") for sp in ("train", "test")],
    *[(f"rescue2-ce-{sp}-{v}", ["rescue2.py", "ce", sp, v]) for v in RESCUE_EXTRA.split(",") for sp in ("train", "test")],
    ("rescue-model-big", ["rescue.py", "model"], dict(ER_RESCUE_CE_EXTRA=RESCUE_EXTRA)),
    ("rescue2-model-big", ["rescue2.py", "model"], dict(ER_RESCUE_CE_EXTRA=RESCUE_EXTRA)),
    ("final", ["build_v13.py", "--old-p", "feat/test_pfinal.npy", "--new-p", "feat/test_pfinal_big.npy",
               "--old-r1", "rescue/test_rescue_pred.parquet", "--new-r1", f"rescue/test_rescue_pred_{RESCUE_EXTRA.replace(',', '_')}.parquet",
               "--old-r2", "rescue2/test_rescue_pred.parquet", "--new-r2", f"rescue2/test_rescue_pred_{RESCUE_EXTRA.replace(',', '_')}.parquet",
               "--thr", "0.8", "--rank-t1", "0.65"]),
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
    env["USE_TF"] = "0"                    # transformers: PyTorch only
    env["TRANSFORMERS_NO_TF"] = "1"

    names = [s for s, _ in STEPS]
    todo = names
    if args.start:
        todo = names[names.index(args.start):]
    if args.only:
        todo = [n for n in names if n in args.only]
    for name, cmd, *over in STEPS:
        if name not in todo:
            continue
        over = dict(over[0]) if over else {}
        if over.pop("ONLY_IF", None) not in (None, *EXTRA_CE.split(",")):
            continue
        t = time.time()
        work = Path(args.work_dir).resolve()
        v10c_dir, v12_dir = str(work / "out_v10c"), str(work / "out_v12")
        if name == "final-v10c":
            cmd = [*cmd, "--out", v10c_dir]
        if name == "final-v12":
            cmd = [*cmd, "--v10c", v10c_dir, "--out", v12_dir]
        if name == "final":
            cmd = [*cmd, "--v12", v12_dir, "--out", str(Path(args.out_dir).resolve())]
        print(f"=== step {name}: {' '.join(cmd)}", flush=True)
        step_env = dict(env, ER_CE_VARIANT="r") if name.endswith("-r") else env
        step_env = dict(step_env, **over)
        r = subprocess.run([sys.executable, "-W", "ignore", *cmd], cwd=SRC, env=step_env)
        if r.returncode != 0:
            raise SystemExit(f"step {name} failed with exit code {r.returncode}")
        print(f"=== step {name} done in {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
