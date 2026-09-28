#!/bin/bash
# e5-large cross-encoder on the same training set as e5-base (ce/train_fitb.parquet), lr 3e-5, 1 epoch, two fold models
# run after run_e5base.sh (same ER_WORK_DIR); ~13 GB GPU memory, ER_CE_GC=1 for smaller GPUs
set -e
: "${ER_WORK_DIR:?set ER_WORK_DIR to the pipeline work dir}"
cd "$(dirname "$0")/code/business_entity_resolution/src"
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
export ER_CE_VARIANT=l ER_CE_FIT=fitb ER_CE_LR=3e-5 ER_CE_GC=${ER_CE_GC:-0} ER_CE_INFER_TOKENS=32768
python -W ignore crossenc.py tokenize
python -W ignore crossenc.py train A
python -W ignore crossenc.py train B
python -W ignore crossenc.py score
echo E5LARGE_DONE
