#!/bin/bash
# reference-style e5-base cross-encoder: wider band + confident sample, lr 5e-5, 1 epoch, two fold models
# (the same steps as run_pipeline.py's ce-select-big / ce-*-b); needs a work dir in which stage 2 has run
set -e
: "${ER_WORK_DIR:?set ER_WORK_DIR to the pipeline work dir}"
cd "$(dirname "$0")/code/business_entity_resolution/src"
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
export ER_CE_VARIANT=b ER_CE_FIT=fitb ER_CE_LR=5e-5 ER_CE_GC=0 ER_CE_INFER_TOKENS=49152
python -W ignore crossenc.py select_fit
python -W ignore crossenc.py tokenize
python -W ignore crossenc.py train A
python -W ignore crossenc.py train B
python -W ignore crossenc.py score
echo E5BASE_DONE
