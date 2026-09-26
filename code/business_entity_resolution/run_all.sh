#!/usr/bin/env bash
# End-to-end: dataset/ -> work/ (cache, model) -> output/{matching_results,candidate_pairs}.tsv
# Usage: bash run_all.sh [extra args for train.py]
set -euo pipefail
cd "$(dirname "$0")"
python src/train.py "$@"
python src/predict.py
