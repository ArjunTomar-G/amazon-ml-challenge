# Business Entity Resolution — ML Challenge 2026

Match every Source-1 business to its records in Sources 2 and 3 (one-to-many, macro F0.5).
The challenge statement is in [`CHALLENGE.md`](CHALLENGE.md).

| Folder | What it is |
|---|---|
| [`code/aiagents_v1/`](code/aiagents_v1) | **Team pipeline as submitted** (public leaderboard 0.97, validation 0.9906), unchanged — the base to build on |
| [`code/aiagents_v2/`](code/aiagents_v2) | **Improvements on top of v1**: test-like training data, France (unlabelled country) remedies, decision variants, label-free audit — see its README for findings and run order |
| [`code/business_entity_resolution/`](code/business_entity_resolution) | Our first pipeline (validation 0.957), superseded by v1; kept for reference |
| [`eda/`](eda) | Exploratory analysis: data profiling, blocking-strategy comparison, similarity-feature study ([`eda/REPORT.md`](eda/REPORT.md)) |
| [`utils/validate_submission.py`](utils/validate_submission.py) | Official format validator |
| [`Documentation_template.md`](Documentation_template.md) | Methodology write-up (to fill in for the submission zip) |

## Quick start (teammates)

For the current team pipeline use [`code/aiagents_v1`](code/aiagents_v1) and the steps in
[`code/aiagents_v2/README.md`](code/aiagents_v2/README.md). The commands below run our first
(superseded) pipeline.

```bash
# 1. put the challenge data here (not in git):
#    dataset/train/train_source{1,2,3}.tsv, dataset/train/train_ground_truth.tsv
#    dataset/test/test_source{1,2,3}.tsv
# 2. install (Python 3.10+)
pip install -r code/business_entity_resolution/requirements.txt
# 3. train, then predict the test set
cd code/business_entity_resolution
python src/train.py        # -> work/model.txt, work/model_config.json (prints holdout macro F0.5)
python src/predict.py      # -> output/matching_results.tsv, output/candidate_pairs.tsv (+ validator)
```

Details, parameters, runtimes and memory needs: [`code/business_entity_resolution/README.md`](code/business_entity_resolution/README.md).

## Submission package

```
<team_name>_submission.zip
├── output/                          # matching_results.tsv + candidate_pairs.tsv from predict.py
├── code/business_entity_resolution/ # this folder as-is
└── Documentation_template.md
```
