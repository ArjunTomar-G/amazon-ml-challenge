# Business Entity Resolution — AIAgents final approach (v13)

Amazon ML Challenge 2026, team **AIAgents** (Vasu Mahajan, Saarthak Gupta, Vedant Krish Shanker).
Every Source-1 business has to be matched to its records in Sources 2 and 3; the metric is macro F0.5 per
Source-1 entity. The challenge statement is in [`CHALLENGE.md`](CHALLENGE.md).

This branch holds the team's latest and best-validated pipeline, **v13**.

## Results

Validation is fold V: 440 k US / India Source-1 entities that no model was trained on.

| version | what it adds | validation F0.5 | public leaderboard |
|---|---|---|---|
| v7c | stage-2 LightGBM + e5-small cross-encoders + stacker, rescue top-3, France rules A, B, C, D, T2 | 0.99162 | 0.990 |
| v10c | 60 pairwise features in the stacker, France rule A0, fine-tuned rescue retriever (top 5) | 0.99181 | 0.99015 |
| v11 | rescue v2 (records without an address, top 10, name-uniqueness features), France rule T3 | 0.99189 | 0.9903 |
| v12 | rank rule: an S1 with no link takes its strongest record at p >= 0.65 | 0.99197 | not uploaded |
| **v13** | **multilingual-e5-base and e5-large cross-encoders as stacker and rescue features** | **0.99243** | not uploaded |

- v11-v13 validation numbers come from the rebuilt pipeline in this branch (v10c-equivalent 0.99176).
- The leaderboard closed before v12 and v13 were uploaded. The projection for v13 is about **0.9907**: the
  validated gain per changed decision times the 9,962 test decisions v13 changes.
- By country, v12 -> v13: US 0.99153 -> 0.99184, India 0.99264 -> 0.99332. On the density-augmented, test-like
  fold the stacked score goes 0.99104 -> 0.99147.

Full change log with the evidence for every change, and the ideas that were tested and not shipped:
[`TEAM_README.md`](TEAM_README.md). Methodology write-up: [`Documentation_template.md`](Documentation_template.md).

## How it works

1. **Normalisation**: names and addresses (legal forms, OCR digits, house numbers, units, states), an
   Indic -> Latin transliteration dictionary and spelling variants learned from the training pairs.
2. **Blocking**: two-channel IDF inverted index (Numba) plus a learned LightGBM pruner, about 1.5 candidates per
   record at 98.95 % pair recall on train.
3. **Stage 1**: about 145 pairwise features, cross-fitted LightGBM (two fold models, out-of-fold probabilities).
4. **Stage 2**: density-robust context features (competition between candidates, consensus), with the decision
   rule chosen on a density-augmented, test-like validation fold (test has about twice as many look-alike
   businesses per entity).
5. **Cross-encoders** on the uncertain band (0.005 < p < 0.995): four fine-tuned multilingual-e5-small models,
   plus e5-base and e5-large in v13 (1 epoch on 2.27 M pairs, lr 5e-5 / 3e-5, bf16).
6. **Stacker**: LightGBM over stage 2, the cross-encoder logits and 60 pairwise features.
7. **Decision**: each record goes to its most probable S1 if p >= 0.8. The rank rule (US / India) lets an S1
   with no link take its strongest record at p >= 0.65.
8. **Two rescue passes** for true pairs that blocking missed: fine-tuned e5-small retrievers (top 5 and top 10),
   then a cross-encoder and a LightGBM rescue model; links at p >= 0.8 for records still unlinked.
9. **France** (no labels): label-free rules A, A0, B, C, D, T2 and T3 that unlink sibling businesses posing as
   copies (category-word swaps such as club -> ecole, shifted house numbers).

Models: multilingual-e5-small / -base / -large (MIT; 118 M / 278 M / 560 M parameters) and LightGBM (MIT). No
external data. All models are trained on the labelled training data only (no pseudo-labels). Unlabelled
statistics such as IDF weights and France's category-word list are computed from the provided files.

## Repository layout

| path | contents |
|---|---|
| [`code/business_entity_resolution/`](code/business_entity_resolution) | the pipeline: `src/`, [`README.md`](code/business_entity_resolution/README.md) (every step, runtimes, hardware), `requirements.txt` |
| [`TEAM_README.md`](TEAM_README.md) | change log v10c -> v13 with validation evidence, reference-approach review, rejected ideas |
| [`Documentation_template.md`](Documentation_template.md) | methodology write-up (challenge template) |
| [`experiments/`](experiments) | analysis scripts behind the numbers in TEAM_README |
| [`logs/`](logs) | training logs of the v13 cross-encoders, stacker and rescue models |
| `run_e5base.sh`, `run_e5large.sh` | stand-alone runs of the two v13 cross-encoders (`ER_WORK_DIR` must point at a work dir) |
| `validate_v11.py` | format check plus "no record linked twice" and "no cross-country link" |
| [`utils/validate_submission.py`](utils/validate_submission.py) | official format validator |
| [`eda/`](eda) | early exploratory analysis (from `main`) |

The baseline pipeline on `main` (`src/ber`, `train.py`, `predict.py`) is replaced here by the final pipeline.

## Run

```bash
pip install -r code/business_entity_resolution/requirements.txt   # PyTorch with CUDA: see the code README
cd code/business_entity_resolution
python src/run_pipeline.py --data-dir <dataset> --work-dir <work> --out-dir <out>
```

`<dataset>` must contain `train/` and `test/` with the original TSV files. The full run takes about 6 h on an
RTX PRO 4000 Blackwell (24 GB), 64 GB RAM and an 8-core CPU, and needs about 50 GB for the work dir. Training
e5-large needs about 13 GB of GPU memory (`ER_CE_GC=1` turns on gradient checkpointing for smaller GPUs).
Every step persists its results, so a run can resume with `--from <step>`. The steps are listed in the
[code README](code/business_entity_resolution/README.md).

## Output files and reproducibility

- The output TSVs (98 MB and 230 MB) and the trained weights (about 8 GB) are too large for git. They are
  attached to the release
  [`best-approach-v13`](https://github.com/ArjunTomar-G/amazon-ml-challenge/releases/tag/best-approach-v13):
  both TSVs (the v13 `matching_results.tsv` has MD5 `00e6fd366ec819dff43810d2440e6117`), the LightGBM models, the
  fine-tuned e5-small / e5-base / e5-large cross-encoders and the rescue retrievers, with SHA-256 checksums.
  Each zip unpacks to `models/...` as in the pipeline's work dir.
- The v13 file was built as deltas on the team's scored v10c file (`src/build_v11.py`, then
  `src/build_v13.py`), so each change is measured apart from retraining noise. The exact commands are in the
  code README.
- `run_pipeline.py` retrains everything from scratch. GPU training and multi-threaded LightGBM are not
  bit-reproducible: a full retrain of v11 changed about 19,500 of the 10 M test records and moved validation by
  about 5e-5. Rebuilding the TSVs from saved model outputs gives the identical file.
