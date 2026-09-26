# Business Entity Resolution — reproducible pipeline

Blocking + learned pruning + two-stage cross-fitted LightGBM + expected-F0.5
decision layer. No external data, APIs or pretrained models are used: every
dictionary (transliteration, spelling variants) is learned from the provided
training data, and every statistic is computed from the provided files.

## Environment

* Python 3.12 (tested 3.12.4 on Windows 11; Linux/macOS work the same)
* 16 CPU threads recommended; **~16 GB free RAM**; ~40 GB free disk for the work dir
* `pip install -r requirements.txt`

All libraries are permissively licensed (MIT / Apache-2.0 / BSD / ISC). The
matching model is LightGBM (MIT) — gradient-boosted trees, far below the 8B
parameter limit.

## Run end-to-end

```bash
python src/run_pipeline.py --data-dir <path>/student_resource/dataset --work-dir ./work --out-dir ./output
```

`--data-dir` must contain `train/` and `test/` with the original TSV files.
Outputs: `output/matching_results.tsv` and `output/candidate_pairs.tsv`.

Resume from a step with `--from <step>`, or run a subset with `--only <step> ...`.
Every step runs in its own process and persists its results in `--work-dir`.

| step | module | what it does |
|---|---|---|
| `prep` | `prep.py`, `normalize.py`, `translit.py`, `geo.py` | TSV→parquet; learn the Indic→Latin transliteration dictionary from train pairs; normalise names/addresses of all 24M records; segment glued domain names |
| `pruner` | `candidates.py` | train the LightGBM candidate pruner on a 4% train sample |
| `cand-train`, `cand-test` | `blocking.py`, `candidates.py` | two-channel IDF inverted-index retrieval (Numba) + learned pruning → final candidate set |
| `feats-sample` | `model.py`, `features.py` | ~150 pairwise features for a 15% query sample |
| `stage1` | `model.py` | cross-fitted stage-1 LightGBM models (A/B) |
| `oof-train`, `oof-test` | `model.py` | features + out-of-fold stage-1 probabilities for every candidate pair |
| `ctx-train`, `ctx-test` | `stage2.py`, `context.py` | context features (competition + consensus) from stage-1 probabilities |
| `stage2` | `stage2.py`, `decide.py` | stage-2 LightGBM, validation on held-out S1 entities, choice of decision rule |
| `predict-test` | `stage2.py` | stage-2 probabilities on test |
| `output` | `output.py` | exclusivity + expected-F0.5 decisions → the two TSV files |

## Validate the output format

From `student_resource/`:

```bash
python utils/validate_submission.py --matching <out>/matching_results.tsv --candidate <out>/candidate_pairs.tsv --test-dir dataset/test
```

## Source layout (`src/`)

* `common.py` – paths (env vars `ER_DATA_DIR`, `ER_WORK_DIR`, `ER_OUT_DIR`), IO helpers
* `translit.py` – learned Indic-script → Latin token dictionary (+ anyascii fallback)
* `geo.py` – state / region canonicalisation tables (US, India, France; open-set safe)
* `normalize.py` – name and address normalisation (aliases/DBA, legal forms, OCR digits, house numbers, units, states …)
* `prep.py` – stage 1 driver (parquet, normalisation, glued-name segmentation)
* `canon.py` – learned spelling-variant / OCR / glued-word canonicalisation
* `blocking.py` – inverted-index retrieval kernel (Numba), key families, two channels
* `candidates.py` – learned candidate pruner → final candidate set
* `features.py` – token spaces, Numba token-alignment kernels, rapidfuzz and house-number features
* `model.py` – folds, stage-1 cross-fitting, out-of-fold probabilities
* `context.py` – stage-2 context features
* `stage2.py` – stage-2 model, validation, decision-rule selection
* `decide.py` – exclusivity, expected-F0.5 maximisation, macro-F0.5 metric
* `output.py` – TSV writer
* `run_pipeline.py` – end-to-end driver

Development / analysis tools (not needed to reproduce the outputs):

* `analyze.py` – false-positive / false-negative samples on the validation fold
* `diagnose.py` – label-free per-country comparison of test predictions with validation
* `loco.py` – leave-one-country-out experiment (simulates an unlabelled country such as France)

## Runtime (reference machine: 12-core i5-12500H, 24 GB RAM)

| step | time |
|---|---|
| prep (24M records) | ~12 min (first run) |
| pruner | ~5 min |
| cand-train / cand-test | ~35 min each |
| feats-train / feats-test | ~3 min each |
| stage1 (+ probabilities) | ~40 min |
| ctx, stage2, predict, output | ~20 min |
