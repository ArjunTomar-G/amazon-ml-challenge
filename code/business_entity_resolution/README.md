# Business Entity Resolution — reproducible pipeline

Blocking + learned pruning + two-stage cross-fitted LightGBM + a fine-tuned
transformer cross-encoder on the uncertain band (with a LightGBM stacker) +
embedding-based rescue retrieval + label-free France rules. No external data or
APIs are used: every dictionary (transliteration, spelling variants) is learned
from the provided training data, and every statistic is computed from the
provided files. The only pretrained model is `intfloat/multilingual-e5-small`
(MIT licence, 118M parameters), downloaded once by the `fetch-model` step and
fine-tuned on the training pairs.

## Environment

* Python 3.12 (tested 3.12.4 on Windows 11; Linux/macOS work the same)
* 16 CPU threads recommended; **~16 GB free RAM**; ~50 GB free disk for the work dir
* a CUDA GPU with >= 4 GB for stage 3 (tested on an RTX 3050 Laptop, 4 GB, bf16)
* `pip install -r requirements.txt` (PyTorch with CUDA:
  `pip install torch==2.4.1+cu124 --extra-index-url https://download.pytorch.org/whl/cu124`)
* v11 was rebuilt and run with Python 3.11.9, torch 2.13.0+cu130 and transformers 5.15 on an
  RTX PRO 4000 Blackwell (24 GB; torch 2.4.1 has no kernels for Blackwell GPUs), 64 GB RAM, 8-core i7-10700.
* v13's e5-large cross-encoder needs about 13 GB of GPU memory during training (`ER_CE_GC=1` turns on gradient
  checkpointing for smaller GPUs, at some speed cost).
  All other pins as in `requirements.txt`.

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
| `feats-train`, `feats-test` | `model.py`, `features.py` | ~145 pairwise features for every candidate pair (universe-invariant token statistics, label-free word-behaviour features) |
| `stage1` | `model.py` | cross-fitted stage-1 LightGBM models (A/B) + out-of-fold / averaged probabilities |
| `ctx-train`, `ctx-test` | `stage2.py`, `context.py` | context features (competition + consensus) from stage-1 probabilities; for train also the *density-augmented* universe (a synthetic twin for every distractor record) |
| `stage2` | `stage2.py`, `decide.py` | stage-2 LightGBM on density-invariant context features; validation on held-out S1 entities (plain and density-augmented); stage-1-vs-stage-2 and decision rule chosen on the augmented, test-like universe |
| `predict-test` | `stage2.py` | stage-2 probabilities on test |
| `fetch-model` | `fetch_model.py` | one-time download of multilingual-e5-small (MIT) into `<work>/hf` |
| `ce-select`, `ce-tokenize` | `crossenc.py` | uncertain-band pairs (train: out-of-fold stage-1 p in (0.005, 0.995); validation / test: stage-2 p), pre-tokenised |
| `ce-train-A`, `ce-train-B` | `crossenc.py` | fine-tune the cross-encoder on each fold's pairs (1 epoch, bf16, ~12 min each on a 4 GB GPU) |
| `ce-score` | `crossenc.py` | cross-encoder logits (validation: model(s) that never saw the S1 entity; test: mean of both) |
| `stack-cv`, `stack-fit` | `stack.py` | LightGBM stacker on the validation fold (5-fold CV, plain + density-augmented validation) → stacked test probabilities |
| `rescue-retriever` | `retriever.py` | contrastively fine-tune the e5-small bi-encoder on train folds A/B (~33 min) |
| `rescue-*` | `rescue.py` | embeddings of S1 entities and unlinked records (fine-tuned retriever), top-5 cosine retrieval, pre-filter, cross-encoder, rescue model |
| `rescue2-retriever` | `retriever2.py` | v11: second bi-encoder, trained on records with and without an address (~16 min) |
| `rescue2-*` | `rescue2.py` | v11: rescue v2 - every US / India record embedded, top-10 retrieval for the unlinked ones, name-uniqueness features, pre-filter, cross-encoder, rescue model |
| `final-v10c` | `final.py`, `france.py` | v10c decisions: France rules A,A0,B,C,D,T2, exclusivity + threshold 0.8, v10c rescue links -> `<work>/out_v10c` |
| `final-v12` | `build_v11.py`, `france.py` | v11: France rule T3 and rescue v2 (records still unlinked, p >= 0.8) on top of the v10c decisions; v12: rank rule (`--rank-t1 0.65`, US / India: an S1 with no link takes its strongest record at p >= 0.65) -> `<work>/out_v12` |
| `fetch-base`, `fetch-large` | `fetch_model.py` | v13: multilingual-e5-base / -large (MIT) -> `<work>/hf` |
| `ce-select-big` | `crossenc.py select_fit` | v13: larger training set: stage-1 band 0.002-0.998 of folds A/B plus 600 k confident pairs (2.27 M) |
| `ce-*-b`, `ce-*-l` | `crossenc.py` | v13: e5-base (lr 5e-5, 13-15 min per fold model) and e5-large (lr 3e-5, 40-45 min) cross-encoders, tokenise / 2 fold models / scoring |
| `stack-cv-big`, `stack-fit-big` | `stack.py` | v13: stacker with the e5-base and e5-large logits as extra features -> `feat/test_pfinal_big.npy` |
| `rescue-ce-*-b/l`, `rescue2-ce-*-b/l`, `rescue*-model-big` | `rescue.py`, `rescue2.py` | v13: e5-base and e5-large logits as extra features of both rescue models -> `test_rescue_pred_b_l.parquet` |
| `final` | `build_v13.py` | v13: the US / India decisions that the larger cross-encoders change (old vs new pipeline of this work dir), applied where the v12 file agrees with the old decision -> the two TSV files |

The submitted v13 file was made from the scored v12 file (itself the scored v10c file plus the v11 / v12 changes;
a full retrain moves validation by a few 1e-5 by itself):

```bash
python src/build_v11.py --v10c <dir with v10c TSVs> --out ./output_v12 --rank-t1 0.65
python src/build_v13.py --v12 ./output_v12 --out ./output     --old-p feat/test_pfinal.npy --new-p feat/test_pfinal_big.npy     --old-r1 rescue/test_rescue_pred.parquet --new-r1 rescue/test_rescue_pred_b_l.parquet     --old-r2 rescue2/test_rescue_pred.parquet --new-r2 rescue2/test_rescue_pred_b_l.parquet
```

(`ER_WORK_DIR` must point at a work dir that holds `feat/`, `norm/`, `rescue/` and `rescue2/` of the same data.
Without `--rank-t1` build_v11.py writes the v11 file.)

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
* `output.py` – TSV writer helpers
* `fetch_model.py` – one-time download of the pretrained transformer
* `crossenc.py` – stage-3 cross-encoder (band selection, tokenisation, training, scoring)
* `stack.py` – stacker over stage-2 probability + cross-encoder logit
* `retriever.py` – contrastive fine-tuning of the rescue retriever (v11: its recall gate no longer needs the rescue embeddings, so a fresh run works)
* `rescue.py` – embedding-based rescue retrieval (`ER_RESCUE_DIR` selects the output folder)
* `retriever2.py`, `rescue2.py` – v11 rescue v2 (records with and without an address, K = 10)
* `france.py` – France signatures and link rules (v11: rule T3)
* `final.py` – final decisions and TSV files (v11: several rescue folders, `--pfile`)
* `build_v11.py` – v11 changes (and the v12 rank rule, `--rank-t1`) applied to v10c decisions
* `build_v13.py` – v13: decisions changed by the e5-base / e5-large cross-encoders applied to the v12 file
* `run_pipeline.py` – end-to-end driver

Development / analysis tools (not needed to reproduce the outputs):

* `analyze.py` – false-positive / false-negative samples on the validation fold
* `diagnose.py` – label-free per-country comparison of test predictions with validation
* `loco.py` – leave-one-country-out experiment (simulates an unlabelled country such as France)
* `val_combo.py` – validation of rescue combinations on fold V

## Runtime (reference machine: 12-core i5-12500H, 24 GB RAM)

| step | time |
|---|---|
| prep (24M records) | ~12 min (first run) |
| pruner | ~5 min |
| cand-train / cand-test | ~35 min each |
| feats-train / feats-test | ~8 min each |
| stage1 (+ probabilities) | ~60 min |
| ctx (incl. augmented universe), stage2, predict | ~25 min |
| cross-encoder (tokenise, 2 fold models, scoring) | ~40 min (RTX 3050 Laptop) |
| stacker, rescue (embeddings, retrieval, scoring), final | ~55 min |

v11 rebuild (8-core i7-10700, 64 GB RAM, RTX PRO 4000 Blackwell): prep 14 min, pruner 7 min, cand-train /
cand-test 43 / 42 min, feats 5 + 5 min, stage1 49 min, ctx + stage2 + predict 21 min, four cross-encoders
+ scoring 24 min, stacker 2 min, rescue (both) about 45 min, final 1 min. v13 on the same machine: e5-base
select + tokenise + 2 fold models + scoring 41 min, e5-large 2 h 15 min, stacker 8 min, rescue e5-base / e5-large
features and models 35 min, build_v13 1 min.
