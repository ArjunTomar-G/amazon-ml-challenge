# Business entity resolution pipeline

End-to-end: raw TSVs → normalised Parquet → candidate generation (blocking) →
pair features → LightGBM match probability → threshold → submission files.
Everything is learned from the provided training data only (no external data,
no pretrained models); the only model is LightGBM (MIT licence).

## Run it

```bash
pip install -r requirements.txt            # Python 3.10+; tested with 3.12
python src/train.py                        # train + choose threshold on a holdout
python src/predict.py                      # write output/*.tsv and run the validator
# or: bash run_all.sh
```

Paths default to the repository layout (`<repo>/dataset`, `<repo>/work`,
`<repo>/output`). Override with `--data-dir / --work-dir / --out-dir` or the
environment variables `BER_DATA`, `BER_WORK`, `BER_OUT`.

`predict.py` needs the three files `train.py` writes to the work dir:
`model.txt`, `model_config.json` (features, threshold, blocking settings) and
`translit.json`. Copy that folder to another machine to predict without retraining.

| Step | Output | Time* | Peak RAM* |
|---|---|---|---|
| normalise train / test (first run only, cached) | `work/{train,test}_s{1,2,3}.parquet` | ~3 + 5 min | < 1 GB |
| `train.py` (150k + 50k Source-1 entities → 7.2M labelled pairs) | `work/model.txt`, `model_config.json` | ~22 min (LightGBM ~12 of it) | ~4 GB |
| `predict.py` (1.73M test entities, ~36 candidates each) | `output/matching_results.tsv`, `output/candidate_pairs.tsv` | ~1 h 45 min | ~3.5 GB |

*Measured on a MacBook Air M1, 8 GB RAM, 8 threads. More cores help almost linearly
(numba kernels and LightGBM are multi-threaded; set `--threads`). If memory is
tight, lower `--chunk` (predict / train) or `--n-train`.

Useful flags: `--threads N`, `--chunk N` (Source-1 entities per feature batch),
`train.py --n-train/--n-holdout`, retrieval settings (`--k-joint`, `--k-mh`,
`--k-empty`, `--cap-joint`, ...; stored in `model_config.json` and reused by
`predict.py`), `predict.py --limit N` (quick smoke run on the first N entities),
`predict.py --threshold T` (override the tuned threshold).

## Method

**Normalisation** (`ber/textnorm.py`, `ber/data.py`): Unicode NFKC + unidecode
(accents and nine Indic scripts → ASCII), lower-case, alias markers (aka/dba/
formerly), URL scaffolding (`www.`, `.com`), phone numbers and punctuation
removed; addresses split into components, NULL placeholders dropped, street
types / ordinals / US & Indian state names canonicalised.

**Indic dictionary** (`ber/translit.py`): 23% of India Source-2 names are
written in Indic scripts; `unidecode` gives distorted spellings
(`phyuucr investtmentts praaivett limittedd`). Aligning those names token-by-token
with their matched Source-1 names in the training ground truth yields a
760-entry dictionary (`innnvesttmenntts → investments`), applied to all names.

**Candidate generation** (`ber/retrieval.py`), per country (records never
match across `country` labels), union of three channels:

1. *joint retrieval* — IDF-weighted overlap of name tokens and canonical address
   tokens (keys in more than `cap_joint` records are purged); top `k_joint`;
2. *fuzzy-name channel* — char-3-gram MinHash (16 bands × 2 rows) of the compact
   name, ranked by colliding bands (≥ 2); top `k_mh` — catches concatenated /
   domain-style names and typos;
3. *no-address channel* — joint scores restricted to Source-2/3 records without an
   address; top `k_empty`.

On a 50k-entity training holdout: 36.1 candidates per Source-1 entity, 96.5% of
true pairs retrieved, F0.5 ceiling 0.987.

**Features** (`ber/features.py`, 44): Jaro-Winkler, token sort/set ratio,
Levenshtein (rapidfuzz) on core names (legal suffixes and honorifics removed, Indic
dictionary applied), full names, suffix-canonicalised names and canonical
addresses; token and char-3-gram Jaccard; IDF-weighted char-3-gram and token
cosines (IDF per country, computed from that split's own records);
house-number agreement; legal-form agreement; missing-address / Indic flags;
retrieval scores and ranks; and entity-relative gaps (feature minus the best
value among the same entity's candidates), which let the model recognise
Source-1 entities with no true match.

**Model and decision**: LightGBM binary classifier trained on labelled
candidate pairs of a random entity sample; early stopping on a disjoint
holdout, where the probability threshold maximising macro F0.5 (singletons
included) is chosen. `predict.py` outputs every candidate with probability
≥ threshold.

**Holdout result** (50k training entities never used for fitting): **macro F0.5
0.957** at threshold 0.74 (singletons 0.953, matched entities 0.957). LightGBM
was still improving slowly at the 2,000-round cap; `--rounds 4000` (or
`--lr 0.1`) may add a little.

## Files

```
src/train.py          training entry point
src/predict.py        test prediction + submission files
src/ber/data.py       paths, TSV streaming, normalisation to Parquet, ground truth
src/ber/textnorm.py   name / address normalisation
src/ber/translit.py   learned Indic→Latin token dictionary
src/ber/retrieval.py  blocking: key functions, postings index, numba top-K kernel
src/ber/features.py   pair features (rapidfuzz + numba kernels)
src/ber/pipeline.py   per-country driver shared by train / predict
src/ber/metrics.py    macro F0.5 and threshold search
```
