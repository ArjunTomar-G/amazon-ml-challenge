# AIAgents - changes from v10c (leaderboard 0.99015) to v11 (0.9903), v12 and v13

Team: Vasu Mahajan, Saarthak Gupta, Vedant Krish Shanker. Amazon ML Challenge 2026, Business Entity
Resolution. Metric: macro F0.5 per Source-1 (S1) entity; only `matching_results.tsv` is scored, and the final
ranking uses the private leaderboard.

## Summary

v13 is the scored v10c file plus four changes, all applied as deltas (`src/build_v11.py`, then `src/build_v13.py`):

1. **Rescue v2** (v11, US / India): a second rescue pass that also covers records **without an address**,
   retrieves the top 10 S1 entities instead of 5 and adds name-uniqueness features. **+2,621 links** on test.
2. **France rule T3** (v11): unlinks France links whose name swaps one *category* word (club, ecole, comite,
   amicale ...) for another. **-730 links**.
3. **Rank rule** (v12, US / India, `--rank-t1 0.65`): an S1 that has no link yet takes its strongest record
   when 0.65 <= p < 0.8. **+179 links.**
4. **Larger cross-encoders** (v13, US / India): multilingual-e5-**base** and multilingual-e5-**large** (both MIT)
   fine-tuned as pair classifiers on a larger training set; their logits are extra features of the stacker
   and of both rescue models. **9,962 US / India decisions change** (-3,345 links, +6,598 links,
   19 re-assigned).

| | US | India | France |
|---|---|---|---|
| v10c links | 2,251,196 | 2,743,662 | 869,291 |
| v11 links (leaderboard 0.9903) | 2,251,923 | 2,745,556 | 868,561 |
| v12 links | 2,252,016 | 2,745,642 | 868,561 |
| **v13 links** | **2,252,487** | **2,748,424** | **868,561** |

Validation (US / India fold V, 440 k S1, never trained on): v10c-equivalent 0.99176 -> v11 0.99189 -> v12
0.99197 -> **v13 0.99243** (US 0.99153 -> 0.99184, India 0.99264 -> 0.99332). On the density-augmented,
test-like fold the stacked score goes 0.99104 -> 0.99147. France has no labels and is unchanged since v11.
**Expected leaderboard for v13: about 0.9907** (v12 about 0.99035, plus +0.00034 for the 9,962 applied changes
at the validated gain per changed decision). v12 and v13 were not scored: the leaderboard closed before
they were uploaded.

## Change 4 (v13) - larger cross-encoders (`src/crossenc.py` variants `b` / `l`, `src/build_v13.py`)

- **What the reference write-up used.** Its cross-encoders were multilingual-e5-base *and* e5-small, trained for
  one epoch on 1-5 M uncertain pairs plus a sample of confident pairs. Our iteration-5 e5-base test (+0.00001)
  used the e5-small recipe on a 4 GB laptop GPU: lr 1e-4, which is high for a larger model, and the e5-small
  training set (1.29 M pairs of the 0.005-0.995 band, versus 2.27 M below).
- **Recipe (v13).** Training set `ce/train_fitb.parquet` (`crossenc.py select_fit`): folds A / B, US / India,
  stage-1 band 0.002 < p1 < 0.998 (1.67 M pairs) plus 600 k confident pairs, 41 % true; two fold models as
  before (A: folds 0-1, B: folds 2-3), 1 epoch, AdamW lr 5e-5 (base) / 3e-5 (large), 3 % warm-up, bf16, frozen
  word embeddings, <= 128 pairs / 8,192 tokens per batch. RTX PRO 4000 Blackwell (24 GB): e5-base 13-15 min per
  fold model, e5-large 40-45 min; scoring the validation and test bands 9 / 25 min.
- **Validation (fold V):**

  | | stacked, plain | stacked, test-like | stacker AUC | full pipeline (rank rule + both rescues) |
  |---|---|---|---|---|
  | v12 (e5-small only) | 0.99145 | 0.99104 | 0.9647 | 0.99197 |
  | + e5-base (stacker) | 0.99168 | 0.99130 | 0.9665 | 0.99218 |
  | + e5-base (stacker and rescues) | | | | 0.99227 |
  | + e5-large (stacker) | 0.99180 | 0.99147 | 0.9676 | 0.99238 |
  | **+ e5-large (rescues) = v13** | | | | **0.99243** |

  With both extra features the first rescue pass adds +0.00043 (97 % of its links correct) and rescue v2
  +0.00054 on their own (e5-small only: +0.00030 / +0.00042). Tested and not used: a larger stacker (127 leaves,
  800 rounds; +0.00002 plain, +0.00000 test-like) and a third extra cross-encoder, e5-base reading the record
  first (stacker AUC 0.9675 vs 0.9676, stacked +0.00002).
- **Applied as a delta (`build_v13.py`).** Both pipelines of this work dir - OLD = the v12 run (e5-small
  features), NEW = the same with the larger cross-encoders - are turned into the same US / India decision (record
  argmax, 0.8, rank rule, rescue pass 1, rescue pass 2). Where they differ, the NEW decision replaces v12's only
  if v12 agreed with OLD: 14,850 records change, v12 agrees with OLD on 9,962 of them and already had the NEW
  decision on 4,777 (the team's v10c run got those right). On validation the NEW pipeline changes 3,410
  decisions for +0.00045, i.e. 0.059 S1-units per changed decision; at that rate the 9,962 applied test changes
  are worth +0.00034.
- **Where the gain comes from.** Most validation errors that remain are generator ambiguities (pseudo-word
  rebrands at an address shared with another business, true copies with a shifted house number, identical names
  without an address). The larger models read the resolvable remainder better. India gains most (+0.00068),
  US +0.00031. In a sample of the changed test links, removals are mostly look-alikes (a shifted house number or
  floor, a mutated brand token) and additions mostly true copies with typos, OCR noise or website names.

## Change 3 (v12) - rank rule (`src/build_v11.py --rank-t1`)

- **Why.** Under macro F0.5 per S1, an S1 with no predicted link scores 0 if it has true copies. A correct
  first link is therefore worth 0.6-1.0 for that S1, while a wrong one costs 1.0 only when the S1 truly has no
  copy. The break-even probability of an S1's *first* link is well below the global 0.8. Our earlier
  entity-aware threshold grid only tried *raising* that threshold. The reference approach the team was given
  uses the same idea for India (strongest link of an S1 at 0.6, further links at 0.7).
- **Validation (fold V, stacked probabilities, record argmax).** Rule "rank 1 of its S1: p >= t1, other
  records: p >= t2":

  | t1 / t2 | plain | density-augmented |
  |---|---|---|
  | 0.6 / 0.8 | +0.00010 | +0.00005 |
  | **0.65 / 0.8** | **+0.00009** | **+0.00006** |
  | 0.6 / 0.7 (reference setting) | +0.00010 | **-0.00005** |

  Gains are positive in both countries on both folds (US +0.00011 / +0.00005, India +0.00007 / +0.00006 at
  0.65 / 0.8). After both rescues the plain fold goes 0.99189 -> 0.99197. A lower threshold for rescue pairs
  of empty S1s does not help.
- **Not France.** The France records the rule would add (455 at 0.6) are almost all look-alike businesses:
  category swaps (Sport -> Club, College -> Ecole), shifted house numbers, other streets.

## Reference approach review (v12, v13)

The team was given the write-up of another system (public 0.990935). Every idea that differs from ours was
tested on our data (scripts in `experiments/b*.py`):

| idea | result on our data | used |
|---|---|---|
| India rank rule | see above; their 0.6 / 0.7 setting loses on the test-like fold, 0.65 / 0.8 gains on both | yes (US and India) |
| France ambiguity cells (per-cell thresholds from leave-one-country-out) | fold V is calibrated in every cell (p ~ precision); France's 0.5-0.8 band is look-alikes, not missed copies | no |
| France rule E (pseudo-word name at the S1's address, no other S1 there) | 20-38 % true on fold V at every margin: the false ones are rebrands of *another* S1 sharing the address | no |
| France rule F (word-swap unlink only for p < 0.99) | our T2 unlinks at any p; the copy-twin rate of the p >= 0.99 swaps is 0.9 % (look-alike range; generic-word copies 4.5-15 %) | kept T2 |
| name-edit signature / brand-mutation features | distractor brand edits are rare; stage 2 alone is 98.8 % precise on them, the stacker 99.5 % | no |
| normalisation (French abbreviations, street parsing) | French addresses with a house number parse correctly (`R.` -> rue, boulevard -> blvd) | nothing to change |
| multilingual-e5-base cross-encoder | with the reference's recipe (larger training set, lr 5e-5): +0.00023 stacked; with e5-large as well +0.00035 stacked, +0.00045 full pipeline (v13) | yes (v13) |
| blending several pipelines | the reference reports +0.0001; the team's v10c run and our rebuild disagree on only ~840 confident records | no |

France's links per S1 are 1.2-1.4 % below US / India. A census of every unlinked France record (by probability,
address relation and name relation, against US / India) shows the excess is look-alike businesses (category-word
swaps at the S1's address), other businesses sharing the S1's address and shared names without an address,
not recoverable copies. France most likely has slightly fewer copies per S1.

## How v11 / v12 were built

The 49 GB work dir was not in the v10c zip, so the whole pipeline was rebuilt on a new machine (RTX PRO 4000
Blackwell 24 GB, 64 GB RAM; about 3.5 h). The rebuild reproduces the team's numbers:

| stage | team (v10c) | rebuild |
|---|---|---|
| train candidate recall | 98.95 % | 98.95 % |
| stage 2, plain / augmented F0.5 at p >= 0.8 | 0.98961 / 0.98866 | 0.98963 / 0.98869 |
| stacker AUC on the V band | 0.965 | 0.9647 |
| stacked, plain / augmented | 0.99151 / 0.99107 | 0.99145 / 0.99104 |
| + v10c rescue (fine-tuned retriever, K=5) | 0.99181 (+1,570 links) | 0.99176 (+1,674 links) |

A full retrain moves validation by about 5e-5 by itself. So the v11 / v12 changes are applied to the **scored
v10c file** rather than shipping a retrained file. `output/` in the zip is that file.

## Change 1 (v11) - rescue v2 (`src/retriever2.py`, `src/rescue2.py`)

- **Retriever** `models/bienc2`: multilingual-e5-small fine-tuned contrastively (InfoNCE, temperature 0.05) on
  train folds A/B. Positives: all A/B blocking misses x4, including 50 k records without an address, plus 420 k
  retrieved true pairs. Hard negative: the record's most probable wrong blocking candidate by pruner probability.
  16 min on this GPU.
- **Rescue model**: v10c's recipe (cheap pre-filter, cross-encoder, cross-fitted LightGBM) with K=10, records
  with and without an address, and three new features: number of S1 with the record's core name, number with
  core name + legal form, and whether this S1 is the only one with both.
- **Validation (V, never seen)**: on its own +0.00042 (+2,198 links, 96.5 % correct), against v10c's rescue
  +0.00030. Applied after the v10c rescue, to records it left unlinked: **+0.00013** (+821 links); threshold
  0.75 / 0.8 / 0.85 give +0.00014 / +0.00013 / +0.00012.
- On records **with** an address, v10c's retriever is better (recall@1 0.73 vs 0.64 on the V blocking misses).
  v2's gain comes from K=10, the name-uniqueness features, and the records without an address.

## Change 2 (v11) - France rule T3 (`src/france.py`)

- **Category vocabulary**: derived label-free, as in `france.signatures`. It is the words that France's
  same-address one-word swaps put in at least 50 times, generic words excluded. 67 words: club, ecole, comite,
  amicale, sportive, amis, union, college, pharmacie ...
- **Rule**: a linked record whose name drops one category word and adds a different, non-typo category word is
  unlinked, whatever else changed. T2 needs an exact one-word swap at identical address numbers.
- **Label-free evidence** on the 730-786 links it removes:
  - Singleton share is 5.1 %, vs 1.9 % for all France links (true links about 2 %, false 5-9 %).
  - Copy-twin rate is 2.3 %, vs 4-9 % for the generator's noise words (fils 9.0 %, france 7.7 %, associes 5.2 %,
    groupe 4.8 %, services 4.2 %) and 0-1.3 % for look-alikes.
  - Both tests point to about two-thirds look-alike businesses, for an expected +0.0003 on France.

## What was tested in v11 and not shipped (all measured; see `experiments/`)

| idea | result |
|---|---|
| Stacker + name-uniqueness / per-order CE / confident-copy features | +0.00002 (augmented fold), within noise |
| Stacker learning curve (half the data) | -0.00002: more stacker data (cross-fitted stage 2) would not help |
| Entity-aware thresholds, per-country thresholds | at most +0.00002 on the augmented fold |
| Cross-encoder stacker applied to France (+ France rules) | +6,717 / -3,925 links, net about 0 by the singleton test; the largest additions are "shifted house number, identical name" records, which test fills with sibling groups |
| France street fix (French street types come first: "rue thiers"; 99.7 % of France records with an address but no house number had no street) | real normalisation bug, but re-scoring France moved only +367 / -217 links with mixed quality; about +0.00001 |
| Legal-form / copy-count tie-break for records without an address and a shared name | the model already uses legal form; copy counts are 53-74 % precise, below F0.5 break-even |
| "Group size" rule against test sibling groups | test's excess links sit where train groups are 82-89 % true; no cut separates them |
| ID / row-order leakage, cross-country pairs, noise shared between copies | none (0.07 % shared rare tokens) |

## Where the loss is (validation V, stacked decision before rescue, oracle gains)

| error | with address | without address |
|---|---|---|
| true pair never retrieved by blocking | +0.00077 | +0.00255 |
| record assigned to another S1 (and left unlinked) | +0.00027 | +0.00190 |
| record linked to another S1 (all records) | +0.00022 | |
| true pair is the argmax but p < 0.8 | +0.00078 | +0.00161 |
| false merges | +0.00071 | |

About 0.006 of the 0.0086 loss comes from records without an address, mostly names shared by several S1s,
which no model can resolve. The rest is spread over ambiguous house-number noise (5167 -> 516 also occurs
between look-alikes).

Test-universe checks (label-free): for US and India, v10c's test link volume per category (house-number
relation x name relation x group size) matches the train true volume within a few per cent. The one visible
excess is about 3 links per 1000 US S1 in "shifted house number, identical name" groups (test sibling groups),
and no cut separates them. France: fitted recall about 1.2 points below US; the gap is spread thinly across
categories.

## How to run

- Full pipeline (about 6 h on an RTX PRO 4000 Blackwell / 64 GB RAM, of which ~2.5 h are the larger
  cross-encoders): `python src/run_pipeline.py --data-dir <dataset> --work-dir <work> --out-dir <out>`. It writes
  the v10c-style file to `<work>/out_v10c`, the v12-style file to `<work>/out_v12` (`build_v11.py`), then trains
  e5-base / e5-large, the v13 stacker and rescue models and writes v13 with `build_v13.py`.
- v13 from the scored v12 file (how the submitted file was made; needs this work dir):
  `python src/build_v13.py --v12 <dir with v12 TSVs> --old-p feat/test_pfinal.npy --new-p feat/test_pfinal_big.npy
  --old-r1 rescue/test_rescue_pred.parquet --new-r1 rescue/test_rescue_pred_b_l.parquet
  --old-r2 rescue2/test_rescue_pred.parquet --new-r2 rescue2/test_rescue_pred_b_l.parquet --out <out>`
  (in the work dir this package was built from, the first rescue pass lives in `rescue_v10/` and the new stacker
  output in `feat/test_pfinal_bl.npy`).
- v12 from the scored v10c file: `python src/build_v11.py --v10c <dir with v10c TSVs> --out <out> --rank-t1 0.65`.
- Format check: the official `validate_submission.py`, or `validate_v11.py` (same rules plus a check that no
  record is linked twice and no link crosses countries). v13 passes.
