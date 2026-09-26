# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** AIAgents
**Team Members:** Vasu Mahajan, Saarthak Gupta, Vedant Krish Shanker
**Submission Date:** 2026-09-25

---

## 1. Executive Summary

We treat entity resolution as **many-to-one retrieval + calibrated classification + decision-theoretic
selection**. Every Source-2/3 record retrieves its most likely Source-1 entities from a Numba
inverted index (two channels, IDF-weighted composite keys), a learned pruner keeps ~1.5 candidates per
record at 98.9 % pair recall, a **cross-fitted two-stage LightGBM** scores the pairs (stage 2 adds
competition and "consensus" context features), and the final match sets are chosen per entity by
**exact expected-F0.5 maximisation**. The key innovations come from reverse-engineering the data
generator: unmatched records are *sibling businesses* (an extra/changed word + a small **upward**
house-number shift), which we detect with a **label-free, universe-specific sibling statistic** that
also works for the unseen country (France), and with signed house-number arithmetic.
Held-out validation (440 k Source-1 entities): **macro F0.5 = 0.9906**.

---

## 2. Methodology

### 2.1 Problem Analysis

Findings from EDA of the 2.2 M / 10.3 M-record training universe (all verified on the data):

* **Strict many-to-one structure.** 7.64 M true pairs; every S2/S3 id occurs in at most one ground-truth
  list. 26 % of S2/S3 records match nothing. 5.6 % of S1 entities are singletons; 3.46 matches/entity on
  average (identical for US and India → the generator is symmetric across countries).
* **The unmatched records are generated hard negatives ("siblings").** Nearly every unmatched record is a
  perturbed copy of a real S1 entity: *name* gets an extra word (Northside, Eastgate, Midtown, Greater,
  Overseas, Infratech, Holdings …), a swapped word (Logistics↔Systems) or another legal form, and the
  *house number is shifted upward by +1…+13* (2611→2613, 515→528, 603→616 …, always upward). Sibling
  marker words never occur in Source 1 (e.g. *northside*: 0 in S1, 0 in true matches, 18 818 in
  unmatched records).
* **Heavy, systematic noise in true matches** (only 22 % of matched names are equal after basic
  normalisation): alias constructions (“X d/b/a Y”, “t/a”, “f/k/a”, “formerly”), gibberish DBA names,
  domains / handles (“ufoods.com”, “@oscarresorts”, glued “Coalitionharborcom”), OCR digit confusions
  (exactly 0→o, 1→l, 5→s, 8→b, 6→g), I→l confusions (“lndia”), ID/phone suffixes, honorifics,
  word shuffles, typos, accents, **nine Indic scripts** (Devanagari, Bengali, Gurmukhi, Gujarati,
  Oriya, Tamil, Telugu, Kannada, Malayalam – ~23 % of Indian S2 names, also mixed-script), spelling
  variants (jai/jay, shree/sree), singular/plural. Addresses: component re-ordering, state code↔name,
  street-type abbreviations and typos (raod, stret, aveune), neighbourhood→city/county substitutions,
  house-number digit deletions / leading zeros / suffixes (1600d, 45bis, 1167-1169), inserted unit /
  PO-box / PMB / “NULL” tokens, and random numeric prefixes in Indian addresses (“#820”, “BLOCK C-900”).
* **No leakage.** Row order and id numbers are independent between matched records (corr ≈ 0).
* **The test universe differs:** it contains France (no labels) and ~2× more sibling records per S1
  entity (sibling-marker counts are the same in absolute terms while S1 is half the size). We verified
  that our predicted statistics on test France (3.33 matches / entity, 5.7 % empty) mirror validation
  (3.38, 5.7 %).
* **Universe-constant pitfall (found and fixed).** Statistics computed per universe take only a few
  distinct values for small vocabularies (legal forms). A tree model memorises such constants; a tiny
  train→test shift (−0.221 → −0.200 for the legal token *ltd*) flipped predictions for “… Private
  Limited” → “… Private” pairs (median probability 1.0 → 0.59). We therefore feed only a clipped,
  shift-robust version of the statistic (benign tokens map to exactly 0).
* **Sibling groups in test (found and guarded).** In training, siblings are single records; in the test
  universe many sibling businesses come as groups of 2-3 noisy records sharing the shifted house number
  (e.g. "Cascade Cardiology LLC / Co / (Ltd)" all at 531 vs the reference's 518). The context-based
  stage-2 model had learned that agreement between records and exact-core-name support signal a true
  entity with a noisy address, so it *raised* such pairs that stage 1 had correctly rejected: these
  upward overrides are 0.02 % of validation pairs (80 % correct) but 0.7 % of test-US pairs. We therefore
  let stage 2 lower any probability but never raise a pair with stage-1 probability < 0.1
  (validation cost 0.00005 F0.5; 40 763 test pairs no longer merged).

### 2.2 Solution Strategy

**Approach Type:** Hybrid — learned blocking + two-stage cross-fitted GBDT + decision-theoretic selection.
**Core Innovation:** generator-aware modelling of hard negatives (label-free sibling statistic, signed
house-number arithmetic, entity-consensus features) combined with exact expected-F0.5 set selection and
record-level exclusivity derived from the many-to-one structure.

Pipeline (each stage is a separate process; see Appendix A):

1. **Normalisation** (24 M records, multiprocessing): learned transliteration, alias/DBA splitting,
   legal-form canonicalisation, OCR repair, domain / handle segmentation, address parsing (house number,
   suffix, street, locality, state, unit), learned canonicalisation of spelling variants.
2. **Blocking**: two-channel IDF retrieval (Numba) → learned pruning.
3. **Stage-1 matcher**: ~150 pairwise features, LightGBM, cross-fitted by S1-entity folds.
4. **Stage-2 matcher**: stage-1 probability + 23 context features + top-60 stage-1 features.
5. **Decisions**: exclusivity + expected-F0.5 maximisation per entity.

---

## 3. Candidate Generation (Blocking)

*Retrieval is done from the S2/S3 side: each record needs exactly one entity (or none).*

- **Blocking keys used** (hashed, per country, IDF-weighted, frequency-capped):
  `n:` name core token, `g:` character 3-grams of the glued name (typos / glued domains),
  `f:` whole name (sorted tokens and glued form — spacing invariant), `b:` unordered name-token pairs,
  `a:` address token, `h:` house number × street token (**plus single-digit-deletion variants of the
  house number**, which catch 820↔20 / 2052↔205 noise), `s:` street token × locality token,
  `l:` name token × locality token, `x:` name token × house number.
- **Scoring:** Σ family-weight × log(1 + N/df) / log(1 + N) over shared keys (IDF normalised by the
  universe size so that scores are comparable between the 1.3 M-entity training US and the 0.26 M-entity
  test France), accumulated with a parallel Numba
  kernel over CSR posting lists. **Adaptive capping:** high-frequency unigram keys (df > 1 500) are only
  used for “starved” records that have < 8 selective keys (2× faster, no recall loss).
- **Two channels:** top-30 over all keys + top-10 over address keys only (so chain names cannot crowd out
  the right location), union → pre-filter on relative score.
- **Learned pruner:** LightGBM on cheap features (Numba token-alignment scores, hash equalities of house
  number / street / name / state / locality, both channel scores and ranks) keeps the top-10 per record
  with probability ≥ 1e-4.
- **Candidate pairs generated:** train 14,465,300 pairs (≈1.4 per record), test 15,762,788 pairs.
- **How true matches were not lost:** recall of the final candidate set on *all* 7.64 M training pairs
  is **98.95 %** (retrieval alone: 99.16 %). Most remaining misses are records with **no address and a
  chain name** (the true entity's exact name is shared by a median of 58 S1 entities) — they are
  unresolvable by construction.

---

## 4. Matching Model

**Features used (≈150 stage-1 + 23 context):**

- *Name features:* Numba soft token alignment over a per-country vocabulary (Levenshtein ≥ 0.75 or
  prefix/abbreviation), IDF mass (IDF / log N, comparable across universes) of matched / unexplained
  tokens on both sides, weighted Jaccard, order
  inversions, first-token match; rapidfuzz ratio / token-set / token-sort / partial / Jaro-Winkler on
  core names, glued names (domain-style names), full names and alias parts (“X d/b/a Y”); legal-form
  agreement and subset flags; **sibling statistic of unexplained tokens** (see below); record flags
  (alias, domain, handle, OCR repair, honorific, ID/phone, Indic-token counts and dictionary misses).
- *Address features:* token alignment and numeric-token agreement; rapidfuzz on full address, street
  and locality; **house-number arithmetic** — equality, absolute and **signed** difference (siblings are
  shifted upward), relative difference, digit Levenshtein / Indel distance, digit-subsequence flag
  (deletion noise), first / last digit equality, suffix agreement; state agreement (US/India codes,
  French departments→regions); unit / PO-box / PMB flags; missing-address flags.
- *Other:* retrieval scores and ranks of both channels; label-free universe statistics (how many S1
  entities share this name / address / house-number+street → chain and multi-tenant ambiguity;
  duplicate count among S2/S3); record source.
- **Sibling statistic** (label-free, per universe and country):
  `ratio(t) = log((c_S2S3(t)+1)/(c_S1(t)+1)) − log(N_S2S3/N_S1)`, used as `clip(ratio − 2, 0, 8)` for the
  unexplained tokens of a pair (computed over all name tokens including legal forms). Sibling words are
  absent from S1, so the statistic is extreme for them in *every* universe — it detects French sibling
  markers (*participations, holding, distribution, SNC*) without any French labels.
- *Stage-2 context features:* competition on the record side (max/second stage-1 probability, margin to
  the best other entity, rank), entity side (number / mass of candidates, rank, strong candidates) and
  **consensus**: probability mass of the entity's *other* candidates sharing this record's house number,
  street, name or address, and the support of the entity's own house number — an isolated record with a
  shifted house number is a sibling, a noisy record that agrees with the other records is a match.

**Model type:** LightGBM (MIT licence, gradient-boosted trees; far below 8 B parameters).
Stage 1: 255 leaves, lr 0.05, early stopping; two models cross-fitted on S1-entity folds (A = folds
0-1, B = folds 2-3); out-of-fold probabilities for training pairs, averaged probabilities for
validation and test. Stage 2: 255 leaves, lr 0.03, trained on folds A∪B, early stopping on 5 % of records.

**Final probability:** p = p2, except that stage 2 may not raise a pair rejected by stage 1
(p = min(p1, p2) when p1 < 0.1) — robust against the test-only sibling groups (Section 2.1).

**Threshold selection method:** no global threshold. (1) *Exclusivity*: a record is only kept for its
highest-probability entity. (2) *Expected-F0.5 maximisation* per entity: with candidate probabilities
p₁ ≥ p₂ ≥ … and a Poisson(λ) number of blocking misses,
E[F(k)] = Σₐ Σ_b P(TP=a)·P(FN=b)·1.25a / (0.25(a+b)+k), E[F(0)] = P(no match at all), computed exactly
with Poisson-binomial dynamic programming; k* = argmax. λ and the rule were chosen on the validation
fold (expected-F beat the best global threshold).

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** **0.9906** on the held-out fold (440 499 S1 entities, 1.53 M true pairs).
  India 0.9904, US 0.9907. Stage-1 alone with the best threshold: 0.9884.
- **Precision / recall (validation):** precision ≈ 0.999, recall ≈ 0.973; 99.4 % of true singletons are
  correctly predicted empty.
- **Common false positives (wrong merges):** very rare (~0.1 % of predicted pairs): siblings whose only
  change is a house-number shift with an otherwise identical name, and records with a gibberish DBA
  name at a multi-tenant address.
- **Common false negatives (missed matches):** records **without an address** whose name is shared by
  several S1 entities (unresolvable), gibberish DBA names at an address, and heavily typo'd names — the
  expected-F0.5 rule deliberately leaves these out because F0.5 punishes false merges twice as hard.

---

## 6. Conclusion

Understanding *how* the data was generated mattered more than model size: modelling the sibling
generator (label-free marker statistic, signed house-number shift, consensus of an entity's records),
exploiting the many-to-one structure (record-side retrieval, exclusivity, competition features) and
optimising the metric exactly (expected-F0.5 per entity) give a precision-first system that transfers
to an unseen country. A key lesson: universe-level statistics must be fed in a shift-robust form, or
tree models memorise them as constants.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/` — run everything with

```
python src/run_pipeline.py --data-dir <student_resource/dataset> --work-dir <scratch> --out-dir <output>
```

| module | role |
|---|---|
| `common.py` | paths (env vars), IO |
| `translit.py` | learned Indic→Latin token dictionary (co-occurrence × phonetic similarity), anyascii fallback |
| `geo.py` | state / region canonicalisation tables (open-set safe) |
| `normalize.py` | name & address normalisation |
| `prep.py` | stage-1 driver: parquet, normalisation (multiprocessing), domain segmentation, canonicalisation |
| `canon.py` | learned spelling-variant / OCR / glue canonicalisation |
| `blocking.py` | key families, Numba retrieval kernel, two channels |
| `candidates.py` | learned pruner → final candidate set |
| `features.py` | token spaces, Numba alignment kernels, rapidfuzz & house-number features |
| `model.py` | folds, stage-1 cross-fitting, out-of-fold probabilities |
| `context.py`, `stage2.py` | context features, stage-2 model, validation, rule choice |
| `decide.py` | exclusivity, expected-F0.5 DP, macro-F0.5, EM prior diagnostics |
| `output.py` | writes `matching_results.tsv` and `candidate_pairs.tsv` |
| `run_pipeline.py` | end-to-end driver |

### B. Additional Results

| experiment | validation macro F0.5 |
|---|---|
| iteration 1 (stage-1, best global threshold) | 0.98883 |
| iteration 1 (stage-2 + expected-F0.5) | 0.99088 |
| iteration 2 (robust sibling statistic, IDF normalised by universe size, canonicalisation, French parsing) | 0.99063 |
| **final** (iteration 2 + stage-2 raise guard) | **0.99058** |

Iteration 1 scored slightly higher on validation (0.99088) because trees exploited universe-constant
statistics; on the test universe this silently depressed India (predicted 3.19 matches / entity, 6.3 %
empty vs 3.38 / 5.7 % expected). Iteration 2 trades 0.0003 of in-distribution validation for
test-universe robustness (India 3.40 / 5.6 %).

Validation detail (final): 1 487 825 predicted pairs, 1 797 false positives (precision 0.9988),
40 763 false negatives (16 080 never reached the candidate set, 24 683 rejected by the model).

Label-free test diagnostics per country (predicted matches per S1 / empty rate): France 3.33 / 5.7 %,
India 3.40 / 5.6 %, US 3.31 / 6.0 %; validation: 3.38 / 5.7 %.

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.
