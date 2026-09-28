# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** AIAgents
**Team Members:** Vasu Mahajan, Saarthak Gupta, Vedant Krish Shanker
**Submission Date:** 2026-09-27

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
After our first leaderboard score (0.970 vs 0.9906 validation) we traced the gap to two train→test
shifts and fixed both (iteration 3): **universe-invariant token statistics** (raw, rounded IDF instead
of IDF / log N, which trees had memorised per word) and a **density-robust stage 2**, selected on a
**density-simulated validation fold** (test has ~2× more distractors per entity, in groups).
Iteration 3 scored 0.9836 on the leaderboard; the remaining gap was the unlabelled country: French
generic-noise words (*… et Fils*, *Compagnie*) were rejected because the model had learnt noise words
by identity. Iteration 4 adds **label-free word-behaviour features** (per universe and country: how
often a word that a record adds keeps the reference house number vs shifts it upward), which transfer
to France: unseen-country proxy (train US only → score India) 0.929 → **0.947**.
Iteration 5 (final) adds a **fine-tuned transformer cross-encoder** (multilingual-e5-small, MIT, 118 M
parameters, trained on a 4 GB laptop GPU) on the uncertain band of stage 2, a **stacker** that combines it
with the stage-2 probability, **embedding-based rescue retrieval** for records blocking left unlinked, and
**France link rules** derived from the generator (same-address swaps to France's generic words are true
copies, swaps to any other descriptor are look-alike businesses).
Iteration 6 (v11) rebuilt the whole pipeline on new hardware and reproduced every validation figure. It then
adds **rescue v2** (a second retriever trained on records with *and without* an address, top 10, name-
uniqueness features; validation +0.00013 on top of the iteration-5 rescue) and **France rule T3**
(category-word swaps that escape T2 are unlinked; checked with the singleton and copy-twin tests). A
label-free audit of the test universe finds US / India test link volumes that match train truth category by
category, so little rule-level headroom is left there.
Iteration 7 (v12) reviews an external write-up of another system idea by idea on our data and adopts one
decision rule: an S1 without any link takes its strongest record at p >= 0.65 instead of 0.8 (US / India;
validation +0.00009, +0.00006 on the test-like fold). Its other ideas did not hold on our data (Section 5).
Iteration 8 (v13) scales the cross-encoder: **multilingual-e5-base and multilingual-e5-large** (MIT, 278 M / 560 M
parameters) fine-tuned on a larger training set (the uncertain band plus confident pairs, 2.27 M pairs) with a
lower learning rate, as extra features of the stacker and of both rescue models: validation +0.00045 (US +0.00031,
India +0.00068), test-like fold +0.00043 for the stacked decision.
Held-out validation (440 k Source-1 entities): **macro F0.5 = 0.9924** plain (iteration 7: 0.9920; iteration 5: 0.9918) and
0.9915 stacked on the test-like density-augmented fold (iteration 5: 0.9911; iteration 4: 0.9899 / 0.9887).
Public leaderboard: iteration 3 0.9836, iteration 5 (first version) 0.990, iteration 5 final (v10c)
0.99015, iteration 6 (v11) **0.9903**.

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
* **Universe-constant pitfall (found and fixed, twice).** Tree models memorise the exact value a
  universe-level statistic takes for a frequent token, and a tiny train→test shift of that value then
  flips predictions. (i) The sibling ratio of the legal token *ltd* moved −0.221 → −0.200 and “… Private
  Limited” → “… Private” pairs dropped from probability 1.0 to 0.59 (fixed in iteration 2 by clipping).
  (ii) After our first leaderboard score (0.970 vs 0.9906 validation) we traced the rest of the gap
  with adversarial validation: true records with the generator's *generic-word noise* (“Regional Steel”
  → “Regional **Services**”, “Kolkata Herbals Ltd” → “Kolkata Ltd **Center**”, French “… SARL **Cie**”,
  “**Groupe** Anatole”) are 6 % of all true pairs, scored ≥ 0.99 in validation but only ~0.70 on test
  (US). The IDF was normalised by log N (to make it universe-size independent), but that *is* the
  shift: test-US has half the entities of train-US, so every IDF moved by ~5 % (*center* 0.271 → 0.283)
  and the memorised thresholds broke (adversarial AUC between validation and test on these pairs
  0.9998). The raw IDF log(N/df) is a relative frequency and is stable between universes (*center*
  3.73 → 3.73, *services* 5.24 → 5.21), so iteration 3 uses the raw IDF rounded to 0.25 and capped at 9,
  the sibling ratio rounded to 0.5, universe counts capped at 3, and drops the absolute retrieval scores
  and candidate counts from the models (adversarial AUC 0.9998 → 0.75).
* **Denser test universe with distractor groups (found and simulated).** Test has 5.75 S2/S3 records
  per S1 entity vs 4.68 in train, i.e. ~2× more distractor records per entity, and its distractor
  businesses come as *groups* of 2-3 noisy records sharing the shifted house number (e.g. "Piedm0nt
  Third Co" and "Piedmont Third Corp", both at 22121 vs the reference's 22120; "August Constructions
  Limited / Public Limited" at 8-B vs 1-B). In training every distractor is a single record, so the
  context-based stage-2 model had learned that agreement between records signals a true entity and it
  *raised* such groups. A **density simulation** on the validation fold (every distractor record gets a
  synthetic twin in the other source with the same keys and noisy stage-1 probabilities) reproduces the
  failure: the iteration-2 stage 2 drops from 0.9906 to 0.9856 and falls *below* stage 1 alone (0.9871).
  Iteration 3 therefore trains stage 2 on the density-augmented universe (which also re-calibrates it to
  the test distractor prior) and chooses the decision rule on the augmented validation fold. The earlier
  guard is kept: stage 2 may lower any probability but never raise a pair with stage-1 probability < 0.1.

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
4. **Stage-2 matcher**: stage-1 probability + 10 density-invariant context features + top-60 stage-1
   features.
5. **Stage-3 cross-encoder** (US / India): multilingual-e5-small fine-tuned on pairs with
   0.005 < p < 0.995, two fold models; a LightGBM **stacker** combines its logit with the stage-2
   probability and the record's competition. Iteration 8: multilingual-e5-base and -large cross-encoders,
   trained on a larger set, as extra stacker features.
6. **Rescue retrieval** (US / India): records without a link are embedded with a contrastively
   fine-tuned e5 bi-encoder, top-5 Source-1 entities by cosine, cheap pre-filter, cross-encoder,
   LightGBM; links added at probability >= 0.8. **Rescue v2** (iteration 6) repeats this with a second
   bi-encoder trained on records with and without an address, top 10 and name-uniqueness features, for
   the records the first rescue left unlinked.
7. **France rules** (no labels): link same-address swaps / insertions of France's generic replacement
   words, unlink same-address swaps to any other descriptor word, links naming another street and
   (iteration 6, rule T3) links that swap one category word for another.
8. **Decisions**: exclusivity + threshold 0.8 (chosen on the density-augmented validation fold);
   iteration 7: an S1 without any link takes its strongest record at p >= 0.65 (US / India).

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
  prefix/abbreviation), IDF mass (raw IDF log(N/df), a relative frequency that is stable between
  universes, rounded to 0.25 and capped at 9 so that no per-token constant can be memorised) of
  matched / unexplained tokens on both sides, weighted Jaccard, order
  inversions, first-token match; rapidfuzz ratio / token-set / token-sort / partial / Jaro-Winkler on
  core names, glued names (domain-style names), full names and alias parts (“X d/b/a Y”); legal-form
  agreement and subset flags; **sibling statistic of unexplained tokens** (see below); record flags
  (alias, domain, handle, OCR repair, honorific, ID/phone, Indic-token counts and dictionary misses).
- *Address features:* token alignment and numeric-token agreement; rapidfuzz on full address, street
  and locality; **house-number arithmetic** — equality, absolute and **signed** difference (siblings are
  shifted upward), relative difference, digit Levenshtein / Indel distance, digit-subsequence flag
  (deletion noise), first / last digit equality, suffix agreement; state agreement (US/India codes,
  French departments→regions); unit / PO-box / PMB flags; missing-address flags.
- *Other:* retrieval ranks and relative scores (score / record's best score) of both channels — the
  absolute scores and candidate counts depend on the universe and are not used; label-free universe
  statistics capped at 3 (how many S1 entities share this name / address / house-number+street → chain
  and multi-tenant ambiguity; duplicate count among S2/S3); record source.
- **Sibling statistic** (label-free, per universe and country):
  `ratio(t) = log((c_S2S3(t)+1)/(c_S1(t)+1)) − log(N_S2S3/N_S1)`, used as `clip(ratio − 2, 0, 8)` rounded to 0.5 for the
  unexplained tokens of a pair (computed over all name tokens including legal forms). Sibling words are
  absent from S1, so the statistic is extreme for them in *every* universe — it detects French sibling
  markers (*participations, holding, distribution, SNC*) without any French labels.
- **Word-behaviour features** (iteration 4; label-free, per universe and country): for every record
  and its pruner-best Source-1 candidate, each word of the record that the candidate's name lacks is an
  *extra word*; per word we measure how often such records keep the candidate's house number
  (same-rate), shift it upward by 1–40 (up-rate) or have none (nohn-rate), from ≥ 30 occurrences,
  rounded to 0.1. Pair features: min same-rate, max up-rate, max nohn-rate over the record's extra
  words and the number of extra words with known / unknown behaviour. Validated on train labels: words
  with same-rate ≥ 0.7 (*center, services, partners, council …*) are 91–98 % true matches, words with
  same-rate ≤ 0.1 (*coastal, summit, group, holdings …*) 0–5 %. Unlike word identity (which trees had
  learnt through IDF), the behaviour of a word is measured in its own country, so French noise words
  (*fils* same-rate 0.89, *compagnie* 0.89, *associés* 0.87) and sibling words (*distribution*,
  *international*: up-rate 0.88–0.89) are recognised without French labels.
- *Stage-2 context features (density-invariant subset):* competition on the record side (max / second
  stage-1 probability, margin to the best other entity, rank, number and mass of competing entities),
  the entity's best *other* candidate, the record's rank within the entity, mutual-best flag, and the
  support of the entity's *own* house number among its other candidates. Entity-side counts / masses
  and the "support of this record's house number / name / address" consensus features were dropped:
  on test, distractor businesses come in groups that support each other, so these features change
  meaning with the distractor density (they raised sibling groups).

**Model type:** LightGBM (MIT licence, gradient-boosted trees; far below 8 B parameters).
Stage 1: 255 leaves, lr 0.05, early stopping; two models cross-fitted on S1-entity folds (A = folds
0-1, B = folds 2-3); out-of-fold probabilities for training pairs, averaged probabilities for
validation and test. Stage 2: 255 leaves, lr 0.03, trained on folds A∪B, early stopping on 5 % of
records. **Density-simulated validation:** every distractor record of the training universe gets a
synthetic twin in the other source (same keys, logit-noise N(0, 0.7) copy of its stage-1
probabilities) → ~2× distractors per entity, in groups, as on test. We use it only to *select*
(stage 1 vs stage 2, decision rule): training stage 2 on it failed because exact twins are trivially
recognisable (plain validation fell to 0.9845).

**Stage 3: transformer cross-encoder + stacker (iteration 5).** *Model:* `intfloat/multilingual-e5-small`
(MIT, 118 M parameters, BERT architecture with the XLM-R tokenizer) with a one-logit head and binary
cross-entropy. *Input:* raw text, lower-cased, `"s1 name | s1 address" </s></s> "record name | record
address"` (at most 62 tokens per side). *Training pairs:* train pairs with out-of-fold stage-1 probability
in (0.005, 0.995), 1.28 M pairs, 33 % true, split by the same S1-entity folds as stage 1: model A on folds
0-1, model B on folds 2-3. One epoch, AdamW (lr 1e-4, 3 % warm-up, linear decay, weight decay 0.01),
bf16 autocast, gradient clipping 1.0, frozen word-embedding matrix, length-sorted batches of at most 128
pairs under an 8 192-token budget (fits a 4 GB GPU with a capped memory fraction); 12 minutes per model
on an RTX 3050 Laptop GPU. A second pair of fold models with the record text first and another seed
forms a 4-model ensemble (mean logit). *Scoring:* validation pairs by the model(s) that never saw their
S1 entity, test pairs by the mean logit of all of them. *Stacker:* LightGBM (63 leaves, lr 0.05, 400 rounds) on the
stage-2 probability and logit, stage-1 probability, cross-encoder logit, its rank and gap among the
record's band pairs, the number of band pairs, the record's best other probability and rank, a few
density-invariant context features, the country and the 60 strongest pairwise stage-1 features (house-
number arithmetic, name alignment, word behaviour ...), so that it can overrule the transformer where
the numbers disagree. The validation fold is scored exactly like test at
every stage (stage 1 and the cross-encoder = mean of two fold models, stage 2 trained on A u B), so the
stacker is trained on the validation fold's 258 k band pairs with 5-fold cross-validation by S1 entity
and applied to test. On the validation band pairs: AUC stage 2 0.938, cross-encoder alone 0.919 (ensemble
0.921), stacker 0.963 (0.965 with the pairwise features). France keeps the stage-2 probability (the cross-encoder does not know France's generic words and
rejects true "... Groupe / ... Developpement" copies).

**Rescue retrieval (iteration 5).** Blocking never retrieves 1.05 % of the true pairs; 78 % of those records
have no address at all (name only, mostly chain names - unresolvable), the rest are the target. A
**contrastively fine-tuned bi-encoder** (multilingual-e5-small, `"query: name | address"`, mean-pooled,
L2-normalised; trained on train folds A/B only: true pairs with an address, the A/B blocking misses
oversampled x4, in-batch negatives + the record's most probable wrong blocking candidate as hard negative,
InfoNCE with temperature 0.05) embeds every US / India S1 entity and every record with an address and no
link; top-5 S1 entities by cosine (chunked GPU matrix product), pairs already produced by blocking dropped.
On the validation fold's blocking misses the true S1 is in the top 3 for **81.6 %** of the records
(untuned e5: 48.8 %; top-10: 88.9 % vs 57.3 %). A cheap LightGBM (cosine, rank / gap, name and address
fuzzy ratios, number agreement, the record's best existing probability; AUC 0.996) keeps 0.7 % of the
pairs and 87 % of the true ones; these are scored by the cross-encoder and a cross-fitted LightGBM, and
each record's best rescue pair is added if its probability is >= 0.8 (validation: +1 570 links, 95.5 %
correct, F0.5 +0.00030; the untuned retriever gave +1 054 links, +0.00019).

**France rules (iteration 5, label-free).** "Signature" = identical sorted address numbers, at least one
shared core word, exactly one added word. The generator's replacement operator swaps a name word for a
country-specific generic word. Two label-free tests identify France's list:
(i) *rates*: same-address swaps to a generic word occur at the same rate in every country (US 149, France
157 per 1 000 S1 entities), while swaps to any *other* real word are linked 6 times per 1 000 S1 in US and
India but 108 times in France by the stage-2 model; (ii) the **copy-twin fingerprint**: the operator picks
its word independently for every copy of an S1, so two copies of the same S1 often carry the same word
(fils 5.8 %, groupe 11.8 %, developpement 12.0 %, france 13.0 %, services 3.7 %, associes 2.9 %;
insertions of groupe / developpement / france 16-18 %), whereas look-alike businesses never pair up
(every other swap word 0-1.3 %, "service" 0.0 %, "compagnie" 0.2 %; removed links 0.3-1.3 % in every
confidence band, also at p >= 0.999). Rules on the record's best S1 entity (pa = p / max(1, sum of the
record's p)): (A) swap to *groupe / developpement / france / fils* -> link (pa > 0.001; the stage-2
model had linked 0 % of the *developpement* swaps); (A0) the same generic swap or insertion on a record
whose address has no number, best S1 at least twice as likely as the next -> link; (B) insertion of
*groupe / developpement / france* -> link (pa > 0.001; same 16-18 % twin rate whatever the model's p);
(C) *cie -> compagnie* -> link; (D) linked with pa < 0.999 but naming a different street (rare street
tokens, fuzzy ratio < 80) -> unlink; (T2) swap to another real word (>= 30 records, not generic, not an
accent / typo variant or an abbreviation of the dropped word: *freres -> frs*, *saint -> st*,
*services -> svcs* are true copies, which the cross-encoder confirms for 97-100 % of them) -> unlink.
Checked and *not* turned into rules (they are generator noise, i.e. true copies): acronyms (*Milieu Seve
Club -> MSC*), pseudo-word rebrands at the same address, website names with a glued legal form
(*zspsportsarl.com*); France's many S1 entities sharing a generic name (*Dunkerque Club* x 212) make
records without a house number genuinely ambiguous. Label-free checks on test: links per S1 entity
(France 3.41 -> 3.35; US 3.39, India 3.38; truth 3.46) and the singleton test.

**France rule T3 (iteration 6).** T2 needs the exact signature (identical address numbers, one word
dropped, one added). Look-alikes whose copy carries extra noise escape it: a second dropped word, extra
address numbers, a house-number change. France's *category* vocabulary is derived label-free per universe:
the words that the T2 population (same-address one-word swaps to a real, non-generic word) puts in at least
50 times. That gives 67 words on test: club, ecole, comite, amicale, sportive, amis, union, college,
pharmacie... T3 unlinks a linked record whose name drops one category word and adds a *different* category
word that is not a typo or abbreviation of it, whatever else changed. It removes 730 of the v10c links.
Checks: their singleton share is 5.1 % (all France links 1.9 %; true links about 2 %, false 5-9 %), and
their copy-twin rate is 2.3 %, against 4-9 % for the generator's noise words (fils 9.0 %, france 7.7 %,
associes 5.2 %, groupe 4.8 %, services 4.2 %) and 0-1.3 % for look-alikes. Both put them at about
two-thirds look-alikes.

**Rescue v2 (iteration 6).** 78 % of the blocking misses are records *without* an address, which the
first rescue skips; for 20 % of those the true entity's name is unique in the country, so the noisy name
alone can identify it ("Eastern Treval Inc" -> "Eastern Travel Inc"). A second bi-encoder (`bienc2`,
multilingual-e5-small, same InfoNCE recipe) is trained on train folds A/B. Positives: every A/B blocking
miss x4, with or without an address, plus 420 k retrieved true pairs. Hard negative: the record's most
probable wrong candidate by the *pruner* probability, available right after blocking. The rescue model
keeps the first rescue's recipe (cheap pre-filter, cross-encoder, cross-fitted LightGBM) with K = 10 and
three new features: number of S1 with the record's core name, with core name + legal form, and whether this
S1 is the only one with both. On validation it adds +0.00042 on its own (2,198 links, 96.5 % correct; the
first rescue +0.00030). Applied after the first rescue, to the records still unlinked, it adds +0.00013
(821 links). On records *with* an address the first retriever stays better (recall@1 on V blocking misses
0.73 vs 0.64). v2's gain comes from K = 10, the name-uniqueness features and the records without an
address. On test it adds 2,621 links to the v10c file.

**Rank rule (iteration 7).** Under macro F0.5 per S1, an S1 with no predicted link scores 0 when it has true
copies, so a correct *first* link is worth 0.6-1.0 for that S1, while a wrong one costs 1.0 only if the S1
truly has no copy. The break-even probability of an S1's first link is therefore well below the global 0.8
(for further links it stays near 0.8: a correct one adds about 0.07, a wrong one costs about 0.19). Rule:
a record that is the strongest (highest p) of the records whose argmax is S1 *e* is linked at p >= t1 when
*e* has no other link; all other records keep p >= 0.8. On the stacked validation probabilities t1 = 0.65
gives +0.00009 plain and +0.00006 on the density-augmented fold (t1 = 0.6: +0.00010 / +0.00005), in both
countries; the external write-up's India setting (0.6 for the first link, 0.7 for the others) gains on the
plain fold but loses 0.00005 on the augmented one. France is excluded: its 0.6-0.8 band at empty S1s is
almost entirely look-alike businesses (category swaps, shifted house numbers, other streets). On test it adds
179 links (US 93, India 86) to the v11 file.

**Larger cross-encoders (iteration 8).** The external write-up used multilingual-e5-base next to e5-small. Our
iteration-5 e5-base test had used the e5-small recipe on the laptop GPU (lr 1e-4 and the e5-small training
set) and gained nothing. Iteration 8 trains e5-base and e5-large the way that write-up describes: training set
= folds A / B, US / India, stage-1 band 0.002 < p1 < 0.998 (1.67 M pairs) plus 600 k confident pairs (41 % true,
`crossenc.py select_fit`); two fold models as before; one epoch; AdamW lr 5e-5 (base) / 3e-5 (large), 3 %
warm-up, bf16, frozen word embeddings, <= 128 pairs / 8,192 tokens per batch; 13-15 min (base) and 40-45 min
(large) per fold model on an RTX PRO 4000 Blackwell. Each model's logit, its rank and its gap among the record's
band pairs are extra stacker features (`ER_CE_EXTRA=b,l`) and extra features of both rescue models
(`ER_RESCUE_CE_EXTRA=b,l`). Stacker AUC on the validation band 0.9647 -> 0.9665 (base) -> 0.9676
(base + large); stacked F0.5 0.99145 -> 0.99180 plain and 0.99104 -> 0.99147 on the density-augmented fold;
with the rescues (first pass +0.00043 alone, 97 % of its links correct; rescue v2 +0.00054) and the rank rule
the full pipeline reaches 0.99243 (v12: 0.99197). A larger stacker (127 leaves, 800 rounds) and a third extra
cross-encoder (e5-base reading the record first) add nothing. The changes are applied to the v12 file as a delta (`build_v13.py`): both pipelines of the work dir
(OLD = e5-small features, NEW = larger cross-encoders) give a US / India decision; the NEW decision replaces
v12's where v12 agrees with OLD (9,962 of 14,850 changed records; v12 already has the NEW decision on 4,777).

**Final probability:** p = p2, except that stage 2 may not raise a pair rejected by stage 1
(p = min(p1, p2) when p1 < 0.1) — robust against the test-only sibling groups (Section 2.1).

**Threshold selection method:** (1) *Exclusivity*: a record is only kept for its highest-probability
entity. (2) The rule is chosen on the density-augmented validation fold among global thresholds and
*expected-F0.5 maximisation* per entity (with candidate probabilities p₁ ≥ p₂ ≥ … and a Poisson(λ)
number of blocking misses, E[F(k)] = Σₐ Σ_b P(TP=a)·P(FN=b)·1.25a / (0.25(a+b)+k), E[F(0)] = P(no
match), computed exactly with Poisson-binomial DP). Chosen: stage-2 probability ≥ 0.8 (best on the
augmented fold, 0.98866; expected-F0.5 was best on the plain fold). Iteration 3 additionally scaled
France's odds so that its expected matches per entity equalled the US/India value (κ = 0.39); the
word-behaviour analysis showed that France's problem was *recall* (true noise-word records rejected),
so iteration 4 removes this anchoring (`stage2.py anchor` is kept only as a diagnostic).

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro, iteration 5 = final):** **0.9917** on the held-out fold (440 499 S1 entities,
  1.53 M true pairs) and **0.9911 on the density-augmented (test-like) fold**: stage 2 0.98961 / 0.98866,
  + cross-encoder stacker 0.99135 / 0.99086, + second cross-encoder pair (ensemble) 0.99142 / 0.99093,
  + 60 pairwise features in the stacker 0.99151 / 0.99107, + rescue with the fine-tuned retriever
  0.99181 (untuned retriever: 0.99170). Leaderboard history:
  iteration 2 0.970, iteration 3 0.9836, iteration 5 0.990, iteration 5 final (v10c) 0.99015, iteration 6
  (v11) 0.9903.
- **Iteration 6 (v11), rebuilt pipeline:** stage 2 0.98963 / 0.98869, stacked 0.99145 / 0.99104, + first
  rescue 0.99176, **+ rescue v2 0.99189**. The rebuild reproduces iteration 5 within retraining noise
  (about 5e-5), so the submitted file is the scored v10c file plus the two iteration-6 changes
  (`build_v11.py`): +2,621 rescue links (US / India), -730 France links (T3).
- **Iteration 7 (v12):** + rank rule **0.99197** (augmented fold +0.00006); submitted file = v11 + 179 links.
- **Iteration 8 (v13):** + e5-base / e5-large cross-encoders **0.99243** (US 0.99153 -> 0.99184, India 0.99264 ->
  0.99332; stacked augmented fold 0.99104 -> 0.99147); submitted file = v12 + 9,962 changed US / India
  decisions (-3,345 / +6,598 links, 19 re-assigned). On validation the new pipeline changes 3,410 decisions for
  +0.00045 (0.059 S1-units each), which puts the 9,962 applied test changes at about +0.00034.
- **Stage-1 learning curve (iteration 8):** one stage-1 model on folds A + B (80 % of train S1) instead of two
  40 % models: stage-1-only F0.5 0.98806 -> 0.98829 (logloss 0.0291 -> 0.0286). Not used: it needs every later
  stage rebuilt, for an expected +0.00003-0.00005 after stacking.
- **Tested and rejected in iteration 7 (external write-up, checked on our data):**
  - per-cell thresholds for France (name sharing x address x margin, learned by leave-one-country-out): the
    validation fold is calibrated in every cell, and France's 0.5-0.8 band is look-alikes, not missed copies;
  - linking a pseudo-word name at the S1's exact address (0.1 <= p < 0.8): 20-38 % true on validation at
    every margin; the false ones are rebrands of *another* S1 sharing the address, which the singleton test
    cannot see (the wrong S1 has copies of its own);
  - unlinking word swaps only below p 0.99: the p >= 0.99 swaps T2 unlinks have a 0.9 % copy-twin rate (the
    look-alike range; generic-word copies 4.5-15 %), so T2 keeps unlinking at any p;
  - name-edit signature features for brand mutations: rare, and already 98.8 % (stage 2) / 99.5 % (stacker)
    precise on validation;
  - France recall: links per S1 are 1.2-1.4 % below US / India, but a census of every unlinked France record
    shows look-alike businesses, other businesses at the S1's address and shared names, not recoverable
    copies.
- **Tested and rejected in iteration 6 (validation, or label-free for France):**
  - new stacker features (name / legal uniqueness, per-order cross-encoder logits, confident-copy count):
    at most +0.00003 plain, +0.00002 augmented;
  - stacker learning curve: half the training band costs only 0.00002, so a bigger stacker training set is
    not worth building;
  - entity-aware thresholds (stricter for entities without another confident copy) and per-country
    thresholds: at most +0.00002 on the augmented fold;
  - cross-encoder stacker applied to France (with the France rules on top): +6,717 / -3,925 links, net
    about 0 by the singleton test. The largest additions are "shifted house number, identical name"
    records, which the test universe fills with sibling groups;
  - French street fix: street types come first in French ("rue thiers"), so 99.7 % of France records with
    an address but no house number had no street. Re-scoring France moved only +367 / -217 links, of mixed
    quality;
  - legal-form or copy-count tie-breaks for records without an address and a shared name: the model
    already uses the legal form, and copy counts are only 53-74 % precise;
  - group-size rules against test sibling groups: test's excess links sit where train's shifted groups are
    82-89 % true.
- **Test-universe structure (iteration 6, label-free):** test distractors are mostly *groups* at a shifted
  house number. US up-shifted records number 1.75 per S1 on test vs 0.9 on train, and identical-name
  groups of 3+ number 114 vs 6 per 1 000 S1. For US and India the test link volume per category (house-number
  relation x name relation x group size) matches the train true volume within a few per cent. The one visible
  excess is about 3 links per 1 000 US S1 in identical-name shifted groups. France's fitted recall is about
  1.2 points below US, spread over many small categories.
- **Where the remaining validation loss is (iteration 6, oracle fixes on the stacked plain fold before
  rescue):** never retrieved +0.00077 with an address and +0.00255 without; assigned to another entity
  +0.00027 / +0.00190, plus +0.00022 where that other link is kept; true argmax below 0.8 +0.00078 /
  +0.00161; false merges +0.00071. About 0.006 of the 0.0086 loss is records without an address, mostly
  names shared by several S1 entities.
- **Tested and rejected in iteration 5 (validation):** expected-F0.5 decisions per entity (+0.00007 plain,
  -0.00015 on the test-like fold); linking the best unlinked candidate of entities left empty (no gain);
  the stacker on France (added and removed links equally doubtful by the singleton test); rescue
  retrieval in France (fooled by France's same-name look-alikes on other streets); a larger cross-encoder
  (multilingual-e5-base, 278 M): +0.0115 AUC alone at equal training data, but +0.00001 F0.5 in the stacker,
  the same as another e5-small model - the stacker is saturated with transformer information.
- **Remaining US / India errors are generator-level ambiguity:** digit deletions (5448 -> 544) and
  pseudo-word names at the same address occur both in true copies and in look-alike businesses.
- **Cross-encoder:** fixes true pairs rejected by stage 2 and records assigned to the wrong entity
  (typos, transliteration, domain names, legal-form changes). The irreducible US / India loss (records
  without an address whose name several entities share) caps validation near 0.9915.
- **Where the remaining validation loss is** (oracle fixes on the plain fold, iteration 3): true pairs
  rejected by the model +0.0040, never retrieved +0.0034, record assigned to another entity +0.0025,
  false positives +0.0012. 68–77 % of the retrieval / assignment misses are records *without an
  address* whose name is shared by several Source-1 entities (unresolvable).
- **France (label-free):** records whose extra word keeps the reference house number are accepted like
  their US/India counterparts (*fils* 3.8 % → 85.6 %, *compagnie* 1.4 % → 58 %, *associés* 82 % → 90 %),
  while sibling words (*développement, groupe, distribution, international, participations*) are
  rejected with a shifted house number and also when they replace a word at the same house number —
  on train labels such replacements are only 20 % true.
- **Test-universe consistency (label-free):** test US and India reproduce the augmented-validation
  statistics (expected matches per entity 3.42 / 3.40 vs 3.41; uncertain pairs per entity 0.09 / 0.11
  vs 0.11–0.14).
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
| `output.py` | TSV writer helpers (iteration-4 decisions) |
| `fetch_model.py` | one-time download of multilingual-e5-small / -base / -large (MIT) into the work dir |
| `crossenc.py` | band selection, tokenisation, cross-encoder fine-tuning (GPU) and scoring; iteration 8: variants `b` / `l` (e5-base / -large), `select_fit` (larger training set), `ER_CE_LR` |
| `stack.py` | stacker: 5-fold CV on the validation fold (plain + density-augmented), test probabilities |
| `retriever.py` | contrastive fine-tuning of the rescue bi-encoder (train folds A/B) |
| `rescue.py` | embedding retrieval, pre-filter, cross-encoder scoring, rescue model |
| `retriever2.py` | iteration 6: rescue bi-encoder for records with and without an address |
| `rescue2.py` | iteration 6: rescue v2 (K = 10, records without an address, name-uniqueness features) |
| `france.py` | France signatures and link rules (iteration 6: rule T3) |
| `final.py` | final decisions -> `matching_results.tsv` and `candidate_pairs.tsv` |
| `build_v11.py` | iteration 6: rule T3 + rescue v2 applied to the v10c decisions; iteration 7: rank rule (`--rank-t1`) |
| `build_v13.py` | iteration 8: US / India decisions changed by the larger cross-encoders applied to the v12 file -> submitted files |
| `val_combo.py` | validation of rescue combinations (development tool) |
| `run_pipeline.py` | end-to-end driver |

### B. Additional Results

| experiment | validation macro F0.5 |
|---|---|
| iteration 1 (stage-1, best global threshold) | 0.98883 |
| iteration 1 (stage-2 + expected-F0.5) | 0.99088 |
| iteration 2 (robust sibling statistic, IDF normalised by universe size, canonicalisation, French parsing) | 0.99063 |
| iteration 2 + stage-2 raise guard (**public leaderboard 0.970**) | 0.99058 (augmented fold 0.9856) |
| iteration 3, stage 1 only (invariant IDF / ratio / counts, no absolute retrieval scores) | 0.98775 (augmented 0.98691) |
| iteration 3, stage 2 trained on the augmented universe (rejected: learns twin artefacts) | 0.98450 |
| iteration 3 final (density-invariant stage 2, rule chosen on augmented fold, France anchoring; **leaderboard 0.9836**) | 0.98943 (augmented 0.98806) |
| iteration 4, stage 1 only (+ word-behaviour features) | 0.98824 (augmented 0.98751) |
| **iteration 4 final** (+ word-behaviour features, no anchoring, stage-2 lr 0.05) | **0.98989 (augmented 0.98866)** |
| rejected: rule dropping same-address one-word swaps between common business words | −0.00500 (the model's selected swaps are 99.8 % precise) |
| iteration 5: + cross-encoder stacker (US / India band) | 0.99135 (augmented 0.99086) |
| iteration 5: + second cross-encoder pair (record-first order, other seed; mean logit) | 0.99142 (augmented 0.99093) |
| iteration 5: + rescue retrieval (+1 049 links, 95 % correct) + France rules (**leaderboard 0.990**) | 0.99162 |
| iteration 5: + 60 pairwise features in the stacker, abbreviation-aware France swap rule | 0.99170 (augmented 0.99107 before rescue) |
| iteration 5: + France rules validated by the copy-twin fingerprint (A0, B at pa > 0.001, "service" / "compagnie" look-alikes) | 0.99170 (France: no labels; +4 819 / -946 France links vs the 0.990 submission) |
| iteration 5 final: + contrastively fine-tuned rescue retriever (recall@3 on blocking misses 0.488 -> 0.816, top-5; **leaderboard 0.99015**) | 0.99181 |
| iteration 6: pipeline rebuilt on new hardware (stacked + first rescue) | 0.99176 |
| rejected: stacker + name-uniqueness / per-order CE / confident-copy features | +0.00002 (augmented) |
| rejected: entity-aware or per-country thresholds | at most +0.00002 (augmented) |
| iteration 6 final: + rescue v2 for the records still unlinked (+821 V links, 96 % correct) + France rule T3 (**leaderboard 0.9903**) | 0.99189 (France: -730 look-alike links, label-free) |
| rejected: external write-up's India rank rule (first link 0.6, further links 0.7) | +0.00010 plain, -0.00005 augmented |
| rejected: pseudo-word name at the S1's address -> link (0.1 <= p < 0.8) | 20-38 % precise on V |
| iteration 7 final: rank rule, first link of an S1 at p >= 0.65 (US / India, +179 test links) | 0.99197 (augmented +0.00006) |
| iteration 8: + e5-base cross-encoder (stacker; larger training set, lr 5e-5) | 0.99218 (stacked augmented 0.99130) |
| iteration 8: + e5-base in both rescue models | 0.99227 |
| rejected: larger stacker (127 leaves, 800 rounds) | +0.00002 plain, +0.00000 augmented |
| iteration 8: + e5-large cross-encoder (stacker) | 0.99238 (stacked augmented 0.99147) |
| rejected: third extra cross-encoder, e5-base with the record text first | stacked +0.00002, AUC 0.9675 vs 0.9676 |
| **iteration 8 final**: + e5-large in both rescue models | **0.99243** |

Unseen-country proxy (stage 1 trained on US only, scored on India's held-out entities): 0.929 with the
iteration-3 features, **0.947 with the word-behaviour features** (in-domain ≈ 0.988). The prior
anchoring of iteration 3 left this proxy unchanged (±0.0005).

Iteration 1 scored slightly higher on validation (0.99088) because trees exploited universe-constant
statistics; on the test universe this silently depressed India (predicted 3.19 matches / entity, 6.3 %
empty vs 3.38 / 5.7 % expected). Iteration 2 trades 0.0003 of in-distribution validation for
test-universe robustness (India 3.40 / 5.6 %).

Label-free test diagnostics per country (predicted matches per S1), iteration 4: France 3.41,
India 3.36, US 3.39 (5 858 642 matched pairs, 100 416 empty entities). Iteration 3 vs 2 added 67 k US
pairs (generic-word noise the IDF shift had pushed below the threshold) and removed 45 k India pairs
(sibling groups raised by the old consensus features); iteration 4 vs 3 adds 32.8 k and removes
13.3 k France pairs (US / India change by < 8 k each).

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.
