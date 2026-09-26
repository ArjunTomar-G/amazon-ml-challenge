# AIAgents v2 — standalone pipeline: test-like training, France remedies, decision variants

Built on the method of [`../aiagents_v1`](../aiagents_v1) (the submission that scored **0.97 on
the public leaderboard**, 0.9906 on its own validation). v2 is **self-contained**: `src/` holds its
own copy of every pipeline module, nothing is imported from `aiagents_v1`, and `run_v2.py` always
trains from a fresh, empty work dir (no cached normalisation, candidates, features or models from
an earlier run are reused). v1 itself is kept unchanged as the reference.

## Run it (one command, fresh start)

```bash
pip install -r requirements.txt
python src/run_v2.py --data-dir <student_resource/dataset> --work-dir <NEW empty dir> --out-dir <out> --selftrain
```
It writes `<out>/matching_results.tsv` + `<out>/candidate_pairs.tsv` (the models trained on test-like
data with the decision rule chosen on the test-like validation fold), and `<out>/variants/<name>/`
for leaderboard comparison: `unseen_thr0.8`, `unseen_thr0.9` (stricter rule for France), `st`,
`st+unseen_thr0.9` (self-trained France). An interrupted run resumes with `--from <step>` on the
same work dir. Needs ~24 GB RAM, ~50 GB disk and about 4–5 h on the v1 reference machine (estimate).

**Fix since the first v2 version (important if you ran it before):** `augment.py` used to create
some synthetic siblings at the entity's *own* address — 20 % in the US and 59 % in India, whenever
the entity had no house number or the sampled shift was 0. They differ from a true record only by
an added or replaced word, which is ordinary generator noise, so the model learned to reject true
matches. Now every synthetic sibling gets an upward shift of at least 1 (`--min-shift 1`), and
entities without a house number are not used as bases. On the 3 % slice, the retrained models now
match v1 on the original validation fold (0.9937 vs 0.9943; the flawed data gave 0.9925), and score
0.9937 on the test-like fold where v1 models score 0.974.

## What is wrong with v1 (measured on the data, no test labels needed)

| | train universe | test universe |
|---|---|---|
| Source-2/3 records that match nothing | 25 % | ~37–40 % (≈ 2× more hard negatives per entity) |
| unmatched records in same-address groups of 2–3 | US 5 %, India 13 % | US 39 %, India 46 %, France 52 % |
| countries with labels | US, India | US, India, **France (15 % of entities, no labels)** |

1. **The validation fold does not look like the test set.** v1 trains and validates in the train
   universe, where sibling businesses are rare single records. In test they are twice as dense
   and come in groups of noisy records sharing the shifted house number. The group records
   "support" each other, which the context model reads as a true entity (v1's raise guard patches
   one symptom). Label-free audit of the submitted file: ≈ 6.5 k (US) + ≈ 6 k (India) predicted
   pairs are up-shifted beyond the training balance, i.e. likely sibling merges.
2. **An unlabelled country loses precision.** `loco_full.py` re-fits the whole pipeline on one
   country and applies it to the other. On a 3 % slice: the held-out country drops from
   **0.994 → 0.932** (precision 0.998 → 0.930, recall unchanged). Its siblings do not carry the
   training country's cue: 40 % of India siblings keep the entity's house number, and many have no
   house number at all. France is in exactly this position.
3. The smaller signals are fine: records without address, exact-name matches and recall per
   entity are close to the training ground truth in all three countries.

On the same slice (fixed augmentation, heavier than test density): **v1 models score 0.974 on the
test-like validation fold** and 0.994 on the original one; models trained by `run_v2.py` score
0.994 / 0.994.

## Tools

| script | needs | what it does |
|---|---|---|
| `run_v2.py` | the dataset folder | the whole v2 pipeline from a fresh start (see above) |
| `audit.py` | test TSVs + any `matching_results.tsv` | label-free per-country statistics (matches / entity, empty share, up- vs down-shifted house numbers → estimated sibling merges); `--compare` profiles the pairs two files disagree on; `--train-dir` adds the ground-truth baseline |
| `variants.py` | a finished work dir | decision variants without retraining; writes `variants/<name>/matching_results.tsv` + `variants_report.json`; the `v1` variant reproduces the run's own `output` file exactly |
| `loco_full.py` | a finished work dir | leave-one-country-out through the whole pipeline; scores the held-out country with the rule it would get in production, plus EM and (`--selftrain`) self-training remedies |
| `selftrain.py` | a finished work dir | one round of cross-fitted self-training for the unlabelled test country (France); US / India probabilities stay unchanged; writes `feat/test_p{1,2}_st.npy` |
| `augment.py` | the dataset folder | writes a copy of the data whose **training** Source-2/3 files contain synthetic sibling groups at test density (Source 1, labels and test files unchanged); step 1 of `run_v2.py` |
| `crosseval.py` | two work dirs | scores one run's models on another run's validation fold (old models on the test-like fold = offline estimate of the leaderboard) |
| `probe.py` | test TSVs + a `matching_results.tsv` | leaderboard probe: blanks one country; the score drop gives that country's real F0.5 (±0.01) |
| `make_mini.py` | the dataset folder | 3 % structure-preserving slice for quick smoke tests |

### Variant grammar (`variants.py --variants ...`)
`+`-joined modifiers: `v1` · `guard<g>` (stage 2 may not raise pairs with p1 < g; v1 = 0.1) ·
`lower_only` · `em` / `emhalf` (per-country prior-shift correction, full / half strength) ·
`unseen_thr<t>` / `unseen_ef<l>` (a different rule for countries without labels, i.e. France) ·
`st` (self-trained France probabilities). Example: `st+unseen_thr0.9`.

### How augmentation works
Learned from the training data only (`augment_stats.json` records everything):
- **Exemplars.** Unmatched records on the same street as a Source-1 entity, with a house number
  0–30 above it and a similar name. The learned shift distribution is +1…+5, +7, +9, +11, +13,
  +21 (US). India exemplars are 40 % unshifted, but only shifts >= `--min-shift` (1) are used
  for synthetic siblings, see the fix above. Also learned: the name edits: added word (Northside, Holdings,
  Infratech…), replaced word, legal-form change (59 % of US siblings).
- **Groups.** A random entity's own true records (which already carry the real generator's noise)
  are used as templates for 1–3 records. One sampled edit and shift is applied identically to all
  of them, giving a group of noisy records of one sibling business.
- **Density.** Enough groups are added per country to reach `--target-unmatched` (default 0.37).

## Tools on a finished run (`WORK` = a work dir created by `run_v2.py`)

```bash
cd src
python audit.py --test-dir $DATA/test --train-dir $DATA/train --matching <out>/matching_results.tsv
python loco_full.py --work-dir $WORK --source US --target India --selftrain   # France simulation, ~2 h
python variants.py --work-dir $WORK --out-dir <out> --variants v1 unseen_thr0.9 emhalf
python probe.py --test-dir $DATA/test --matching <file> --country France --out probe_fr.tsv
```
In `models/loco_full_*.json`: `target_loco` / `target_loco_em` / `selftrain.target` versus
`target_full_model_v1_rule` is the cost of having no labels; `target_oracle_rule` shows which rule an
unlabelled country really wants. `crosseval.py --model-work A --data-work B` compares two finished runs.

Every file must pass the official validator:
`python utils/validate_submission.py --matching <file> --candidate <out>/candidate_pairs.tsv --test-dir dataset/test`

## Where the rest of the gap is (toward 0.99+)

- **US / India.** Beyond the training ground-truth rates, v1's test predictions have about 7 k (US)
  + 7 k (India) extra pairs with a sibling-style shift (+1…5, 7, 9, 11, 13, 21), worth only
  ~0.15 F0.5 in total. If those two countries score ~0.988, the 0.97 leaderboard score implies
  **France ≈ 0.87**. Measure it before investing:
  ```bash
  python probe.py --test-dir $DATA/test --matching <v1 matching_results.tsv> --country France --out probe_fr.tsv
  # submit probe_fr.tsv, then:
  python probe.py --test-dir $DATA/test --matching <v1 file> --country France --lb-original 0.970 --lb-probe <score>
  ```
- **France is structurally different.** Names follow a `<City> <Word> <legal form>` template
  (34 % of names shared by several entities). There are 9.3 entities per street name against
  2.9 in the US, and 78 % of French entities sit on streets with ≥ 10 businesses. A house-number
  typo or a sibling's +d shift often lands on another real entity. Ideas, in order of expected value:
  1. **French pseudo-training universe:** add test France (pseudo-labelled by the current best
     model) as an extra training universe, *and* run `augment.py`-style sibling mining on it. The
     exemplar miner needs only Source 1 + unassigned records, so France's own sibling edits
     (Distribution, Développement, Holding, legal swaps) are learned without labels. Iterate
     2–3 rounds.
  2. **Occupied-number features:** does the record's house number (or ±1 digit edit) belong to
     *another* Source-1 entity on the same street? How many entities sit within ±13 numbers? These
     are label-free universe statistics that transfer to dense streets. They need a feature change
     in v1's `features.py` and a full rerun.
  3. **Cluster-level decisions:** group each entity's candidates into businesses (same shifted
     number + same name edit) and accept or reject whole clusters. Train it on the augmented data.
- **Small, safe gains:** 3–5 seeds of stage 2 averaged (~+0.0005); a per-country λ of the
  expected-F0.5 rule chosen on the augmented validation.
