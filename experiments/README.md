# Experiments

Analysis scripts behind the numbers in [`../TEAM_README.md`](../TEAM_README.md). None of them is needed to
reproduce the output files. They read a finished pipeline work dir and import the pipeline modules, and were run
on the machine the team used: before running one, set `ER_WORK_DIR` to your work dir and change the
`sys.path.insert(...)` line and any `D:/...` paths at the top of the script to your checkout.

| script | question it answers | result (details in TEAM_README) |
|---|---|---|
| `val_priority.py` | v11: rescue v2 after the v10c rescue, on records it left unlinked | +0.00013 on fold V, shipped |
| `stack_exp.py` | v11: extra stacker feature groups (5-fold CV, plain and density-augmented fold V) | within noise, not shipped |
| `france_stack.py` | v11: France through the cross-encoder stacker (no labels) | net about 0 by the singleton test, not shipped |
| `france_street.py` | v11: France re-scored after the street-first normalisation fix | about +0.00001, not shipped |
| `b01_editsig.py` | one-token name edits: copies vs look-alikes, stage-2 precision | already 98.8-99.5 % precise, no feature added |
| `b03_cells.py` | reference idea "ambiguity cells": per-cell calibration | calibrated in every cell, not used |
| `b04_links_deep.py` | where France's high singleton shares come from | look-alikes, not missed copies |
| `b07_rule_e.py` | reference rule E (pseudo-word name at the S1's address) | 20-38 % true on fold V, not used |
| `b08_rank_rule.py` | rank rule thresholds, plain and test-like fold V | 0.65 / 0.8 chosen (v12) |
| `b09_rank_rescue.py` | rank rule together with both rescue passes | v12: 0.99189 -> 0.99197 |
| `b10_t2_by_p.py` | reference rule F (word-swap unlink only below p 0.99) | copy-twin rate 0.9 % at p >= 0.99, kept T2 |
| `b12_residual.py` | residual errors of the v12 decision, with raw text | sized what a stronger model could still fix |
| `b13_lc_stage1.py` | stage-1 learning curve (80 % vs 40 % of train S1) | +0.00023 stage-1 only, needs a full rebuild, not used |
| `b14_full_v.py` | full v12 / v13 decision on fold V for given stacker and rescue outputs | v13 0.99243 |
| `b15_est.py` | projected leaderboard gain of a delta file | v13 about +0.00034 over v12 |
