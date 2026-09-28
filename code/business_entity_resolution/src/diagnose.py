"""Label-free diagnostics: compare test predictions with validation, per country.

The data generator is symmetric across countries (in train, US and India have
identical per-entity statistics), so large per-country differences in the
predicted statistics on test point at a country-specific weakness.

    python diagnose.py
"""
from __future__ import annotations

import json

import numpy as np
import polars as pl

from common import log, wpath
from decide import em_prior, exclusive, expected_f_select, final_prob, threshold_select
from model import VAL_FOLD


def stats(pairs: pl.DataFrame, p: np.ndarray, e_country: np.ndarray, q_country: np.ndarray,
          entities: np.ndarray, rule, tag: str, pi_train: float):
    d = pairs.select("q_row", "e_row").with_columns(pl.Series("p", p))
    ex = exclusive(d, "p")
    sel = (threshold_select(ex, rule[1]) if rule[0] == "thr" else expected_f_select(ex, "p", rule[1]))
    out = []
    ec = e_country[entities]
    for c in np.unique(ec):
        ents = entities[ec == c]
        es = pl.DataFrame({"e_row": ents})
        s = sel.join(es, on="e_row")
        per = es.join(s.group_by("e_row").len(), on="e_row", how="left").fill_null(0)
        dq = d.filter(pl.Series(q_country[d["q_row"].to_numpy()] == c))
        pmax = dq.group_by("q_row").agg(pl.col("p").max())["p"].to_numpy()
        pc = dq["p"].to_numpy()
        n_q = int((q_country == c).sum())
        out.append(dict(
            set=tag, country=c, entities=len(ents),
            matches_per_s1=round(float(per["len"].mean()), 4),
            empty_frac=round(float((per["len"] == 0).mean()), 4),
            q_assigned_frac=round(s.height / max(1, n_q), 4),
            q_with_cand_frac=round(len(pmax) / max(1, n_q), 4),
            ambiguous_mass=round(float(((pc > 0.1) & (pc < 0.9)).mean()), 4),
            mean_p=round(float(pc.mean()), 4),
            em_prior=round(em_prior(pc.astype(np.float64), pi_train), 4),
        ))
    return out


def main():
    dec = json.load(open(wpath("models", "decision.json")))
    rule = (dec["rule"], float(dec["param"]))
    rows = []
    # validation (train universe, fold V entities)
    tp = pl.read_parquet(wpath("feat", "train_pairs.parquet"))
    kt = np.load(wpath("feat", "train_ctxkeys.npz"))
    p2t = final_prob(np.load(wpath("feat", "train_p1.npy")), np.load(wpath("feat", "train_p2.npy")))
    pi_train = float(tp["y"].mean())
    log("pair-level prior in train candidates:", round(pi_train, 4))
    ents_v = np.flatnonzero(kt["e_fold"] == VAL_FOLD)
    isv = tp["fold"].to_numpy() == VAL_FOLD
    # restrict query statistics to queries whose candidates are V entities only (approximation)
    rows += stats(tp.filter(pl.Series(isv)), p2t[isv], kt["e_country"], kt["q_country"], ents_v, rule,
                  "valid", pi_train)
    # test
    te = pl.read_parquet(wpath("feat", "test_pairs.parquet"))
    kte = np.load(wpath("feat", "test_ctxkeys.npz"))
    p2 = final_prob(np.load(wpath("feat", "test_p1.npy")), np.load(wpath("feat", "test_p2.npy")))
    rows += stats(te, p2, kte["e_country"], kte["q_country"], np.arange(len(kte["e_country"])), rule,
                  "test", pi_train)
    pl.Config.set_tbl_width_chars(250)
    print(pl.DataFrame(rows))


if __name__ == "__main__":
    main()
