"""Stacker feature experiments (5-fold CV by S1 on the validation band, plain + density-augmented fold V).

    python stack_exp.py                 baseline (the v10c stacker features) vs extra feature groups

Extra groups (all pair or record-side quantities; none depends on the distractor density):
  nl   record-name uniqueness: #S1 of the country with the record's core name, with core name + legal form,
       and whether this S1 is the only one with the record's core name + legal form
  ce2  the two cross-encoder input orders separately and their disagreement
  conf number of OTHER records of the entity with stage-2 p >= 0.9 (confident copies)
"""
from __future__ import annotations

import os
import sys

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

import stack as S
from blocking import true_pairs_rows
from common import Timer, log, wpath
from model import VAL_FOLD


def name_stats(split: str):
    s1 = pl.read_parquet(wpath("norm", f"{split}_source1.parquet"), columns=["country", "n_core", "n_legal"]).with_row_index("e_row")
    q = pl.concat([pl.read_parquet(wpath("norm", f"{split}_{s}.parquet"), columns=["country", "n_core", "n_legal"])
                   for s in ("source2", "source3")]).with_row_index("q_row")
    a = s1.group_by(["country", "n_core"]).len().rename({"len": "k_name"})
    b = s1.group_by(["country", "n_core", "n_legal"]).len().rename({"len": "k_nl"})
    q = q.join(a, on=["country", "n_core"], how="left").join(b, on=["country", "n_core", "n_legal"], how="left").fill_null(0)
    return s1, q


def nl_block(split, q_rows, e_rows):
    s1, q = name_stats(split)
    d = pl.DataFrame({"q_row": q_rows.astype(np.int64), "e_row": e_rows.astype(np.int64)}).with_row_index("i")
    d = d.join(q.with_columns(pl.col("q_row").cast(pl.Int64)).rename({"n_core": "qn", "n_legal": "ql", "country": "c"}), on="q_row", how="left")
    d = d.join(s1.with_columns(pl.col("e_row").cast(pl.Int64)).rename({"n_core": "en", "n_legal": "el"}).drop("country"), on="e_row", how="left")
    d = d.sort("i").with_columns(((pl.col("qn") == pl.col("en")) & (pl.col("ql") == pl.col("el")) & (pl.col("k_nl") == 1)).cast(pl.Float32).alias("u"))
    return np.column_stack([np.minimum(d["k_name"].to_numpy(), 50), np.minimum(d["k_nl"].to_numpy(), 50), d["u"].to_numpy()]).astype(np.float32)


def conf_block(pairs: pl.DataFrame, pf: np.ndarray, rows: np.ndarray):
    d = pairs.select("e_row").with_columns(pl.Series("c", (pf >= 0.9).astype(np.int32)))
    d = d.with_columns((pl.col("c").sum().over("e_row") - pl.col("c")).alias("k"))
    return d["k"].to_numpy()[rows].astype(np.float32)[:, None]


def main():
    keys = np.load(wpath("feat", "train_ctxkeys.npz"))
    entities = pl.DataFrame({"e_row": np.flatnonzero(keys["e_fold"] == VAL_FOLD)})
    truth = true_pairs_rows().filter(pl.col("e_row").is_in(entities["e_row"].implode()))
    ctx = np.load(wpath("feat", "train_ctx.npy"), mmap_mode="r")
    band = S._band_frame()
    ce0 = np.load(wpath("ce", "train_val_logit.npy"))
    cer = np.load(wpath("ce", "train_val_logitr.npy"))
    pairs, rows, p1, pf = S._val_universe("train")
    pos = np.searchsorted(rows, band["row"].to_numpy())
    india = (keys["q_country"][pairs["q_row"].to_numpy()] == "India")
    with Timer("base features"):
        X0 = S.features(pairs, pf, p1, S._Rows(ctx, rows), pos, band["ce"].to_numpy(), india)
        X0 = np.hstack([X0, S.pair_block("train", band["row"].to_numpy())])
    groups = {
        "nl": nl_block("train", band["q_row"].to_numpy(), band["e_row"].to_numpy()),
        "ce2": np.column_stack([ce0, cer, np.abs(ce0 - cer)]).astype(np.float32),
        "conf": conf_block(pairs, pf, pos),
    }
    # augmented universe (twins copy their source pair's CE logits / pairwise features)
    ap, arows, ap1, apf = S._val_universe("train_aug")
    src = ap["i"].to_numpy()
    srow = band["row"].to_numpy()
    o = np.argsort(srow)
    j = np.minimum(np.searchsorted(srow[o], src), len(srow) - 1)
    hit = srow[o][j] == src
    apos = np.flatnonzero(hit & (apf > 0.005) & (apf < 0.995))
    actx = S._Rows(np.load(wpath("feat", "train_aug_ctx.npy"), mmap_mode="r"), arows)
    Xa0 = S.features(ap, apf, ap1, actx, apos, band["ce"].to_numpy()[o][j[apos]], S._aug_country(keys, ap))
    Xa0 = np.hstack([Xa0, S.pair_block("train", src[apos])])
    bi = o[j[apos]]                        # band index of the source pair of every augmented band pair
    agroups = {"nl": groups["nl"][bi], "ce2": groups["ce2"][bi], "conf": conf_block(ap, apf, apos)}
    y = band["y"].to_numpy()
    isv = band["fold"].to_numpy() == VAL_FOLD
    cvf = S._cv_fold(band["e_row"].to_numpy())
    tr_all = np.flatnonzero(isv)
    ea, fa = ap["e_row"].to_numpy()[apos], ap["fold"].to_numpy()[apos]
    ka = np.where(fa == VAL_FOLD, S._cv_fold(ea), -1)

    def run(extra, frac=1.0):
        X = np.hstack([X0] + [groups[g] for g in extra])
        Xa = np.hstack([Xa0] + [agroups[g] for g in extra])
        P = np.zeros(len(y))
        Pa = np.zeros(len(apos))
        ms = []
        sub = (band["e_row"].to_numpy().astype(np.uint64) * np.uint64(40503) % np.uint64(1000)) < int(frac * 1000)
        for k in range(5):
            tr = tr_all[(cvf[tr_all] != k) & sub[tr_all]]
            te = tr_all[cvf[tr_all] == k]
            m = lgb.train(S.LGB_STACK, lgb.Dataset(X[tr], y[tr]), S.N_ROUNDS)
            P[te] = m.predict(X[te], num_threads=16)
            ms.append(m)
        m_all = lgb.train(S.LGB_STACK, lgb.Dataset(X[tr_all], y[tr_all]), S.N_ROUNDS)
        P[~isv] = m_all.predict(X[~isv], num_threads=16)
        for k in range(-1, 5):
            mk = ka == k
            if mk.any():
                Pa[mk] = (m_all if k < 0 else ms[k]).predict(Xa[mk], num_threads=16)
        pnew = pf.copy(); pnew[pos] = P
        apnew = apf.copy(); apnew[apos] = Pa
        f_p = S._decide_grid(pairs, pnew, entities, truth, f"plain {extra}")
        f_a = S._decide_grid(ap, apnew, entities, truth, f"aug {extra}")
        auc = roc_auc_score(y[tr_all], P[tr_all])
        log(f"RESULT {extra} frac={frac}: AUC {auc:.5f}  plain p>=0.8 {f_p[('p', 0.8)]:.5f}  aug p>=0.8 {f_a[('p', 0.8)]:.5f}")
        return pnew, apnew

    which = sys.argv[1:] or ["curve", "groups"]
    if "curve" in which:
        run([], 0.5)
    for extra in ([], ["nl"], ["ce2"], ["conf"], ["nl", "ce2"], ["nl", "ce2", "conf"]) if "groups" in which else ():
        run(extra)


if __name__ == "__main__":
    main()
