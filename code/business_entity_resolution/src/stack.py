"""Stage 3b: stacker over stage-2 probability + cross-encoder logit (blueprint 4.4).

    python stack.py cv        5-fold (by S1) stacker on the validation fold's band pairs: out-of-fold
                              probabilities, validation F0.5 against stage 2 (plain and density-
                              augmented universe), decision rule
    python stack.py fit       stacker on every validation band pair -> test probabilities
                              (feat/test_pfinal.npy; France and pairs outside the band keep stage 2)

The validation fold (V) is scored exactly like test at every stage: stage 1 = mean of the two
fold models, stage 2 trained on A u B, cross-encoder = mean of the two fold models.  So the
stacker is trained and evaluated on V only (5-fold CV by S1 entity), and applied to test.
Features are pair or record-side quantities (the record's competition between S1 entities),
which do not depend on the distractor density.
"""
from __future__ import annotations

import json
import os
import sys

import lightgbm as lgb
import numpy as np
import polars as pl

from blocking import true_pairs_rows
from common import Timer, log, wpath
from context import CTX_COLS
from decide import final_prob, macro_f05
from model import VAL_FOLD

LGB_STACK = dict(objective="binary", learning_rate=float(os.environ.get("ER_STACK_LR", 0.05)),
                 num_leaves=int(os.environ.get("ER_STACK_LEAVES", 63)),
                 min_data_in_leaf=int(os.environ.get("ER_STACK_MINLEAF", 100)),
                 feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
                 num_threads=16, verbose=-1, seed=2026)
N_ROUNDS = int(os.environ.get("ER_STACK_ROUNDS", 400))
CTX_FEATS = ["q_pmax", "q_psecond", "p_minus_qmax_other", "q_n", "e_pmax_other", "e_rank", "mutual"]
FEATS = ["pf", "lpf", "p1", "ce", "ce_rank", "ce_gap", "n_band", "p_best_other", "p_rank", "is_india"] + CTX_FEATS
TOPK = int(os.environ.get("ER_STACK_TOPK", "60"))      # + this many of the strongest pairwise stage-1 features


def pair_names():
    return json.load(open(wpath("feat", "top_features.json")))[:TOPK] if TOPK else []


def pair_block(split: str, pair_rows: np.ndarray) -> np.ndarray:
    """Top-K pairwise features of the given pair rows (rows of <split>_pairs / <split>_X)."""
    if not TOPK:
        return np.zeros((len(pair_rows), 0), np.float32)
    names = json.load(open(wpath("feat", "feature_names.json")))
    ci = [names.index(t) for t in pair_names()]
    X = np.load(wpath("feat", f"{split}_X.npy"), mmap_mode="r")
    o = np.argsort(pair_rows, kind="stable")
    out = np.empty((len(pair_rows), len(ci)), np.float32)
    step = 1_000_000
    for a in range(0, len(o), step):
        oo = o[a:a + step]
        out[oo] = np.asarray(X[pair_rows[oo]])[:, ci]
    return out


BASE = os.environ.get("ER_CE_BASEFEAT", "1") == "1"   # larger cross-encoders' logits as extra features
# extra cross-encoder variants (crossenc.py ER_CE_VARIANT): "b" = e5-base, "l" = e5-large
EXTRA = [v for v in os.environ.get("ER_CE_EXTRA", "b").split(",") if v]


def _extras() -> list[str]:
    return [v for v in EXTRA if BASE and wpath("ce", f"train_val_logit{v}.npy").exists()
            and wpath("ce", f"test_band_logit{v}.npy").exists()]


def _use_base() -> bool:
    return bool(_extras())


def base_names():
    return [f"ce_{v}{s}" for v in _extras() for s in ("", "_rank", "_gap")]


def base_block(q_rows: np.ndarray, ce_list) -> np.ndarray:
    """Each extra cross-encoder's logit, its rank and gap among the record's band pairs."""
    if not _use_base():
        return np.zeros((len(q_rows), 0), np.float32)
    return np.hstack([_one_block(q_rows, c) for c in ce_list])


def _one_block(q_rows: np.ndarray, ce_b: np.ndarray) -> np.ndarray:
    b = pl.DataFrame({"q_row": q_rows, "ce": ce_b})
    b = b.with_columns(pl.col("ce").rank("ordinal", descending=True).over("q_row").alias("rk"))
    ctop = b.group_by("q_row").agg(pl.col("ce").top_k(2).alias("t2"))
    ctop = ctop.with_columns(pl.col("t2").list.get(0).alias("c1"), pl.col("t2").list.get(1, null_on_oob=True).alias("c2"))
    b = b.join(ctop.select("q_row", "c1", "c2"), on="q_row", how="left", maintain_order="left")
    b = b.with_columns(pl.when(pl.col("ce") >= pl.col("c1")).then(pl.col("ce") - pl.col("c2"))
                       .otherwise(pl.col("ce") - pl.col("c1")).alias("gap"))
    return np.column_stack([b["ce"].to_numpy(), b["rk"].to_numpy(), b["gap"].fill_null(np.nan).to_numpy()]).astype(np.float32)


def _logit(p):
    p = np.clip(p.astype(np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def features(pairs: pl.DataFrame, pf: np.ndarray, p1: np.ndarray, ctx: np.ndarray, band_rows: np.ndarray,
             ce: np.ndarray, india: np.ndarray) -> np.ndarray:
    """pairs: q_row/e_row of ALL pairs of the records concerned (row order = pf/p1/ctx order);
    band_rows: rows (into pairs) of the band pairs, ce: their CE logits.  Returns X for band_rows."""
    d = pairs.select("q_row").with_columns(pl.Series("p", pf), pl.Series("r", np.arange(pairs.height)))
    d = d.with_columns(
        pl.col("p").rank("ordinal", descending=True).over("q_row").alias("p_rank"),
        pl.len().over("q_row").alias("q_all"))
    # best other p of the record (over all its pairs)
    d = d.with_columns(pl.col("p").max().over("q_row").alias("pmax"))
    top2 = d.group_by("q_row").agg(pl.col("p").top_k(2).alias("t2"))
    top2 = top2.with_columns(pl.col("t2").list.get(1, null_on_oob=True).fill_null(0.0).alias("p2nd"))
    d = d.join(top2.select("q_row", "p2nd"), on="q_row", how="left", maintain_order="left")
    d = d.with_columns(pl.when(pl.col("p") >= pl.col("pmax")).then(pl.col("p2nd")).otherwise(pl.col("pmax")).alias("p_best_other"))
    b = d[band_rows].select("q_row", "p_rank", "p_best_other").with_columns(pl.Series("ce", ce))
    b = b.with_columns(
        pl.col("ce").rank("ordinal", descending=True).over("q_row").alias("ce_rank"),
        pl.len().over("q_row").alias("n_band"))
    ctop = b.group_by("q_row").agg(pl.col("ce").top_k(2).alias("t2"))
    ctop = ctop.with_columns(pl.col("t2").list.get(0).alias("c1"), pl.col("t2").list.get(1, null_on_oob=True).alias("c2"))
    b = b.join(ctop.select("q_row", "c1", "c2"), on="q_row", how="left", maintain_order="left")
    b = b.with_columns(pl.when(pl.col("ce") >= pl.col("c1")).then(pl.col("ce") - pl.col("c2"))
                       .otherwise(pl.col("ce") - pl.col("c1")).alias("ce_gap"))
    ci = [CTX_COLS.index(c) for c in CTX_FEATS]
    o = np.argsort(band_rows, kind="stable")
    C = np.empty((len(band_rows), len(ci)), np.float32)
    C[o] = np.asarray(ctx[band_rows[o]])[:, ci]
    X = np.column_stack([
        pf[band_rows], _logit(pf[band_rows]), p1[band_rows], ce, b["ce_rank"].to_numpy(),
        b["ce_gap"].fill_null(np.nan).to_numpy(), b["n_band"].to_numpy(), b["p_best_other"].to_numpy(),
        b["p_rank"].to_numpy(), india[band_rows].astype(np.float32), C]).astype(np.float32)
    return X


def _val_universe(tag: str):
    """Validation rows (records touching V) of the plain or augmented train universe."""
    pr = pl.read_parquet(wpath("feat", f"{tag}_pairs.parquet"))
    vq = pr.filter(pl.col("fold") == VAL_FOLD)["q_row"].unique()
    rows = np.flatnonzero(pr["q_row"].is_in(vq.implode()).to_numpy())
    p1 = np.load(wpath("feat", f"{tag}_p1.npy"))[rows]
    p2 = np.load(wpath("feat", f"{tag}_p2.npy"))[rows]
    return pr[rows], rows, p1, final_prob(p1, p2)


def _decide_grid(pairs, p, entities, truth, label):
    """Exclusivity (argmax per record), then threshold on p or on pa = p / max(1, sum p)."""
    d = pairs.select("q_row", "e_row", "fold").with_columns(pl.Series("p", p))
    d = d.with_columns(pl.col("p").max().over("q_row").alias("pm"), pl.col("p").sum().over("q_row").alias("ps"))
    d = d.filter((pl.col("p") >= pl.col("pm")) & (pl.col("fold") == VAL_FOLD))
    d = d.with_columns((pl.col("p") / pl.max_horizontal(pl.lit(1.0), pl.col("ps"))).alias("pa"))
    res = {}
    for col in ("p", "pa"):
        for t in (0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9):
            res[(col, t)] = macro_f05(d.filter(pl.col(col) >= t).select("e_row", "q_row"), truth, entities)
    best = max(res.items(), key=lambda kv: kv[1])
    log(f"[{label}] " + "  ".join(f"{k[0]}>={k[1]}:{v:.5f}" for k, v in sorted(res.items())))
    log(f"[{label}] best {best[0]} = {best[1]:.5f}   (p>=0.8: {res[('p', 0.8)]:.5f}, pa>=0.8: {res[('pa', 0.8)]:.5f})")
    return res


ENSEMBLE = os.environ.get("ER_CE_ENSEMBLE", "1") == "1"   # mean logit of the cross-encoder variants


def _ce_logits(name: str) -> np.ndarray:
    """Cross-encoder logits (mean over the trained variants: "" and "r" = record-first order)."""
    ls = [np.load(wpath("ce", f"{name}.npy"))]
    alt = wpath("ce", f"{name}r.npy") if name.endswith("logit") else None
    if ENSEMBLE and alt is not None and alt.exists():
        ls.append(np.load(alt))
    log(f"cross-encoder logits {name}: {len(ls)} variant(s)")
    return np.mean(ls, axis=0).astype(np.float32)


def _band_frame():
    """Scored validation band pairs: train pair row, fold, y, CE logit."""
    d = pl.read_parquet(wpath("ce", "train_val.parquet"))
    d = d.with_columns(pl.Series("ce", _ce_logits("train_val_logit")))
    for v in _extras():
        d = d.with_columns(pl.Series(f"ce{v}", np.load(wpath("ce", f"train_val_logit{v}.npy"))))
    return d


def step_cv():
    keys = np.load(wpath("feat", "train_ctxkeys.npz"))
    entities = pl.DataFrame({"e_row": np.flatnonzero(keys["e_fold"] == VAL_FOLD)})
    truth = true_pairs_rows().filter(pl.col("e_row").is_in(entities["e_row"].implode()))
    ctx = np.load(wpath("feat", "train_ctx.npy"), mmap_mode="r")
    band = _band_frame()
    pairs, rows, p1, pf = _val_universe("train")
    pos = np.searchsorted(rows, band["row"].to_numpy())
    assert (rows[pos] == band["row"].to_numpy()).all()
    india = (keys["q_country"][pairs["q_row"].to_numpy()] == "India")
    ctx_v = _Rows(ctx, rows)
    with Timer(f"stacker features: {len(pos)} band pairs"):
        X = features(pairs, pf, p1, ctx_v, pos, band["ce"].to_numpy(), india)
        X = np.hstack([X, pair_block("train", band["row"].to_numpy()),
                       base_block(pairs["q_row"].to_numpy()[pos], [band[f"ce{v}"].to_numpy() for v in _extras()])])
    names_all = FEATS + pair_names() + base_names()
    log(f"stacker features: {len(names_all)} (extra cross-encoders: {_extras() or 'none'})")
    y = band["y"].to_numpy()
    isv = band["fold"].to_numpy() == VAL_FOLD
    e = band["e_row"].to_numpy()
    cvf = _cv_fold(e)
    P = np.zeros(len(y), np.float64)
    tr_all = np.flatnonzero(isv)
    log(f"stacker: {len(tr_all)} V band pairs ({y[tr_all].mean():.3f} true), "
        f"{(~isv).sum()} pairs of V records with A/B entities")
    models = []
    for k in range(5):
        tr = tr_all[cvf[tr_all] != k]
        te = tr_all[cvf[tr_all] == k]
        models.append(lgb.train(LGB_STACK, lgb.Dataset(X[tr], y[tr], feature_name=names_all), N_ROUNDS))
        P[te] = models[k].predict(X[te], num_threads=16)
    m_all = lgb.train(LGB_STACK, lgb.Dataset(X[tr_all], y[tr_all], feature_name=names_all), N_ROUNDS)
    P[~isv] = m_all.predict(X[~isv], num_threads=16)
    m_all.save_model(str(wpath("models", "stacker.txt")))
    imp = sorted(zip(m_all.feature_importance("gain"), names_all), reverse=True)
    log("stacker importance:", [(n, round(float(g))) for g, n in imp])
    from sklearn.metrics import roc_auc_score
    for nm, s_ in (("stage2 pf", X[tr_all, 0]), ("CE logit", X[tr_all, 3]), ("stacker", P[tr_all])):
        log(f"  band V pairs AUC {nm}: {roc_auc_score(y[tr_all], s_):.4f}")
    pnew = pf.copy()
    pnew[pos] = P
    np.save(wpath("ce", "train_val_pstack.npy"), P.astype(np.float32))
    r0 = _decide_grid(pairs, pf, entities, truth, "plain stage-2 (v4)")
    r1 = _decide_grid(pairs, pnew, entities, truth, "plain stacked")
    # V links of the stacked decision (p >= 0.8 on the record's argmax): baseline for rescue.py
    (pairs.select("q_row", "e_row", "fold").with_columns(pl.Series("p", pnew))
     .with_columns(pl.col("p").max().over("q_row").alias("pm"))
     .filter((pl.col("p") >= pl.col("pm")) & (pl.col("p") >= 0.8) & (pl.col("fold") == VAL_FOLD))
     .unique("q_row").select("e_row", "q_row").write_parquet(wpath("ce", "val_links.parquet")))
    # density-augmented universe: a twin copies the CE logit of its source pair; V pairs are
    # predicted by the CV model that did not see their entity
    ap, arows, ap1, apf = _val_universe("train_aug")
    src = ap["i"].to_numpy()
    srow = band["row"].to_numpy()
    o = np.argsort(srow)
    j = np.minimum(np.searchsorted(srow[o], src), len(srow) - 1)
    hit = srow[o][j] == src
    inband = hit & (apf > 0.005) & (apf < 0.995)
    apos = np.flatnonzero(inband)
    ce_aug = band["ce"].to_numpy()[o][j[apos]]
    actx = _Rows(np.load(wpath("feat", "train_aug_ctx.npy"), mmap_mode="r"), arows)
    Xa = features(ap, apf, ap1, actx, apos, ce_aug, _aug_country(keys, ap))
    Xa = np.hstack([Xa, pair_block("train", src[apos]),
                    base_block(ap["q_row"].to_numpy()[apos], [band[f"ce{v}"].to_numpy()[o][j[apos]] for v in _extras()])])
    ea = ap["e_row"].to_numpy()[apos]
    fa = ap["fold"].to_numpy()[apos]
    Pa = np.zeros(len(apos))
    ka = np.where(fa == VAL_FOLD, _cv_fold(ea), -1)
    for k in range(-1, 5):
        mk = ka == k
        if mk.any():
            Pa[mk] = (m_all if k < 0 else models[k]).predict(Xa[mk], num_threads=16)
    apnew = apf.copy()
    apnew[apos] = Pa
    log(f"augmented: {len(apos)} band pairs with a CE logit, "
        f"{(~hit & (apf > 0.005) & (apf < 0.995)).sum()} band pairs without (keep stage 2)")
    # final probabilities of the plain and augmented validation universes (for decision-layer experiments)
    np.savez(wpath("ce", "val_pfinal.npz"), rows=rows, p=pnew.astype(np.float32),
             aug_rows=arows, aug_p=apnew.astype(np.float32))
    r2 = _decide_grid(ap, apf, entities, truth, "augmented stage-2 (v4)")
    r3 = _decide_grid(ap, apnew, entities, truth, "augmented stacked")
    best = max(r3.items(), key=lambda kv: kv[1])
    dec = {"rule": best[0][0], "param": best[0][1], "f05_augmented": best[1],
           "f05_plain": r1[best[0]], "v4_plain_p08": r0[("p", 0.8)], "v4_aug_p08": r2[("p", 0.8)]}
    log("stacker decision:", dec)
    json.dump(dec, open(wpath("models", "decision_stack.json"), "w"))


def _cv_fold(e_rows: np.ndarray) -> np.ndarray:
    return (e_rows.astype(np.uint64) * np.uint64(2654435761) % np.uint64(5)).astype(np.int64)


def _aug_country(keys, ap):
    """India flag of the augmented pairs (twins keep their source record's country)."""
    qc = keys["q_country"]
    q = ap["q_row"].to_numpy()
    out = np.zeros(len(q), bool)
    base = q < len(qc)
    out[base] = qc[q[base]] == "India"
    # twins: country of the S1 entity (country is a hard partition)
    ec = keys["e_country"][ap["e_row"].to_numpy()]
    out[~base] = ec[~base] == "India"
    return out


class _Rows:
    """Row-subset view of a memmap: X[rows[i]] for i."""

    def __init__(self, X, rows):
        self.X, self.rows = X, rows

    def __getitem__(self, i):
        return np.asarray(self.X[self.rows[i]])


def step_fit():
    """Stacker on all V band pairs was saved by step_cv; apply it to the test band pairs."""
    m = lgb.Booster(model_file=str(wpath("models", "stacker.txt")))
    keys = np.load(wpath("feat", "test_ctxkeys.npz"))
    pr = pl.read_parquet(wpath("feat", "test_pairs.parquet"))
    p1 = np.load(wpath("feat", "test_p1.npy"))
    pf = final_prob(p1, np.load(wpath("feat", "test_p2.npy")))
    band = pl.read_parquet(wpath("ce", "test_band.parquet"))
    ce = _ce_logits("test_band_logit")
    ctx = np.load(wpath("feat", "test_ctx.npy"), mmap_mode="r")
    india = keys["q_country"][pr["q_row"].to_numpy()] == "India"
    pos = band["row"].to_numpy()
    with Timer(f"test stacker: {len(pos)} band pairs"):
        X = features(pr, pf, p1, _Rows(ctx, np.arange(pr.height)), pos, ce, india)
        X = np.hstack([X, pair_block("test", pos),
                       base_block(pr["q_row"].to_numpy()[pos], [np.load(wpath("ce", f"test_band_logit{v}.npy")) for v in _extras()])])
        P = m.predict(X, num_threads=16)
    out = pf.copy()
    out[pos] = P
    np.save(wpath("feat", os.environ.get("ER_STACK_OUT", "test_pfinal.npy")), out.astype(np.float32))
    log(f"test: {len(pos)} band pairs re-scored; mean p {pf[pos].mean():.4f} -> {P.mean():.4f}")


if __name__ == "__main__":
    s = sys.argv[1]
    if s == "cv":
        step_cv()
    elif s == "fit":
        step_fit()
    else:
        raise SystemExit(f"unknown step {s}")
