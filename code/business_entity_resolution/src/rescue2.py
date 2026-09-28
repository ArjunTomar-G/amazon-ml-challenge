"""Rescue retrieval v2 for records that blocking left without a link (US / India).

    python rescue2.py embed <split>     retriever embeddings of every US / India S1 and every unlinked record
                                        (with OR without an address; v10c: with an address only)
    python rescue2.py retrieve <split>  top-K S1 by cosine per record (same country), blocking pairs dropped
    python rescue2.py feats <split>     cheap pair features + name-uniqueness features
    python rescue2.py cheap             pre-filter model (folds A u B), recall check on V
    python rescue2.py ce <split>        cross-encoder logits for the pairs the pre-filter keeps
    python rescue2.py model             cross-fitted LightGBM (A -> B, B -> A, V / test <- mean), V validation
                                        against the stacked decisions, test predictions
                                        -> rescue2/test_rescue_pred.parquet

Differences to rescue.py (v10c):
  * records without an address are rescued too (78 % of the blocking misses; resolvable when the noisy
    name has a clear best entity).  They are embedded as "query: name | " like in training.
  * retriever = models/bienc2 (retriever2.py: trained on records with and without an address), K = 10
  * name-uniqueness features: how many S1 of the country share the record's core name, how many share
    core name and legal form, whether this S1 is the only one with the record's core name + legal form.
"""
from __future__ import annotations

import os
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from common import Timer, log, wpath
from crossenc import CE_COUNTRIES, MODEL_DIR, _batches, _device
from model import FOLD_A, FOLD_B, VAL_FOLD
from rescue import _decision_p, _fill1, _linked_records, _raw_texts, LGB_R

K = int(os.environ.get("ER_RESCUE_K", "10"))
RETRIEVER = os.environ.get("ER_RESCUE_MODEL", "bienc2")
THR = 0.8
D = "rescue2"
# extra (larger) cross-encoders as rescue features, e.g. "b" = e5-base (crossenc.py ER_CE_VARIANT); their models,
# predictions and V report go to files with the suffix OUT, so the v11 / v12 rescue files stay as they are
EXTRA = [v for v in os.environ.get("ER_RESCUE_CE_EXTRA", "").split(",") if v]
OUT = "".join(f"_{v}" for v in EXTRA)
HF = {"": "multilingual-e5-small", "r": "multilingual-e5-small", "b": "multilingual-e5-base", "l": "multilingual-e5-large"}


def step_embed(split: str):
    import torch
    from transformers import AutoModel, AutoTokenizer
    dev = _device()
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    path = str(wpath("models", RETRIEVER))
    log(f"retriever: {path}")
    m = AutoModel.from_pretrained(path, attn_implementation="sdpa").to(dev).eval().half()
    s1n = pl.read_parquet(wpath("norm", f"{split}_source1.parquet"), columns=["country"])
    e_rows = np.flatnonzero(np.isin(s1n["country"].to_numpy(), CE_COUNTRIES))
    qn = pl.concat([pl.read_parquet(wpath("norm", f"{split}_{s}.parquet"), columns=["country", "n_core", "a_all"])
                    for s in ("source2", "source3")])
    q_ok = np.isin(qn["country"].to_numpy(), CE_COUNTRIES)
    q_ok &= ((qn["a_all"].fill_null("").str.len_chars() > 0) | (qn["n_core"].fill_null("").str.len_chars() > 0)).to_numpy()
    # every US / India record is embedded (independent of the final decision); retrieve keeps the unlinked ones
    q_rows = np.flatnonzero(q_ok)
    log(f"{split}: embed {len(e_rows)} S1 and {len(q_rows)} records")
    from crossenc import _tokenize
    cls, sep, pad = tok.cls_token_id, tok.sep_token_id, tok.pad_token_id
    dim = m.config.hidden_size
    for tag, rows, srcs in (("e", e_rows, ("source1",)), ("q", q_rows, ("source2", "source3"))):
        texts = sum((_raw_texts(split, s) for s in srcs), [])
        with Timer(f"tokenise {tag}: {len(rows)} texts"):
            flat, off = _tokenize(tok, [texts[i] for i in rows])
        del texts
        lens = np.diff(off) + 2
        E = np.zeros((len(rows), dim), np.float16)
        bs = _batches(lens, 64 * 1024, 2048)
        t0 = time.time()
        with torch.inference_mode():
            for k, b in enumerate(bs):
                x = np.full((len(b), int(lens[b].max())), pad, np.int64)
                _fill1(x, b.astype(np.int64), flat, off, cls, sep)
                x = torch.from_numpy(x).to(dev, non_blocking=True)
                att = (x != pad)
                h = m(input_ids=x, attention_mask=att.long()).last_hidden_state
                h = (h * att.unsqueeze(-1)).sum(1) / att.sum(1, keepdim=True)
                h = torch.nn.functional.normalize(h.float(), dim=-1)
                E[b] = h.half().cpu().numpy()
        np.save(wpath(D, f"{split}_emb_{tag}.npy"), E)
        np.save(wpath(D, f"{split}_rows_{tag}.npy"), rows)
        log(f"  {tag}: {len(rows)} embeddings ({(time.time() - t0) / 60:.1f} min)")


def step_retrieve(split: str):
    import torch
    dev = _device()
    kt = np.load(wpath("feat", f"{split}_ctxkeys.npz"))
    Ee, er = np.load(wpath(D, f"{split}_emb_e.npy")), np.load(wpath(D, f"{split}_rows_e.npy"))
    Eq, qr = np.load(wpath(D, f"{split}_emb_q.npy"), mmap_mode="r"), np.load(wpath(D, f"{split}_rows_q.npy"))
    keep = ~np.isin(qr, _linked_records(split))
    log(f"{split}: {keep.sum()} of {len(qr)} embedded records are unlinked")
    Eq, qr = np.asarray(Eq[np.flatnonzero(keep)]), qr[keep]
    ec = kt["e_country"][er]
    qc = kt["q_country"][qr]
    out = []
    for c in CE_COUNTRIES:
        ie, iq = np.flatnonzero(ec == c), np.flatnonzero(qc == c)
        S = torch.from_numpy(Ee[ie]).to(dev)
        top_i = np.zeros((len(iq), K + 1), np.int64)
        top_s = np.zeros((len(iq), K + 1), np.float32)
        step = 512
        with Timer(f"{split} {c}: top-{K} of {len(ie)} S1 for {len(iq)} records"), torch.inference_mode():
            for a in range(0, len(iq), step):
                Q = torch.from_numpy(Eq[iq[a:a + step]]).to(dev)
                v, i = torch.topk(Q @ S.T, K + 1, dim=1)
                top_i[a:a + step] = i.cpu().numpy()
                top_s[a:a + step] = v.float().cpu().numpy()
        del S
        torch.cuda.empty_cache()
        d = pl.DataFrame({
            "q_row": np.repeat(qr[iq], K), "e_row": er[ie][top_i[:, :K]].ravel(),
            "cos": top_s[:, :K].ravel(), "cos_rank": np.tile(np.arange(1, K + 1), len(iq)),
            "cos_gap": (top_s[:, :K] - np.where(np.arange(K) == 0, top_s[:, 1:2], top_s[:, :1])).ravel(),
            "cos_next": np.repeat(top_s[:, K], K)})
        out.append(d)
    d = pl.concat(out).with_columns(pl.col("q_row").cast(pl.Int64), pl.col("e_row").cast(pl.Int64))
    pr = pl.read_parquet(wpath("feat", f"{split}_pairs.parquet")).select(pl.col("q_row").cast(pl.Int64), pl.col("e_row").cast(pl.Int64))
    n0 = d.height
    d = d.join(pr, on=["q_row", "e_row"], how="anti")
    if split == "train":
        from blocking import true_pairs_rows, load_universe
        from model import entity_folds
        s1, _ = load_universe("train", ["entity_id"])
        fold = entity_folds(s1["entity_id"])
        tp = true_pairs_rows("train").select(pl.col("q_row").cast(pl.Int64), pl.col("e_row").cast(pl.Int64)).with_columns(pl.lit(1, pl.Int8).alias("y"))
        d = d.join(tp, on=["q_row", "e_row"], how="left").with_columns(pl.col("y").fill_null(0),
                                                                          pl.Series("fold", fold[d["e_row"].to_numpy()]))
        log(f"train rescue pairs: {d.height} ({n0 - d.height} already candidates), true {int(d['y'].sum())}")
    else:
        log(f"test rescue pairs: {d.height} ({n0 - d.height} already candidates)")
    d.write_parquet(wpath(D, f"{split}_pairs.parquet"))


def _name_stats(split: str):
    """Per record: #S1 (same country) with the same core name, and with the same core name + legal form."""
    s1 = pl.read_parquet(wpath("norm", f"{split}_source1.parquet"), columns=["country", "n_core", "n_legal"])
    a = s1.group_by(["country", "n_core"]).len().rename({"len": "k_name"})
    b = s1.group_by(["country", "n_core", "n_legal"]).len().rename({"len": "k_name_legal"})
    return a, b


def step_feats(split: str):
    from rapidfuzz import fuzz, process
    d = pl.read_parquet(wpath(D, f"{split}_pairs.parquet"))
    cols = ["country", "n_core", "n_legal", "a_tok", "a_hn", "a_nums"]
    s1 = pl.read_parquet(wpath("norm", f"{split}_source1.parquet"), columns=cols).with_row_index("e_row")
    q = pl.concat([pl.read_parquet(wpath("norm", f"{split}_{s}.parquet"), columns=cols) for s in ("source2", "source3")]).with_row_index("q_row")
    ka, kb = _name_stats(split)
    q = (q.join(ka, on=["country", "n_core"], how="left").join(kb, on=["country", "n_core", "n_legal"], how="left")
         .with_columns(pl.col("k_name").fill_null(0), pl.col("k_name_legal").fill_null(0)))
    x = (d.select("q_row", "e_row").with_row_index("i")
         .join(q.with_columns(pl.col("q_row").cast(pl.Int64)).rename({c: "q_" + c for c in cols + ["k_name", "k_name_legal"]}), on="q_row")
         .join(s1.with_columns(pl.col("e_row").cast(pl.Int64)).rename({c: "e_" + c for c in cols}), on="e_row")
         .sort("i"))
    with Timer(f"{split}: fuzzy features for {x.height} rescue pairs"):
        qn, en = x["q_n_core"].fill_null("").to_list(), x["e_n_core"].fill_null("").to_list()
        qa, ea = x["q_a_tok"].fill_null("").to_list(), x["e_a_tok"].fill_null("").to_list()
        f_nr = process.cpdist(qn, en, scorer=fuzz.ratio, workers=-1)
        f_nts = process.cpdist(qn, en, scorer=fuzz.token_set_ratio, workers=-1)
        f_npr = process.cpdist(qn, en, scorer=fuzz.partial_ratio, workers=-1)
        f_ats = process.cpdist(qa, ea, scorer=fuzz.token_set_ratio, workers=-1)
    x = x.with_columns(
        (pl.col("q_a_hn") == pl.col("e_a_hn")).and_(pl.col("q_a_hn") != "").cast(pl.Int8).alias("hn_eq"),
        pl.col("q_a_nums").str.split(" ").list.set_intersection(pl.col("e_a_nums").str.split(" "))
        .list.eval(pl.element().filter(pl.element() != "")).list.len().alias("num_shared"),
        (pl.col("q_a_hn") == "").cast(pl.Int8).alias("q_no_hn"),
        (pl.col("q_a_tok").fill_null("") == "").cast(pl.Int8).alias("q_no_addr"),
        (pl.col("q_n_core") == pl.col("e_n_core")).cast(pl.Int8).alias("name_eq"),
        (pl.col("q_n_legal") == pl.col("e_n_legal")).cast(pl.Int8).alias("legal_eq"),
        pl.col("q_k_name").clip(0, 50).alias("k_name"),
        pl.col("q_k_name_legal").clip(0, 50).alias("k_name_legal"))
    x = x.with_columns(((pl.col("name_eq") == 1) & (pl.col("legal_eq") == 1) & (pl.col("k_name_legal") == 1)).cast(pl.Int8).alias("unique_nl"))
    pr, p = _decision_p(split)
    best = pr.select("q_row").with_columns(pl.Series("p", p)).group_by("q_row").agg(pl.col("p").max().alias("p_exist")) \
        .with_columns(pl.col("q_row").cast(pl.Int64))
    x = x.join(best, on="q_row", how="left", maintain_order="left").with_columns(pl.col("p_exist").fill_null(0.0))
    F = d.with_columns(pl.Series("nr", f_nr), pl.Series("nts", f_nts), pl.Series("npr", f_npr), pl.Series("ats", f_ats),
                       x["hn_eq"], x["num_shared"], x["q_no_hn"], x["q_no_addr"], x["name_eq"], x["legal_eq"],
                       x["k_name"], x["k_name_legal"], x["unique_nl"], x["p_exist"])
    F.write_parquet(wpath(D, f"{split}_feats.parquet"))
    log(f"{split}: wrote rescue features {F.shape}")


CHEAP = ["cos", "cos_rank", "cos_gap", "cos_next", "nr", "nts", "npr", "ats", "hn_eq", "num_shared", "q_no_hn",
         "q_no_addr", "name_eq", "legal_eq", "k_name", "k_name_legal", "unique_nl", "p_exist"]
PREFILTER = float(os.environ.get("ER_RESCUE_PREFILTER", "0.02"))


def step_cheap():
    from sklearn.metrics import roc_auc_score
    tr = pl.read_parquet(wpath(D, "train_feats.parquet"))
    f, y = tr["fold"].to_numpy(), tr["y"].to_numpy()
    X = tr.select(CHEAP).to_numpy().astype(np.float32)
    ab, v = np.isin(f, FOLD_A + FOLD_B), f == VAL_FOLD
    m = lgb.train(LGB_R, lgb.Dataset(X[ab], y[ab], feature_name=CHEAP), 400)
    m.save_model(str(wpath("models", "rescue2_cheap.txt")))
    pv = m.predict(X[v], num_threads=16)
    keep = pv >= PREFILTER
    na = tr["q_no_addr"].to_numpy()[v] == 1
    log(f"cheap rescue model: V pairs {v.sum()} ({y[v].sum()} true), AUC {roc_auc_score(y[v], pv):.4f}, "
        f"pre-filter keeps {keep.mean():.4f} of pairs and {y[v][keep].sum() / max(1, y[v].sum()):.3f} of true pairs "
        f"(no-address true kept {y[v][keep & na].sum()} of {y[v][na].sum()})")


def step_ce(split: str, v: str = ""):
    """Cross-encoder logits (variant v: "" = e5-small fold models ce_A / ce_B, "b" = ce_Ab / ce_Bb ...)."""
    import torch
    from crossenc import Tokens, _load_model, _score, _texts, _tokenize
    from transformers import AutoTokenizer
    F = pl.read_parquet(wpath(D, f"{split}_feats.parquet"))
    cheap = lgb.Booster(model_file=str(wpath("models", "rescue2_cheap.txt")))
    pc = cheap.predict(F.select(CHEAP).to_numpy().astype(np.float32), num_threads=16)
    keep = np.flatnonzero(pc >= PREFILTER)
    log(f"{split}: {len(keep)} of {F.height} rescue pairs pass the pre-filter")
    sub = F[keep].select("q_row", "e_row")
    tok = AutoTokenizer.from_pretrained(MODEL_DIR if not v else str(wpath("hf", HF[v])))
    ts = v if v in ("b", "l") else ""
    for tag, col, srcs in (("e", "e_row", ("source1",)), ("q", "q_row", ("source2", "source3"))):
        rows = np.unique(sub[col].to_numpy())
        texts = sum((_texts(split, s) for s in srcs), [])
        flat, off = _tokenize(tok, [texts[i] for i in rows])
        np.savez(wpath("ce", f"{split}_r2tok_{tag}{ts}.npz"), rows=rows.astype(np.int64), flat=flat, off=off)
    T = Tokens.__new__(Tokens)
    T.side = {}
    for tag in ("e", "q"):
        z = np.load(wpath("ce", f"{split}_r2tok_{tag}{ts}.npz"))
        T.side[tag] = (z["rows"], z["flat"], z["off"])
    T.cls, T.sep, T.pad = np.load(wpath("ce", f"special{ts}.npy")).tolist()
    ei, qi = T.index("e", sub["e_row"].to_numpy()), T.index("q", sub["q_row"].to_numpy())
    dev = _device()
    logits = {}
    for tag in ("A", "B"):
        m = _load_model(str(wpath("models", f"ce_{tag}{v}")), dev).eval()
        if split == "train":
            f = F["fold"].to_numpy()[keep]
            need = ~np.isin(f, FOLD_A if tag == "A" else FOLD_B)
        else:
            need = np.ones(len(keep), bool)
        lg = np.full(len(keep), np.nan, np.float32)
        with Timer(f"{split}: CE {tag} on {need.sum()} rescue pairs"):
            lg[need] = _score(m, T, ei[need], qi[need], dev)
        logits[tag] = lg
        del m
        torch.cuda.empty_cache()
    if split == "train":
        f = F["fold"].to_numpy()[keep]
        ce = np.where(np.isin(f, FOLD_A), logits["B"], np.where(np.isin(f, FOLD_B), logits["A"], 0.5 * (logits["A"] + logits["B"])))
    else:
        ce = 0.5 * (logits["A"] + logits["B"])
    out = np.full(F.height, np.nan, np.float32)
    out[keep] = ce
    np.save(wpath(D, f"{split}_ce{v}.npy"), out)


FULL = CHEAP + ["ce", "ce_rank", "ce_gap"] + [f"ce{v}{s}" for v in EXTRA for s in ("", "_rank", "_gap")]


def _rank_gap(F: pl.DataFrame, c: str) -> pl.DataFrame:
    F = F.with_columns(pl.col(c).rank("ordinal", descending=True).over("q_row").alias(f"{c}_rank"),
                       (pl.col(c) - pl.col(c).max().over("q_row")).alias("gap0"))
    top2 = F.group_by("q_row").agg(pl.col(c).top_k(2).alias("t"))
    top2 = top2.with_columns(pl.col("t").list.get(1, null_on_oob=True).alias("c2"))
    F = F.join(top2.select("q_row", "c2"), on="q_row", how="left", maintain_order="left")
    return F.with_columns(pl.when(pl.col("gap0") >= 0).then(pl.col(c) - pl.col("c2")).otherwise(pl.col("gap0"))
                          .fill_null(np.nan).alias(f"{c}_gap")).drop("gap0", "c2")


def _full_frame(split: str) -> pl.DataFrame:
    F = pl.read_parquet(wpath(D, f"{split}_feats.parquet")).with_columns(pl.Series("ce", np.load(wpath(D, f"{split}_ce.npy"))))
    for v in EXTRA:
        F = F.with_columns(pl.Series(f"ce{v}", np.load(wpath(D, f"{split}_ce{v}.npy"))))
    F = F.filter(pl.col("ce").is_not_nan())
    for c in ["ce"] + [f"ce{v}" for v in EXTRA]:
        F = _rank_gap(F, c)
    return F


def step_model():
    from blocking import true_pairs_rows
    from decide import macro_f05
    tr = _full_frame("train")
    f, y = tr["fold"].to_numpy(), tr["y"].to_numpy()
    X = tr.select(FULL).to_numpy().astype(np.float32)
    ms = {}
    for tag, folds in (("A", FOLD_A), ("B", FOLD_B)):
        idx = np.isin(f, folds)
        ms[tag] = lgb.train(LGB_R, lgb.Dataset(X[idx], y[idx], feature_name=FULL), 400)
        ms[tag].save_model(str(wpath("models", f"rescue2_{tag}{OUT}.txt")))
    pa_, pb_ = ms["A"].predict(X, num_threads=16), ms["B"].predict(X, num_threads=16)
    P = np.where(np.isin(f, FOLD_A), pb_, np.where(np.isin(f, FOLD_B), pa_, 0.5 * (pa_ + pb_)))
    imp = sorted(zip(ms["A"].feature_importance("gain"), FULL), reverse=True)
    log("rescue2 importance:", [(n, round(float(g))) for g, n in imp])
    tr = tr.with_columns(pl.Series("prob", P))
    tr.select("q_row", "e_row", "fold", "y", "prob", "q_no_addr").write_parquet(wpath(D, f"train_rescue_pred{OUT}.parquet"))
    best = tr.sort("prob", descending=True).unique("q_row", keep="first")
    keys = np.load(wpath("feat", "train_ctxkeys.npz"))
    entities = pl.DataFrame({"e_row": np.flatnonzero(keys["e_fold"] == VAL_FOLD)})
    truth = true_pairs_rows().filter(pl.col("e_row").is_in(entities["e_row"].implode()))
    base = pl.read_parquet(wpath("ce", "val_links.parquet")).select(pl.col("e_row").cast(pl.Int64), pl.col("q_row").cast(pl.Int64))
    f0 = macro_f05(base, truth, entities)
    tt = truth.select(pl.col("e_row").cast(pl.Int64), pl.col("q_row").cast(pl.Int64))
    for sub_name, sub in (("all", best), ("with address", best.filter(pl.col("q_no_addr") == 0)),
                          ("without address", best.filter(pl.col("q_no_addr") == 1))):
        for t in (0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95):
            add = sub.filter((pl.col("prob") >= t) & (pl.col("fold") == VAL_FOLD)).select("e_row", "q_row")
            f1 = macro_f05(pl.concat([base, add]), truth, entities)
            ntrue = add.join(tt, on=["e_row", "q_row"]).height
            log(f"V rescue2 [{sub_name}] thr {t}: +{add.height} links ({ntrue} true), F0.5 {f0:.5f} -> {f1:.5f} ({f1 - f0:+.5f})")
    te = _full_frame("test")
    Xt = te.select(FULL).to_numpy().astype(np.float32)
    pt = 0.5 * (ms["A"].predict(Xt, num_threads=16) + ms["B"].predict(Xt, num_threads=16))
    te.select("q_row", "e_row", "q_no_addr").with_columns(pl.Series("prob", pt)).write_parquet(wpath(D, f"test_rescue_pred{OUT}.parquet"))
    bt = te.with_columns(pl.Series("prob", pt)).sort("prob", descending=True).unique("q_row", keep="first")
    log(f"test rescue2: {te.height} scored pairs, records with best prob >= 0.8: {(bt['prob'] >= 0.8).sum()} "
        f"(without address {((bt['prob'] >= 0.8) & (bt['q_no_addr'] == 1)).sum()})")


if __name__ == "__main__":
    s = sys.argv[1]
    if s == "embed":
        step_embed(sys.argv[2])
    elif s == "retrieve":
        step_retrieve(sys.argv[2])
    elif s == "feats":
        step_feats(sys.argv[2])
    elif s == "cheap":
        step_cheap()
    elif s == "ce":
        step_ce(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "")
    elif s == "model":
        step_model()
    else:
        raise SystemExit(f"unknown step {s}")
