"""Stage 5: context features, stage-2 model, validation and decision-rule choice.

    python stage2.py ctx <split>      context features for every pair -> memmap
                                      (train: also the density-augmented universe)
    python stage2.py train            stage-2 model on A u B pairs; validation on V
    python stage2.py predict <split>  stage-2 probabilities for every pair
    python stage2.py validate         re-run the validation grid only

Density-robust stage 2.  The test universe has ~2x more distractor records per
Source-1 entity than train (5.75 vs 4.68 records per S1) and its distractor
businesses come as *groups* of 2-3 noisy records (e.g. a "+7 house number, extra
word" sibling in both S2 and S3), whereas every train distractor is a single
record.  Entity-side context features (candidate counts / probability mass,
"support" of the record's house number / name / address by the entity's other
candidates) therefore mean something else on test: a stage-2 model trained on
train raises test sibling groups.  Stage 2 only uses context features that do not
depend on the distractor density (CTX_USE: competition for the record between
Source-1 entities, the entity's best other candidate, support of the entity's own
house number).

Density simulation (validation only).  Every distractor record of the train
universe gets a synthetic twin in the other source with the same keys and a noisy
copy of its stage-1 probabilities (~2x distractors per entity, in groups, as on
test).  The decision rule - and whether stage 2 is used at all - is chosen on the
held-out fold of this augmented universe.  (Training stage 2 on the augmented
universe does not work: exact twins are trivially recognisable, e.g. their
pairwise duplicate counts disagree with their context, and the model learns the
artefact - plain validation fell from 0.9878 to 0.9845.)
"""
from __future__ import annotations

import json
import sys

import lightgbm as lgb
import numpy as np
import polars as pl

from blocking import true_pairs_rows
from common import Timer, log, wpath
from context import CTX_COLS, context_features
from decide import exclusive, expected_f_select, final_prob, macro_f05, threshold_select
from model import FOLD_A, FOLD_B, LGB_STAGE2, VAL_FOLD, query_hash

AUG_FRAC = 1.0      # fraction of distractor records that get a twin
AUG_SIGMA = 0.7     # logit-scale noise of the twin's stage-1 probabilities
AUG_SEED = 2026
KEY_NAMES = ("q_a_hn", "q_a_street", "q_n_core", "q_a_tok", "q_hn_empty", "q_src", "q_country")


def load_keys(split):
    return dict(np.load(wpath("feat", f"{split}_ctxkeys.npz"), allow_pickle=False))


def _context(pairs: pl.DataFrame, p1: np.ndarray, keys, tag: str) -> np.ndarray:
    ctry = keys["q_country"][pairs["q_row"].to_numpy()]
    C = np.zeros((pairs.height, len(CTX_COLS)), np.float32)
    for country in np.unique(ctry):
        idx = np.flatnonzero(ctry == country)
        with Timer(f"[{tag}] context {country}: {len(idx)} pairs"):
            c = pairs[idx].select("q_row", "e_row").with_columns(pl.Series("p1", p1[idx]))
            C[idx] = context_features(c, keys, "p1").to_numpy()
    return C


def augment(pairs: pl.DataFrame, p1: np.ndarray, keys):
    """Density-augmented train universe: a synthetic twin for every distractor record.
    Returns (aug pairs with column i = source pair row, aug p1, extended keys)."""
    rng = np.random.default_rng(AUG_SEED)
    base = pairs.select("q_row", "e_row", "fold", "y").with_row_index("i")
    qmax = base.group_by("q_row").agg(pl.col("y").max().alias("ymax"))
    distr = np.sort(qmax.filter(pl.col("ymax") == 0)["q_row"].to_numpy())
    pick = distr[rng.random(len(distr)) < AUG_FRAC]
    nq = len(keys["q_src"])
    twin = pl.DataFrame({"q_row": pick, "q_new": np.arange(nq, nq + len(pick), dtype=np.int64)})
    twin = twin.with_columns(pl.col("q_row").cast(base["q_row"].dtype))
    dup = base.join(twin, on="q_row")
    noise = rng.normal(0.0, AUG_SIGMA, len(pick))
    q_noise = noise[dup["q_new"].to_numpy() - nq]
    p = np.clip(p1[dup["i"].to_numpy()].astype(np.float64), 1e-6, 1 - 1e-6)
    p_twin = 1.0 / (1.0 + np.exp(-(np.log(p / (1 - p)) + q_noise)))
    aug = pl.concat([base, dup.select("i", pl.col("q_new").cast(base["q_row"].dtype).alias("q_row"),
                                      "e_row", "fold", "y")])
    p_aug = np.concatenate([p1, p_twin]).astype(np.float32)
    k2 = dict(keys)
    for kname in KEY_NAMES:
        ext = keys[kname][pick]
        if kname == "q_src":           # the twin lives in the other source
            ext = np.where(ext == 2, 3, 2).astype(keys[kname].dtype)
        k2[kname] = np.concatenate([keys[kname], ext])
    log(f"augmentation: {len(distr)} distractor records, {len(pick)} twins, "
        f"{aug.height - pairs.height} extra pairs")
    return aug, p_aug, k2


def step_ctx(split: str):
    pairs = pl.read_parquet(wpath("feat", f"{split}_pairs.parquet"))
    p1 = np.load(wpath("feat", f"{split}_p1.npy"))
    keys = load_keys(split)
    np.save(wpath("feat", f"{split}_ctx.npy"), _context(pairs, p1, keys, split))
    if split == "train":
        aug, p_aug, k2 = augment(pairs, p1, keys)
        aug.write_parquet(wpath("feat", "train_aug_pairs.parquet"))
        np.save(wpath("feat", "train_aug_p1.npy"), p_aug)
        np.save(wpath("feat", "train_aug_ctx.npy"), _context(aug, p_aug, k2, "train-aug"))


# context features that are invariant to the distractor density (see module docstring)
CTX_USE = ["q_pmax", "q_psecond", "p_minus_qmax_other", "q_rank", "q_n", "q_psum",
           "e_pmax_other", "e_rank", "e_own_hn_other", "mutual"]
CTX_IDX = [CTX_COLS.index(c) for c in CTX_USE]


def stage2_names():
    return json.load(open(wpath("feat", "top_features.json"))) + ["p1"] + CTX_USE


def stage2_matrix(split: str, rows: np.ndarray, aug: bool = False, src_rows: np.ndarray | None = None):
    """Rows of the stage-2 design matrix.  aug=True: rows index the augmented train
    universe and src_rows maps them to original pair rows (pairwise features)."""
    names = json.load(open(wpath("feat", "feature_names.json")))
    top = json.load(open(wpath("feat", "top_features.json")))
    cols = [names.index(t) for t in top]
    X = np.load(wpath("feat", f"{split}_X.npy"), mmap_mode="r")
    tag = "train_aug" if aug else split
    C = np.load(wpath("feat", f"{tag}_ctx.npy"), mmap_mode="r")
    p1 = np.load(wpath("feat", f"{tag}_p1.npy"), mmap_mode="r")
    out = np.empty((len(rows), len(cols) + 1 + len(CTX_IDX)), np.float32)
    step = 2_000_000
    for a in range(0, len(rows), step):
        r = rows[a:a + step]
        xr = r if src_rows is None else src_rows[r]
        o = np.argsort(xr, kind="stable")
        Xr = np.empty((len(r), len(cols)), np.float32)
        Xr[o] = np.asarray(X[xr[o]])[:, cols]
        out[a:a + len(r), :len(cols)] = Xr
        out[a:a + len(r), len(cols)] = p1[r]
        out[a:a + len(r), len(cols) + 1:] = np.asarray(C[r])[:, CTX_IDX]
    return out


def evaluate(ex: pl.DataFrame, entities: pl.DataFrame, truth: pl.DataFrame, pcol="p"):
    """Grid over decision rules on (already exclusive) validation pairs."""
    res = {}
    for t in (0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9):
        res[("thr", t)] = macro_f05(threshold_select(ex, t, pcol), truth, entities)
    for lam in (0.0, 0.01, 0.03, 0.06, 0.1):
        res[("ef", lam)] = macro_f05(expected_f_select(ex, pcol, lam), truth, entities)
    for k, v in sorted(res.items(), key=lambda kv: -kv[1])[:10]:
        log(f"  {k[0]:4s} {k[1]:<5}: F0.5 = {v:.5f}")
    return max(res.items(), key=lambda kv: kv[1])


def step_train():
    pairs = pl.read_parquet(wpath("feat", "train_pairs.parquet"))
    fold = pairs["fold"].to_numpy()
    q = pairs["q_row"].to_numpy()
    y = pairs["y"].to_numpy()
    names = stage2_names()
    idx = np.flatnonzero(np.isin(fold, FOLD_A + FOLD_B))
    es = idx[query_hash(q[idx], 20, 40503) == 0]
    fit = np.setdiff1d(idx, es)
    with Timer(f"stage-2 fit on {len(fit)} rows"):
        dtr = lgb.Dataset(stage2_matrix("train", fit), y[fit], feature_name=names, free_raw_data=True)
        des = lgb.Dataset(stage2_matrix("train", es), y[es], reference=dtr)
        m = lgb.train(LGB_STAGE2, dtr, 5000, valid_sets=[des],
                      callbacks=[lgb.early_stopping(150), lgb.log_evaluation(250)])
    m.save_model(str(wpath("models", "stage2.txt")))
    imp = sorted(zip(m.feature_importance("gain"), names), reverse=True)
    log("stage-2 top features:", [(n, round(float(g))) for g, n in imp[:30]])
    del dtr, des
    step_predict("train")
    step_predict_aug()
    validate()


def validation_rows(pairs: pl.DataFrame) -> np.ndarray:
    """Rows of every record that has a candidate among the held-out (fold V) entities: the
    only rows validation needs (exclusivity is decided per record)."""
    vq = pairs.filter(pl.col("fold") == VAL_FOLD)["q_row"].unique()
    return np.flatnonzero(pairs["q_row"].is_in(vq.implode()).to_numpy())


def step_predict_aug():
    aug = pl.read_parquet(wpath("feat", "train_aug_pairs.parquet"))
    src = aug["i"].to_numpy()
    m = lgb.Booster(model_file=str(wpath("models", "stage2.txt")))
    P = np.zeros(aug.height, np.float32)
    need = validation_rows(aug)
    step = 3_000_000
    for a in range(0, len(need), step):
        rows = need[a:a + step]
        P[rows] = m.predict(stage2_matrix("train", rows, aug=True, src_rows=src), num_threads=16)
    np.save(wpath("feat", "train_aug_p2.npy"), P)


def validate():
    """Exclusivity over ALL pairs (as on test), then score the fold-V entities, on the
    plain train universe and on the density-augmented (test-like) one.  Whether the
    final probability is stage 1 alone or stage 2, and the decision rule, are chosen
    on the augmented universe."""
    keys = load_keys("train")
    entities = pl.DataFrame({"e_row": np.flatnonzero(keys["e_fold"] == VAL_FOLD)})
    truth = true_pairs_rows().filter(pl.col("e_row").is_in(entities["e_row"].implode()))
    out = {}
    for tag in ("train", "train_aug"):
        pairs = pl.read_parquet(wpath("feat", f"{tag}_pairs.parquet"))
        p1 = np.load(wpath("feat", f"{tag}_p1.npy"))
        p2 = np.load(wpath("feat", f"{tag}_p2.npy"))
        isv = pl.Series(pairs["fold"].to_numpy() == VAL_FOLD)
        base = pairs.select("q_row", "e_row")
        ex1 = exclusive(base.with_columns(pl.Series("p", p1)), "p").filter(isv)
        ex2 = exclusive(base.with_columns(pl.Series("p", final_prob(p1, p2))), "p").filter(isv)
        if tag == "train":
            ex2.write_parquet(wpath("feat", "val_pred.parquet"))
            rec = truth.join(ex2.select("q_row", "e_row"), on=["q_row", "e_row"]).height / max(1, truth.height)
            log(f"validation: {entities.height} S1 entities, {truth.height} true pairs, "
                f"{ex2.height} candidate pairs, candidate recall {rec:.5f}")
        log(f"[{tag}] stage-1 p1 decisions:")
        best1 = evaluate(ex1, entities, truth, "p")
        log(f"[{tag}] stage-2 p2 decisions:")
        best2 = evaluate(ex2, entities, truth, "p")
        log(f"[{tag}] BEST stage-1:", best1, " BEST stage-2:", best2)
        out[tag] = {"p1": best1, "final": best2}
    # choose on the density-augmented (test-like) fold: stage-1 alone or stage 2, and the rule
    prob = max(("p1", "final"), key=lambda k: out["train_aug"][k][1])
    rule, f_aug = out["train_aug"][prob]
    dec = {"prob": prob, "rule": rule[0], "param": rule[1], "f05_augmented": f_aug,
           "f05_plain_same_source_best": out["train"][prob][1]}
    log("decision:", dec)
    json.dump(dec, open(wpath("models", "decision.json"), "w"))


def step_anchor():
    """Expected number of matches per entity on the held-out fold of the density-augmented
    (test-like) universe, for the chosen probability source; output.py anchors every test
    country to it (decide.anchor_kappa)."""
    dec = json.load(open(wpath("models", "decision.json")))
    keys = load_keys("train")
    n_val = int((keys["e_fold"] == VAL_FOLD).sum())
    pairs = pl.read_parquet(wpath("feat", "train_aug_pairs.parquet")).select("q_row", "e_row", "fold")
    p = np.load(wpath("feat", "train_aug_p1.npy"))
    if dec.get("prob", "final") != "p1":
        p = final_prob(p, np.load(wpath("feat", "train_aug_p2.npy")))
    ex = exclusive(pairs.with_columns(pl.Series("p", p)), "p").filter(pl.col("fold") == VAL_FOLD)
    dec["anchor_target"] = float(ex["p"].sum()) / n_val
    log("anchor target (expected matches per entity, augmented validation):", round(dec["anchor_target"], 4))
    json.dump(dec, open(wpath("models", "decision.json"), "w"))


def step_predict(split: str):
    pairs = pl.read_parquet(wpath("feat", f"{split}_pairs.parquet"))
    m = lgb.Booster(model_file=str(wpath("models", "stage2.txt")))
    P = np.zeros(pairs.height, np.float32)
    need = validation_rows(pairs) if split == "train" else np.arange(pairs.height)
    step = 3_000_000
    for a in range(0, len(need), step):
        rows = need[a:a + step]
        P[rows] = m.predict(stage2_matrix(split, rows), num_threads=16)
    np.save(wpath("feat", f"{split}_p2.npy"), P)


if __name__ == "__main__":
    s = sys.argv[1]
    if s == "ctx":
        step_ctx(sys.argv[2])
    elif s == "train":
        step_train()
    elif s == "predict":
        step_predict(sys.argv[2]) if sys.argv[2] != "train_aug" else step_predict_aug()
    elif s == "validate":
        validate()
    elif s == "anchor":
        step_anchor()
