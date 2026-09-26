"""Decision-layer variants on an existing v1 work dir (no retraining, minutes).

Starts from the stage-1 / stage-2 probabilities that the v1 run already saved
(feat/{train,test}_{p1,p2}.npy) and writes one matching_results.tsv per variant,
so that leaderboard submissions can compare them.  Every variant is also scored
on the v1 validation fold (where the change applies there) and summarised per
test country with label-free statistics (matches per entity, empty rate, pairs
added / removed compared with the first variant).

A variant is a '+'-joined list of modifiers (order does not matter):

    v1             the v1 decision (raise guard 0.1); alone it reproduces the submitted file
    guard<g>       stricter raise guard: stage 2 may not raise pairs with p1 < g
    lower_only     stage 2 may only lower probabilities: p = min(p1, p2)
    em / emhalf    per-country prior-shift correction (Saerens EM), full or half strength
                   (half = prior moved half-way in log-odds).  The test universe has ~2x
                   more hard negatives per entity; full EM assumes they look like the
                   training negatives, which sibling groups violate - hence the half version
    unseen_thr<t>  countries without training labels (France) use threshold t instead of
                   the v1 rule; loco_full.py shows an unlabelled country loses precision,
                   and a stricter rule recovers most of it
    unseen_ef<l>   same with expected-F0.5 and missed-match rate l
    st             use the self-trained probabilities for unlabelled countries
                   (feat/test_p{1,2}_st.npy written by selftrain.py)

    python variants.py --work-dir <v1 work> --out-dir <out>
                       [--variants v1 guard0.3 emhalf unseen_thr0.9 st st+unseen_thr0.9 ...]
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

import v1path

DEFAULT = ["v1", "guard0.3", "lower_only", "emhalf", "unseen_thr0.8", "unseen_thr0.9"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def parse(name: str) -> dict:
    v = {"guard": 0.1, "lower_only": False, "em": None, "unseen": None, "st": False}
    for m in name.split("+"):
        if m == "v1":
            continue
        if m.startswith("guard"):
            v["guard"] = float(m[5:])
        elif m == "lower_only":
            v["lower_only"] = True
        elif m in ("em", "emhalf"):
            v["em"] = 1.0 if m == "em" else 0.5
        elif m.startswith("unseen_thr"):
            v["unseen"] = ("thr", float(m[10:]))
        elif m.startswith("unseen_ef"):
            v["unseen"] = ("ef", float(m[9:]))
        elif m == "st":
            v["st"] = True
        else:
            raise SystemExit(f"unknown modifier {m!r} in variant {name!r}")
    return v


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work-dir", required=True, help="work dir of a finished v1 run")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--variants", nargs="*", default=DEFAULT)
    a = ap.parse_args()
    v1path.use(work_dir=a.work_dir)

    from pathlib import Path

    import polars as pl
    from blocking import load_universe, true_pairs_rows
    from common import wpath
    from decide import adjust_prior, em_prior, exclusive, expected_f_select, macro_f05, threshold_select
    from model import VAL_FOLD
    from output import _id_lists

    out = Path(a.out_dir)
    dec = json.load(open(wpath("models", "decision.json")))
    v1_rule = (dec["rule"], float(dec["param"]))
    log("v1 decision rule:", dec)

    def select(ex, rule):
        if rule[0] == "thr":
            return threshold_select(ex, rule[1], "p")
        return expected_f_select(ex, "p", rule[1])

    def prob(p1, p2, v):
        if v["lower_only"]:
            return np.minimum(p1, p2).astype(np.float32)
        return np.where(p1 < v["guard"], np.minimum(p1, p2), p2).astype(np.float32)

    def em_correct(p, country, pi_train, strength):
        q = p.astype(np.float64).copy()
        info = {}
        lo = lambda x: np.log(x / (1 - x))
        for c in np.unique(country):
            m = country == c
            pi_new = em_prior(q[m], pi_train)
            z = lo(pi_train) + strength * (lo(pi_new) - lo(pi_train))
            pi_new = float(1 / (1 + np.exp(-z)))
            q[m] = adjust_prior(q[m], pi_train, pi_new)
            info[str(c)] = round(pi_new, 4)
        return q.astype(np.float32), info

    # ---- data ----
    tr = pl.read_parquet(wpath("feat", "train_pairs.parquet"))
    tr_p1, tr_p2 = np.load(wpath("feat", "train_p1.npy")), np.load(wpath("feat", "train_p2.npy"))
    ktr = dict(np.load(wpath("feat", "train_ctxkeys.npz")))
    te = pl.read_parquet(wpath("feat", "test_pairs.parquet"))
    te_p = {False: (np.load(wpath("feat", "test_p1.npy")), np.load(wpath("feat", "test_p2.npy")))}
    if wpath("feat", "test_p1_st.npy").exists():
        te_p[True] = (np.load(wpath("feat", "test_p1_st.npy")), np.load(wpath("feat", "test_p2_st.npy")))
    kte = dict(np.load(wpath("feat", "test_ctxkeys.npz")))
    y = tr["y"].to_numpy()
    fold = tr["fold"].to_numpy()
    pi_train = float(y[fold != VAL_FOLD].mean())
    tr_country = ktr["q_country"][tr["q_row"].to_numpy()]
    te_country = kte["q_country"][te["q_row"].to_numpy()]
    unseen = sorted(set(np.unique(te_country)) - set(np.unique(tr_country)))
    val_ents = pl.DataFrame({"e_row": np.flatnonzero(ktr["e_fold"] == VAL_FOLD)})
    truth = true_pairs_rows().filter(pl.col("e_row").is_in(val_ents["e_row"].implode()))
    isv = pl.Series(fold == VAL_FOLD)
    tr_base, te_base = tr.select("q_row", "e_row"), te.select("q_row", "e_row")
    s1, q = load_universe("test", ["entity_id"])
    s1_ids = s1.select(pl.col("row").cast(pl.Int32).alias("e_row"), "entity_id")
    q_ids = q.select(pl.col("row").cast(pl.Int32).alias("q_row"), pl.col("entity_id").alias("q_id"))
    e_country = kte["e_country"]
    e_unseen = pl.Series(np.isin(e_country, unseen))
    log(f"pair prior in train candidates (folds != V): {pi_train:.4f}; countries without labels: {unseen}")

    out.mkdir(parents=True, exist_ok=True)
    cand = _id_lists(s1_ids, te_base.select(pl.col("q_row").cast(pl.Int32), pl.col("e_row").cast(pl.Int32)),
                     q_ids, "candidate_entity_ids")
    cand.write_csv(out / "candidate_pairs.tsv", separator="\t", quote_style="never")

    report, ref = [], None
    for name in a.variants:
        v = parse(name)
        if v["st"] and True not in te_p:
            log(f"skip {name}: run selftrain.py first (no feat/test_p1_st.npy)")
            continue
        pv = prob(tr_p1, tr_p2, v)
        pt = prob(*te_p[v["st"]], v)
        info = {}
        if v["em"] is not None:
            pv, _ = em_correct(pv, tr_country, pi_train, v["em"])      # validation: ~ the train prior
            pt, info = em_correct(pt, te_country, pi_train, v["em"])
        # validation fold (train universe; exclusivity over all pairs, as in v1)
        exv = exclusive(tr_base.with_columns(pl.Series("p", pv)), "p").filter(isv)
        f_val = macro_f05(select(exv, v1_rule), truth, val_ents)
        # test: exclusivity over all pairs, then the rule of each entity's country
        ex = exclusive(te_base.with_columns(pl.Series("p", pt)), "p")
        if v["unseen"] and unseen:
            is_u = pl.Series(np.isin(e_country[ex["e_row"].to_numpy()], unseen))
            sel = pl.concat([select(ex.filter(~is_u), v1_rule), select(ex.filter(is_u), v["unseen"])])
        else:
            sel = select(ex, v1_rule)
        sel = sel.select(pl.col("e_row").cast(pl.Int32), pl.col("q_row").cast(pl.Int32))
        per = np.bincount(sel["e_row"].to_numpy(), minlength=len(e_country))
        row = {"variant": name, "val_f05": round(f_val, 5), "test_pairs": sel.height}
        for c in np.unique(e_country):
            m = e_country == c
            row[f"{c}_matches_per_s1"] = round(float(per[m].mean()), 4)
            row[f"{c}_empty"] = round(float((per[m] == 0).mean()), 4)
        if info:
            row["em_prior_test"] = info
        if ref is None:
            ref = sel
        else:
            row["pairs_added_vs_first"] = sel.join(ref, on=["e_row", "q_row"], how="anti").height
            row["pairs_removed_vs_first"] = ref.join(sel, on=["e_row", "q_row"], how="anti").height
        d = out / "variants" / name
        d.mkdir(parents=True, exist_ok=True)
        _id_lists(s1_ids, sel, q_ids, "matched_entity_ids").write_csv(
            d / "matching_results.tsv", separator="\t", quote_style="never")
        report.append(row)
        log(json.dumps(row))
    json.dump(report, open(out / "variants_report.json", "w"), indent=1)
    log("wrote", out / "variants", "and variants_report.json")


if __name__ == "__main__":
    main()
