"""v13 = the v12 files + the US / India decisions that the larger cross-encoders change.

    python build_v13.py --v12 <dir with v12 TSVs> --old-p <test_pfinal.npy of the v12 run> --out <dir>
                        [--new-p feat/test_pfinal.npy] [--old-r1 rescue_v10/test_rescue_pred.parquet ...]

Both pipelines are turned into the same US / India decision, as on validation: record argmax of the stacked
probability, p >= thr, the v12 rank rule (the strongest record of an S1 at p >= rank-t1), then the first rescue
pass and the second rescue pass (best pair >= thr, records still unlinked).  OLD = this work dir's v12 run (stacker
and rescue models with e5-small features), NEW = the same with the e5-base cross-encoder features.  A record whose
decision differs between OLD and NEW takes the NEW decision only where the v12 file agrees with OLD, so the change is
isolated from the differences between the team's v10c run and this rebuild (as build_v11.py does for its changes).
France is left as in v12.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

from blocking import load_universe
from common import Timer, log, wpath
from output import _id_lists


def explode(path, col):
    d = pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False)
    return (d.with_columns(pl.col(col).fill_null("").str.split(",")).explode(col)
            .filter(pl.col(col) != "").rename({"source1_entity_id": "entity_id", col: "q_id"}))


def decide(pairs: pl.DataFrame, p: np.ndarray, thr: float, t1: float | None) -> pl.DataFrame:
    """US / India stacked decisions: (q_row, e_row) of the linked records."""
    d = pairs.with_columns(pl.Series("p", p[pairs["i"].to_numpy()]))
    d = (d.sort(["q_row", "p"], descending=[False, True]).group_by("q_row", maintain_order=True)
          .agg(pl.col("e_row").first(), pl.col("p").first()))
    if t1 is None:
        return d.filter(pl.col("p") >= thr).select("q_row", "e_row")
    d = d.with_columns(pl.col("p").rank("ordinal", descending=True).over("e_row").alias("rk"))
    return d.filter(pl.when(pl.col("rk") == 1).then(pl.col("p") >= t1).otherwise(pl.col("p") >= thr)).select("q_row", "e_row")


def add_rescue(L: pl.DataFrame, path: str | None, thr: float) -> pl.DataFrame:
    """Best rescue pair (>= thr) of every record not linked yet."""
    if not path:
        return L
    r = pl.read_parquet(wpath(*path.split("/"))).select(pl.col("q_row").cast(pl.Int32), pl.col("e_row").cast(pl.Int32), "prob")
    best = r.sort("prob", descending=True).unique("q_row", keep="first").filter(pl.col("prob") >= thr)
    return pl.concat([L, best.join(L.select("q_row"), on="q_row", how="anti").select("q_row", "e_row")])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v12", required=True)
    ap.add_argument("--old-p", required=True)
    ap.add_argument("--new-p", default=None, help="default: feat/test_pfinal.npy")
    ap.add_argument("--old-r1", default=None)
    ap.add_argument("--new-r1", default=None)
    ap.add_argument("--old-r2", default=None)
    ap.add_argument("--new-r2", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--thr", type=float, default=0.8)
    ap.add_argument("--rank-t1", type=float, default=0.65)
    a = ap.parse_args()
    s1, q = load_universe("test", ["entity_id", "country"])
    s1_ids = s1.select(pl.col("row").cast(pl.Int32).alias("e_row"), "entity_id")
    q_ids = q.select(pl.col("row").cast(pl.Int32).alias("q_row"), pl.col("entity_id").alias("q_id"))
    links = (explode(Path(a.v12) / "matching_results.tsv", "matched_entity_ids")
             .join(s1_ids, on="entity_id").join(q_ids, on="q_id").select("e_row", "q_row"))
    cands = (explode(Path(a.v12) / "candidate_pairs.tsv", "candidate_entity_ids")
             .join(s1_ids, on="entity_id").join(q_ids, on="q_id").select("e_row", "q_row"))
    pairs = pl.read_parquet(wpath("feat", "test_pairs.parquet")).select(pl.col("q_row").cast(pl.Int32), pl.col("e_row").cast(pl.Int32))
    kt = np.load(wpath("feat", "test_ctxkeys.npz"))
    ok = np.isin(kt["q_country"][pairs["q_row"].to_numpy()], ["US", "India"])
    pairs = pairs.with_row_index("i").filter(pl.Series(ok))
    wp = lambda x: x if Path(x).is_absolute() else str(wpath(*x.split("/")))   # absolute, or relative to the work dir
    new_p = wp(a.new_p or "feat/test_pfinal.npy")
    old = decide(pairs, np.load(wp(a.old_p)), a.thr, a.rank_t1)
    old = add_rescue(add_rescue(old, a.old_r1, a.thr), a.old_r2, a.thr)
    new = decide(pairs, np.load(new_p), a.thr, a.rank_t1)
    new = add_rescue(add_rescue(new, a.new_r1, a.thr), a.new_r2, a.thr)
    usin = q_ids.select("q_row").filter(pl.Series(np.isin(kt["q_country"][q_ids["q_row"].to_numpy()], ["US", "India"])))
    d = (usin.join(old.rename({"e_row": "eo"}), on="q_row", how="left")
             .join(new.rename({"e_row": "en"}), on="q_row", how="left")
             .join(links.rename({"e_row": "ev"}), on="q_row", how="left"))
    F = lambda c: pl.col(c).fill_null(-1)
    changed = d.filter(F("eo") != F("en"))
    agree = changed.filter(F("ev") == F("eo"))
    log(f"US / India records whose decision changes: {changed.height}; the v12 file agrees with the old decision on "
        f"{agree.height}, already has the new one on {changed.filter(F('ev') == F('en')).height}")
    drop = agree.filter(pl.col("ev").is_not_null()).select("q_row")
    add = agree.filter(pl.col("en").is_not_null()).select(pl.col("en").alias("e_row"), "q_row")
    re = add.join(drop, on="q_row").height
    links = pl.concat([links.join(drop, on="q_row", how="anti"), add.select("e_row", "q_row")])
    log(f"removed {drop.height - re} links, added {add.height - re}, re-assigned {re}")
    assert links["q_row"].n_unique() == links.height, "a record linked twice"
    extra = [add.select("e_row", "q_row")]
    for path in (a.new_r1, a.new_r2):
        if path:
            extra.append(pl.read_parquet(wpath(*path.split("/"))).select(pl.col("e_row").cast(pl.Int32), pl.col("q_row").cast(pl.Int32)))
    cands = pl.concat([cands] + extra).unique()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with Timer("write TSVs"):
        _id_lists(s1_ids, links, q_ids, "matched_entity_ids").write_csv(out / "matching_results.tsv", separator="\t", quote_style="never")
        _id_lists(s1_ids, cands, q_ids, "candidate_entity_ids").write_csv(out / "candidate_pairs.tsv", separator="\t", quote_style="never")
    log(f"wrote {out}: {links.height} links, {cands.height} candidate pairs")


if __name__ == "__main__":
    main()
