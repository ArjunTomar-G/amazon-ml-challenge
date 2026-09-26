"""Build a small, structure-preserving copy of the challenge data for smoke tests.

Samples a fraction of Source-1 entities, keeps their true Source-2/3 records and
every *unmatched* record on the same streets (so sibling businesses stay next to
the entity they were derived from).  The test split is sampled the same way,
without labels.  Output has the original layout (<out>/train, <out>/test).

    python make_mini.py --data-dir <dataset> --out-dir <dataset_mini> [--frac 0.03]
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pcsv

from tsvio import norm_text, read_tsv, street_key, write_tsv


def city_of(addr) -> pa.ChunkedArray:
    """Second address component (the city in the clean Source-1 format)."""
    parts = pc.split_pattern(pc.fill_null(addr, ""), ",")
    second = pc.if_else(pc.greater(pc.list_value_length(parts), 1), pc.list_element(parts, 1), "")
    return norm_text(second)


def same_street_and_city(q: pa.Table, s1_keys: pd.DataFrame) -> np.ndarray:
    """Records on a street of a sampled entity whose address also names that entity's city."""
    k = pc.binary_join_element_wise(q["country"], street_key(q["business_address"]), "|")
    pre = pc.is_in(k, value_set=pa.array(s1_keys["k"].unique()))
    idx = np.flatnonzero(pre.to_numpy(zero_copy_only=False))
    d = pd.DataFrame({"i": idx, "k": k.take(idx).to_numpy(zero_copy_only=False),
                      "a": norm_text(q["business_address"].take(idx)).to_numpy(zero_copy_only=False)})
    d = d.merge(s1_keys, on="k")
    ok = [c != "" and f" {c} " in f" {a} " for a, c in zip(d["a"], d["city"])]
    out = np.zeros(q.num_rows, bool)
    out[d["i"].to_numpy()[np.array(ok, bool)]] = True
    return out


def sample_split(data_dir, out_dir, split, frac, seed):
    rng = np.random.default_rng(seed)
    s1 = read_tsv(os.path.join(data_dir, split, f"{split}_source1.tsv"))
    keep = rng.random(s1.num_rows) < frac
    s1 = s1.filter(pa.array(keep))
    sk = street_key(s1["business_address"])
    s1_keys = pd.DataFrame({
        "k": pc.binary_join_element_wise(s1["country"], sk, "|").to_numpy(zero_copy_only=False),
        "sk": sk.to_numpy(zero_copy_only=False),
        "city": city_of(s1["business_address"]).to_numpy(zero_copy_only=False)})
    s1_keys = s1_keys[(s1_keys.sk.str.len() >= 4) & (s1_keys.city != "")][["k", "city"]].drop_duplicates()
    gt_ids = None
    if split == "train":
        gt = read_tsv(os.path.join(data_dir, split, "train_ground_truth.tsv"))
        gt = gt.filter(pc.is_in(gt["source1_entity_id"], value_set=s1["entity_id"]))
        write_tsv(gt, os.path.join(out_dir, split, "train_ground_truth.tsv"))
        all_gt = read_tsv(os.path.join(data_dir, split, "train_ground_truth.tsv"))["matched_entity_ids"]
        matched = pc.list_flatten(pc.split_pattern(pc.fill_null(all_gt, ""), ","))
        gt_ids = matched
        mine = pc.list_flatten(pc.split_pattern(pc.fill_null(gt["matched_entity_ids"], ""), ","))
    write_tsv(s1, os.path.join(out_dir, split, f"{split}_source1.tsv"))
    for s in ("source2", "source3"):
        q = read_tsv(os.path.join(data_dir, split, f"{split}_{s}.tsv"))
        on_street = pa.array(same_street_and_city(q, s1_keys))
        if split == "train":
            unmatched = pc.invert(pc.is_in(q["entity_id"], value_set=gt_ids))
            sel = pc.or_(pc.is_in(q["entity_id"], value_set=mine), pc.and_(on_street, unmatched))
        else:
            sel = pc.or_(on_street, pa.array(rng.random(q.num_rows) < frac * 0.3))
        q = q.filter(sel)
        write_tsv(q, os.path.join(out_dir, split, f"{split}_{s}.tsv"))
        print(f"{split}/{s}: {q.num_rows} records", flush=True)
    print(f"{split}/source1: {s1.num_rows} entities", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--frac", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    for split in ("train", "test"):
        os.makedirs(os.path.join(a.out_dir, split), exist_ok=True)
        sample_split(a.data_dir, a.out_dir, split, a.frac, a.seed)


if __name__ == "__main__":
    main()
