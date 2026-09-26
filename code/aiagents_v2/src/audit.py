"""Label-free audit of a matching_results.tsv (and optional comparison with another).

Needs only the raw test TSVs.  Per country it reports statistics that are
stable in the training ground truth, so a submission that departs from them has
a visible weakness:

* matches per Source-1 entity and share of empty entities;
* house-number shift of predicted pairs: +1..+13 ("up") vs -13..-1 ("down").
  In the training ground truth (same parser) US has up = down = 0.50 % and
  India up 1.01 % / down 1.13 %; siblings are shifted UP, so an excess of "up"
  over the ground-truth balance estimates the number of predicted sibling merges
  (--train-dir adds the ground-truth row for this comparison);
* sibling-group merges: predicted records whose (street, house number) is
  shared with another record that is NOT predicted for the entity while the
  pair's house number is shifted up - the test-only group pattern.

With --compare <other matching_results.tsv> it also profiles the pairs that
differ between the two files (how many added / removed pairs are up-shifted),
which tells whether a change mostly removed siblings or true matches.

    python audit.py --test-dir <dataset>/test --matching <file> [--compare <file>]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import pyarrow.compute as pc

from tsvio import norm_text, read_tsv, street_key


def load_records(test_dir, split="test"):
    def frame(path):
        t = read_tsv(path)
        first = norm_text(pc.list_element(pc.split_pattern(t["business_address"], ",", max_splits=1), 0))
        hn = pc.struct_field(pc.extract_regex(first, r"(?P<h>\d+)"), [0])
        return pd.DataFrame({
            "id": t["entity_id"].to_numpy(zero_copy_only=False),
            "country": t["country"].to_numpy(zero_copy_only=False),
            "sk": street_key(t["business_address"]).to_numpy(zero_copy_only=False),
            "hn": pd.to_numeric(pd.Series(hn.to_numpy(zero_copy_only=False)), errors="coerce").to_numpy(),
        })
    s1 = frame(f"{test_dir}/{split}_source1.tsv").set_index("id")
    q = pd.concat([frame(f"{test_dir}/{split}_source{s}.tsv") for s in (2, 3)]).set_index("id")
    return s1, q


def pairs_of(path):
    t = read_tsv(path).to_pandas()
    col = t.columns[1]
    t[col] = t[col].str.split(",")
    t = t.explode(col)
    t = t[t[col].notna() & (t[col] != "")]
    return pd.DataFrame({"eid": t.iloc[:, 0].to_numpy(), "rid": t[col].to_numpy()})


def annotate(p, s1, q):
    p = p.copy()
    p["country"] = s1.country.reindex(p.eid).to_numpy()
    d = q.hn.reindex(p.rid).to_numpy() - s1.hn.reindex(p.eid).to_numpy()
    p["up"] = (d >= 1) & (d <= 13)
    p["down"] = (d <= -1) & (d >= -13)
    p["sk_q"] = q.sk.reindex(p.rid).to_numpy()
    p["hn_q"] = q.hn.reindex(p.rid).to_numpy()
    return p


def audit(pred, s1, q, cand=None):
    out = {}
    per = pred.groupby("eid").size()
    if cand is not None:
        # records on the same street + house number as a predicted up-shifted record,
        # among the entity's candidates but not predicted -> the pair is part of a group
        c = cand.merge(pred[["eid", "rid"]].assign(pred=1), how="left").fillna({"pred": 0})
        c["sk_q"] = q.sk.reindex(c.rid).to_numpy()
        c["hn_q"] = q.hn.reindex(c.rid).to_numpy()
        grp = c.groupby(["eid", "sk_q", "hn_q"]).size().rename("n_same_addr")
        pred = pred.join(grp, on=["eid", "sk_q", "hn_q"])
    for country, ents in s1.groupby("country"):
        e = ents.index
        k = per.reindex(e).fillna(0)
        p = pred[pred.country == country]
        row = {
            "entities": len(e),
            "matches_per_s1": round(float(k.mean()), 4),
            "empty_share": round(float((k == 0).mean()), 4),
            "pairs": len(p),
            "up_shift_share": round(float(p.up.mean()), 5),
            "down_shift_share": round(float(p.down.mean()), 5),
            "est_sibling_merges(up-down)": int(p.up.sum() - p.down.sum()),
            "entities_whose_matches_are_all_up": int(p.groupby("eid").up.all().sum()),
        }
        if "n_same_addr" in p:
            row["up_pairs_in_same_address_group"] = int((p.up & (p.n_same_addr >= 2)).sum())
        out[country] = row
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test-dir", required=True)
    ap.add_argument("--matching", required=True)
    ap.add_argument("--candidates", default=None, help="candidate_pairs.tsv (enables the group statistic)")
    ap.add_argument("--compare", default=None, help="another matching_results.tsv")
    ap.add_argument("--json", default=None, help="write the report here")
    ap.add_argument("--train-dir", default=None,
                    help="<dataset>/train: also report the same statistics for the training ground truth")
    a = ap.parse_args()
    rep = {}
    if a.train_dir:
        s1t, qt = load_records(a.train_dir, "train")
        truth = annotate(pairs_of(f"{a.train_dir}/train_ground_truth.tsv"), s1t, qt)
        rep["train_ground_truth"] = audit(truth, s1t, qt)
        del s1t, qt, truth
    s1, q = load_records(a.test_dir)
    pred = annotate(pairs_of(a.matching), s1, q)
    cand = pairs_of(a.candidates) if a.candidates else None
    rep.update({"file": a.matching, "per_country": audit(pred, s1, q, cand)})
    if a.compare:
        other = annotate(pairs_of(a.compare), s1, q)
        k = ["eid", "rid"]
        added = pred.merge(other[k], how="left", indicator=True).query("_merge == 'left_only'")
        removed = other.merge(pred[k], how="left", indicator=True).query("_merge == 'left_only'")
        rep["vs"] = a.compare
        rep["diff"] = {}
        for c in sorted(s1.country.unique()):
            ad, rm = added[added.country == c], removed[removed.country == c]
            rep["diff"][c] = {"added": len(ad), "added_up_share": round(float(ad.up.mean()), 4) if len(ad) else None,
                              "removed": len(rm), "removed_up_share": round(float(rm.up.mean()), 4) if len(rm) else None}
    print(json.dumps(rep, indent=1))
    if a.json:
        json.dump(rep, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
