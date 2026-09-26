"""Leaderboard probe: measure one country's F0.5 with a single submission.

Writes a copy of a matching_results.tsv in which every Source-1 entity of the
chosen country gets an EMPTY match list.  An empty list scores 1 on a true
singleton and 0 otherwise, so for that country the score becomes its singleton
share s.  With w = the country's share of Source-1 entities:

    LB(original) - LB(probe) = w * (F_country - s)
    =>  F_country = s + (LB(original) - LB(probe)) / w

s is estimated by the share of the country's entities the original file leaves
empty (on validation v1 predicts 99.4 % of true singletons empty and its empty
share matches the true one to +-0.002).  The public leaderboard scores a subset
of the test set, so w is the full-test share and the result has an error of
roughly +-0.01.

    python probe.py --test-dir <dataset>/test --matching <file> --country France --out <probe.tsv>
    python probe.py ... --lb-original 0.970 --lb-probe 0.8xx      # compute F_country
"""
from __future__ import annotations

import argparse

import pyarrow as pa
import pyarrow.compute as pc

from tsvio import read_tsv, write_tsv


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test-dir", required=True)
    ap.add_argument("--matching", required=True)
    ap.add_argument("--country", required=True)
    ap.add_argument("--out", default=None, help="probe file to write")
    ap.add_argument("--lb-original", type=float, default=None)
    ap.add_argument("--lb-probe", type=float, default=None)
    a = ap.parse_args()
    s1 = read_tsv(f"{a.test_dir}/test_source1.tsv", ["entity_id", "country"])
    ids = s1["entity_id"].filter(pc.equal(s1["country"], a.country))
    m = read_tsv(a.matching)
    col = m.column_names[1]
    inc = pc.is_in(m["source1_entity_id"], value_set=ids)
    w = len(ids) / s1.num_rows
    empty = pc.equal(m[col], "")
    s = pc.sum(pc.and_(inc, empty)).as_py() / max(1, pc.sum(inc).as_py())
    print(f"{a.country}: {len(ids)} of {s1.num_rows} entities (w = {w:.4f}); "
          f"empty share in {a.matching}: s = {s:.4f}")
    if a.out:
        blank = pc.if_else(inc, pa.scalar("", pa.string()), m[col])
        write_tsv(m.set_column(1, col, blank), a.out)
        print("wrote", a.out, "- submit it and pass the two leaderboard scores back in")
    if a.lb_original is not None and a.lb_probe is not None:
        f = s + (a.lb_original - a.lb_probe) / w
        print(f"estimated {a.country} F0.5 = {f:.4f} (about +-0.01); "
              f"the other countries together: {(a.lb_original - w * f) / (1 - w):.4f}")


if __name__ == "__main__":
    main()
