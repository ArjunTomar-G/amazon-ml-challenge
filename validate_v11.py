"""Local re-implementation of the challenge's submission checks (the official validate_submission.py is not on
this machine).  Checks every rule of the problem statement plus two consistency checks.

    python validate_v11.py <out_dir> <test_dataset_dir>
"""
import sys

import polars as pl

out, test = sys.argv[1], sys.argv[2]
ok = True


def fail(msg):
    global ok
    ok = False
    print("FAIL:", msg)


s1 = pl.read_csv(f"{test}/test_source1.tsv", separator="\t", quote_char=None, infer_schema=False, columns=["entity_id", "country"])
q = pl.concat([pl.read_csv(f"{test}/test_source{i}.tsv", separator="\t", quote_char=None, infer_schema=False, columns=["entity_id", "country"]) for i in (2, 3)])
qids = set(q["entity_id"].to_list())
country = dict(zip(s1["entity_id"].to_list(), s1["country"].to_list())) | dict(zip(q["entity_id"].to_list(), q["country"].to_list()))
lists = {}
for fname, col in (("matching_results.tsv", "matched_entity_ids"), ("candidate_pairs.tsv", "candidate_entity_ids")):
    with open(f"{out}/{fname}", encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n")
    if header != f"source1_entity_id\t{col}":
        fail(f"{fname}: header {header!r}")
    d = pl.read_csv(f"{out}/{fname}", separator="\t", quote_char=None, infer_schema=False)
    if d.columns != ["source1_entity_id", col]:
        fail(f"{fname}: columns {d.columns}")
    if d.height != s1.height or d["source1_entity_id"].n_unique() != d.height:
        fail(f"{fname}: {d.height} rows, {d['source1_entity_id'].n_unique()} unique, {s1.height} test S1")
    if set(d["source1_entity_id"].to_list()) != set(s1["entity_id"].to_list()):
        fail(f"{fname}: S1 ids differ from test_source1")
    ex = d.with_columns(pl.col(col).fill_null("").str.split(",")).explode(col).filter(pl.col(col) != "")
    dup = ex.group_by(["source1_entity_id", col]).len().filter(pl.col("len") > 1).height
    if dup:
        fail(f"{fname}: {dup} duplicate ids inside lists")
    bad = [x for x in ex[col].unique().to_list() if x not in qids]
    if bad:
        fail(f"{fname}: {len(bad)} ids not in test S2/S3 (e.g. {bad[:3]})")
    if ex[col].str.starts_with("S1-").any():
        fail(f"{fname}: S1 ids inside lists")
    lists[fname] = ex.rename({col: "q"})
    print(f"{fname}: {d.height} rows, {ex.height} ids, {(d[col].fill_null('') == '').sum()} empty lists")
m, c = lists["matching_results.tsv"], lists["candidate_pairs.tsv"]
miss = m.join(c, on=["source1_entity_id", "q"], how="anti").height
if miss:
    fail(f"{miss} matched ids are not candidates of their S1")
multi = m.group_by("q").len().filter(pl.col("len") > 1).height
if multi:
    fail(f"{multi} records linked to more than one S1")
cross = sum(country[a] != country[b] for a, b in zip(m["source1_entity_id"].to_list(), m["q"].to_list()))
if cross:
    fail(f"{cross} cross-country links")
per = m.with_columns(pl.col("source1_entity_id").replace_strict(country, default=None).alias("country")).group_by("country").len()
print("links by country:", dict(zip(per["country"].to_list(), per["len"].to_list())))
print("PASS" if ok else "NOT OK")
sys.exit(0 if ok else 1)
