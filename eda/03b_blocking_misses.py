"""What does the recommended candidate union (U8@K) miss? Categorise the true pairs that fall
outside a union of saved candidate lists and show examples.

    python 03b_blocking_misses.py [--k 25]   -> results/blocking_misses.json
"""

import argparse
import json
import os
import sys

import numpy as np
import pyarrow as pa
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

from erlib import blocking as B
from erlib import data

bl = __import__("03_blocking")


def best_union_lists(corpus, lists_dir, k, k_empty=10):
    """U8@k: joint name+address IDF top-k (R2)  U  name-MinHash top-k (R5)
    U  name-IDF top-k_empty among S2/S3 records that have no address (from R3's top-200)."""
    def topk_csr(sid, kk):
        return B.topk_lists({"topk": np.load(os.path.join(lists_dir, f"{sid}_topk.npy"))[:, :kk]})
    aempty = corpus.column("a_empty").combine_chunks().to_numpy(zero_copy_only=False)
    t3 = np.load(os.path.join(lists_dir, "R3_topk.npy"))
    emp = np.full((t3.shape[0], k_empty), -1, np.int32)
    for i in range(t3.shape[0]):
        row = t3[i][t3[i] >= 0]
        row = row[aempty[row]][:k_empty]
        emp[i, :len(row)] = row
    return [topk_csr("R2", k), topk_csr("R5", k), B.topk_lists({"topk": emp})]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=25)
    args = ap.parse_args()
    lists_dir = os.path.join(data.CACHE_DIR, "blocking")
    corpus = B.Corpus("train")
    s1_all = data.read_prepared("train", 1, columns=bl.S1_COLS + ["name_raw", "addr_raw"])
    rng = np.random.default_rng(42)
    pick = np.sort(rng.choice(s1_all.num_rows, size=20000, replace=False))
    s1 = s1_all.take(pa.array(pick))
    del s1_all
    _, pair_s1, pair_src, pair_id = data.load_ground_truth()
    sample = B.Sample(s1, corpus, pair_s1, pair_src, pair_id)

    lists = best_union_lists(corpus, lists_dir, args.k)
    res = B.union(lists, sample)
    missed = ~res["pair_found"]
    mi = np.nonzero(missed)[0]
    recs = sample.pair_rec[mi]
    rows = sample.pair_sample[mi]
    cand = corpus.take(recs, ["name_raw", "addr_raw", "name_core", "addr_core", "name_script", "a_empty"])
    A = s1.take(pa.array(rows))
    a_name, b_name = A.column("name_core").to_pylist(), cand.column("name_core").to_pylist()
    a_addr, b_addr = A.column("addr_core").to_pylist(), cand.column("addr_core").to_pylist()
    ntset = cpdist(a_name, b_name, scorer=fuzz.token_set_ratio, workers=-1) / 100
    atset = cpdist(a_addr, b_addr, scorer=fuzz.token_set_ratio, workers=-1) / 100
    empty = np.asarray(cand.column("a_empty").to_pylist(), bool)
    indic = ~np.isin(np.asarray(cand.column("name_script").to_pylist()), ["ASCII", "LATIN_EXT"])
    cats = {
        "address empty on S2/S3 side": empty,
        "Indic-script name": indic,
        "name token-set sim < 0.5": ntset < 0.5,
        "address token-set sim < 0.5 (non-empty)": (~empty) & (atset < 0.5),
        "name >= 0.8 but lost in ranking": ntset >= 0.8,
        "address >= 0.8 but lost in ranking": (~empty) & (atset >= 0.8),
        "both name < 0.5 and (address empty or < 0.5)": (ntset < 0.5) & (empty | (atset < 0.5)),
    }
    out = {"k": args.k, "n_true_pairs": int(len(sample.pair_rec)), "n_missed": int(len(mi)),
           "recall": float(res["pair_found"].mean()),
           "categories": {k: float(v.mean()) for k, v in cats.items()},
           "gt_size_of_missed": {str(k): int(v) for k, v in zip(*np.unique(np.minimum(sample.gt_count[rows], 6),
                                                                               return_counts=True))}}
    ex_idx = rng.choice(len(mi), size=min(40, len(mi)), replace=False)
    out["examples"] = [{
        "s1_name": A.column("name_raw")[int(i)].as_py(), "s1_addr": A.column("addr_raw")[int(i)].as_py(),
        "m_name": cand.column("name_raw")[int(i)].as_py(), "m_addr": cand.column("addr_raw")[int(i)].as_py(),
        "name_tset": float(ntset[i]), "addr_tset": float(atset[i]),
    } for i in ex_idx]
    with open(os.path.join(data.RESULTS_DIR, "blocking_misses.json"), "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in out.items() if k != "examples"}, indent=1))
    for e in out["examples"][:25]:
        print(f"- S1: {e['s1_name']!r} | {e['s1_addr']!r}\n  M : {e['m_name']!r} | {e['m_addr']!r}  "
              f"(name {e['name_tset']:.2f}, addr {e['addr_tset']:.2f})")


if __name__ == "__main__":
    sys.exit(main())
