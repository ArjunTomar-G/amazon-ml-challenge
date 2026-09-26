"""Data profiling: field lengths, formats, scripts, noise flags, missing fields,
within-source duplicates, ground-truth match-set sizes and exact-agreement rates
of true pairs. A label-free format check of the unseen test country (France)
is included to see how US/India-specific rules generalise.

    python 02_profile.py            -> results/profile.json
"""

import json
import os
import sys
import time
from collections import Counter

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from erlib import data
from erlib import textnorm as tn

PCTS = [1, 5, 25, 50, 75, 95, 99]
FLAG_COLS = ["f_alias", "f_domain", "f_phone", "f_hashat", "f_allcaps", "f_brackets", "f_honor",
             "f_legal", "f_dash", "f_concat", "a_empty", "a_null", "a_landmark", "a_unit", "a_pobox",
             "a_co", "a_hash", "a_startnum"]


def hash_cols(tbl, cols):
    """uint64 hash of the concatenation of several string columns."""
    import pandas as pd
    joined = tbl.column(cols[0])
    for c in cols[1:]:
        joined = pc.binary_join_element_wise(joined, tbl.column(c), "\x1f")
    joined = joined.combine_chunks() if isinstance(joined, pa.ChunkedArray) else joined
    d = pc.dictionary_encode(joined)
    h = pd.util.hash_array(d.dictionary.to_numpy(zero_copy_only=False), categorize=False)
    return h[d.indices.to_numpy(zero_copy_only=False)]


def dup_rate(h):
    """Fraction of records whose key occurs more than once, and 1 - unique/total."""
    _, inv, cnt = np.unique(h, return_inverse=True, return_counts=True)
    return {"in_dup_group": float(np.mean(cnt[inv] > 1)), "redundant": float(1 - len(cnt) / len(h))}


def pct(arr):
    return {f"p{p}": float(v) for p, v in zip(PCTS, np.percentile(arr, PCTS))} | {"mean": float(np.mean(arr))}


def _top_tokens(col, mask, n, k=30):
    toks = pc.list_flatten(pc.utf8_split_whitespace(col.filter(pa.array(mask))))
    vc = pc.value_counts(toks)
    vals = vc.field("values").to_pylist()
    cnts = vc.field("counts").to_numpy()
    order = np.argsort(-cnts)[:k]
    return [(vals[i], float(cnts[i] / n)) for i in order]


def _vc(arr):
    vc = pc.value_counts(arr)
    return dict(zip(vc.field("values").to_pylist(), vc.field("counts").to_numpy().tolist()))


def profile_source(src):
    small = ["cty", "name_script", "addr_script", "state", "zip5", "pin6", "a_pattern", "name_len",
             "name_ntok", "name_ncore", "addr_len", "addr_ncomp"] + FLAG_COLS
    t = data.read_prepared("train", src, columns=small)
    cty_col = t.column("cty").combine_chunks()
    countries = sorted(_vc(cty_col))
    masks = {c: pc.equal(cty_col, c).to_numpy(zero_copy_only=False) for c in countries}
    out = {c: {"n": int(m.sum())} for c, m in masks.items()}
    for c, m in masks.items():
        r = out[c]
        am = pa.array(m)
        tm = t.filter(am)
        for col in ("name_len", "name_ntok", "name_ncore", "addr_len", "addr_ncomp"):
            r[col] = pct(tm.column(col).to_numpy())
        for col in FLAG_COLS:
            r[col] = float(np.mean(tm.column(col).to_numpy(zero_copy_only=False)))
        for col in ("name_script", "addr_script"):
            vc = _vc(tm.column(col).combine_chunks())
            r[col] = {k: v / r["n"] for k, v in sorted(vc.items(), key=lambda x: -x[1])}
        for col in ("state", "zip5", "pin6"):
            r[col + "_found"] = float(np.mean(pc.not_equal(tm.column(col), "").to_numpy(zero_copy_only=False)))
        pats = tm.column("a_pattern").combine_chunks()
        vc = sorted(_vc(pats).items(), key=lambda x: -x[1])[:8]
        r["a_pattern_top"] = [(k if k else "(empty)", v / r["n"]) for k, v in vc]
        first = _vc(pc.utf8_slice_codeunits(pats, 0, 1))
        r["first_comp_type"] = {(k if k else "(empty)"): v / r["n"] for k, v in sorted(first.items(), key=lambda x: -x[1])}
        del tm
    del t
    # within-source duplicates (hash one column group at a time)
    for label, cols in (("dup_exact_raw", ["name_raw", "addr_raw"]),
                        ("dup_norm_name_addr", ["name_n", "addr_n"]),
                        ("dup_core_name", ["name_core"]),
                        ("dup_core_addr", ["addr_core"])):
        tt = data.read_prepared("train", src, columns=cols)
        h = hash_cols(tt, cols)
        nonempty = None
        if label == "dup_core_addr":
            nonempty = tt.column("addr_core").to_numpy(zero_copy_only=False) != ""
        del tt
        for c, m in masks.items():
            mm = m if nonempty is None else (m & nonempty)
            out[c][label] = dup_rate(h[mm])
        del h
    # empty normalised fields + most frequent tokens
    for col, key in (("name_n", "top_name_tokens"), ("addr_core", "top_addr_tokens")):
        tt = data.read_prepared("train", src, columns=[col])
        cc = tt.column(col).combine_chunks()
        empt = cc.to_numpy(zero_copy_only=False) == ""
        for c, m in masks.items():
            out[c][col + "_empty"] = float(np.mean(empt[m]))
            out[c][key] = _top_tokens(cc, m, out[c]["n"])
        del tt, cc
    return out


def profile_gt():
    s1 = data.read_prepared("train", 1, columns=["id", "cty"])
    s1_ids = s1.column("id").to_numpy()
    s1_cty = np.asarray(s1.column("cty").to_pylist())  # 2.2M short strings
    gt_s1, pair_s1, pair_src, pair_id = data.load_ground_truth()
    order = np.argsort(s1_ids)
    pos_gt = order[np.searchsorted(s1_ids[order], gt_s1)]
    n2 = np.zeros(len(s1_ids), np.int64)
    n3 = np.zeros(len(s1_ids), np.int64)
    ppos = order[np.searchsorted(s1_ids[order], pair_s1)]
    np.add.at(n2, ppos[pair_src == 2], 1)
    np.add.at(n3, ppos[pair_src == 3], 1)
    tot = n2 + n3
    out = {"n_s1": int(len(gt_s1)), "n_pairs": int(len(pair_s1)),
           "gt_rows_cover_all_s1": bool(len(np.unique(pos_gt)) == len(s1_ids))}
    for c in ["ALL"] + sorted(set(s1_cty.tolist())):
        m = np.ones(len(tot), bool) if c == "ALL" else s1_cty == c
        vc = np.bincount(np.minimum(tot[m], 8), minlength=9)
        out[c] = {
            "n": int(m.sum()),
            "size_dist": {("8+" if i == 8 else str(i)): float(v / m.sum()) for i, v in enumerate(vc)},
            "singleton_rate": float(np.mean(tot[m] == 0)),
            "one_rate": float(np.mean(tot[m] == 1)),
            "many_rate": float(np.mean(tot[m] >= 2)),
            "mean_size": float(tot[m].mean()),
            "mean_size_nonzero": float(tot[m][tot[m] > 0].mean()),
            "n_s2_dist": {str(i): float(v / m.sum()) for i, v in enumerate(np.bincount(np.minimum(n2[m], 5), minlength=6))},
            "n_s3_dist": {str(i): float(v / m.sum()) for i, v in enumerate(np.bincount(np.minimum(n3[m], 5), minlength=6))},
            "has_s2": float(np.mean(n2[m] > 0)), "has_s3": float(np.mean(n3[m] > 0)),
        }
    return out, (s1_ids, s1_cty, order, pair_s1, pair_src, pair_id)


def profile_pairs(gtinfo):
    """Exact agreement of true pairs on normalised fields + corpus 'distractor' share."""
    s1_ids, s1_cty, order, pair_s1, pair_src, pair_id = gtinfo
    ppos = order[np.searchsorted(s1_ids[order], pair_s1)]
    fields = ["name_n", "name_core", "addr_n", "addr_core", "cty"]
    s1 = data.read_prepared("train", 1, columns=fields)
    h1 = {f: hash_cols(s1, [f]) for f in fields}
    del s1
    out = {}
    matched_any = {}
    for src in (2, 3):
        t = data.read_prepared("train", src, columns=["id"] + fields + ["name_script", "a_empty"])
        ids = t.column("id").to_numpy()
        o = np.argsort(ids)
        sel = pair_src == src
        rpos = o[np.searchsorted(ids[o], pair_id[sel])]
        sp = ppos[sel]
        r = {"n_pairs": int(sel.sum())}
        for f in fields:
            hc = hash_cols(t, [f])
            r[f"eq_{f}"] = float(np.mean(h1[f][sp] == hc[rpos]))
        scr_col = t.column("name_script").combine_chunks().take(pa.array(rpos))
        indic = pc.invert(pc.is_in(scr_col, value_set=pa.array(["ASCII", "LATIN_EXT"]))).to_numpy(
            zero_copy_only=False)
        del scr_col
        r["match_name_indic"] = float(np.mean(indic))
        r["match_addr_empty"] = float(np.mean(t.column("a_empty").to_numpy(zero_copy_only=False)[rpos]))
        is_matched = np.zeros(len(ids), bool)
        is_matched[rpos] = True
        cty_col = t.column("cty").combine_chunks()
        matched_any[f"S{src}"] = {c: float(np.mean(is_matched[pc.equal(cty_col, c).to_numpy(zero_copy_only=False)]))
                                  for c in sorted(_vc(cty_col))}
        # by S1 country
        for c in sorted(set(s1_cty.tolist())):
            m = s1_cty[sp] == c
            r[f"eq_name_core_{c}"] = float(np.mean(h1["name_core"][sp][m] == hash_cols(t, ["name_core"])[rpos][m]))
            r[f"eq_addr_core_{c}"] = float(np.mean(h1["addr_core"][sp][m] == hash_cols(t, ["addr_core"])[rpos][m]))
            r[f"match_name_indic_{c}"] = float(np.mean(indic[m]))
        out[f"S{src}"] = r
        del t
    out["corpus_share_matched"] = matched_any
    return out


def profile_test_country(country="France", per_source=100_000):
    """Label-free format profile of an unseen test country, using the same
    normaliser as training, to check which rules transfer."""
    prep = __import__("01_prepare")
    res = {}
    cty_counts = {}
    for src in (1, 2, 3):
        path = data.raw_path("test", src)
        reader = data.open_tsv(path)
        ids, names, addrs, ctys = [], [], [], []
        cnt = Counter()
        for rb in reader:
            cols = [rb.column(i).to_pylist() for i in range(4)]
            cnt.update(cols[3])
            for e, n, a, c in zip(*cols):
                if c == country and len(ids) < per_source:
                    ids.append(e); names.append(n); addrs.append(a); ctys.append(c)
        cty_counts[f"S{src}"] = dict(cnt)
        t = prep.process((ids, names, addrs, ctys))
        r = {"n_profiled": t.num_rows}
        for col in ("name_len", "name_ntok", "addr_len", "addr_ncomp"):
            r[col] = pct(t.column(col).to_numpy())
        for col in FLAG_COLS:
            r[col] = float(np.mean(t.column(col).to_numpy(zero_copy_only=False)))
        r["name_script"] = dict(Counter(t.column("name_script").to_pylist()).most_common())
        r["state_found_with_US_India_lexicon"] = float(np.mean(t.column("state").to_numpy(zero_copy_only=False) != ""))
        r["zip5"] = float(np.mean(t.column("zip5").to_numpy(zero_copy_only=False) != ""))
        r["a_pattern_top"] = Counter(t.column("a_pattern").to_pylist()).most_common(6)
        toks = pc.list_flatten(pc.utf8_split_whitespace(t.column("name_n")))
        vc = pc.value_counts(toks)
        vals, cnts = vc.field("values").to_pylist(), vc.field("counts").to_numpy()
        o = np.argsort(-cnts)[:30]
        r["top_name_tokens"] = [(vals[i], float(cnts[i] / t.num_rows)) for i in o]
        in_hand = np.isin(np.asarray(vals, dtype=object), list(tn.NAME_HAND_STOP))
        r["name_token_share_removed_by_hand_list"] = float(cnts[in_hand].sum() / cnts.sum())
        atoks = pc.list_flatten(pc.utf8_split_whitespace(
            pc.replace_substring(t.column("addr_n"), ",", " ")))
        vc = pc.value_counts(atoks)
        vals, cnts = vc.field("values").to_pylist(), vc.field("counts").to_numpy()
        o = np.argsort(-cnts)[:30]
        r["top_addr_tokens"] = [(vals[i], float(cnts[i] / t.num_rows)) for i in o]
        comps = pc.list_flatten(pc.split_pattern(t.column("addr_n"), ","))
        vc = pc.value_counts(comps)
        vals, cnts = vc.field("values").to_pylist(), vc.field("counts").to_numpy()
        o = np.argsort(-cnts)[:15]
        r["top_addr_components"] = [(vals[i], float(cnts[i] / t.num_rows)) for i in o]
        avc = pc.value_counts(atoks)
        abbr_hits = np.isin(np.asarray(avc.field("values").to_pylist(), dtype=object),
                            list(tn.STREET_ABBR) + list(tn.ADDR_HAND_STOP))
        acnt = avc.field("counts").to_numpy()
        r["addr_token_share_in_US_street_lexicon"] = float(acnt[abbr_hits].sum() / acnt.sum())
        res[f"S{src}"] = r
        # a few raw examples for the report
        res[f"S{src}_examples"] = list(zip(names[:6], addrs[:6]))
    res["country_counts"] = cty_counts
    return res


def main():
    t0 = time.time()
    out = {}
    for src in (1, 2, 3):
        out[f"S{src}"] = profile_source(src)
        print(f"profiled S{src} [{time.time() - t0:.0f}s]", flush=True)
    out["gt"], gtinfo = profile_gt()
    print(f"profiled GT [{time.time() - t0:.0f}s]", flush=True)
    out["pairs"] = profile_pairs(gtinfo)
    print(f"profiled pairs [{time.time() - t0:.0f}s]", flush=True)
    out["test_france"] = profile_test_country("France")
    print(f"profiled test France [{time.time() - t0:.0f}s]", flush=True)
    os.makedirs(data.RESULTS_DIR, exist_ok=True)
    with open(os.path.join(data.RESULTS_DIR, "profile.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    sys.exit(main())
