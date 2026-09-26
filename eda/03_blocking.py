"""Blocking / candidate-generation strategies: recall ceiling vs reduction ratio.

Every strategy is evaluated for the same random sample of Source-1 entities
against the FULL Source-2 + Source-3 corpus (exact block sizes), so recall,
candidates per entity, reduction ratio and false-candidate rate are directly
comparable.

    python 03_blocking.py [--sample 20000] [--only REGEX] [--threads 8]

Writes results/blocking_results.json (+ .md tables) and caches per-strategy
candidate lists needed for unions in ER_CACHE/blocking/.
"""

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict

import jellyfish
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from erlib import blocking as B
from erlib import data
from erlib import textnorm as tn

NAME_STOP = pa.array(sorted(tn.NAME_HAND_STOP))
ADDR_STOP = pa.array(sorted(tn.ADDR_HAND_STOP))
S1_COLS = ["id", "cty", "name_n", "name_core", "addr_n", "addr_core", "zip5", "pin6",
           "name_script", "a_empty"]


# --------------------------------------------------------------------------
# key-function builders
# --------------------------------------------------------------------------

def _country(tbl, rows, keys, country):
    if country:
        keys = B.mix(keys, B.country_hash(tbl)[rows])
    return rows, keys


def map_unique(arr, fn):
    """Apply a Python function to each distinct value of an Arrow string array."""
    if isinstance(arr, pa.ChunkedArray):
        arr = arr.combine_chunks()
    d = pc.dictionary_encode(arr)
    mapped = pa.array([fn(v) for v in d.dictionary.to_pylist()], pa.string())
    return mapped.take(d.indices)


def single(fn_arr, country=True):
    """One key per record from a derived string column ('' = no key)."""
    def keyfn(tbl):
        s = fn_arr(tbl)
        if isinstance(s, pa.ChunkedArray):
            s = s.combine_chunks()
        rows = np.nonzero(pc.not_equal(s, "").to_numpy(zero_copy_only=False))[0]
        h = B.hash_str_array(s)[rows]
        return _country(tbl, rows, h, country)
    return keyfn


def tokens(source, stop=None, min_len=1, tok_map=None, tag="", country=True, digits=None):
    """Multi-key: every whitespace token of a (derived) string column."""
    tag_h = B.const_hash(tag) if tag else None

    def keyfn(tbl):
        col = source(tbl) if callable(source) else tbl.column(source)
        rows, toks = B.tokenize(col)
        if tok_map is not None:
            toks = tok_map(toks)
        mask = pc.not_equal(toks, "")
        if stop is not None:
            mask = pc.and_(mask, pc.invert(pc.is_in(toks, value_set=stop)))
        if min_len > 1:
            mask = pc.and_(mask, pc.greater_equal(pc.utf8_length(toks), min_len))
        if digits is not None:
            has_d = pc.match_substring_regex(toks, r"\d")
            mask = pc.and_(mask, has_d if digits else pc.invert(has_d))
        m = mask.to_numpy(zero_copy_only=False)
        rows, toks = rows[m], toks.filter(mask)
        h = B.hash_str_array(toks) if len(toks) else np.zeros(0, np.uint64)
        if tag_h is not None:
            h = B.mix(h, np.full(len(h), tag_h, np.uint64))
        return _country(tbl, rows, h, country)
    return keyfn


def union_keys(*fns):
    def keyfn(tbl):
        parts = [f(tbl) for f in fns]
        return (np.concatenate([p[0] for p in parts]),
                np.concatenate([np.asarray(p[1], np.uint64) for p in parts]))
    return keyfn


def compact(col, n=None):
    s = pc.replace_substring(col, " ", "")
    return pc.utf8_slice_codeunits(s, 0, n) if n else s


def phon(fn):
    def tok_map(toks):
        return map_unique(toks, lambda t: (fn(t) or t) if t.isalpha() else t)
    return tok_map


def first_token_code(fn):
    def f(v):
        for t in v.split():
            if t.isalpha():
                return fn(t)
        return v.split()[0] if v else ""
    return f


def street_key(addr_n):
    """House number + first street word of the first component with a number."""
    for comp in addr_n.split(","):
        toks = comp.split()
        for i, t in enumerate(toks):
            if t[0].isdigit():
                for w in toks[i + 1:]:
                    w = tn.canon_addr_token(w)
                    if w not in tn.ADDR_HAND_STOP and not w[0].isdigit():
                        return tn.canon_addr_token(t) + "_" + w
                    if w not in tn.ADDR_HAND_STOP and w.isdigit():
                        return tn.canon_addr_token(t) + "_" + w
                break
    return ""


def first_line_tokens(addr_n):
    """Canonical tokens of the first component that contains a number."""
    for comp in addr_n.split(","):
        if any(ch.isdigit() for ch in comp):
            out = [tn.canon_addr_token(t) for t in comp.split()]
            return " ".join(t for t in out if t not in tn.ADDR_HAND_STOP)
    return ""


def locality_components(tbl):
    """Alphabetic address components (no digits), spaces removed, as a token string."""
    col = tbl.column("addr_n")
    return map_unique(col, lambda v: " ".join(
        c.replace(" ", "") for c in v.split(",") if len(c) >= 3 and not any(ch.isdigit() for ch in c)))


# --------------------------------------------------------------------------
# learned transliteration dictionary (Indic-script token -> Latin token)
# --------------------------------------------------------------------------

def learn_translit(corpus, s1_all, pair_s1, pair_src, pair_id, exclude_s1, min_count=3):
    """Align tokens of S1 names (Latin) with matched corpus names written in an Indic
    script (same token count, positional alignment) and keep confident mappings.
    Entities in exclude_s1 (the evaluation sample) are not used."""
    script = corpus.column("name_script").combine_chunks()
    indic = pc.invert(pc.is_in(script, value_set=pa.array(["ASCII", "LATIN_EXT"]))).to_numpy(
        zero_copy_only=False)
    idx = np.nonzero(indic)[0]
    names = corpus.column("name_n").combine_chunks().take(pa.array(idx)).to_pylist()
    rec_name = dict(zip(idx.tolist(), names))
    s1_ids = s1_all.column("id").to_numpy()
    s1_order = np.argsort(s1_ids)
    s1_names = s1_all.column("name_n")
    keep = ~np.isin(pair_s1, exclude_s1)
    recs = corpus.index_of(pair_src[keep], pair_id[keep])
    m = indic[recs]
    recs, ps1 = recs[m], pair_s1[keep][m]
    s1_idx = s1_order[np.searchsorted(s1_ids[s1_order], ps1)]
    s1_n = s1_names.take(pa.array(s1_idx)).to_pylist()
    counts = defaultdict(Counter)
    used = 0
    for r, latin in zip(recs.tolist(), s1_n):
        a, b = rec_name[r].split(), latin.split()
        if len(a) == len(b) and a:
            used += 1
            for x, y in zip(a, b):
                if x != y:
                    counts[x][y] += 1
    mapping = {}
    for x, c in counts.items():
        y, n = c.most_common(1)[0]
        tot = sum(c.values())
        if n >= min_count and n / tot >= 0.5:
            mapping[x] = y
    return mapping, used, int(len(recs))


def translit_map(mapping):
    keys = pa.array(list(mapping.keys()), pa.string())
    vals = pa.array(list(mapping.values()), pa.string())

    def tok_map(toks):
        idx = pc.index_in(toks, value_set=keys)
        mapped = vals.take(pc.fill_null(idx, 0))
        return pc.if_else(pc.is_null(idx), toks, mapped)
    return tok_map


# --------------------------------------------------------------------------
# reporting helpers
# --------------------------------------------------------------------------

def breakdown(found, attrs):
    out = {}
    for name, labels in attrs.items():
        for lab in np.unique(labels):
            m = labels == lab
            out[f"{name}={lab}"] = (float(found[m].mean()), int(m.sum()))
    return out


def fmt_pct(x, d=2):
    return f"{100 * x:.{d}f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--only", default=None, help="regex on strategy ids")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    t0 = time.time()
    out_json = os.path.join(data.RESULTS_DIR, "blocking_results.json")
    results = json.load(open(out_json)) if os.path.exists(out_json) else {}
    lists_dir = os.path.join(data.CACHE_DIR, "blocking")
    os.makedirs(lists_dir, exist_ok=True)

    corpus = B.Corpus("train")
    s1_all = data.read_prepared("train", 1, columns=S1_COLS)
    n_s1_full = s1_all.num_rows
    rng = np.random.default_rng(args.seed)
    pick = np.sort(rng.choice(n_s1_full, size=args.sample, replace=False))
    s1 = s1_all.take(pa.array(pick))
    _, pair_s1, pair_src, pair_id = data.load_ground_truth()
    sample = B.Sample(s1, corpus, pair_s1, pair_src, pair_id)
    print(f"corpus={corpus.n:,} sample={sample.n:,} true pairs in sample={len(sample.pair_rec):,} "
          f"singletons={int((sample.gt_count == 0).sum()):,}  [{time.time() - t0:.0f}s]", flush=True)

    # per-pair attributes for recall breakdowns
    cty_s1 = np.asarray(s1.column("cty").to_pylist(), dtype=object)
    scr = corpus.column("name_script").combine_chunks().take(pa.array(sample.pair_rec)).to_pylist()
    aemp = corpus.column("a_empty").combine_chunks().take(pa.array(sample.pair_rec)).to_numpy(
        zero_copy_only=False)
    attrs = {
        "country": cty_s1[sample.pair_sample].astype(str),
        "script": np.where(np.isin(scr, ["ASCII", "LATIN_EXT"]), "latin", "indic"),
        "addr": np.where(aemp, "empty", "present"),
        "src": np.where(sample.pair_src == 2, "S2", "S3"),
    }
    n_by_cty = Counter(corpus.column("cty").combine_chunks().to_pylist())

    # translit dictionary learned outside the evaluation sample
    tmap_path = os.path.join(lists_dir, "translit_map.json")
    if os.path.exists(tmap_path):
        tinfo = json.load(open(tmap_path))
    else:
        full_s1 = data.read_prepared("train", 1, columns=["id", "name_n"])
        mapping, used, n_ind = learn_translit(corpus, full_s1, pair_s1, pair_src, pair_id,
                                              s1.column("id").to_numpy())
        tinfo = {"mapping": mapping, "aligned_pairs": used, "indic_pairs": n_ind}
        json.dump(tinfo, open(tmap_path, "w"))
        del full_s1
    tmap = translit_map(tinfo["mapping"])
    print(f"translit dictionary: {len(tinfo['mapping']):,} tokens from "
          f"{tinfo['aligned_pairs']:,}/{tinfo['indic_pairs']:,} aligned Indic pairs", flush=True)
    del pair_s1, pair_src, pair_id

    name_tok = tokens("name_core")
    name_tok_tr = tokens("name_n", stop=NAME_STOP, tok_map=tmap)
    addr_tok = tokens("addr_core")
    idf = B.idf_weight(corpus.n)

    # (id, group, description, keyfn, corpus columns, hard_cap, [eval configs])
    # eval config: dict(k=.., cap=.., label=..)
    S = []

    def add(sid, group, desc, keyfn, cols, hard_cap=None, evals=({"k": 1},), save_lists=None):
        S.append((sid, group, desc, keyfn, cols, hard_cap, list(evals), save_lists))

    # ---- 1. exact / normalised name keys -----------------------------------
    add("N1", "name-key", "first 4 chars of normalised name (no suffix removal)",
        single(lambda t: compact(t.column("name_n"), 4)), ["cty", "name_n"])
    for n in (3, 5, 8):
        add(f"N2.{n}", "name-key", f"first {n} chars of core name (legal suffix/honorific removed, spaces dropped)",
            single(lambda t, n=n: compact(t.column("name_core"), n)), ["cty", "name_core"])
    add("N3", "name-key", "sorted token set of core name (exact)",
        single(lambda t: map_unique(t.column("name_core"), lambda v: " ".join(sorted(set(v.split()))))),
        ["cty", "name_core"])
    add("N4", "name-key", "Soundex of first core-name token",
        single(lambda t: map_unique(t.column("name_core"), first_token_code(jellyfish.soundex))),
        ["cty", "name_core"])
    add("N5", "name-key", "Metaphone of first core-name token",
        single(lambda t: map_unique(t.column("name_core"), first_token_code(jellyfish.metaphone))),
        ["cty", "name_core"])
    add("N6", "name-key", "sorted set of Metaphone codes of core tokens (exact)",
        single(lambda t: map_unique(t.column("name_core"), lambda v: " ".join(
            sorted({(jellyfish.metaphone(x) or x) if x.isalpha() else x for x in v.split()})))),
        ["cty", "name_core"])
    add("N7", "name-key", "any shared Metaphone code of core tokens (multi-key)",
        tokens("name_core", tok_map=phon(jellyfish.metaphone)), ["cty", "name_core"],
        hard_cap=100_000, evals=[{"k": 1, "cap": 1000}, {"k": 1, "cap": 10_000}, {"k": 2, "cap": 100_000}])

    # ---- 2. address keys -----------------------------------------------------
    add("A1", "address", "postal code (5-digit ZIP component or 6-digit PIN)",
        single(lambda t: pc.if_else(pc.not_equal(t.column("zip5"), ""), t.column("zip5"), t.column("pin6"))),
        ["cty", "zip5", "pin6"])
    add("A2", "address", "city/locality components (alphabetic address components)",
        tokens(locality_components), ["cty", "addr_n"], hard_cap=200_000,
        evals=[{"k": 1, "cap": 5000}, {"k": 1, "cap": 50_000}, {"k": 1, "cap": 200_000}])
    add("A3", "address", "street key: house number + first street word",
        single(lambda t: map_unique(t.column("addr_n"), street_key)), ["cty", "addr_n"], save_lists=[{"k": 1}])
    add("A4", "address", "first-line (street component) tokens",
        tokens(lambda t: map_unique(t.column("addr_n"), first_line_tokens)), ["cty", "addr_n"],
        hard_cap=100_000, evals=[{"k": 2, "cap": 100_000}, {"k": 3, "cap": 100_000}])
    add("A5", "address", "house-number tokens (tokens containing a digit)",
        tokens("addr_core", digits=True), ["cty", "addr_core"], hard_cap=100_000,
        evals=[{"k": 1, "cap": 1000}, {"k": 2, "cap": 100_000}])

    # ---- 3. token inverted index ----------------------------------------------
    add("T1", "token-index", "core-name tokens (hand stop list), DF-capped",
        name_tok, ["cty", "name_core"], hard_cap=200_000,
        evals=[{"k": 1, "cap": 1000}, {"k": 1, "cap": 10_000}, {"k": 1, "cap": 100_000},
               {"k": 2, "cap": 200_000}], save_lists=[{"k": 1, "cap": 1000}, {"k": 2, "cap": 200_000}])
    add("T1n", "token-index", "name tokens, DF-based stop removal only (no hand list)",
        tokens("name_n"), ["cty", "name_n"], hard_cap=200_000,
        evals=[{"k": 1, "cap": 1000}, {"k": 1, "cap": 10_000}, {"k": 2, "cap": 200_000}])
    add("T1t", "token-index", "name tokens + learned Indic->Latin token dictionary",
        name_tok_tr, ["cty", "name_n"], hard_cap=200_000,
        evals=[{"k": 1, "cap": 1000}, {"k": 1, "cap": 10_000}, {"k": 2, "cap": 200_000}],
        save_lists=[{"k": 1, "cap": 1000}, {"k": 2, "cap": 200_000}])
    add("T2", "token-index", "address tokens (canonicalised, hand stop list), DF-capped",
        addr_tok, ["cty", "addr_core"], hard_cap=200_000,
        evals=[{"k": 2, "cap": 20_000}, {"k": 3, "cap": 200_000}, {"k": 4, "cap": 200_000}],
        save_lists=[{"k": 3, "cap": 200_000}])
    add("T3", "token-index", "joint name+address tokens (field-tagged)",
        union_keys(tokens("name_core", tag="n"), tokens("addr_core", tag="a")),
        ["cty", "name_core", "addr_core"], hard_cap=200_000,
        evals=[{"k": 2, "cap": 20_000}, {"k": 3, "cap": 200_000}, {"k": 4, "cap": 200_000}])

    # ---- 4. MinHash LSH ----------------------------------------------------------
    for b, r in ((32, 2), (20, 3), (16, 4), (10, 6)):
        add(f"L1.{b}x{r}", "minhash-lsh",
            f"name char-3gram MinHash, {b} bands x {r} rows (J*~{B.lsh_threshold(b, r):.2f})",
            B.minhash_keyfn("name_core", b, r, q=3, skip_space=True), ["cty", "name_core"],
            hard_cap=200_000, save_lists=[{"k": 1}] if (b, r) in ((20, 3), (16, 4)) else None)
    for b, r in ((20, 3), (16, 4), (10, 6)):
        add(f"L2.{b}x{r}", "minhash-lsh",
            f"address char-3gram MinHash, {b} bands x {r} rows (J*~{B.lsh_threshold(b, r):.2f})",
            B.minhash_keyfn("addr_core", b, r, q=3, skip_space=False), ["cty", "addr_core"],
            hard_cap=200_000, save_lists=[{"k": 1}] if (b, r) == (10, 6) else None)

    # ---- 5. country-aware vs country-agnostic --------------------------------
    add("C1", "country", "N3 sorted token set WITHOUT country in key",
        single(lambda t: map_unique(t.column("name_core"), lambda v: " ".join(sorted(set(v.split())))),
               country=False), ["cty", "name_core"])
    add("C2", "country", "T1 core-name tokens WITHOUT country in key",
        tokens("name_core", country=False), ["cty", "name_core"], hard_cap=200_000,
        evals=[{"k": 1, "cap": 1000}, {"k": 1, "cap": 10_000}, {"k": 2, "cap": 200_000}])
    add("C3", "country", "L1 name MinHash 16x4 WITHOUT country in key",
        B.minhash_keyfn("name_core", 16, 4, q=3, skip_space=True, country=False), ["cty", "name_core"],
        hard_cap=200_000)

    # ---- 6. weighted retrieval (IDF-scored inverted index, keep top-K) -------------
    add("R1", "ranked", "IDF-weighted name+address token retrieval, top-K per S1",
        union_keys(tokens("name_core", tag="n"), tokens("addr_core", tag="a")),
        ["cty", "name_core", "addr_core"], hard_cap=100_000,
        evals=[{"k": 1, "cap": 100_000, "weighted": True}])
    add("R2", "ranked", "R1 + Indic->Latin dictionary on names, top-K per S1",
        union_keys(tokens("name_n", stop=NAME_STOP, tok_map=tmap, tag="n"), tokens("addr_core", tag="a")),
        ["cty", "name_n", "addr_core"], hard_cap=100_000,
        evals=[{"k": 1, "cap": 100_000, "weighted": True}])
    add("R3", "ranked", "name-only IDF retrieval (with Indic dictionary), top-K per S1",
        tokens("name_n", stop=NAME_STOP, tok_map=tmap), ["cty", "name_n"], hard_cap=100_000,
        evals=[{"k": 1, "cap": 100_000, "weighted": True}])
    add("R4", "ranked", "address-only IDF retrieval, top-K per S1",
        tokens("addr_core"), ["cty", "addr_core"], hard_cap=100_000,
        evals=[{"k": 1, "cap": 100_000, "weighted": True}])
    add("R5", "ranked", "name char-3gram MinHash 32x2, ranked by number of colliding bands",
        B.minhash_keyfn("name_core", 32, 2, q=3, skip_space=True), ["cty", "name_core"],
        hard_cap=100_000, evals=[{"k": 1, "weighted": "count"}])

    only = re.compile(args.only) if args.only else None
    KS = [1, 5, 10, 20, 50, 100, 200]
    for sid, group, desc, keyfn, cols, hard_cap, evals, save_lists in S:
        if only and not only.search(sid):
            continue
        ts = time.time()
        index = B.Index(keyfn, sample, corpus, cols, hard_cap=hard_cap)
        t_index = time.time() - ts
        for ev in evals:
            k, cap = ev.get("k", 1), ev.get("cap")
            label = f"{sid} k>={k}" + (f" cap={cap:,}" if cap else "")
            te = time.time()
            if ev.get("weighted"):
                wfn = idf if ev["weighted"] is True else (lambda df, keys: np.ones(len(keys)))
                res = index.evaluate(k=k, cap=cap, weight_fn=wfn, n_threads=args.threads, topk_max=200)
                summ = B.summarize(res, sample, corpus.n, n_s1_full)
                entry = {"id": sid, "group": group, "desc": desc, "k": k, "cap": cap, **summ,
                         "recall_at_k": B.recall_at_k(res, sample, KS),
                         "by": {}, "by_at_k": {}}
                for kk in (10, 20, 50, 200):
                    hit = (res["pair_rank"] > 0) & (res["pair_rank"] <= kk)
                    entry["by_at_k"][str(kk)] = breakdown(hit, attrs)
                entry["by"] = breakdown(res["pair_found"], attrs)
                np.save(os.path.join(lists_dir, f"{sid}_topk.npy"), res["topk"])
                np.save(os.path.join(lists_dir, f"{sid}_rank.npy"), res["pair_rank"])
            else:
                res = index.evaluate(k=k, cap=cap, n_threads=args.threads)
                summ = B.summarize(res, sample, corpus.n, n_s1_full)
                entry = {"id": sid, "group": group, "desc": desc, "k": k, "cap": cap, **summ,
                         "by": breakdown(res["pair_found"], attrs)}
            entry.update({"keys_per_s1": res["keys_per_s1"], "s1_no_key": res["s1_no_key"],
                          "t_index_s": round(t_index, 1), "t_eval_s": round(time.time() - te, 1)})
            results[label] = entry
            print(f"{label:28s} recall={fmt_pct(entry['recall_pairs'])} F05ceil={entry['f05_ceiling']:.4f} "
                  f"cand/S1={entry['cand_mean']:.1f} RR={entry['reduction_ratio']:.6f} "
                  f"falseC={fmt_pct(entry['false_cand_rate'])} maxblock={entry['max_block']:,} "
                  f"[{t_index:.0f}s+{time.time() - te:.0f}s]", flush=True)
            if save_lists and any(sl.get("k", 1) == k and sl.get("cap") == cap for sl in save_lists):
                ip, rc = index.candidate_lists(k=k, cap=cap, n_threads=args.threads)
                tag = f"{sid}_k{k}_cap{cap}"
                np.save(os.path.join(lists_dir, f"{tag}_indptr.npy"), ip)
                np.save(os.path.join(lists_dir, f"{tag}_recs.npy"), rc)
        del index
        json.dump(results, open(out_json, "w"), indent=1, default=float)

    # ---- 7. unions of complementary strategies ----------------------------------
    def load_lists(tag):
        return (np.load(os.path.join(lists_dir, f"{tag}_indptr.npy")),
                np.load(os.path.join(lists_dir, f"{tag}_recs.npy")))

    def topk_csr(sid, kk):
        topk = np.load(os.path.join(lists_dir, f"{sid}_topk.npy"))[:, :kk]
        return B.topk_lists({"topk": topk})

    unions = {
        "U1": ("name tokens k>=1 cap 1k  U  street key  U  name MinHash 16x4",
               lambda: [load_lists("T1_k1_cap1000"), load_lists("A3_k1_capNone"), load_lists("L1.16x4_k1_capNone")]),
        "U2": ("U1 with Indic dictionary tokens (T1t) instead of T1",
               lambda: [load_lists("T1t_k1_cap1000"), load_lists("A3_k1_capNone"), load_lists("L1.16x4_k1_capNone")]),
        "U3": ("U2  U  address tokens k>=3  U  address MinHash 10x6",
               lambda: [load_lists("T1t_k1_cap1000"), load_lists("A3_k1_capNone"), load_lists("L1.16x4_k1_capNone"),
                        load_lists("T2_k3_cap200000"), load_lists("L2.10x6_k1_capNone")]),
    }
    for kk in (10, 25, 50, 100):
        unions[f"U4@{kk}"] = (f"name-only top-{kk} (R3)  U  address-only top-{kk} (R4)",
                              lambda kk=kk: [topk_csr("R3", kk), topk_csr("R4", kk)])
        unions[f"U5@{kk}"] = (f"R3 top-{kk}  U  R4 top-{kk}  U  name-MinHash top-{kk} (R5)",
                              lambda kk=kk: [topk_csr("R3", kk), topk_csr("R4", kk), topk_csr("R5", kk)])
        unions[f"U6@{kk}"] = (f"joint top-{kk} (R2)  U  name-MinHash top-{kk} (R5)",
                              lambda kk=kk: [topk_csr("R2", kk), topk_csr("R5", kk)])
        unions[f"U7@{kk}"] = (f"U5@{kk}  U  street key (A3)",
                              lambda kk=kk: [topk_csr("R3", kk), topk_csr("R4", kk), topk_csr("R5", kk),
                                             load_lists("A3_k1_capNone")])
    for uid, (desc, get) in unions.items():
        if only and not only.search(uid):
            continue
        try:
            lists = get()
        except FileNotFoundError as e:
            print(f"[skip {uid}] missing {e.filename}")
            continue
        te = time.time()
        res = B.union(lists, sample, n_threads=args.threads)
        summ = B.summarize(res, sample, corpus.n, n_s1_full)
        results[uid] = {"id": uid, "group": "union", "desc": desc, "k": None, "cap": None, **summ,
                        "by": breakdown(res["pair_found"], attrs),
                        "keys_per_s1": None, "s1_no_key": res["s1_no_key"],
                        "t_index_s": 0, "t_eval_s": round(time.time() - te, 1)}
        e = results[uid]
        print(f"{uid:28s} recall={fmt_pct(e['recall_pairs'])} F05ceil={e['f05_ceiling']:.4f} "
              f"cand/S1={e['cand_mean']:.1f} RR={e['reduction_ratio']:.6f} "
              f"falseC={fmt_pct(e['false_cand_rate'])}", flush=True)
        json.dump(results, open(out_json, "w"), indent=1, default=float)

    meta = {"sample": sample.n, "seed": args.seed, "corpus": corpus.n, "s1_full": n_s1_full,
            "true_pairs_sample": int(len(sample.pair_rec)),
            "singletons_sample": int((sample.gt_count == 0).sum()),
            "corpus_by_country": dict(n_by_cty),
            "translit_tokens": len(tinfo["mapping"]),
            "translit_aligned_pairs": tinfo["aligned_pairs"]}
    results["_meta"] = meta
    json.dump(results, open(out_json, "w"), indent=1, default=float)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    sys.exit(main())
