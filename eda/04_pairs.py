"""Build labelled candidate pairs and compute similarity features.

Candidates come from the blocking run (03_blocking.py must have run first): the
recommended union U8@K = joint name+address IDF top-K  U  name-MinHash top-K
U  empty-address name channel, for a sub-sample of the same Source-1 sample.

Pair classes
  true         candidate is a ground-truth match
  hard_neg     candidate of an entity that has matches, but not one of them
               ("same group but wrong match")
  single_cand  candidate of an entity with no match at all ("no match" entities)
  random       random same-country record (easy negative baseline)
  true_missed  ground-truth match that blocking did NOT retrieve

    python 04_pairs.py [--n-s1 4000] [--k 25] [--embed]
Writes ER_CACHE/pairs.parquet
"""

import argparse
import os
import re
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein
from rapidfuzz.process import cpdist

from erlib import blocking as B
from erlib import data
from erlib import textnorm as tn

bl = __import__("03_blocking")
S1_COLS = ["id", "cty", "name_raw", "addr_raw", "name_n", "name_core", "addr_n", "addr_core",
           "name_script", "a_empty"]


def load_sample(seed=42, n=20000):
    s1_all = data.read_prepared("train", 1, columns=S1_COLS)
    rng = np.random.default_rng(seed)
    pick = np.sort(rng.choice(s1_all.num_rows, size=n, replace=False))
    return s1_all.take(pa.array(pick)), s1_all.num_rows


def jaccard_tokens(a, b):
    out = np.empty(len(a), np.float32)
    for i, (x, y) in enumerate(zip(a, b)):
        sx, sy = set(x.split()), set(y.split())
        u = len(sx | sy)
        out[i] = len(sx & sy) / u if u else np.nan
    return out


def jaccard_qgrams(a, b, q=3):
    def grams(s):
        s = s.replace(" ", "")
        return {s[i:i + q] for i in range(max(1, len(s) - q + 1))} if s else set()
    out = np.empty(len(a), np.float32)
    for i, (x, y) in enumerate(zip(a, b)):
        gx, gy = grams(x), grams(y)
        u = len(gx | gy)
        out[i] = len(gx & gy) / u if u else np.nan
    return out


def tfidf_cosine(a, b, fit_texts, analyzer, ngram_range):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range, min_df=2,
                          sublinear_tf=True, dtype=np.float32)
    vec.fit(fit_texts)
    uniq = sorted(set(a) | set(b))
    pos = {s: i for i, s in enumerate(uniq)}
    X = vec.transform(uniq)
    ia = np.fromiter((pos[s] for s in a), np.int64, len(a))
    ib = np.fromiter((pos[s] for s in b), np.int64, len(b))
    sim = np.asarray(X[ia].multiply(X[ib]).sum(axis=1)).ravel().astype(np.float32)
    empty = np.array([not x or not y for x, y in zip(a, b)])
    sim[empty] = np.nan
    return sim


def rf(scorer, a, b):
    """rapidfuzz element-wise scores in [0, 1]; NaN when either side is empty."""
    s = cpdist(a, b, scorer=scorer, workers=-1).astype(np.float32)
    if getattr(scorer, "__module__", "").endswith("fuzz") or scorer in (
            fuzz.ratio, fuzz.partial_ratio, fuzz.token_sort_ratio, fuzz.token_set_ratio):
        s /= 100.0
    empty = np.array([not x or not y for x, y in zip(a, b)])
    s[empty] = np.nan
    return s


def translit(mapping, s):
    return " ".join(mapping.get(t, t) for t in s.split())


def embed(texts, model_name, prefix="", batch=256, device=None):
    import torch
    import transformers.utils.import_utils as tiu
    tiu._torchvision_available = False  # the local torchvision build does not match torch; not needed
    from transformers import AutoModel, AutoTokenizer
    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    out = np.zeros((len(texts), model.config.hidden_size), np.float32)
    order = np.argsort([len(t) for t in texts])
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            idx = order[i:i + batch]
            enc = tok([prefix + (texts[j] or " ") for j in idx], padding=True, truncation=True,
                      max_length=64, return_tensors="pt").to(device)
            h = model(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).float()
            v = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
            v = torch.nn.functional.normalize(v, dim=-1)
            out[idx] = v.float().cpu().numpy()
    return out


def embed_cosine(a, b, model_name, prefix=""):
    uniq = sorted(set(a) | set(b))
    pos = {s: i for i, s in enumerate(uniq)}
    E = embed(uniq, model_name, prefix)
    ia = np.fromiter((pos[s] for s in a), np.int64, len(a))
    ib = np.fromiter((pos[s] for s in b), np.int64, len(b))
    sim = np.einsum("ij,ij->i", E[ia], E[ib]).astype(np.float32)
    empty = np.array([not x or not y for x, y in zip(a, b)])
    sim[empty] = np.nan
    return sim


EMBED_MODELS = {
    # name in outputs: (HF id, licence, params, text prefix)
    "minilm_l6": ("sentence-transformers/all-MiniLM-L6-v2", "Apache-2.0", "22.7M", ""),
    "mminilm_l12": ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "Apache-2.0", "118M", ""),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-s1", type=int, default=4000)
    ap.add_argument("--k", type=int, default=25)
    ap.add_argument("--embed", action="store_true")
    ap.add_argument("--embed-n-s1", type=int, default=1200)
    args = ap.parse_args()
    t0 = time.time()
    lists_dir = os.path.join(data.CACHE_DIR, "blocking")
    tinfo = __import__("json").load(open(os.path.join(lists_dir, "translit_map.json")))
    mapping = tinfo["mapping"]

    corpus = B.Corpus("train")
    s1, n_s1_full = load_sample()
    _, pair_s1, pair_src, pair_id = data.load_ground_truth()
    sample = B.Sample(s1, corpus, pair_s1, pair_src, pair_id)
    del pair_s1, pair_src, pair_id

    # sub-sample: first n random S1 rows + every singleton of the 20k sample
    rng = np.random.default_rng(7)
    base = np.sort(rng.choice(sample.n, size=args.n_s1, replace=False))
    single = np.nonzero(sample.gt_count == 0)[0]
    sub = np.union1d(base, single)
    in_base = np.isin(sub, base)
    print(f"sub-sample: {len(sub)} S1 ({int((sample.gt_count[sub] == 0).sum())} singletons)", flush=True)

    # candidate union for every sample row, restricted to the sub-sample
    lists = __import__("03b_blocking_misses").best_union_lists(corpus, lists_dir, args.k)
    rows, recs = [], []
    for li, (ip, rc) in enumerate(lists):
        for i in sub:
            seg = rc[ip[i]:ip[i + 1]]
            rows.append(np.full(len(seg), i, np.int64))
            recs.append(seg.astype(np.int64))
    rows, recs = np.concatenate(rows), np.concatenate(recs)
    key = rows * corpus.n + recs
    _, first = np.unique(key, return_index=True)
    rows, recs = rows[first], recs[first]
    is_true = sample.gt_owner[recs] == rows
    label = np.where(is_true, "true", np.where(sample.gt_count[rows] == 0, "single_cand", "hard_neg"))
    # missed true matches
    in_sub = np.isin(sample.pair_sample, sub)
    got = set(zip(rows[is_true].tolist(), recs[is_true].tolist()))
    miss = [(i, r) for i, r in zip(sample.pair_sample[in_sub].tolist(), sample.pair_rec[in_sub].tolist())
            if (i, r) not in got]
    # random same-country negatives (5 per S1 of the base sub-sample)
    cty_all = np.asarray(corpus.column("cty").combine_chunks().to_pylist())
    s1_cty = np.asarray(s1.column("cty").to_pylist())
    rand_rows, rand_recs = [], []
    by_c = {c: np.nonzero(cty_all == c)[0] for c in np.unique(cty_all)}
    del cty_all
    for i in sub[in_base]:
        pool = by_c[s1_cty[i]]
        rr = pool[rng.integers(0, len(pool), 5)]
        rr = rr[sample.gt_owner[rr] != i]
        rand_rows.append(np.full(len(rr), i)); rand_recs.append(rr)
    rows = np.concatenate([rows, np.array([m[0] for m in miss], np.int64), np.concatenate(rand_rows)])
    recs = np.concatenate([recs, np.array([m[1] for m in miss], np.int64), np.concatenate(rand_recs)])
    label = np.concatenate([label, np.full(len(miss), "true_missed"), np.full(sum(map(len, rand_rows)), "random")])
    print(f"pairs: {len(rows):,}  " + ", ".join(f"{l}={int((label == l).sum()):,}" for l in np.unique(label))
          + f"  [{time.time() - t0:.0f}s]", flush=True)

    # fetch strings
    cand = corpus.take(recs, ["name_raw", "addr_raw", "name_n", "name_core", "addr_n", "addr_core",
                              "name_script", "a_empty"])
    s1s = s1.take(pa.array(rows))
    A = {c: s1s.column(c).to_pylist() for c in s1s.column_names}
    Bc = {c: cand.column(c).to_pylist() for c in cand.column_names}
    n = len(rows)

    # normalisation variants for the suffix study
    def raw_norm(s):
        return re.sub(r"\s+", " ", tn.ascii_fold(s).lower()).strip()

    variants = {
        "raw": ([raw_norm(x) for x in A["name_raw"]], [raw_norm(x) for x in Bc["name_raw"]]),
        "norm": (A["name_n"], Bc["name_n"]),
        "canon": ([tn.name_canon(x) for x in A["name_n"]], [tn.name_canon(x) for x in Bc["name_n"]]),
        "core": (A["name_core"], Bc["name_core"]),
        "core_tr": (A["name_core"], [" ".join(t for t in translit(mapping, x).split()
                                              if t not in tn.NAME_HAND_STOP) for x in Bc["name_n"]]),
    }
    feats = {}
    fit_names = variants["core_tr"][1][:200000] + variants["core"][0]
    for v, (a, b) in variants.items():
        feats[f"name_{v}_lev"] = rf(Levenshtein.normalized_similarity, a, b)
        feats[f"name_{v}_jw"] = rf(JaroWinkler.similarity, a, b)
        feats[f"name_{v}_tsort"] = rf(fuzz.token_sort_ratio, a, b)
        feats[f"name_{v}_tset"] = rf(fuzz.token_set_ratio, a, b)
        feats[f"name_{v}_jac"] = jaccard_tokens(a, b)
        feats[f"name_{v}_jac3"] = jaccard_qgrams(a, b)
        print(f"  name/{v} done [{time.time() - t0:.0f}s]", flush=True)
    a, b = variants["core_tr"]
    feats["name_core_tr_partial"] = rf(fuzz.partial_ratio, a, b)
    feats["name_core_tr_tfidf_char"] = tfidf_cosine(a, b, fit_names, "char_wb", (2, 4))
    feats["name_core_tr_tfidf_word"] = tfidf_cosine(a, b, fit_names, "word", (1, 1))
    feats["name_core_tfidf_char"] = tfidf_cosine(variants["core"][0], variants["core"][1], fit_names, "char_wb", (2, 4))
    feats["name_norm_tfidf_char"] = tfidf_cosine(variants["norm"][0], variants["norm"][1],
                                                 variants["norm"][1][:200000] + variants["norm"][0], "char_wb", (2, 4))
    # addresses: normalised (commas -> spaces) and canonicalised core tokens
    addr_n_a = [x.replace(",", " ") for x in A["addr_n"]]
    addr_n_b = [x.replace(",", " ") for x in Bc["addr_n"]]
    for v, (a, b) in {"norm": (addr_n_a, addr_n_b), "core": (A["addr_core"], Bc["addr_core"])}.items():
        feats[f"addr_{v}_lev"] = rf(Levenshtein.normalized_similarity, a, b)
        feats[f"addr_{v}_jw"] = rf(JaroWinkler.similarity, a, b)
        feats[f"addr_{v}_tsort"] = rf(fuzz.token_sort_ratio, a, b)
        feats[f"addr_{v}_tset"] = rf(fuzz.token_set_ratio, a, b)
        feats[f"addr_{v}_jac"] = jaccard_tokens(a, b)
        feats[f"addr_{v}_jac3"] = jaccard_qgrams(a, b)
    fit_addr = A["addr_core"] + Bc["addr_core"][:200000]
    feats["addr_core_tfidf_char"] = tfidf_cosine(A["addr_core"], Bc["addr_core"], fit_addr, "char_wb", (2, 4))
    feats["addr_core_tfidf_word"] = tfidf_cosine(A["addr_core"], Bc["addr_core"], fit_addr, "word", (1, 1))
    # house number agreement (first number token of each address)
    def first_num(s):
        m = re.search(r"\b\d+[a-z]?\b", s)
        return m.group(0) if m else ""
    fa, fb = [first_num(x) for x in A["addr_core"]], [first_num(x) for x in Bc["addr_core"]]
    feats["addr_housenum_eq"] = np.array([np.nan if not x or not y else float(x == y) for x, y in zip(fa, fb)],
                                         np.float32)
    # combined name+address strings
    comb_a = [x + " | " + y for x, y in zip(variants["core_tr"][0], A["addr_core"])]
    comb_b = [x + " | " + y for x, y in zip(variants["core_tr"][1], Bc["addr_core"])]
    feats["comb_tfidf_char"] = tfidf_cosine(comb_a, comb_b, comb_a + comb_b[:200000], "char_wb", (2, 4))
    feats["comb_tset"] = rf(fuzz.token_set_ratio, comb_a, comb_b)
    print(f"  features done [{time.time() - t0:.0f}s]", flush=True)

    # embeddings on a smaller subset of S1 entities
    if args.embed:
        emb_rows = set(sub[:args.embed_n_s1].tolist())  # sub is sorted: a random mix incl. singletons
        em = np.isin(rows, list(emb_rows))
        idx = np.nonzero(em)[0]
        for tag, (mid, _lic, _p, prefix) in EMBED_MODELS.items():
            te = time.time()
            for fld, (a, b) in {
                "name_raw": ([A["name_raw"][i] for i in idx], [Bc["name_raw"][i] for i in idx]),
                "name_core_tr": ([variants["core_tr"][0][i] for i in idx], [variants["core_tr"][1][i] for i in idx]),
                "addr_raw": ([A["addr_raw"][i] for i in idx], [Bc["addr_raw"][i] for i in idx]),
            }.items():
                col = np.full(n, np.nan, np.float32)
                col[idx] = embed_cosine(a, b, mid, prefix)
                feats[f"emb_{tag}_{fld}"] = col
            print(f"  embeddings {tag} done [{time.time() - te:.0f}s]", flush=True)

    out = {
        "s1_row": rows, "rec": recs, "label": label,
        "in_base": np.isin(rows, sub[in_base]),
        "gt_count": sample.gt_count[rows],
        "cty": s1_cty[rows],
        "s1_name": A["name_raw"], "s1_addr": A["addr_raw"],
        "c_name": Bc["name_raw"], "c_addr": Bc["addr_raw"],
        "c_script": Bc["name_script"], "c_addr_empty": np.asarray(Bc["a_empty"], bool),
        "c_src": cand.column("src").to_numpy(),
        **feats,
    }
    tbl = pa.table({k: (pa.array(v) if not isinstance(v, pa.Array) else v) for k, v in out.items()})
    pq.write_table(tbl, os.path.join(data.CACHE_DIR, "pairs.parquet"), compression="zstd")
    print(f"done: {tbl.num_rows:,} pairs x {len(feats)} features in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    sys.exit(main())
