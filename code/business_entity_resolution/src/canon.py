"""Learned token canonicalisation (normalisation pass 3).

Systematic noise that generic rules miss is learned from the training ground
truth and from the Source-1 vocabulary:

1. Substitution mining.  For true pairs whose token sets differ by at most two
   tokens on each side, count (record token x -> Source-1 token y) swaps.  A
   swap becomes a synonym if it is frequent, highly predictive (P(y|x) high)
   and orthographically plausible (same phonetic skeleton / high Jaro-Winkler)
   or x is not a Source-1 word at all.  Synonym groups are merged with
   union-find and mapped onto their most frequent Source-1 spelling
   (jay/jai, shree/sree, laxmi/lakshmi, centre/center, raod/rd, lndia/india...).
2. OCR letter confusions for out-of-vocabulary tokens (l<->i, rn->m, vv->w,
   cl->d): accepted only if the repaired token is a frequent Source-1 word.
3. Glued words ("nursinghome", "sevasamiti"): an out-of-vocabulary token that
   splits into two frequent Source-1 words is split.

Rules 2-3 only use the (unlabelled) Source-1 vocabulary of each universe, so
they also apply to countries without training labels.
"""
from __future__ import annotations

import pickle
from collections import Counter

import polars as pl
from rapidfuzz.distance import JaroWinkler

from common import Timer, log, wpath
from translit import skeleton

NAME_COLS = ("n_all", "n_core", "n_alt")
ADDR_COLS = ("a_tok", "a_street", "a_loc", "a_all")


class UnionFind:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def _is_subsequence(a: str, b: str) -> bool:
    it = iter(b)
    return all(ch in it for ch in a)


def _is_glue(x: str, y: str, s1_counts: Counter, min_count=20) -> bool:
    """x = y + r or r + y where the remainder r is itself a real word."""
    if len(x) < len(y) + 3:
        return False
    if x.startswith(y) and s1_counts.get(x[len(y):], 0) >= min_count:
        return True
    if x.endswith(y) and s1_counts.get(x[:-len(y)], 0) >= min_count:
        return True
    return False


def mine_synonyms(q_toks, e_toks, s1_counts: Counter, min_count=30, min_p=0.8):
    """Mine systematic (noisy token -> reference token) substitutions.

    * glue fragments (x contains y or vice versa with a long remainder) are
      skipped - glued words are split by rule_repairs instead;
    * two REAL Source-1 words are merged (union-find) only if they are
      orthographic variants (jay/jai, industry/industries);
    * a token that is NOT a Source-1 word is mapped directly (never chained)
      onto its reference form (lndia->india, stret->st, intl->international)."""
    sub, dq = Counter(), Counter()
    for a, b in zip(q_toks, e_toks):
        A, B = set(a.split()), set(b.split())
        Dq, De = A - B, B - A
        for t in Dq:
            dq[t] += 1
        if 1 <= len(Dq) <= 2 and 1 <= len(De) <= 2:
            for x in Dq:
                for y in De:
                    sub[(x, y)] += 1
    uf = UnionFind()
    direct = {}
    accepted = []
    best_for_x = {}
    glue = {}
    for (x, y), c in sub.items():
        if c < min_count or x.isdigit() or y.isdigit() or len(x) < 2 or len(y) < 2:
            continue
        p = c / dq[x]
        if _is_glue(x, y, s1_counts):
            # x = y + r (or r + y) in TRUE pairs: a glued word -> learn its split
            if p >= 0.3 and s1_counts.get(x, 0) == 0:
                split = (y + " " + x[len(y):]) if x.startswith(y) else (x[:-len(y)] + " " + y)
                if c > glue.get(x, (0, None))[0]:
                    glue[x] = (c, split)
            continue
        if p < min_p or _is_glue(y, x, s1_counts):
            continue
        jw = JaroWinkler.normalized_similarity(x, y)
        ortho = (skeleton(x) == skeleton(y) or jw >= 0.9) and abs(len(x) - len(y)) <= 3
        cx, cy = s1_counts.get(x, 0), s1_counts.get(y, 0)
        x_real = cx >= 5 and cx >= 0.02 * max(cy, 1)
        if x_real:
            if ortho and cy >= 5:
                uf.union(x, y)
                accepted.append(("variant", x, y, c, round(p, 3)))
        elif ortho or jw >= 0.85 or p >= 0.9 or (len(x) <= len(y) and _is_subsequence(x, y)):
            if c > best_for_x.get(x, (0, None))[0]:
                best_for_x[x] = (c, y)
    groups = {}
    for x in list(uf.p):
        groups.setdefault(uf.find(x), []).append(x)
    mapping = {}
    for members in groups.values():
        canon = max(members, key=lambda t: (s1_counts.get(t, 0), -len(t)))
        for m in members:
            if m != canon:
                mapping[m] = canon
    for x, (c, y) in best_for_x.items():
        if x not in mapping:
            direct[x] = mapping.get(y, y)
            accepted.append(("noise", x, direct[x], c))
    mapping.update(direct)
    for x, (c, split) in glue.items():
        if x not in mapping:
            mapping[x] = split
            accepted.append(("glue", x, split, c))
    return mapping, accepted


RULE_MAX_COUNT = 50


def rule_repairs(tokens: Counter, s1_counts: Counter, min_s1=20):
    """OCR confusions and glued words for RARE tokens absent from Source 1.

    Frequent tokens absent from Source 1 are deliberately left alone: those are
    the generator's sibling-marker words (northside, eastgate, participations
    ...) - splitting "northside" into "north side" would erase the strongest
    hard-negative signal.  Typos / OCR slips are idiosyncratic, hence rare."""
    out = {}
    for t, n in tokens.items():
        if n > RULE_MAX_COUNT or s1_counts.get(t, 0) > 0 or not t.isalpha() or len(t) < 4:
            continue
        cands = []
        if t[0] == "l":
            cands.append("i" + t[1:])
        for a, b in (("rn", "m"), ("vv", "w"), ("cl", "d"), ("ii", "u"), ("l", "i")):
            if a in t:
                cands.append(t.replace(a, b, 1))
        best = max(cands, key=lambda c: s1_counts.get(c, 0), default=None)
        if best and s1_counts.get(best, 0) >= min_s1:
            out[t] = best
            continue
        if len(t) >= 7:
            best_split, best_score = None, 0
            for k in range(3, len(t) - 2):
                a, b = t[:k], t[k:]
                ca, cb = s1_counts.get(a, 0), s1_counts.get(b, 0)
                if ca >= min_s1 and cb >= min_s1 and min(ca, cb) > best_score:
                    best_split, best_score = a + " " + b, min(ca, cb)
            if best_split:
                out[t] = best_split
    return out


def learn(split_frames, pairs_rows):
    """split_frames: dict src -> normalised train frames; pairs_rows: (q_row,e_row)."""
    s1 = split_frames["source1"]
    q = pl.concat([split_frames["source2"], split_frames["source3"]])
    maps = {}
    for field, cols in (("name", "n_core"), ("addr", "a_tok")):
        s1c = Counter(t for s in s1[cols].to_list() for t in s.split())
        j = pairs_rows.sample(n=min(3_000_000, pairs_rows.height), seed=7)
        qa = q[cols].to_numpy()[j["q_row"].to_numpy()]
        ea = s1[cols].to_numpy()[j["e_row"].to_numpy()]
        m, acc = mine_synonyms(qa, ea, s1c)
        maps[field] = m
        log(f"{field}: {len(m)} learned synonyms; examples: {acc[:25]}")
    with open(wpath("models", "canon_maps.pkl"), "wb") as f:
        pickle.dump(maps, f)
    return maps


def apply(frames: dict, maps: dict) -> dict:
    """Apply learned maps + universe-specific rule repairs to all sources."""
    s1 = frames["source1"]
    for field, cols in (("name", NAME_COLS), ("addr", ADDR_COLS)):
        base = cols[1] if field == "name" else cols[0]
        s1c = Counter(t for s in s1[base].to_list() for t in s.split())
        allc = Counter()
        for df in frames.values():
            for s in df[base].to_list():
                allc.update(s.split())
        m = dict(maps.get(field, {}))
        rep = rule_repairs(allc, s1c)
        for k, v in rep.items():
            m.setdefault(k, v)
        log(f"{field}: applying {len(m)} token maps ({len(rep)} rule repairs)")
        if not m:
            continue
        for src, df in frames.items():
            upd = []
            for c in cols:
                vals = df[c].to_list()
                if c == "a_loc":
                    new = [" | ".join(" ".join(m.get(t, t) for t in comp.split()) for comp in v.split(" | "))
                           if v else v for v in vals]
                elif c == "a_all":
                    new = [" , ".join(" ".join(m.get(t, t) for t in comp.split()) for comp in v.split(" , "))
                           if v else v for v in vals]
                else:
                    new = [" ".join(m.get(t, t) for t in v.split()) if v else v for v in vals]
                upd.append(pl.Series(c, new))
            frames[src] = df.with_columns(upd)
    return frames


if __name__ == "__main__":
    from blocking import true_pairs_rows
    from prep import norm_path
    frames = {s: pl.read_parquet(norm_path("train", s)) for s in ("source1", "source2", "source3")}
    with Timer("learn canonical maps"):
        learn(frames, true_pairs_rows())
