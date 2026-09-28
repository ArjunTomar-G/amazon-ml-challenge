"""Learned Indic-script -> Latin transliteration dictionary.

About a quarter of Indian Source-2/3 names (and many state names in addresses)
are rendered in one of nine Indic scripts (Devanagari, Bengali, Gurmukhi,
Gujarati, Oriya, Tamil, Telugu, Kannada, Malayalam).  Generic romanisers are
lossy ("रियल मॉडर्न फूड" -> "riyl modrn phud"), so we LEARN the mapping from the
training ground truth only:

  for every true pair (S1 record e, S2/S3 record r) and every Indic token t in
  r's name (address), count co-occurrence with every Latin token w of e's name
  (address).  The translation of t is the w maximising

        P(w | t) * phonetic_similarity(romanise(t), w)

  The phonetic factor breaks ties between tokens that always co-occur
  (e.g. "प्राइवेट" co-occurs with both "private" and "limited").

Tokens never seen in training fall back to a rule-based romanisation (anyascii,
ISC licence) followed by light phonetic fixes.
"""
from __future__ import annotations

import pickle
import re
from collections import Counter, defaultdict

import polars as pl
from anyascii import anyascii
from rapidfuzz.distance import JaroWinkler

from common import (INDIC_RE, RAW_TOKEN_RE, ZW_RE, Timer, load_pairs, load_raw, log,
                    strip_accents, wpath)

DICT_PATH = "translit_dict.pkl"

_LATIN_TOKEN_RE = re.compile(r"[a-z0-9]+")


def raw_tokens(s: str) -> list[str]:
    s = ZW_RE.sub("", s)
    return RAW_TOKEN_RE.findall(s)


def latin_tokens(s: str) -> list[str]:
    return _LATIN_TOKEN_RE.findall(strip_accents(s).lower())


def _clean_indic_token(t: str) -> str:
    # strip leading / trailing dots, danda etc.
    return t.strip(".।॥-'")


_SKEL_SUBS = [("ph", "f"), ("bh", "b"), ("dh", "d"), ("th", "t"), ("kh", "k"), ("gh", "g"),
              ("ch", "c"), ("sh", "s"), ("jh", "j"), ("q", "k"), ("x", "ks"), ("z", "j"),
              ("w", "v"), ("y", "i"), ("ck", "k")]


def skeleton(s: str) -> str:
    """Crude phonetic skeleton: fold aspirates, drop non-initial vowels,
    collapse repeats.  Used for tie-breaking and fallback matching."""
    s = s.lower()
    for a, b in _SKEL_SUBS:
        s = s.replace(a, b)
    s = s.replace("c", "k")
    if not s:
        return s
    out = [s[0]]
    for ch in s[1:]:
        if ch in "aeiou":
            continue
        if ch != out[-1]:
            out.append(ch)
    return "".join(out)


def romanise(t: str) -> str:
    r = anyascii(t).lower()
    # anusvara is rendered as 'm' before any consonant; before non-labials it is 'n'
    r = re.sub(r"m(?=[kgcjtdnszlrh])", "n", r)
    return re.sub(r"[^a-z0-9]", "", r)


def phon_sim(a: str, b: str) -> float:
    return max(JaroWinkler.normalized_similarity(a, b),
               JaroWinkler.normalized_similarity(skeleton(a), skeleton(b)))


def learn_dictionary(min_count: int = 2):
    pairs = load_pairs()
    s1 = load_raw("train", "source1").select("entity_id", "business_name", "business_address")
    s23 = pl.concat([load_raw("train", "source2"), load_raw("train", "source3")]).select(
        "entity_id", "business_name", "business_address")
    s23 = s23.filter(
        pl.col("business_name").str.contains(r"[ऀ-෿]")
        | pl.col("business_address").fill_null("").str.contains(r"[ऀ-෿]"))
    j = (pairs.join(s23, left_on="mid", right_on="entity_id")
         .join(s1, left_on="s1", right_on="entity_id", suffix="_1"))
    log("pairs with Indic text:", j.height)

    c_t = Counter()
    c_tw = defaultdict(Counter)
    for field in ("business_name", "business_address"):
        for r_txt, e_txt in zip(j[field].to_list(), j[field + "_1"].to_list()):
            if not r_txt or not e_txt or not INDIC_RE.search(r_txt):
                continue
            T = {_clean_indic_token(t) for t in raw_tokens(r_txt) if INDIC_RE.search(t)}
            T.discard("")
            W = set(latin_tokens(e_txt))
            for t in T:
                c_t[t] += 1
                ct = c_tw[t]
                for w in W:
                    ct[w] += 1

    mapping = {}
    for t, n in c_t.items():
        if n < min_count:
            continue
        rom = romanise(t)
        best, best_s = None, 0.0
        for w, k in c_tw[t].most_common(12):
            p = k / n
            if p < 0.25:
                break
            s = p * (0.25 + 0.75 * phon_sim(rom, w))
            if s > best_s:
                best, best_s = w, s
        if best is not None and best_s >= 0.35:
            mapping[t] = best
    log(f"learned {len(mapping)} Indic token translations from {len(c_t)} distinct tokens")
    with open(wpath("models", DICT_PATH), "wb") as f:
        pickle.dump(mapping, f)
    return mapping


_DICT = None


def get_dict():
    global _DICT
    if _DICT is None:
        with open(wpath("models", DICT_PATH), "rb") as f:
            _DICT = pickle.load(f)
    return _DICT


def transliterate_text(s: str, d: dict | None = None) -> tuple[str, int, int]:
    """Replace every Indic token of `s` by its learned Latin translation.

    Returns (text, n_indic_tokens, n_dictionary_hits)."""
    if not s or not INDIC_RE.search(s):
        return s, 0, 0
    d = d if d is not None else get_dict()
    s = ZW_RE.sub("", s)
    n_ind = n_hit = 0

    def rep(m):
        nonlocal n_ind, n_hit
        tok = m.group(0)
        if not INDIC_RE.search(tok):
            return tok
        n_ind += 1
        core = _clean_indic_token(tok)
        w = d.get(core)
        if w is not None:
            n_hit += 1
            return w
        return romanise(core)

    out = RAW_TOKEN_RE.sub(rep, s)
    return out, n_ind, n_hit


if __name__ == "__main__":
    with Timer("learn transliteration dictionary"):
        m = learn_dictionary()
    for k in list(m)[:40]:
        print(k, "->", m[k])
