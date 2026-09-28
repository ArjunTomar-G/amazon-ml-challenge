"""Record normalisation: business names and addresses.

Every record (S1/S2/S3, train/test) is turned into a set of canonical string
fields that the blocking and feature stages work on.  Everything here is
deterministic, country-agnostic in structure (country only selects the
state-canonicalisation table) and learned/derived exclusively from the
provided data plus generic domain knowledge (legal-form and street-type
abbreviations).

Name fields
    n_all     all canonical tokens (legal forms canonicalised) in original order
    n_core    tokens without legal forms / honorifics / stop words
    n_alt     core tokens of the *other* side of an alias marker ("X d/b/a Y"),
              or of the " | " suffix (usually a domain) - empty if none
    n_legal   sorted legal-form tokens
    n_flags   bit flags (see NF_* constants)
    n_indic   number of Indic-script tokens, n_indic_hit = dictionary hits
Address fields
    a_all     canonical address string, components joined by " , "
    a_tok     canonical tokens (state removed) in original order
    a_hn      house number digits (leading zeros stripped) or ""
    a_hnsfx   house number suffix (letter / bis->b / range end) or ""
    a_street  street tokens of the numbered component (no house number)
    a_loc     locality components (city / district ...) joined by " | "
    a_state   canonical state key or ""
    a_unit    unit / po box / pmb tokens
    a_nums    all numeric tokens (canonicalised) space separated
    a_flags   bit flags (see AF_* constants)
"""
from __future__ import annotations

import re
import unicodedata

from common import strip_accents
from geo import STATE_LOOKUP
from translit import get_dict, transliterate_text

# ----------------------------------------------------------------------------
# name normalisation
# ----------------------------------------------------------------------------
NF_ALIAS = 1        # "X d/b/a Y", "X formerly Y", ...
NF_DOMAIN = 2       # www / .com / trailing 'com'
NF_HANDLE = 4       # @handle or #handle
NF_ID = 8           # "#12345", "(ID: 8138)"
NF_PHONE = 16       # long digit run
NF_HONOR = 32       # Mr / Dr / Shri / The / M/s ...
NF_PIPE = 64        # "name | www.x.com"
NF_UPPER = 128      # raw name all upper-case
NF_OCR = 256        # digit->letter OCR repair applied
NF_PERSON_INV = 512  # "Smith, John" style comma inversion

_ALIAS_RE = re.compile(
    r"\s(?:trading\s+as|t/a|d/b/a|d\.b\.a\.?|dba:?|doing\s+business\s+as|f/k/a|fka|"
    r"formerly\s+known\s+as|formerly|a/k/a|aka|also\s+known\s+as)\s", re.I)
_ID_RE = re.compile(r"\(\s*id\s*[:#]?\s*\d+\s*\)|\bid\s*[:#]\s*\d+|#\s*\d{3,}\b", re.I)
_PHONE_RE = re.compile(r"(?<![\w])\+?\d[\d\s\-().]{7,}\d(?![\w])")
_WWW_RE = re.compile(r"\bwww\.", re.I)
_DOMAIN_RE = re.compile(r"\b([a-z0-9][a-z0-9\-]*)\.(?:com|net|org|co\.in|in|fr|io|biz|info|co)\b")
_HANDLE_RE = re.compile(r"(?<![\w])[@#]([a-z][a-z0-9_.]*)")
_ABBR_DOTS_RE = re.compile(r"\b(?:[a-z]\.){2,}[a-z]?\b|\b[a-z]\.[a-z]\b")
_ORDINAL_RE = re.compile(r"^\d+(?:st|nd|rd|th)$")
# digit->letter OCR confusions used by the noise generator (measured on train):
# 0->O, 1->l, 5->S, 8->B, 6->G.  Other digits inside words come from glued handles.
_OCR_MAP = str.maketrans({"0": "o", "1": "l", "5": "s", "8": "b", "6": "g"})
_NONWORD_RE = re.compile(r"[^a-z0-9 ]+")
_WS_RE = re.compile(r"\s+")

LEGAL_MAP = {
    "llc": "llc", "inc": "inc", "incorporated": "inc", "lnc": "inc", "incorp": "inc",
    "corp": "corp", "corporation": "corp", "corpn": "corp", "co": "co", "company": "co",
    "ltd": "ltd", "limited": "ltd", "ltda": "ltd", "pvt": "pvt", "private": "pvt",
    "prvt": "pvt", "pvte": "pvt", "llp": "llp", "lp": "lp", "plc": "plc", "pllc": "pllc",
    "pc": "pc", "pa": "pa", "psc": "psc", "opc": "opc", "gmbh": "gmbh", "ag": "ag",
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "sa": "sa", "eurl": "eurl", "sci": "sci",
    "snc": "snc", "scp": "scp", "scm": "scm", "selarl": "selarl", "selas": "selas",
    "gie": "gie", "sca": "sca", "scop": "scop", "sem": "sem", "cie": "cie",
    "ei": "ei",     # entreprise individuelle: ends 4 182 test-France S1 names
}
# One-directional name abbreviations of the noise generator: the reference source (S1)
# spells the word out, S2/S3 abbreviate it (test France: "frs" 3x in S1 vs 1 391x in S2/S3,
# "svcs" 0x vs 115x; same-address "saint" -> "st" swaps are left unlinked by v7c).
NAME_ABBREV = {"st": "saint", "frs": "freres", "svcs": "services", "svc": "service"}
HONORIFICS = {"the", "mr", "mrs", "ms", "dr", "shri", "sri", "smt", "messrs", "ms", "mx"}
NAME_STOP = {"and", "of", "de", "la", "le", "les", "des", "du", "et", "d", "l", "en", "a"}

_LLC_PHRASE_RE = re.compile(r"\blimited\s+liability\s+(?:company|co)\b")
_PVT_P_RE = re.compile(r"\(\s*p\s*\)")
_MS_RE = re.compile(r"\bm\s*/\s*s\b")


def _ocr_fix(tok: str) -> tuple[str, bool]:
    if tok.isdigit() or tok.isalpha() or _ORDINAL_RE.match(tok):
        return tok, False
    n_alpha = sum(c.isalpha() for c in tok)
    if n_alpha < 2 or n_alpha <= len(tok) - n_alpha:
        return tok, False
    return tok.translate(_OCR_MAP), True


def _clean_name_part(s: str, flags: list) -> list[str]:
    """Lower-cased latin string -> canonical token list."""
    s = _LLC_PHRASE_RE.sub(" llc ", s)
    s = _PVT_P_RE.sub(" pvt ", s)
    if _MS_RE.search(s):
        s = _MS_RE.sub(" ", s)
        flags[0] |= NF_HONOR
    if _ID_RE.search(s):
        s = _ID_RE.sub(" ", s)
        flags[0] |= NF_ID
    if _PHONE_RE.search(s):
        s = _PHONE_RE.sub(" ", s)
        flags[0] |= NF_PHONE
    if _WWW_RE.search(s):
        s = _WWW_RE.sub(" ", s)
        flags[0] |= NF_DOMAIN
    if _DOMAIN_RE.search(s):
        s = _DOMAIN_RE.sub(r" \1 ", s)
        flags[0] |= NF_DOMAIN
    if _HANDLE_RE.search(s):
        s = _HANDLE_RE.sub(r" \1 ", s)
        flags[0] |= NF_HANDLE
    s = _ABBR_DOTS_RE.sub(lambda m: m.group(0).replace(".", ""), s)
    s = s.replace("&", " and ").replace("+", " and ")
    s = s.replace("'", "").replace("’", "").replace("`", "")
    s = _NONWORD_RE.sub(" ", s)
    toks = []
    for t in s.split():
        t2, ch = _ocr_fix(t)
        if ch:
            flags[0] |= NF_OCR
        toks.append(t2)
    return toks


def normalize_name(raw: str, tdict=None) -> dict:
    if raw is None:
        raw = ""
    flags = [0]
    s = unicodedata.normalize("NFKC", raw)
    s, n_ind, n_hit = transliterate_text(s, tdict)
    if raw.isupper():
        flags[0] |= NF_UPPER
    s = strip_accents(s).lower()
    if re.search(r"\w+,\s*\w+", s) and "," in s:
        flags[0] |= NF_PERSON_INV
    alt = ""
    parts = _ALIAS_RE.split(" " + s + " ")
    if len(parts) >= 2:
        flags[0] |= NF_ALIAS
        main_txt = parts[-1]
        alt = " ".join(parts[:-1])
    else:
        main_txt = s
    if "|" in main_txt:
        flags[0] |= NF_PIPE
        a, b = main_txt.split("|", 1)
        main_txt, alt = a, (alt + " " + b).strip()
    toks = _clean_name_part(main_txt, flags)
    alt_toks = _clean_name_part(alt, flags) if alt else []

    def canon(tokens):
        allt, core, legal = [], [], []
        for t in tokens:
            t = NAME_ABBREV.get(t, t)
            if t in HONORIFICS:
                flags[0] |= NF_HONOR
                continue
            lt = LEGAL_MAP.get(t)
            if lt is not None:
                allt.append(lt)
                legal.append(lt)
                continue
            allt.append(t)
            if t not in NAME_STOP:
                core.append(t)
        return allt, core, legal

    allt, core, legal = canon(toks)
    _, alt_core, alt_legal = canon(alt_toks)
    if not core and alt_core:           # e.g. "LLC | www.x.com"
        core, alt_core = alt_core, []
    return {
        "n_all": " ".join(allt),
        "n_core": " ".join(core),
        "n_alt": " ".join(alt_core),
        "n_legal": " ".join(sorted(set(legal))),
        "n_flags": flags[0],
        "n_indic": n_ind,
        "n_indic_hit": n_hit,
    }


# ----------------------------------------------------------------------------
# address normalisation
# ----------------------------------------------------------------------------
AF_MISSING = 1
AF_NULL = 2          # had null / <NULL> / N/A placeholders
AF_POBOX = 4
AF_PMB = 8
AF_UNIT = 16
AF_REORDER = 32      # first component is a state or locality (not the street)
AF_NOHN = 64         # no house number found
AF_HASH = 128        # '#'-prefixed number
AF_RANGE = 256       # house number range 12-14
AF_HALF = 512        # 1/2 house number
AF_INDIC = 1024

_NULL_RE = re.compile(r"<\s*null\s*>|\bnull\b|\bnone\b|\bn/a\b|\bna\b(?=\s*(?:,|$))", re.I)
_ADDR_KEEP_RE = re.compile(r"[^a-z0-9/\- ]+")
_NUMTOK_RE = re.compile(r"\d")
_LEADZERO_RE = re.compile(r"(?<![0-9])0+(?=\d)")

STREET_TYPES = {
    "street": "st", "st": "st", "str": "st", "saint": "st", "avenue": "ave", "ave": "ave",
    "av": "ave", "avn": "ave", "aven": "ave", "avnue": "ave", "boulevard": "blvd", "blvd": "blvd",
    "bd": "blvd", "boul": "blvd", "bld": "blvd", "road": "rd", "rd": "rd", "drive": "dr",
    "dr": "dr", "drv": "dr", "court": "ct", "ct": "ct", "place": "pl", "pl": "pl", "lane": "ln",
    "ln": "ln", "circle": "cir", "cir": "cir", "cove": "cv", "cv": "cv", "trail": "trl",
    "trl": "trl", "terrace": "ter", "ter": "ter", "parkway": "pkwy", "pkwy": "pkwy",
    "highway": "hwy", "hwy": "hwy", "square": "sq", "sq": "sq", "plaza": "plz", "plz": "plz",
    "expressway": "expy", "expy": "expy", "freeway": "fwy", "fwy": "fwy", "alley": "aly",
    "crossing": "xing", "xing": "xing", "heights": "hts", "hts": "hts", "junction": "jct",
    "mount": "mt", "mt": "mt", "mountain": "mtn", "ridge": "rdg", "point": "pt",
    "pt": "pt", "route": "rte", "rte": "rte", "turnpike": "tpke", "crescent": "cres",
    "grove": "grv", "harbor": "hbr", "village": "vlg", "center": "ctr", "centre": "ctr",
    "fort": "ft", "ft": "ft", "sainte": "ste", "ste": "ste",
    # french
    "rue": "rue", "r": "rue", "allee": "allee", "all": "allee", "impasse": "imp", "imp": "imp",
    "chemin": "chem", "chem": "chem", "che": "chem", "ch": "chem", "quai": "quai",
    "cours": "cours", "crs": "cours", "residence": "res", "res": "res", "faubourg": "fbg",
    "fbg": "fbg", "lotissement": "lot", "rond": "rpt", "rpt": "rpt", "sentier": "sen",
    "passage": "pass", "hameau": "ham", "lieu": "lieu", "dit": "dit",
    # india
    "marg": "marg", "nagar": "nagar", "ngr": "nagar", "colony": "colony", "sector": "sector",
    "sec": "sector", "bazar": "bazaar", "bazaar": "bazaar", "gali": "gali",
}
DIRECTIONS = {"north": "n", "south": "s", "east": "e", "west": "w", "northeast": "ne",
              "northwest": "nw", "southeast": "se", "southwest": "sw", "n": "n", "s": "s",
              "e": "e", "w": "w", "ne": "ne", "nw": "nw", "se": "se", "sw": "sw"}
ORDINAL_WORDS = {"first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5",
                 "sixth": "6", "seventh": "7", "eighth": "8", "ninth": "9", "tenth": "10",
                 "eleventh": "11", "twelfth": "12", "thirteenth": "13", "fourteenth": "14",
                 "fifteenth": "15", "sixteenth": "16", "seventeenth": "17", "eighteenth": "18",
                 "nineteenth": "19", "twentieth": "20", "six": "6"}
UNIT_WORDS = {"unit", "apt", "apartment", "suite", "ste", "pmb", "box", "po", "fl", "floor",
              "rm", "room", "bldg", "building", "trlr", "spc", "lot", "dept", "#"}
HN_PREFIX_WORDS = {"no", "h", "hn", "hno", "house", "door", "dno", "plot", "flat", "shop",
                   "office", "survey", "sy", "s", "khasra", "kh", "block", "blk", "number",
                   "num", "nos", "ward", "n", "nr", "numero"}
_SFX_LETTERS = set("abcdfghq")    # single-letter house-number suffixes (not n/s/e/w directions)
_ORD_SUFFIX_RE = re.compile(r"^(\d+)(?:st|nd|rd|th)$")
_STREET_TYPE_VALUES = {"st", "ave", "blvd", "rd", "dr", "ct", "pl", "ln", "cir", "cv", "trl", "ter",
                       "pkwy", "hwy", "sq", "plz", "expy", "fwy", "aly", "xing", "way", "run",
                       "loop", "pike", "path", "walk", "row", "rue", "allee", "imp", "chem",
                       "quai", "cours", "rte", "marg"}
_HN_RE = re.compile(r"^#*(\d+)(?:\s*-\s*(\d+))?([a-z]{0,6})?$")
_FR_SFX = {"bis": "b", "ter": "t", "quater": "q", "b": "b", "t": "t", "q": "q", "a": "a",
           "c": "c", "d": "d"}


def _canon_num(tok: str) -> str:
    # strip leading zeros inside every digit run: c-0373 -> c-373, 009 -> 9
    return _LEADZERO_RE.sub("", tok)


def _addr_tokens(comp: str) -> list[str]:
    comp = comp.replace(".", " ").replace("'", "")
    comp = _ADDR_KEEP_RE.sub(" ", comp)
    out = []
    for t0 in comp.split():
        t0 = t0.strip("-/")
        if not t0:
            continue
        pieces = [t0] if _NUMTOK_RE.search(t0) else [p for p in re.split(r"[-/]", t0) if p]
        for t in pieces:
            if _NUMTOK_RE.search(t):
                m = _ORD_SUFFIX_RE.match(t)
                if m:
                    out.append(_canon_num(m.group(1)) + "th")
                    continue
                out.append(_canon_num(t))
                continue
            if t in ORDINAL_WORDS:
                out.append(ORDINAL_WORDS[t] + "th")
                continue
            t2 = STREET_TYPES.get(t)
            if t2 is None:
                t2 = DIRECTIONS.get(t, t)
            out.append(t2)
    return out


def _state_key(comp_norm: str, table: dict):
    if not table:
        return None
    return table.get(comp_norm)


_COMP_CLEAN_RE = re.compile(r"[^a-z0-9 ]+")


def normalize_address(raw: str, country: str, tdict=None) -> dict:
    empty = {"a_all": "", "a_tok": "", "a_hn": "", "a_hnsfx": "", "a_street": "", "a_loc": "",
             "a_state": "", "a_unit": "", "a_nums": "", "a_flags": AF_MISSING | AF_NOHN}
    if raw is None or not raw.strip():
        return empty
    flags = 0
    s = unicodedata.normalize("NFKC", raw)
    s, n_ind, _ = transliterate_text(s, tdict)
    if n_ind:
        flags |= AF_INDIC
    s = strip_accents(s).lower()
    if _NULL_RE.search(s):
        s = _NULL_RE.sub(" ", s)
        flags |= AF_NULL
    table = STATE_LOOKUP.get((country or "").strip().lower(), {})
    comps_raw = [c.strip() for c in s.split(",")]
    comps_raw = [c for c in comps_raw if c and _COMP_CLEAN_RE.sub("", c).strip()]
    if not comps_raw:
        return empty

    state = ""
    street_idx = -1
    comps = []           # list of (kind, tokens)
    cnorm = [_WS_RE.sub(" ", _COMP_CLEAN_RE.sub(" ", c.replace("-", " "))).strip() for c in comps_raw]
    st_idx = [i for i, cn in enumerate(cnorm)
              if not _NUMTOK_RE.search(cn) and _state_key(cn, table) is not None]
    if st_idx:
        codes = [i for i in st_idx if len(cnorm[i]) <= 3]
        chosen = codes[-1] if codes else st_idx[-1]
        state = _state_key(cnorm[chosen], table)
        # a second state-like component whose key equals the chosen one is a duplicate
        st_idx = {i for i in st_idx if i == chosen or _state_key(cnorm[i], table) == state}
    for i, c in enumerate(comps_raw):
        if i in st_idx:
            comps.append(("state", []))
            continue
        toks = _addr_tokens(c)
        if not toks:
            continue
        kind = "loc"
        if toks[0] in ("po", "pmb") or (len(toks) > 1 and toks[0] == "p" and toks[1] == "o"):
            kind = "unit"
            flags |= AF_POBOX if toks[0] != "pmb" else AF_PMB
        elif toks[0] in ("unit", "apt", "apartment", "suite", "ste", "#") and len(toks) <= 4:
            kind = "unit"
            flags |= AF_UNIT
        elif any(_NUMTOK_RE.search(t) for t in toks):
            kind = "num"
        comps.append((kind, toks))

    # locate the street component: first numbered component that starts with a
    # number (or a number-prefix word), else first numbered component.
    hn, hnsfx, street = "", "", []
    num_comps = [i for i, (k, t) in enumerate(comps) if k == "num"]
    for i in num_comps:
        toks = comps[i][1]
        j = 0
        while j < len(toks) - 1 and toks[j] in HN_PREFIX_WORDS:
            j += 1
        if j < len(toks) and _NUMTOK_RE.search(toks[j]):
            street_idx = i
            break
    if street_idx < 0 and num_comps:
        street_idx = num_comps[0]
    if street_idx >= 0:
        toks = comps[street_idx][1]
        raw_comp = comps_raw[street_idx] if street_idx < len(comps_raw) else ""
        if "#" in raw_comp:
            flags |= AF_HASH
        j = 0
        while j < len(toks) - 1 and toks[j] in HN_PREFIX_WORDS:
            j += 1
        if j < len(toks) and not _NUMTOK_RE.search(toks[j]):
            k = next((q for q in range(len(toks)) if _NUMTOK_RE.search(toks[q])), None)
            j = k if k is not None else len(toks)
        if j < len(toks):
            first = toks[j]
            m = None if _ORD_SUFFIX_RE.match(first) else _HN_RE.match(first)
            before = [t for t in toks[:j] if t not in HN_PREFIX_WORDS]
            rest = before + toks[j + 1:]
            if m:
                hn = m.group(1).lstrip("0") or "0"
                sfx = m.group(3) or ""
                if m.group(2):
                    flags |= AF_RANGE
                    hnsfx = "-" + m.group(2)
                elif sfx:
                    hnsfx = _FR_SFX.get(sfx, sfx[:1])
                # "19 1/2", "5 bis", "15 b" style suffixes
                if rest and rest[0] in ("1/2",):
                    flags |= AF_HALF
                    hnsfx = hnsfx or "h"
                    rest = rest[1:]
                elif rest and len(rest) > 1 and rest[0] in ("bis", "ter", "quater"):
                    hnsfx = hnsfx or _FR_SFX[rest[0]]
                    rest = rest[1:]
                elif (rest and len(rest) > 1 and len(rest[0]) == 1 and rest[0] in _SFX_LETTERS
                      and rest[1] in _STREET_TYPE_VALUES):
                    hnsfx = hnsfx or rest[0]
                    rest = rest[1:]
                street = rest
            else:
                # alphanumeric first token (india style: a-304, 8-2-120/1, wz-340)
                hn = first if not _ORD_SUFFIX_RE.match(first) else ""
                street = rest if hn else toks
            # canonical house-number token inside the street component
            if hn:
                hn_tok = hn + (hnsfx if hnsfx and hnsfx[0] != "-" and hnsfx != "h" else "")
                comps[street_idx] = ("num", [hn_tok] + street)
    else:
        # no numbered component: a component ending in a street type is the street
        for i, (k, toks) in enumerate(comps):
            if k == "loc" and len(toks) >= 2 and toks[-1] in _STREET_TYPE_VALUES:
                street_idx = i
                street = toks
                comps[i] = ("street", toks)
                break
    if not hn:
        flags |= AF_NOHN

    # reorder flag: first component is not the street / unit component
    if comps and comps[0][0] in ("state", "loc") and street_idx > 0:
        flags |= AF_REORDER

    all_toks, locs, unit_toks, nums = [], [], [], []
    for i, (k, toks) in enumerate(comps):
        if k == "state":
            continue
        if k == "unit":
            unit_toks.extend(toks)
            continue
        all_toks.extend(toks)
        if k == "loc":
            locs.append(" ".join(toks))
        for t in toks:
            if _NUMTOK_RE.search(t):
                nums.append(t)
    a_all = " , ".join(" ".join(t) for k, t in comps if k != "state" and t)
    return {
        "a_all": a_all,
        "a_tok": " ".join(all_toks),
        "a_hn": hn,
        "a_hnsfx": hnsfx,
        "a_street": " ".join(street),
        "a_loc": " | ".join(locs),
        "a_state": state,
        "a_unit": " ".join(unit_toks),
        "a_nums": " ".join(nums),
        "a_flags": flags,
    }


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    d = get_dict()
    tests_n = ["Aviriza Labs trading as Kimbra Juarez Reliable Telephone P.C.",
               "A Unified Bóx L.L.C. | www.aunified.com", "Jarvis Unified-8arings",
               "The Jewel Lange Trusted Home Improvements Inc. (ID: 8138)",
               "Apex Pennymac - 4350152116", "M/s CP TRADING PRIVATE LIMITED", "@kirklandbergstr0m",
               "Smith, Rickie Sílver Medicine", "हाईटेक Logistics प्राइवेट लिमिटेड",
               "Shri MGM IT PRIVATE LTD", "Hungry + Sio Corporation", "Orchid Technologies (India) Pvt Ltd",
               "Onyxjax f/k/a Al Solutions Private Limited", "Kelojaxlum d/b/a Mgm It Private Limited",
               "Ectodova DBA: Painters Local 865", "Barca (France) SARL [Développement]",
               "Petit & Cie International", "lavandouamicalesasu.com", "Coalitionharborcom",
               "Blume, Darla K.,  DO, DDS PC #14675", "Susana Muller, O.D., M.D., P.C.", "E 2 Advanced Jersey",
               "1st Choice 24x7 Services"]
    for t in tests_n:
        print(f"{t!r:60} -> {normalize_name(t, d)}")
    tests_a = [("GREENSBORO, NC, 19 1/2 STARDUST TRAIL", "US"), ("##4 HIGHLAND DR, BRONSVILLE, TX", "US"),
               ("HN 942. A-304, THIRD FLOOR, DOUBLE STOREY, KALKAJI, NEW DELHI, Delhi", "India"),
               ("1167-1169 ST CLAIR DRIVE, SIDNEEY, OH", "US"), ("NULL, AL, 1314 SEVENTH AVE, DECATUR", "US"),
               ("2100 Cameron Drive, Unit APARTMENT G, Dundalk, MD", "US"),
               ("1 Ivanhoe Ave, PO Box 6009, Cincinnati, Ohio", "US"),
               ("5 bis Rue Pierre Dignac, La Teste-de-Buch, Nouvelle-Aquitaine", "France"),
               ("45BIS RUE DE MISÉRICORDE, Nantes", "France"), ("Gironde, 0020 Av Des Bruyeres Pyla, La Teste-de-buch", "France"),
               ("1T AVE DU PONDY, LA BAULE-ESCOUBLAC", "France"), ("ST.-NAZAIRE, 12 R. DE LA PAIX", "France"),
               ("Door No 183, 41St Cross, 22Nd Main 9Th Block Jayanagar, Bengaluru Urban, Bangalore, ಕರ್ನಾಟಕ", "India"),
               ("BLOCK C-900 RAGAS FLAT, PLOT NO.53, NEAR JAYA COLLEGE, CHENNAI, Tamil Nadu", "India"),
               ("C-004/131, Safdarjung Development Area, New Delhi, दिल्ली", "India"),
               ("3703 151st Avenue E, Tulsa, OK", "US"), ("OK, TULSA, 151ST AVE EAST", "US"),
               ("206 Green Saint, Unit 458, Winston-Salem, North Carolina", "US"), (None, "US"),
               ("Rua Augusta 100, Lisboa", "Portugal")]
    for t, c in tests_a:
        print(f"{str(t)!r:70} -> {normalize_address(t, c, d)}")
