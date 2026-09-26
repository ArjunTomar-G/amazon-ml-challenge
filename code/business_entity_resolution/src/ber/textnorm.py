"""Text normalisation for business names and addresses.

Everything here is country-agnostic except the explicitly-labelled US/India
lexicons (legal suffixes, street types, state names). Those lexicons are used
only by the "hand-list" variants so the EDA can measure how much they help
compared with data-driven (document-frequency based) stop-token removal, which
also works for countries that are absent from training (e.g. France).
"""

import re
import unicodedata

from unidecode import unidecode

# --------------------------------------------------------------------------
# Script detection / ASCII folding
# --------------------------------------------------------------------------

_ZERO_WIDTH = dict.fromkeys([0x200B, 0x200C, 0x200D, 0xFEFF, 0x00AD], None)

# Unicode block ranges for the scripts seen in the data.
_SCRIPT_RANGES = [
    (0x0900, 0x097F, "DEVANAGARI"),
    (0x0980, 0x09FF, "BENGALI"),
    (0x0A00, 0x0A7F, "GURMUKHI"),
    (0x0A80, 0x0AFF, "GUJARATI"),
    (0x0B00, 0x0B7F, "ORIYA"),
    (0x0B80, 0x0BFF, "TAMIL"),
    (0x0C00, 0x0C7F, "TELUGU"),
    (0x0C80, 0x0CFF, "KANNADA"),
    (0x0D00, 0x0D7F, "MALAYALAM"),
]


def script_of(s):
    """Return 'ASCII', 'LATIN_EXT' (accented Latin) or the dominant Indic script."""
    if s.isascii():
        return "ASCII"
    counts = {}
    latin_ext = 0
    for ch in s:
        o = ord(ch)
        if o < 128:
            continue
        for lo, hi, name in _SCRIPT_RANGES:
            if lo <= o <= hi:
                counts[name] = counts.get(name, 0) + 1
                break
        else:
            if 0x00C0 <= o <= 0x024F:
                latin_ext += 1
    if counts:
        return max(counts, key=counts.get)
    return "LATIN_EXT" if latin_ext else "OTHER"


def ascii_fold(s):
    """NFKC + drop zero-width chars + unidecode (accents and Indic scripts -> ASCII)."""
    if s.isascii():
        return s
    s = unicodedata.normalize("NFKC", s).translate(_ZERO_WIDTH)
    return unidecode(s)


# --------------------------------------------------------------------------
# Names
# --------------------------------------------------------------------------

_DOTTED_ACRONYM = re.compile(r"\b(?:[a-z]\.){2,}[a-z]?\b\.?")
_ALIAS = re.compile(r"\b(?:aka|a/k/a|dba|d/b/a|fka|f/k/a|formerly known as|formerly)\b")
_URL = re.compile(r"https?://|\bwww\.")
_TLD = re.compile(r"\.(?:co\.in|com|net|org|in|co|biz|info|us|fr|io)\b")
_PHONE = re.compile(r"\+?\d[\d\s\-]{6,}\d")
_MS = re.compile(r"\bm/s\b")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_DIGIT_WORD = re.compile(r"(\d)([a-z]{3,})")

# US/India legal suffixes (hand list).
LEGAL_SUFFIXES = frozenset(
    """inc incorporated llc lc ltd limited pvt private corp corporation co company
    llp lp pllc plc pc pa l c p""".split()
)
# Canonical forms used for the "canonicalised" (not removed) variant.
LEGAL_CANON = {
    "inc": "incorporated", "incorporated": "incorporated",
    "ltd": "limited", "limited": "limited",
    "pvt": "private", "private": "private",
    "corp": "corporation", "corporation": "corporation",
    "co": "company", "company": "company",
    "llc": "llc", "lc": "llc", "llp": "llp", "lp": "lp", "pllc": "pllc",
    "plc": "plc", "pc": "pc", "pa": "pa",
}
HONORIFICS = frozenset("mr mrs ms dr shri sri shree smt m s the sir kumari".split())
STOPWORDS = frozenset("and of the for a an in at by on to with".split())
NAME_HAND_STOP = LEGAL_SUFFIXES | HONORIFICS | STOPWORDS


def norm_name(raw):
    """ASCII-folded, lower-cased, punctuation-free name with alias markers,
    URL scaffolding and phone numbers removed. All content tokens are kept."""
    s = ascii_fold(raw).lower()
    s = s.replace("|", " ")
    s = _ALIAS.sub(" ", s)
    s = _URL.sub(" ", s)
    s = _TLD.sub(" ", s)
    s = _PHONE.sub(" ", s)
    s = _MS.sub(" ", s)
    s = _DOTTED_ACRONYM.sub(lambda m: m.group(0).replace(".", ""), s)
    s = s.replace("'", "").replace("’", "")
    s = s.replace("&", " and ").replace("+", " and ")
    s = _NON_ALNUM.sub(" ", s).strip()
    return s


def name_core_tokens(name_n):
    """Tokens after removing the US/India hand list (legal suffixes, honorifics, stopwords)."""
    return [t for t in name_n.split() if t not in NAME_HAND_STOP]


def name_canon(name_n):
    """Legal suffixes canonicalised (pvt->private, ltd->limited, ...) instead of removed."""
    return " ".join(LEGAL_CANON.get(t, t) for t in name_n.split())


# --------------------------------------------------------------------------
# Addresses
# --------------------------------------------------------------------------

_NULLS = re.compile(r"<\s*null\s*>|\bnull\b|\bn/a\b|\bnone\b")
_ORDINAL_SUFFIX = re.compile(r"\b(\d+)(?:st|nd|rd|th)\b")

ORDINAL_WORDS = {
    "first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5",
    "sixth": "6", "seventh": "7", "eighth": "8", "ninth": "9", "tenth": "10",
    "eleventh": "11", "twelfth": "12", "thirteenth": "13", "fourteenth": "14",
    "fifteenth": "15", "sixteenth": "16", "seventeenth": "17", "eighteenth": "18",
    "nineteenth": "19", "twentieth": "20",
}
# US street-type / directional abbreviations (hand list).
STREET_ABBR = {
    "street": "st", "str": "st", "road": "rd", "avenue": "ave", "av": "ave",
    "drive": "dr", "lane": "ln", "court": "ct", "circle": "cir",
    "boulevard": "blvd", "parkway": "pkwy", "highway": "hwy", "place": "pl",
    "terrace": "ter", "trail": "trl", "alley": "aly", "square": "sq",
    "north": "n", "south": "s", "east": "e", "west": "w", "suite": "ste",
    "apartment": "apt", "nr": "near", "opp": "opposite",
}
ADDR_HAND_STOP = frozenset(
    """st rd ave dr ln ct cir blvd pkwy hwy pl ter trl aly sq way n s e w
    unit apt ste fl floor no number near opposite behind po box h hno house
    flat plot door block shop bldg building c o the of and null na""".split()
)

US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "district of columbia": "dc",
    "florida": "fl", "georgia": "ga", "hawaii": "hi", "idaho": "id", "illinois": "il",
    "indiana": "in", "iowa": "ia", "kansas": "ks", "kentucky": "ky", "louisiana": "la",
    "maine": "me", "maryland": "md", "massachusetts": "ma", "michigan": "mi",
    "minnesota": "mn", "mississippi": "ms", "missouri": "mo", "montana": "mt",
    "nebraska": "ne", "nevada": "nv", "new hampshire": "nh", "new jersey": "nj",
    "new mexico": "nm", "new york": "ny", "north carolina": "nc", "north dakota": "nd",
    "ohio": "oh", "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa",
    "rhode island": "ri", "south carolina": "sc", "south dakota": "sd", "tennessee": "tn",
    "texas": "tx", "utah": "ut", "vermont": "vt", "virginia": "va", "washington": "wa",
    "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy", "puerto rico": "pr",
}
INDIA_STATES = {
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br",
    "chhattisgarh": "cg", "goa": "ga", "gujarat": "gj", "haryana": "hr",
    "himachal pradesh": "hp", "jharkhand": "jh", "karnataka": "ka", "kerala": "kl",
    "keralam": "kl", "madhya pradesh": "mp", "maharashtra": "mh", "manipur": "mn",
    "meghalaya": "ml", "mizoram": "mz", "nagaland": "nl", "odisha": "od", "orissa": "od",
    "punjab": "pb", "rajasthan": "rj", "sikkim": "sk", "tamil nadu": "tn",
    "telangana": "ts", "telengana": "ts", "tripura": "tr", "uttar pradesh": "up",
    "uttarakhand": "uk", "uttaranchal": "uk", "west bengal": "wb", "delhi": "dl",
    "jammu and kashmir": "jk", "jammu kashmir": "jk", "ladakh": "la",
    "chandigarh": "ch", "puducherry": "py", "pondicherry": "py",
    "andaman and nicobar islands": "an", "dadra and nagar haveli": "dn",
    "daman and diu": "dd", "lakshadweep": "ld",
}


def addr_components(raw):
    """Split on commas, ASCII-fold, drop NULL placeholders, clean each component."""
    s = ascii_fold(raw).lower()
    s = _NULLS.sub(" ", s)
    out = []
    for part in s.split(","):
        part = part.replace("'", "")
        part = _NON_ALNUM.sub(" ", part)
        part = _DIGIT_WORD.sub(r"\1 \2", part).strip()
        if part:
            out.append(part)
    return out


def canon_addr_token(t):
    t = ORDINAL_WORDS.get(t, t)
    m = _ORDINAL_SUFFIX.fullmatch(t)
    if m:
        t = m.group(1)
    return STREET_ABBR.get(t, t)


_STATE_CODE = {**US_STATES, **INDIA_STATES}
_STATE_CODES = frozenset(US_STATES.values()) | frozenset(INDIA_STATES.values())


def state_of_component(comp):
    """Canonical state code if the whole component is a US/India state name or code."""
    if comp in _STATE_CODE:
        return _STATE_CODE[comp]
    if len(comp) == 2 and comp in _STATE_CODES:
        return comp
    return ""


def addr_core_tokens(comps):
    """Canonicalised address tokens with the hand stop-list removed; whole-component
    US/India state names are replaced by their 2-letter code."""
    toks = []
    for c in comps:
        st = state_of_component(c)
        if st:
            toks.append(st)
            continue
        for t in c.split():
            t = canon_addr_token(t)
            if t not in ADDR_HAND_STOP:
                toks.append(t)
    return toks


# A component that is only a postal code, optionally preceded by a 2-letter
# state code ("NC 27601", "27601-1234", "75002"), or a 6-digit Indian PIN anywhere.
_ZIP_COMPONENT = re.compile(r"(?:[a-z]{2}\s)?(\d{5})(?:\s?\d{4})?")
_PIN = re.compile(r"(?<!\d)([1-9]\d{5})(?!\d)")


def zip5(comps):
    """5-digit postal code standing alone in a component (US ZIP / FR code postal)."""
    for comp in comps:
        m = _ZIP_COMPONENT.fullmatch(comp)
        if m:
            return m.group(1)
    return ""


def pin6(comps):
    """6-digit Indian-style PIN code anywhere in the address."""
    for comp in comps:
        m = _PIN.search(comp)
        if m:
            return m.group(1)
    return ""
