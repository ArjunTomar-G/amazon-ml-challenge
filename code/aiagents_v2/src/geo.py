"""Small geographic normalisation tables (domain knowledge, no external lookup).

Only *canonicalisation* tables are hard-coded here (state code <-> state name,
French department -> region).  Nothing in the pipeline depends on the country
being one of these: unknown countries simply get no state canonicalisation.
"""

US_STATES = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas", "ca": "california",
    "co": "colorado", "ct": "connecticut", "de": "delaware", "fl": "florida", "ga": "georgia",
    "hi": "hawaii", "id": "idaho", "il": "illinois", "in": "indiana", "ia": "iowa",
    "ks": "kansas", "ky": "kentucky", "la": "louisiana", "me": "maine", "md": "maryland",
    "ma": "massachusetts", "mi": "michigan", "mn": "minnesota", "ms": "mississippi",
    "mo": "missouri", "mt": "montana", "ne": "nebraska", "nv": "nevada", "nh": "new hampshire",
    "nj": "new jersey", "nm": "new mexico", "ny": "new york", "nc": "north carolina",
    "nd": "north dakota", "oh": "ohio", "ok": "oklahoma", "or": "oregon", "pa": "pennsylvania",
    "ri": "rhode island", "sc": "south carolina", "sd": "south dakota", "tn": "tennessee",
    "tx": "texas", "ut": "utah", "vt": "vermont", "va": "virginia", "wa": "washington",
    "wv": "west virginia", "wi": "wisconsin", "wy": "wyoming", "dc": "district of columbia",
    "pr": "puerto rico", "gu": "guam", "vi": "virgin islands",
}

IN_STATES = {
    "ap": "andhra pradesh", "ar": "arunachal pradesh", "as": "assam", "br": "bihar",
    "cg": "chhattisgarh", "ga": "goa", "gj": "gujarat", "hr": "haryana", "hp": "himachal pradesh",
    "jh": "jharkhand", "ka": "karnataka", "kl": "kerala", "mp": "madhya pradesh",
    "mh": "maharashtra", "mn": "manipur", "ml": "meghalaya", "mz": "mizoram", "nl": "nagaland",
    "od": "odisha", "pb": "punjab", "rj": "rajasthan", "sk": "sikkim", "tn": "tamil nadu",
    "tg": "telangana", "tr": "tripura", "up": "uttar pradesh", "uk": "uttarakhand",
    "wb": "west bengal", "dl": "delhi", "jk": "jammu and kashmir", "la": "ladakh",
    "ch": "chandigarh", "py": "puducherry", "an": "andaman and nicobar islands",
    "dn": "dadra and nagar haveli", "dd": "daman and diu", "ld": "lakshadweep",
}
IN_STATE_VARIANTS = {
    "orissa": "od", "keralam": "kl", "uttaranchal": "uk", "pondicherry": "py", "tamilnadu": "tn",
    "telengana": "tg", "chattisgarh": "cg", "chhatisgarh": "cg", "chhattisgarh": "cg",
    "ts": "tg", "or": "od", "ct": "cg", "ut": "uk", "jammu kashmir": "jk", "jammu & kashmir": "jk",
    "new delhi": None,  # a city, not a state
    "nct of delhi": "dl", "national capital territory of delhi": "dl",
    "andhrapradesh": "ap", "uttarpradesh": "up", "madhyapradesh": "mp", "westbengal": "wb",
    "paschimbang": "wb", "bangla": "wb", "gujrat": "gj", "karnatak": "ka", "maharastra": "mh",
    "odisa": "od", "panjab": "pb",
}

# France: regions and the departments that belong to them.  Department names
# are mapped onto their region so that "Bordeaux, Gironde" == "Bordeaux,
# Nouvelle-Aquitaine".
FR_REGIONS = {
    "hauts de france": ["nord", "pas de calais", "somme", "oise", "aisne"],
    "nouvelle aquitaine": ["gironde", "landes", "pyrenees atlantiques", "dordogne",
                           "lot et garonne", "charente", "charente maritime", "vienne",
                           "haute vienne", "deux sevres", "correze", "creuse"],
    "pays de la loire": ["loire atlantique", "maine et loire", "mayenne", "sarthe", "vendee"],
    "ile de france": ["paris", "seine et marne", "yvelines", "essonne", "hauts de seine",
                      "seine saint denis", "val de marne", "val d oise"],
    "bretagne": ["finistere", "morbihan", "ille et vilaine", "cotes d armor"],
    "normandie": ["calvados", "manche", "orne", "eure", "seine maritime"],
    "grand est": ["bas rhin", "haut rhin", "moselle", "meurthe et moselle", "marne", "aube",
                  "ardennes", "vosges", "meuse", "haute marne"],
    "occitanie": ["haute garonne", "herault", "gard", "aude", "pyrenees orientales", "tarn",
                  "tarn et garonne", "gers", "lot", "aveyron", "lozere", "ariege",
                  "hautes pyrenees"],
    "auvergne rhone alpes": ["rhone", "isere", "loire", "ain", "savoie", "haute savoie",
                             "drome", "ardeche", "puy de dome", "allier", "cantal",
                             "haute loire"],
    "provence alpes cote d azur": ["bouches du rhone", "var", "alpes maritimes", "vaucluse",
                                   "alpes de haute provence", "hautes alpes"],
    "bourgogne franche comte": ["cote d or", "doubs", "saone et loire", "yonne", "nievre",
                                "jura", "haute saone", "territoire de belfort"],
    "centre val de loire": ["loiret", "indre et loire", "cher", "eure et loir", "indre",
                            "loir et cher"],
    "corse": ["corse du sud", "haute corse"],
}


def build_state_lookup():
    """Map (normalised) state strings -> canonical key, per country label."""
    us = {}
    for code, name in US_STATES.items():
        us[code] = "us_" + code
        us[name] = "us_" + code
    ind = {}
    for code, name in IN_STATES.items():
        ind[code] = "in_" + code
        ind[name] = "in_" + code
        ind[name.replace(" ", "")] = "in_" + code
    for v, code in IN_STATE_VARIANTS.items():
        if code is not None:
            ind[v] = "in_" + code
    fr = {}
    for region, depts in FR_REGIONS.items():
        key = "fr_" + region.replace(" ", "_")
        fr[region] = key
        for d in depts:
            fr[d] = key
    return {"us": us, "india": ind, "france": fr}


STATE_LOOKUP = build_state_lookup()
