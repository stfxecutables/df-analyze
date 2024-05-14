from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import os
import re
import sys
import traceback
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
    no_type_check,
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
from joblib import Memory, Parallel, delayed
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import ndarray
from pandas import DataFrame, Series
from rapidfuzz.distance.Levenshtein import distance as lev_dist
from rapidfuzz.process import cdist
from tqdm import tqdm
from typing_extensions import Literal

SOURCE = ROOT / "traffic_violations_complete.csv"
DATA_OUT = ROOT / "traffic_data"
DATA_OUT.mkdir(exist_ok=True)
MIN_CLEAN = DATA_OUT / "min_cleaned.parquet"
MEMOIZER = Memory(location=ROOT / "__JOBLIB_CACHE__")

DESC_OUT = DATA_OUT / "unique_descriptions.txt"

YN_BINS = [
    "belts",  # [No, Yes]
    "pers_injury",  # [No, Yes]
    "prop_dmg",  # [No, Yes]
    "fatal",  # [No, Yes]
    "comm_license",  # [No, Yes]
    "hazmat",  # [No, Yes]
    "comm_vehicle",  # [No, Yes]
    "accident",  # [No, Yes]
    "alcohol",  # [No, Yes]
    "work_zone",  # [No, Yes]
]
TF_BINS = [
    "acc_blame",  # False, True
]

COLS = {
    "date_of_stop": "ord",
    "time_of_stop": "ord",
    "agency": "const",
    "subagency": "cat",
    "description": "text",  # 16 660 unique entries
    "location": "ord",
    "latitude": "ord",
    "longitude": "ord",
    "accident": "bin",
    "belts": "bin",
    "personal_injury": "bin",
    "property_damage": "ord",
    "fatal": "ord",
    "commercial_license": "ord",
    "hazmat": "ord",
    "commercial_vehicle": "ord",
    "alcohol": "ord",
    "work_zone": "ord",
    "state": "ord",
    "vehicletype": "ord",
    "year": "ord",
    "make": "ord",
    "model": "ord",
    "color": "ord",
    "violation_type": "ord",
    "charge": "ord",
    "article": "ord",
    "contributed_to_accident": "ord",
    "race": "ord",
    "gender": "ord",
    "driver_city": "ord",
    "driver_state": "ord",
    "dl_state": "ord",
    "arrest_type": "ord",
    "geolocation": "ord",
}
DROPS = [
    "time_of_stop",
    "date_of_stop",
    "agency",
    # could be lemma-tized and clustered, but also this gives away the
    # charge, so maybe should just not be used fully
    "description",
    # Most common location is only 2341 times, so this is too undersampled
    "location",
    # Duplicates lat/long features
    "geolocation",
]
NaN = float("nan")

STATE_COORDS = {  # obtained through geopy via Nominatim, using query "{XX}, USA"
    "AK": (64.4459613, -149.680909),  # AK: Alaska
    "KS": (38.27312, -98.5821872),  # KS: Kansas
    "VT": (44.5990718, -72.5002608),  # VT: Vermont
    "OR": (43.9792797, -120.737257),  # OR: Oregon
    "IA": (41.9216734, -93.3122705),  # IA: Iowa
    "UT": (39.4225192, -111.714358),  # UT: Utah
    "NV": (39.5158825, -116.853722),  # NV: Nevada
    "NH": (43.4849133, -71.6553992),  # NH: New Hampshire
    "NM": (34.5802074, -105.996047),  # NM: New Mexico
    "AR": (35.2048883, -92.4479108),  # AR: Arkansas
    "RI": (41.7962409, -71.5992372),  # RI: Rhode Island
    "MN": (45.9896587, -94.6113288),  # MN: Minnesota
    "ME": (45.709097, -68.8590201),  # ME: Maine
    "WI": (44.4308975, -89.6884637),  # WI: Wisconsin
    "LA": (34.0536909, -118.242766),  # LA: Los Angeles
    "MS": (32.9715285, -89.7348497),  # MS: Mississippi
    "CO": (38.7251776, -105.607716),  # CO: Colorado
    "WA": (47.2868352, -120.212613),  # WA: Washington
    "MO": (38.7604815, -92.5617875),  # MO: Missouri
    "OK": (34.9550817, -97.2684063),  # OK: Oklahoma
    "KY": (37.5726028, -85.1551411),  # KY: Kentucky
    # "ND": (NaN, NaN),  # ND: Unknown
    "AL": (33.2588817, -86.8295337),  # AL: Alabama
    "CT": (41.6500201, -72.7342163),  # CT: Connecticut
    "IN": (40.3270127, -86.1746933),  # IN: Indiana
    "AZ": (34.395342, -111.763275),  # AZ: Arizona
    "MI": (43.6211955, -84.6824346),  # MI: Michigan
    "TN": (35.7730076, -86.2820081),  # TN: Tennessee
    "SC": (33.6874388, -80.4363743),  # SC: South Carolina
    "IL": (40.0796606, -89.4337288),  # IL: Illinois
    "MA": (42.3788774, -72.032366),  # MA: Massachusetts
    "OH": (40.2253569, -82.6881395),  # OH: Ohio
    "CA": (36.7014631, -118.755997),  # CA: California
    "DE": (38.6920451, -75.4013315),  # DE: Delaware
    "GA": (32.3293809, -83.1137366),  # GA: Georgia
    "NJ": (40.0757384, -74.4041622),  # NJ: New Jersey
    "NY": (43.1561681, -75.8449946),  # NY: New York
    "NC": (35.6729639, -79.0392919),  # NC: North Carolina
    "WV": (38.4758406, -80.8408415),  # WV: West Virginia
    "FL": (27.7567667, -81.4639835),  # FL: Florida
    "TX": (31.2638905, -98.5456116),  # TX: Texas
    "PA": (40.9699889, -77.7278831),  # PA: Pennsylvania
    # "XX": (NaN, NaN),  # XX: Unknown
    "DC": (39.7940527, -100.472769),  # DC: Decatur County, Kansas
    "VA": (37.1232245, -78.4927721),  # VA: Virginia
    "MD": (39.5162401, -76.9382069),  # MD: Maryland
}


def renamer(s: str) -> str:
    s = str(s).lower().replace(" ", "_")
    remaps = {
        "personal_injury": "pers_injury",
        "property_damage": "prop_dmg",
        "commercial_license": "comm_license",
        "commercial_vehicle": "comm_vehicle",
        "contributed_to_accident": "acc_blame",
        "year": "vehicle_year",
        "make": "vehicle_make",
        "model": "vehicle_model",
        "color": "vehicle_color",
    }
    if s in remaps:
        return remaps[s]
    return s


def compute_distance(state: str, lat: float, long: float) -> float:
    if state not in STATE_COORDS or state == "XX":
        return NaN
    state_point = STATE_COORDS[state]
    md_point = (lat, long)
    return geodesic(md_point, state_point).km


# @MEMOIZER.cache()
def states_to_distances(df: DataFrame) -> DataFrame:
    df = df.copy()
    df["latitude"]
    # (0, 0) is used to represent unknown, XX is unknown state, so we need to axe these
    idx_nan = (df.latitude.abs() < 0.1) | (df.longitude.abs() < 0.1)
    df.loc[idx_nan, "latitude"] = NaN
    df.loc[idx_nan, "longitude"] = NaN
    df.loc[idx_nan, "state"] = "XX"
    states, lats, longs = df["state"], df["latitude"], df["longitude"]

    kms = Parallel(n_jobs=-1)(
        delayed(compute_distance)(state, lat, long)
        for (state, lat, long) in tqdm(
            zip(states, lats, longs), total=len(states), desc="Computing geodesics"
        )
    )

    df["state_km"] = kms
    return df


def rename_makes_1(s: str) -> str:
    # These are the common makes, and their most common miss-spellings
    # fmt: off
    spellings = {
        "acura": ("acur",),
        "buick": ("buic",),
        "cadillac": ("cad", "cadi", "cadilac"),
        "chevy": ("chev", "cheverolet", "chevorlet", "chevrolet"),
        "chrysler": ("chry", "chrys", "chrystler", "crysler", "cry"),
        "dodge": ("dodg", "ram"),
        "freightliner": ("frht",),
        "honda": ("hinda", "hino", "hond"),
        "hummer": ("humm", "hyun", "hyund", "hyunda", "hyundai", "hyundi", "hyundia"),
        "infinity": ("inf", "infi", "infin", "infiniti"),
        "international": ("intl",),
        "isuzu": ("isu", "isuz"),
        "jaguar": ("jag", "jagu"),
        "kawasaki": ("kawk",),
        "kenworth": ("kw",),
        "landrover": ("land rover", "lndr"),
        "lexus": ("lex", "lexs", "lexu"),
        "lincoln": ("linc",),
        "mazda": ("maz", "mazada", "mazd"),
        "mercury": ("merc",),
        "mercedes": ("mercedes benz", "mercedes-benz", "mercedez", "merz", "merz benz", "benz",),
        "mini": ("mini cooper", "mnni"),
        "mitsubishi": ("mits", "mitsu", "mitz"),
        "nissan": ("niss", "nissa", "nissian"),
        "none": ("unknown",),
        "oldsmobile": ("olds",),
        "peterbilt": ("pete", "peterbuilt", "ptrb"),
        "plymouth": ("plym",),
        "pontiac": ("pont",),
        "porsche": ("pors",),
        "range rover": ("rang",),
        "saab": ("saa",),
        "saturn": ("satr", "satu", "strn"),
        "scion": ("scio",),
        "subaru": ("sub", "suba", "subu", "suburu"),
        "suzuki": ("suzi", "suzu"),
        "taotao": ("tao tao",),
        "tesla": ("tesl",),
        "toyota": ("toty", "toy", "toyo", "toyot", "toyt", "toyta", "toytoa"),
        "volkswagen": ("volk", "volks", "volkswagon", "vw"),
        "volvo": ("volv",),
        "yamaha": ("yama",),
    }
    # fmt: on
    for correct, misspellings in spellings.items():
        if s in misspellings:
            return correct
    return s


def rename_makes_2(s: str) -> str:
    # These are the common makes after the first renaming, and their most common miss-spellings
    # fmt: off
    spellings = {
        "acura": ("accura", "acrua"),
        "alfa": ("alfa romeo"),
        "bmw": ("bwm",),
        "cadillac": ("caddilac", ),
        "chevy": ("cehvy", "cheve", "cheverlot", "chevey", ),
        "chrysler": ("chrylser", "chyrsler", "chysler", "crys", "crystler",),
        "ducati": ("duca",),
        "ford": ("for", ),
        "freightliner": ("frei", "freight", ),
        "gillig": ("gill", ),
        "harley": ("harley davidson", "hd",),
        "honda": ("hnda", "hoda", "hon", "hona", ),
        "hyundai": ("hundai", "huyn", "huyndai", "hyandai", "hyn", "hynd", "hyndai", "hyu", "hyudai", "hyunday",),
        "infinity": ("infini", "infinit", "infinite", "infinti", "inifiniti",),
        "international": ("int", "inte", ),
        "izuzu": ("isuzu", ),
        "kenworth": ("kenw", ),
        "landrover": ("land", ),
        "mazda": ("madza",),
        "maserati":  ("mase", "mast",),
        "mercury": ("mer", ),
        "mercedes": ("merzedes", "mb", "merc benz", "mercades", "mercedez benz",),
        "mitsubishi": ("mistubishi", "mit", "mitsibishi", "mitsubushi",),
        "nissan": ("nisan", ),
        "porsche": ("porche", "porshe",),
        "rangerover": ("range", "range rover", "rov",),
        "saturn": ("sat", ),
        "smart": ("smrt",),
        "subaru": ("subura",),
        "suzuki": ("suz", ),
        "toyota": ("totota", "toyoa", "toyoya", "toyoyta", "toytota", "tyota",),
        "unknown": ("unk", ),
        "volkswagen": ("volks wagon", "volkswaggon", "volkwagen", "volkwagon", "volswagon", "voltswagon",),
    }
    # fmt: on
    for correct, misspellings in spellings.items():
        if s in misspellings:
            return correct
    return s


def rename_makes_3(s: str) -> str:
    # fmt: off
    spellings = {
        "4s": ["4d", "4dr", "4x4"],
        "acura": ["a ura", "aacura", "accur", "acira", "aciura", "acora", "acra", "acru", "acruaa",
            "acrura", "acu", "acua", "acuar", "acuara", "acuea", "acufra", "acuira", "acur g",
            "acura", "acuras", "acurax", "acurra", "acurs", "acuru", "acurua", "acurva", "acurval",
            "acuta", "acutra", "acuura", "acyr", "acyra", "acyua", "aqcura", "arcura", "aruca",
            "aucra", "aur", "aura", "aurca", "aurra", "avcura", "avora", "avur", "avura", "axura",
        ],
        "alfa": ["alpha"],
        "audi": ["adui", "aidi", "au", "aud", "aud1", "auddi", "audi", "audie", "audii", "audio",
            "audiq", "audo", "audu", "audui", "audy", "aufi", "augi", "aui", "auid", "auidi", "ausi",
        ],
        "bentley": ["bart", "bartley", "bear", "beck", "belm", "bend", "bent", "bentley", "bently",
            "bently", "bett", "bigt", "bnw", "bnz", "bolt", "bona", "bwn",
        ],
        "bmw": ["b,w", "b0w", "big", "bm", "bm,w", "bma", "bmc", "bmd", "bme", "bmew", "bmi", "bmq",
            "bms", "bmv", "bmw", "bmw2s", "bmw4", "bmw`", "bmw`1", "bmws", "bmwv", "bmwx1", "bmx",
            "bnmw", "bnw", "bomw", "box", "bri", "brma", "brmr", "brp", "bu", "bui", "bus", "buw",
            "bw", "bwi", "bwn", "bws", "bww",
        ],
        "buick": ["bbuick", "beck", "big", "biick", "biuck", "biuick", "bruik", "bu", "bucik",
            "buck", "bui ck", "bui", "buic2", "buicj", "buick \\", "buick`", "buicl", "buiick",
            "buiik", "buik", "buix", "buuick",
        ],
        "cadillac": ["caadillac", "cadailac", "cadaillic", "cadalac", "cadallac", "cadalliac",
            "cadallic", "caddiliac", "caddillac", "cadi;ac", "cadiac", "cadiallac", "cadiallic",
            "cadiilac", "cadiillac", "cadikkac", "cadilaac", "cadilack", "cadiliac", "cadilic",
            "cadilkac", "cadilla", "cadillac", "cadillace", "cadillacq", "cadillad", "cadillaic",
            "cadillas", "cadillav", "cadilliac", "cadillic", "cadilllac", "cadllac", "cadullac",
            "cafillac", "caidilac", "caidllac", "caillac", "cdillac",
        ],
        "chevy": [
            "c hev", "cady", "camy", "cary", "cchevrolet", "cchevy", "cehev", "cehevrolet", "cehevy",
            "cehv", "cehvrolet", "cev", "ceverolet", "cevrolet", "cevy", "cgev", "cgevorlet",
            "cgevy", "ch", "chan", "chavy", "chchevy", "chdvy", "che v", "che vrolet", "che vy",
            "che", "cheb", "chebrolet", "cheby", "chec", "check", "checorolet", "checrolet", "checvy",
            "checy", "chedvy", "cheevrolet", "cheevy", "cheey", "chek", "chen", "cheny", "cher",
            "cherolet", "cherovet", "cherovlet", "chervolet", "chervrolet", "chery", "chev i",
            "chev.", "chev1", "chev2d", "chevc", "cheverlet", "cheverloet", "chevf", "chevl",
            "chevolet", "chevolete", "chevollet", "chevolret", "chevorelet", "chevorlete", "chevorlett",
            "chevorley", "chevorolet", "chevq", "chevr", "chevrelet", "chevreloet", "chevrelot",
            "chevrilet", "chevriolet", "chevrlet", "chevrlete", "chevrlette", "chevrlo", "chevrloet",
            "chevrlot", "chevrlote", "chevro;et", "chevro", "chevroelet", "chevroelt", "chevroet",
            "chevrole", "chevrole6t", "chevroleet", "chevrolegt", "chevroler", "chevrolert",
            "chevrolet`", "chevrolete", "chevrolette", "chevrolety", "chevroley", "chevroliet",
            "chevrollet", "chevrollete", "chevroloet", "chevrolret", "chevrolrt", "chevrolt",
            "chevrolte", "chevroltet", "chevroltt", "chevrolwt", "chevrooet", "chevroolet",
            "chevrotet", "chevrp;et", "chevrplet", "chevry", "chevt", "chevtolet", "chevu",
            "chevvrolet", "chevvy", "chevy c", "chevy i", "chevy p", "chevy", "chevy0", "chevyc",
            "chevyq", "chevyrolet", "chevyt", "chevyy", "chey", "cheyv", "cheyy", "chez",
            "chhevrolet", "chhevy", "chivy", "chmm", "chon", "chr", "chrevrolet", "chrey", "chrl",
            "chrs", "chrsy", "chrv", "chrverolet", "chrvrolet", "chrvy", "chry.", "chrya", "chryl",
            "chty", "chtys", "chv", "chve", "chveolet", "chverolet", "chvey", "chvorlet",
            "chvrlolet", "chvrolet", "chvy", "chvye", "chwvy", "chy", "chyl", "chyr", "chys", "civc",
            "cjev", "cjevrolet", "cjevy", "cjhevy", "cjry", "cnev", "cnry", "cuevy", "cvehrolet",
            "cvev", "cvevrolet",
        ],
        "chrysler": [
            "chr", "chrl", "chrs", "chry.", "chrya", "chryl", "chy", "chyl", "chyr", "chyrl", "chyrs",
            "chys", "chyst", "crhy", "cry", "cryh", "cryl", "cryn", "cgrysler", "cheysler", "chhrysler",
            "chreysler", "chrisl", "chrisler", "chrisler", "christler", "chrrysler", "chrsler",
            "chrslyer", "chrsyler", "chrsysler", "chrtsler", "chrusler", "chrustler", "chryler",
            "chryseler", "chryselr", "chryser", "chrysker", "chrysl", "chryslar", "chrysleer",
            "chrysler", "chrysler\\", "chryslerq", "chryslert", "chryslet", "chryslewr", "chrysley",
            "chrysller", "chryslr", "chryslter", "chryslyer", "chryster", "chrystlar", "chrystle",
            "chrystlet", "chrysyler", "chryxler", "chryysler", "chtysler", "chyrler", "chyrstler",
            "chyrysler", "chyslyer", "chystler", "crhrsler", "crhsyler", "crhysler", "crrysler",
            "cryser", "cryslr", "cyrsler", "cysler",
        ],
        "cooper": ["coop", "cooper", "copper"],
        "daewoo": ["daew", "daewo", "daewoo", "daewood"],
        "dodge": [
            "d0dge", "dadg", "dadge", "dddge", "ddge", "ddodge", "ddoge", "didg", "didge", "diodge",
            "dod", "dodb", "dodde", "doddge", "dode", "dodeg", "dodege", "dodfe", "dodga", "dodgd",
            "dodge", "dodgec", "dodgee", "dodgeg", "dodgei", "dodgeq", "dodger", "dodgge", "dodgr",
            "dodgw", "dodhe", "dodke", "dododge", "dodoge", "doege", "dofge", "dog", "dogd", "dogde",
            "dogdge", "doge", "dogg", "dogge", "donde", "donf", "dong", "doodge", "dord", "dors",
            "dosg", "dosge", "dudge",
        ],
        "ducati": ["ducadi", "ducat", "ducati", "ducato", "ducatti", "ducoti", "dukati"],
        "eagle": ["eagl", "eagle"],
        "ferrari": ["ferarri", "ferrari", "ferreri"],
        "fiat": ["fia", "fiar", "fiat", "fiiatt", "flat"],
        "ford": [
            "f0rd", "fd", "fford", "fiord", "fird", "flrd", "fod", "fodd", "fode", "fodr", "foed",
            "foedq", "foird", "food", "foord", "forc", "ford .", "ford t", "ford", "ford`", "ford1",
            "ford5", "fordcn", "fordd", "forde", "fordf", "fordm", "fordq", "fore", "fored", "forf",
            "forg", "forj", "fork", "form", "forn", "forod", "fors", "forsd", "fotd", "fprd", "frd",
            "frg", "frh", "frod", "frord", "frrw", "frt", "ftr", "ftwd",
        ],
        "freightliner": [
            "feightliner", "fraight liner", "freighliner", "freight liner", "freightiner",
            "freightlier", "freightline", "freightlineer", "freightliner", "freightlinrt",
            "freightlinter", "freightlnr", "freigtliner", "freithliner", "frghtliner",
            "frieghtliner", "friehtliner", "frightliner", ],
        "genesis": [
            "genesis", "genisis", "gensis", "gen", "gena", "gene", "gens", "genu",
        ],
        "geo": [
            "geel", "geio", "geo", "gi", "gil", "gio", "goe",
        ],
        "gillig": [
            "giilig", "gilg", "gili", "gilig", "gillag", "gilleg", "gillian", "gillig", "gillis",
        ],
        "gmc": [
            "gc", "gm", "gma", "gmc", "gmc s", "gmc?", "gmcc", "gmcv", "gmd", "gmec", "gmf", "gmg",
            "gms", "gmt", "gmv", "gmx", "gmz", "gnc", "gnmc",
        ],
        "harley": [
            "haeley", "hand", "hari", "harl", "harl", "harley d", "harley", "harley", "haul",
            "hawk", "hcry", "heil",
        ],
        "honda": [
            "h d", "h-d", "h0nd", "h0nda", "h6nda", "hand", "handa", "hando", "hhond", "hhonda",
            "hind", "hindo", "hiond", "hionda", "hiunda", "hiundai", "hiunday", "hiyunda", "hlms",
            "hmc", "hmd", "hmde", "hmw", "hnd", "hnhonda", "hnoda", "ho nda", "ho0nda",
            "hoada", "hobda", "hobnda", "hocnda", "hod", "hodd", "hodda", "hodna", "hoha",
            "hohd", "hohda", "hoidna", "hoind", "hoinda", "holda", "holm", "holnda", "homd",
            "homda", "home", "homnda", "homs", "hon da", "honad", "honada", "honca", "hond`",
            "hond4d", "hond4s", "honda b", "honda", "honda]", "honda`", "honda1", "honda2d", "honda4",
            "honda4d", "hondaa", "hondac", "hondad", "hondai", "hondaq", "hondas", "honday", "hondd",
            "hondda", "hondfa", "hondm", "hondq", "honds", "hondsa", "hondtk", "hondva", "hondval",
            "hondy", "honf", "honfa", "hong", "honga", "honhda", "honida", "honnd", "honnda",
            "honoda", "hons", "honsa", "hontd", "hoond", "hoonda", "hooonda", "hpnda", "hum",
            "hund", "hunda", "hunday", "hundi", "hundy", "hunyda", "huyna", "huynda", "hyd",
            "hyinda", "hynda", "hynday", "hyndi", "hyndia", "hyud", "hyuda", "hyuna",
        ],
        "hyundai": [
            "h0nda", "h6nda", "handa", "hayundai", "hayundi", "hhyunda", "hhyundai", "hionda",
            "hiudai", "hiuday", "hiunda", "hiundai", "hiunday", "hiyunda", "honda", "honda]",
            "honda`", "honda1", "honda4", "hondaa", "hondac", "hondad", "hondai", "hondaq",
            "hondas", "honday", "hondy", "hoyundai", "hpnda", "htudai", "htundai", "huand",
            "huandai", "hudayi", "huinday", "hund", "hunda", "hunday", "hundayi", "hundi",
            "hundy", "hundyai", "hundyi", "huni", "huny", "hunyda", "huudai", "huundai",
            "huyandai", "huyandi", "huydai", "huynda", "huynday", "huyndi", "huyunda", "huyundai",
            "huyundi", "hyandau", "hyandi", "hyandia", "hyanduai", "hyandui", "hyaundai", "hyaundi",
            "hydai", "hydunai", "hydundai", "hyhundai", "hyidai", "hyinda", "hyindai", "hyiundai",
            "hynda", "hyndaui", "hynday", "hyndi", "hynduai", "hyni", "hynndai", "hynudai",
            "hynundai", "hyu ndai", "hyu ndia", "hyua", "hyuadai", "hyuan", "hyuand1", "hyuanda",
            "hyuandai", "hyuandi", "hyub=ndai", "hyubdai", "hyubndai", "hyuda", "hyudao", "hyuday",
            "hyuddai", "hyudnai", "hyuindai", "hyumdai", "hyun dai", "hyuna", "hyunadai", "hyunadi",
            "hyunai", "hyunandi", "hyund.", "hyund1", "hyunda5", "hyundah", "hyundai", "hyundai/",
            "hyundai`", "hyundaia", "hyundaie", "hyundaii", "hyundain", "hyundaiq", "hyundair", "hyundaiw",
            "hyundao", "hyundaoi", "hyundap", "hyundaqi", "hyundau", "hyundaui", "hyunddai", "hyunddia",
            "hyunde", "hyundhi", "hyundoia", "hyundri", "hyundui", "hyundy", "hyunfai", "hyuni",
            "hyunndai", "hyunsai", "hyunudai", "hyunundai", "hyunva", "hyusai", "hyuundai",
            "hyyndai", "hyyunda", "hyyundai",
        ],
        "hudson": ["hdsn", "huds", "hudson", "husdon"],
        "hummer": ["humer", "hummer"],
        "infinity": [
            "i finiti", "ifini", "ifiniti", "ifinity", "iinfiniti", "indi", "inff",
            "infi nti", "infii", "infiiniti", "infiinti", "infiit", "infiiti", "infiity",
            "infim", "infimiti", "infinidi", "infiniete", "infinifty", "infinii", "infiniiti",
            "infininti", "infininty", "infinit1", "infinita", "infinitie", "infinitif", "infinitii",
            "infinitit", "infinitt", "infinitti", "infinitu", "infinity", "infinityi", "infinityq",
            "infinityy", "infiniy", "infiniyi", "infinnity", "infinte", "infintit", "infinty",
            "infit", "infiti", "infititi", "infitnite", "infity", "infn", "infni",
            "infniity", "infniti", "infnity", "infnti", "infoniti", "infonity", "inft",
            "inginiti", "inginity", "ini", "inif", "inifi", "inifinit", "inifinity",
            "inifinti", "inifinty", "inifiti", "inifity", "inifnity", "ininiti", "ininity",
            "init", "inti", "intiniti", "inviniti",
        ],
        "international": [
            "int'l", "intel", "intenational", "inter", "intern", "interna", "internati0nal",
            "internatinal", "internatioal", "internation", "internationa", "international d",
            "international", "internationl", "internl", "interntional", "intertational", "inti",
            "intn", "intnl", "into", "intr",
        ],
        "izuzu": [
            "isizu", "issuzu", "isszu", "isuku", "isusu", "isuza", "isuze", "isuzi", "isuzo",
            "isuzue", "isuzui", "iszu", "iszuu", "iszuzu", "iusuzu", "iuzu", "izu", "izus",
            "izusu", "izuz", "izuzi", "izuzu",
        ],
        "jaguar": [
            "jaduar", "jagjuar", "jagr", "jagua", "jaguar", "jaguuar", "jahuar", "jajuar",
            "janguar", "januar", "jaquar", "jargua", "jauar", "jauguar", "jugar", "juguar",
        ],
        "jeep": [
            "jee", "jeed", "jeedp", "jeeep", "jeef", "jeeg", "jeek", "jeem", "jeeo", "jeep",
            "jeep l", "jeep1", "jeepd", "jeepp", "jeepq", "jeer", "jees", "jeesp", "jeet", "jeff",
            "jep", "jepp", "jevw", "jjeep", "joop", "jp", "jrrp", "jwwp",
        ],
        "kawasaki": [
            "kaasaki", "kawasacki", "kawasake", "kawasaki", "kawasawki", "kawaski", "kawkasaki",
            "kawkaski", "kawsaki", "kowasaki", "kwasaki",
        ],
        "kenworth": [
            "ken worth", "kenalworth", "kenilworth", "keniwerth", "keniworth", "kenoworth",
            "kentworth", "kenworh", "kenwork", "kenwort", "kenworth", "kenworthy", "kenwoth",
            "keworth", "kwenworth",
        ],
        "kia": [
            "ka", "kai", "kais", "kait", "kaka", "kara", "ken", "ki", "kia", "kia4", "kiav", "kida",
            "kiia", "kinc", "kino", "kio", "kis", "kisa", "kiw",
        ],
        "kymco": ["kmco", "kyco", "kymc", "kymco", "kymoco"],
        "landrover": [
            "la nd rover", "labd rover", "lad rover", "lamd rover", "lan rover", "land  rover",
            "land eover", "land over", "land raover", "land roaver", "land roer", "land rove",
            "land rovr", "land rovwr", "land rver", "land-rover", "landdrover", "landover",
            "landr rover", "landroer", "landrov", "landrove", "landrover", "landrvr", "lanrover",
            "lmand rover", "lnd rover", "lndrover",
        ],
        "lexis": [
            "lecxus", "ledxus", "leexus", "leik", "lemus", "les", "lesus", "lesux", "lesxus",
            "leus", "leux", "levu", "levus", "lexas", "lexcus", "lexes", "lexi", "lexis", "lexiu",
            "lexius", "lexo", "lexstk", "lexsus", "lexua", "lexuas", "lexuc", "lexucs", "lexud",
            "lexue", "lexues", "lexus", "lexus`", "lexus4d", "lexusq", "lexuss", "lexustr",
            "lexuus", "lexux", "lexuxs", "lexuxus", "lexuz", "lexuzs", "lexxs", "lexxus", "lexy",
            "lexys", "lexz", "lezus", "llexus", "lotus", "lrxus", "luxu", "luxus", "lxs", "lxus",
        ],
        "lincoln": [
            "licn", "licolin", "licoln", "liincoln", "linciln", "lincl", "lincln", "linclon",
            "linco", "lincokn", "lincol", "lincolb", "lincolcn", "lincolin", "lincolm", "lincoln",
            "lincolnn", "lincon", "linconl", "linconln", "linlcon", "linncoln", "linvoln", "loncoln",
        ],
        "mack": [
            "mac", "macda", "mach", "mack", "macl",
        ],
        "maserati": [
            "masarati", "masaratti", "maseati", "maseradi", "maseraii", "maserat", "maserati",
            "maseratti", "maseriti", "maseritti", "masersti", "maserti", "masserati", "mazeradi",
            "mazeratti",
        ],
        "mazda": [
            "maazda", "mac", "macda", "mach", "mack", "macl", "mad", "mada", "madaza", "madd",
            "madz", "madzda", "mail", "make", "marc", "marm", "marz", "mas", "masd", "masda",
            "masdia", "masr", "masz", "maszda", "matl", "matr", "mavda", "mawo", "maxd", "maxda",
            "mayb", "maz4s", "maza", "mazad", "mazc", "mazd3", "mazda 3", "mazda 6", "mazda", "mazda`",
            "mazda3", "mazda4s", "mazda5", "mazda6", "mazdad", "mazdai", "mazddda", "mazds",
            "mazdva", "mazdz", "mazfa", "mazra", "mazs", "mazsa", "mazzda", "mcla",
            "mera", "mez", "mezda", "miazda", "misa", "mita", "mizz", "monda",
            "mozda", "mrz", "mrzd", "mtz", "mz", "mzad", "mzada", "mzd", "mzda",
        ],
        "mercury": [
            "mecrury", "mecurt", "mecury", "mer ury", "merccury", "mercedy", "mercery",
            "merchury", "mercruy", "mercry", "mercu", "mercucy", "mercur", "mercuray",
            "mercurey", "mercuri", "mercurt", "mercury", "mercuty", "mercuy", "mercy",
            "mercyrt", "mercyry", "merury", "mrcury", "mrecury", "murcary", "murcry",
            "murcury",
        ],
        "mercedes": [
            "marcedes", "marcedez", "mecedes", "mecedez", "mecerdes", "meercedes", "merc edes",
            "merc edez", "mercadez", "mercdes", "mercdez", "merceades", "merced", "mercede",
            "mercede z", "mercedea", "mercedec", "merceded", "mercededs", "mercedees", "mercedes",
            "mercedese", "mercedex", "mercedezq", "mercedies", "mercedis", "merceds", "mercedses",
            "mercedy", "mercedz", "merceedes", "mercees", "mercendes", "mercerdes", "merces",
            "mercesdes", "mercesed", "mercidez", "mercsdez", "merczdez", "merdedes", "merecdes",
            "merecedes", "meredes", "mersede", "mersedes", "mersedez", "mervedes", "mervedez",
            "merzades", "merzcedes", "merzede", "merzeds", "mmercedes", "mrcedes", "mrcedez",
            "mrecedes", "mrercedes",
        ],
        "mifu": ["mi/f", "mi/fu", "mifu" ],
        "mini": [
            "mimi", "min", "minc", "mini c", "mini", "mini2d", "minia", "minii", "minin", "minji",
            "minni", "minnian", "mino", "misti", "miti", "mitsi", "mni",
        ],
        "mitsubishi": [
            "miits", "miitsubishi", "mis", "misbshi", "mishibishi", "miss", "missibishi",
            "missubishi", "mist", "misth", "misti", "mists", "mistsubishi", "mistu", "mistubuishi",
            "mistusubishi", "misubishi", "mita", "mitbubishi", "mitch", "miti", "mitibishi",
            "mitibshi", "mitis", "mitisbishi", "mitishbshi", "mitishibishi", "mitisibishi",
            "mitisubishi", "mitisubshi", "mitisubushi", "mits.", "mitsabushi", "mitsb", "mitsbishi",
            "mitsbishii", "mitsbuishi", "mitsbushi", "mitsh", "mitshbishi", "mitshibishi",
            "mitshibshi", "mitshubishi", "mitshubitshi", "mitsi", "mitsibish", "mitsibushi",
            "mitsiubishi", "mitso", "mitssubishi", "mitsubashi", "mitsubeshi", "mitsubhi",
            "mitsubi", "mitsubichi", "mitsubighi", "mitsubihi", "mitsubihshi", "mitsubiishi",
            "mitsubis", "mitsubisgu", "mitsubish", "mitsubishi", "mitsubishi`", "mitsubishie",
            "mitsubishii", "mitsubishit", "mitsubisho", "mitsubishu", "mitsubishui", "mitsubisi",
            "mitsubisih", "mitsubisihi", "mitsubisiu", "mitsubisshi", "mitsubisui", "mitsubitshi",
            "mitsubitu", "mitsuboshi", "mitsubshi", "mitsubuishi", "mitsubushu", "mitsuhishi",
            "mitsuibishi", "mitsuishi", "mitsunshi", "mitsushi", "mitsuubishi", "mitt", "mitts",
            "mittsubishi", "mitu", "mitubish", "mitubishi", "mitubishi`", "mitusbishi", "mitusbuishi",
            "mitushishi", "mitxs", "mitxubishi", "mitzibishi", "mitzibushi", "mitzl", "mitzu",
            "mitzubishi", "mizubishi", "motsubishi", "mtis", "mtsbshi", "mtsubishi",
        ],
        "mustang": ["mustang"],
        "ndmc": ["nc", "ndmc"],
        "new flyer": ["new fluer", "new flyer", "new glyer", "newflyer"],
        "nissan": [
            "n5ssan", "n9ssan", "niaan", "niaasan", "niassan", "niddan", "nidssan",
            "nii", "niis", "niisan", "niiss", "niissan", "niissian", "nimr",
            "ninsan", "ninssan", "niro", "nis", "nisaan", "nisano", "nisasan",
            "nisd", "nisian", "nisn", "niss.", "niss4", "nissa n", "nissaan",
            "nissab", "nissabn", "nissain", "nissak", "nissal", "nissam", "nissan",
            "nissan`", "nissana", "nissane", "nissang", "nissanm", "nissann", "nissano",
            "nissans", "nissas", "nissasn", "nissasnm", "nissav", "nissen", "nissi",
            "nissia", "nissiam", "nissin", "nissina", "nission", "nissn", "nissna",
            "nisson", "nisss", "nisssan", "nisssn", "nissvan", "nisu", "nits",
            "niu", "nixx", "nizzan", "nnisan", "nniss", "nnissan", "nnt",
            "noissan", "nosan", "nossan", "nss", "nssan", "nssian", "nsssan",
            "nuissan", "nus", "nussan",
        ],
        "oldsmobile": [
            "old mobile", "oldesmobile", "oldmobile", "olds mobile", "oldsmobil", "oldsmobile",
            "oldsmobile`", "oldsmoble", "oldsmoblie", "oldsmobole", "oldsmoile", "olsmobile",
        ],
        "orion": ["ori", "orino", "orio", "orion"],
        "peterbilt": [
            "peerbilt", "pererbuilt", "peter built", "peterbelt", "peterbilt", "peterblt",
            "peterbu;lt", "peterbulit", "peterbult", "piterbuilt", "pterbilt",
        ],
        "plymouth": [
            "plmonth", "plmouth", "plumouth", "plymith", "plymonth", "plymoth", "plymounth",
            "plymouth", "plymouyh", "plymuth", "pylmouth", "pymouth",
        ],
        "pontiac": [
            "p0ntiac", "pntiac", "pobtiac", "pointiac", "pomtiac", "ponatiac", "poniac",
            "ponic", "ponitac", "ponitiac", "pontac", "pontaic", "ponti", "pontia",
            "pontiac", "pontiac`", "pontiacc", "pontiace", "pontiacq", "pontian", "pontiav",
            "pontic", "pontica", "ponticas", "pontoac", "ponyiac", "poontiac", "popntiac",
            "potiac",
        ],
        "porsche": [
            "poesche", "porch", "porcha", "porchse", "porcsche", "porcse", "porcshe",
            "poreche", "porsc", "porsce", "porscge", "porsch", "porscha", "porsche",
            "porse", "porseche", "porsh", "porsh1", "porshce", "posch", "prosche",
            "prrsche", "prsche", "pursche",
        ],
        "prem": ["perm", "prei", "prem", "prim"],
        "rangerover": [
            "rage rover", "rainge rover", "rand rover", "randge rover", "ranfe rover", "rang rover",
            "range  rover", "range over", "range roger", "range rovery", "range rovr", "range rver",
            "ranger rover", "rangerover", "rangervr", "rangevrover", "rangr rover", "rangrover",
            "rng rover",
        ],
        "saab": [
            "saab", "saabb", "sab", "sabb", "sabu", "sag", "sarn", "satn", "saub", "sba", "smar",
            "spal", "ssa", "ssab", "sssb", "star", "sua", "surb",
        ],
        "saturn": [
            "sarn", "sarurn", "satarn", "satern", "satn", "satrn", "satrun", "satun", "satur",
            "saturan", "saturb", "saturen", "saturm", "saturn", "saturn`", "saturne", "saturni",
            "saturnq", "saturrn", "satutn", "saurn", "sautn", "sayurn", "staturn", "staurn",
            "sturn", "suturn",
        ],
        "scion": [
            "sarn", "satn", "sci", "sciaon", "scid", "scin", "scion", "scione", "scionia",
            "scionq", "scoin", "scon", "scwinn", "shor", "si", "sican", "sicion", "sico",
            "sicon", "sien", "sion", "sizi", "slin", "sol", "spor", "sun", "sxion", "syion",
            "sion",
        ],
        "scooter": ["scooter"],
        "smart": ["sarn", "sbaru", "smar", "smart", "smartc", "star", "suaru"],
        "sterling": [
            "steerling", "ster", "sterl", "sterlig", "sterling", "stirling", "stlg", "str",
            "strg", "sts",
        ],
        "subaru": [
            "sabaru", "saburu", "sba", "sbaru", "sibaru", "su baru", "sua", "suaru", "sub aru",
            "sub4dr", "subaa", "subaaru", "subabu", "subaca", "subaeu", "subai", "subar", "subara",
            "subarau", "subarbu", "subarh", "subari", "subariu", "subaro", "subarru",
            "subaru", "subarua", "subarue", "subarui", "subarus", "subarusw", "subaruu",
            "subary", "subaryu", "subau", "subaur", "subauru", "subbaru", "subburu",
            "subero", "suberu", "subi", "subie", "subr", "subra", "subraru",
            "subrau", "subru", "subrur", "subs", "subsru", "subuaru", "subur",
            "suburau", "suburi", "suda", "sudaru", "suna", "sunaru", "supr",
            "suraru", "surbaru", "susbaru", "susubaru",
        ],
        "susuki": [
            "sazuki", "sizuki", "suburi", "suki", "susk", "suski", "susu", "susuki",
            "suzik", "suziki", "suziuki", "suzk", "suzki", "suzuiki", "suzuk", "suzuki",
            "suzuki`", "suzukia", "suzukii", "suzukki", "suzuku", "suzuky", "suzuzi",
        ],
        "taizhou": ["taizhou", "taizou"],
        "taotao": [
            "taoato", "taot", "taota", "taotan", "taotao", "taotao50", "taotaro", "taoto", "tautau",
            "toota", "toto",
        ],
        "tesla": [
            "telsa", "temsa", "tes", "tesal", "tesca", "tesla", "tesla4", "teslda",
            "teslla", "tess", "tessla", "test", "testla", "texa", "tresla", "tsla", "tusla",
        ],
        "thomas": ["thms", "thoas", "thom", "thoma", "thomas", "thomos", "toma", "tomas"],
        "toyota": [
            "t0y0ta", "t0yot", "t0yota", "t6oyota", "ti=oyota", "tiyita", "tiyot", "tiyota",
            "tloyota", "toat", "toatoa", "toiyota", "toota", "tooyota", "tooyt", "tooyta",
            "tora", "torota", "tot", "toto", "totoa", "totora", "totot", "totoya",
            "totoyot", "totoyta", "tott", "totya", "totyoa", "totyota", "totyt", "totyta",
            "touota", "tout", "toy0a", "toy0ta", "toya", "toyata", "toyato", "toyiooa",
            "toyiota", "toyita", "toyo scion", "toyo0ta", "toyo4d", "toyoat", "toyoata", "toyoato",
            "toyoota", "toyopta", "toyora", "toyorta", "toyot a", "toyota (scion)",
            "toyota / scion", "toyota c", "toyota p", "toyota s", "toyota scion", "toyota-scion",
            "toyota", "toyota/scion", "toyota`", "toyota``", "toyota2", "toyota4s", "toyotaa",
            "toyotal", "toyotao", "toyotaq", "toyotas", "toyotat", "toyoto", "toyotoa", "toyotq",
            "toyotra", "toyots", "toyotsa", "toyotta", "toyoty", "toyotya", "toyoua", "toyouta",
            "toyova", "toyoval", "toyovan", "toyoy", "toyoyt", "toypta", "toyr", "toyt  scion",
            "toyt scion", "toyt]ota", "toyt`", "toyt=ota", "toyto", "toytq", "toyts", "toytta",
            "toyttk", "toyuota", "toyut", "toyuta", "toyyota", "tpoyota", "tpyota", "tpypta",
            "tpyta", "ttoy", "ttoyota", "tyot", "tyotao", "tyoyta", "tyta",
        ],
        "triumph": ["trimuph", "trium", "triump", "triumph"],
        "unknown": ["unknown", "unkown"],
        "vespa": ["versa", "vesp", "vespa"],
        "volkswagen": [
            "v0lkswagen", "valks", "vikswagen", "vilks", "vilkswagon", "vilkw", "violks",
            "vlks", "vlkswa", "vlkswagon", "vo;ks", "voiks", "voilks", "voks wagen",
            "voks", "vokswagen", "vokswagon", "voljs", "volk sw", "volk swagon", "volk w",
            "volk wagan", "volk wagon", "volk.", "volka", "volkawagen", "volkawagon", "volkcswagon",
            "volkd", "volke", "volkes", "volkeswagen", "volkeswagon", "volkks", "volkkswagon",
            "volks w", "volks wagan", "volks wagaon", "volks wagen", "volksagen", "volksagon", "volksewagon",
            "volksswagon", "volksvagen", "volksvagon", "volksw", "volkswa", "volkswaegn", "volkswaen",
            "volkswag", "volkswagan", "volkswagaon", "volkswage", "volkswageb", "volkswageg", "volkswagem",
            "volkswagen", "volkswagewn", "volkswaggo", "volkswagin", "volkswagkon", "volkswagn", "volkswago",
            "volkswagog", "volkswagom", "volkswagonq", "volkswagoon", "volkswagpn", "volkswagwen", "volkswahen",
            "volkswaon", "volkswaton", "volkswg", "volkswgan", "volkswgen", "volkswgn", "volkswgon",
            "volkw", "volkwage", "volkwaggen", "volkwasgen", "volkxs", "volkz", "volkzwagen",
            "vollkswagen", "vollkswagon", "vollswagon", "vols", "volsk", "volsks", "volskwagen",
            "volskwagon", "volsswagon", "volsw", "volswagen", "volswaggen", "volswago", "volts",
            "voltswagen", "voltwagon", "volvs", "volvswagen", "volvswagon", "volwagon", "vowlks",
            "vvolkswagen", "vvolkswagon", "vwolks",
        ],
        "volvo": [
            "v olvo", "v0lv", "vilv", "vilvo", "vllv", "vlovo", "vlv", "vlvo", "vol vo",
            "volco", "vollvo", "volo", "volov", "vols", "volv0", "volva", "volve", "volvi",
            "volvl", "volvo", "volvo`", "volvoe", "volvoo", "volvot", "volvs", "volz", "vov",
            "vovl", "vovlo", "vovlv", "vovlvo", "vovo", "vovol", "vvolvo",
        ],
        "west": ["weha", "well", "wesr", "west", "whit", "wort", "wstr"],
        "white": ["whi", "whit", "white"],
        "xx": ["x", "x5", "xb", "xx", "xxx", "xxxx"],
        "yamaha": ["yahaha", "yahama", "yahmaha", "yamah", "yamaha", "yamaya"],
    }
    # fmt: on
    for correct, misspellings in spellings.items():
        if s in misspellings:
            return correct
    return s


def rename_makes_4(s: str) -> str:
    # fmt: off
    spellings = {
        'accord': ['acco', 'accord'],
        'alfa': ['adva', 'aga', 'alfa'],
        'apollo': ['apol', 'apollo', 'appolo'],
        'aston martin': ['asto martin',
                        'aston martin',
                        'astro martin',
                        'austin martin'],
        'buick': ['black', 'bruick', 'buick'],
        'cadillac': ['cadillac', 'cadal', 'cadalic', 'cadli', 'catalac'],
        'camry': ['caddy', 'cam', 'cama', 'camaro', 'camery', 'camry', 'car',
                'carg', 'carm', 'carr', 'carry', 'catr', 'crhry', 'crry'],
        'civic': ['cidi', 'cimc', 'citi', 'civic'],
        'dodge': ['dodge ram', 'dodge van'],
        'dongfang': ['dong fang', 'dongfang'],
        'ferrari': [ 'ferr', 'ferrari'],
        'ford': ['ford', 'fort', 'forte'],
        'freightliner': ['freightliner' 'fhrt', 'fht', 'fhtl', 'fre', 'fret',
                'frf', 'frgh', 'frgt', 'frhy', 'frie', 'frli', 'frnt', 'fron',
                'frth', 'frtl', 'ftl' 'ftl', 'ftlr', 'ftlr'],
        'honda': ['ond', 'onda', 'ondah'],
        'kawasaki': ['kaw', 'kawa', 'kawa', 'kawak', 'kawas', 'kawc', 'kawi',
                'kazda', 'kwak'],
        'landrover': ['lanr', 'landr', 'landro', 'lanr', 'lanro', 'laro', 'lnd',
                'lnrr', 'lnrv'],
        'mercedes': ['mercedes', 'm benz', 'm-benz', 'm/benz', 'marcz', 'mbenz',
                'mebe', 'mebz', 'mec', 'mece', 'merb', 'merbenz', 'merc.',
                'mercb', 'mercben', 'mercbz', 'mercd', 'mercdz', 'merce',
                'merce', 'mercez', 'mercq', 'mercs', 'mercz', 'merd', 'merdz',
                'merec', 'merece', 'merez', 'merk', 'merq', 'merrz', 'mers',
                'mertz', 'merv', 'merx', 'merxz', 'merzb', 'merzd', 'merzds',
                'merze', 'merzs', 'merzz', 'merzzz', 'metz', 'mmerz', 'modz',
                'mrc', 'mrcb', 'mrcd', 'mrcy', 'mrez', 'mrrz', 'mrrz'],
        'mitsubishi': ['mitsubishi', 'mitsub', 'mitsuh.'],
        'moped': ['mobed', 'moped'],
        'nissan': ['nissan', 'missan'],
        'none': ['unknown'],
        'rav4': ['rav', 'rav 4', 'rav4', 'rave'],
        'shanghai': ['shanghai', 'shanhaie'],
        'sunny': ['sunl', 'sunny', 'surly'],
        'taotao': ['taotao', 'tayotayo' 'tao', 'taoi', 'tatt', 'toay'],
        'toyota': ['yoyota', 'yotoa', 'yotota', 'youota', 'yoyo', 'yoyota',
                    'yoyt', 'ytoyota'],
        'volkswagen': ['vlkwg', 'volk wag', 'volks wag', 'volkswagen',
                'volkval', 'volkwag', 'volkwgn', 'volwag', 'wilkswagon',
                'wolks wagon', 'wolkswagen', 'wolkswagon', 'wolkwagen',
                'wolskwagen', 'wolswagon', 'wvolkswagen'],
        'volvo': ['vol', 'volvo'],
        'western star': ['western sta', 'western star', 'west'],
        'yongfu': ['yonfu', 'yong', 'yongfu', 'youngfu'],
        'zhejiang': ['zhe jiang', 'zhejiang'],
        'zhejiang jiajue': ['zhejiang jiajue', 'zhej', 'zheng', 'zhil', 'zhng']
    }
    # fmt: on
    for correct, misspellings in spellings.items():
        if s in misspellings:
            return correct
    return s


def fix_makes(df: DataFrame) -> DataFrame:
    """Use Levenshtein distance for now to heuristically reduce the makes"""
    df = df.copy()
    makes = df["vehicle_make"].str.lower().str.strip().apply(str)
    print("Cleaning vehicle makes round 1")
    makes = makes.apply(rename_makes_1)
    print("Cleaning vehicle makes round 2")
    makes = makes.apply(rename_makes_2)
    print("Cleaning vehicle makes round 3")
    makes = makes.apply(rename_makes_3)
    print("Cleaning vehicle makes round 4")
    makes = makes.apply(rename_makes_4).apply(str).convert_dtypes()
    unqs, cnts = np.unique(makes.str.lower(), return_counts=True)
    idx = np.argsort(-cnts)
    unqs, cnts = unqs[idx], cnts[idx]

    """
    After all this, a few common categories cover most of the data

    >>> for n in [5, 10, 20, 25, 30, 40, 50]: print(n, np.sum(cnts[:n])/N)
     5 0.569058997726716
    10 0.7255442541459739
    20 0.8987631899619767
    25 0.9390410392512049
    30 0.9642817207891328
    40 0.9840417647673746
    50 0.9915798517658647

    >>> print(unqs[:40])
    ['toyota' 'honda' 'ford' 'nissan' 'chevy' 'hummer' 'dodge' 'acura' 'bmw'
    'mercedes' 'volkswagen' 'jeep' 'lexis' 'mazda' 'subaru' 'chrysler' 'kia'
    'gmc' 'infinity' 'mitsubishi' 'audi' 'cadillac' 'volvo' 'buick' 'mercury'
    'pontiac' 'none' 'lincoln' 'saturn' 'scion' 'izuzu' 'susuki' 'landrover'
    'mini' 'jaguar' 'porsche' 'saab' 'tesla' 'oldsmobile' 'international'
    'hyundai' 'freightliner' 'rangerover' 'plymouth' 'yamaha' 'mack'
    'kawasaki' 'fiat' 'peterbilt' 'kenworth']
    """

    common_makes = unqs.tolist()[:30]
    idx = makes.isin(common_makes)
    makes[~idx] = "unknown"

    # all_dists = cdist(common_makes, unqs, scorer=lev_dist)
    # all_spellings = {}
    # for m, common_make in enumerate(common_makes):
    #     dists = all_dists[m]  # len == len(unqs)
    #     idx = dists <= 2
    #     matches = sorted(unqs[idx].tolist())
    #     matches = [mtch for mtch in matches if mtch[0] == common_make[0]]
    #     all_spellings[common_make] = matches

    # for make, spellings in all_spellings.items():
    #     ...

    # commons = sorted(all_spellings.keys())

    # for some reason we are left with "none" despite above code
    ix = makes == "none"
    makes[ix] = "unknown"
    assert (makes == "none").sum() == 0
    df["vehicle_make"] = makes

    return df


def clean_vehicle_info(df: DataFrame) -> DataFrame:
    RARE_COLORS = {
        "BEIGE": "TAN",
        "None": "OTHER",
        "CAMOUFLAGE": "OTHER",
        "CHROME": "OTHER",
        "PINK": "OTHER",
        "COPPER": "OTHER",
        "CREAM": "OTHER",
        "MULTICOLOR": "OTHER",
        "PURPLE": "OTHER",
        "BRONZE": "OTHER",
    }
    df = df.copy()

    # the year column is full of total garbage and nonsense, number from 0 to 10000
    # just assume anything in 1960 to 2025 (range based on actual numbers present)
    df["vehicle_year"] = df["vehicle_year"].astype(float)
    idx_invalid = (df["vehicle_year"] < 1960) | (df["vehicle_year"] > 2025)
    df.loc[idx_invalid, "vehicle_year"] = NaN

    # We could manually limit the color classes because df-analyze will not
    # so limit them, which may cause a cost explosion in feature selection
    # There could also be the argument for dropping this feature if it has
    # real association with anything.
    df["vehicle_color"] = df["vehicle_color"].apply(
        lambda c: RARE_COLORS[c] if c in RARE_COLORS else c
    )

    # there are over 22 000 unique vehicle models recorded, the the most
    # common "45" (???) and "TK" (???), followed be "ACCORD", "CIVIC", ...,
    # "SUV".
    #
    # Models should probably be properly cleaned to "sedan", "truck", "suv",
    # "sport" or otherwise the column should be broken into price and class,
    # or other things This is a lot of work, so for now we drop to prevent
    # exploding compute costs.
    df.drop(columns="vehicle_model", inplace=True)

    df = fix_makes(df)

    return df


def to_hour(s: str) -> float:
    reg = r"(?P<hour>\d\d):(?P<min>\d\d):(?P<secs>\d\d)"
    result = re.search(reg, s)
    if result is None:
        return float("nan")
    if "hour" not in reg:
        return float("nan")
    d = result.groupdict()
    hrs = int(d["hour"])
    mins = 0 if "min" not in d else int(d["min"])
    secs = 0 if "secs" not in d else int(d["secs"])
    return hrs + mins / 60 + secs / 3600


def clean(s: str) -> str:
    reg = r"[^\w ]"
    s = str(s).strip().lower().replace("\\", "")  # don't intepret excapes
    try:
        return re.sub(reg, "", s).strip()
    except re.error:
        return s


def desc_remaps(s: str) -> str:
    s = s.replace("lamp", "light")
    # s = re.sub(r"(:?)|(:?)")
    s = re.sub(
        r"(:?inoperable)|(:?inoperative)|(:?in opt)|(:?inopt)|(:?inop)|(:?broken)",
        "inoperative",
        s,
    )
    s = re.sub(r"\d+ ?of ? \d+ ?", "", s)

    if ("light" in s) and ("inoperative" in s):
        return "000 inoperative lights"
    return s


def reduce_descriptions(df: DataFrame) -> DataFrame:
    print("Reducing descriptions")
    df["description"] = df["description"].apply(desc_remaps)
    descs, cnts = np.unique(df["description"], return_counts=True)
    info = DataFrame(dict(desc=descs, count=cnts))
    info.sort_values(by=["count", "desc"], ascending=False, inplace=True)
    print("Assembling description lines")
    lines = []
    for i in range(len(info)):
        desc, cnt = info.iloc[i]
        lines.append(f"[{cnt}] {desc}")
    desc = "\n".join(lines)
    DESC_OUT.write_text(desc)
    return df


if __name__ == "__main__":
    # # with open(DATA_OUT / "makes_alphasort.txt", "r") as handle:
    # #     makes = [line.replace("\n", "").strip() for line in handle.readlines()]
    # with open(DATA_OUT / "makes.txt", "r") as handle:
    #     countlines = [line.replace("\n", "").strip() for line in handle.readlines()]

    # countlines = list(reversed(countlines))

    # counts = [
    #     int(re.search(r"\[(?P<cnt>\d+)\]", line).groupdict()["cnt"])
    #     for line in countlines
    # ]
    # makes = np.array([re.sub(r"\[\d+\]", "", line).strip() for line in countlines])
    # common_makes = makes.tolist()[:150]
    # all_dists = cdist(common_makes, makes, scorer=lev_dist)
    # spellings = {}
    # for m, common_make in enumerate(common_makes):
    #     dists = all_dists[m]  # len == len(makes)
    #     idx = dists <= 2
    #     matches = sorted(makes[idx].tolist())
    #     matches = [mtch for mtch in matches if mtch[0] == common_make[0]]
    #     spellings[common_make] = matches

    use_cached = False
    use_cached = True
    if MIN_CLEAN.exists() and use_cached:
        print("Loading cleaned data...")
        df = pd.read_parquet(MIN_CLEAN)
    else:
        print("Loading ...")
        df = pd.read_csv(SOURCE, low_memory=False)
        df.rename(columns=renamer, inplace=True)
        # convert times to something useful
        print("Converting times")
        df["time_of_stop"]
        df["hour_of_stop"] = df["time_of_stop"].apply(to_hour)
        df.drop(columns="time_of_stop", inplace=True)

        # convert dates
        print("Converting dates to periodic features")
        t = pd.to_datetime(df["date_of_stop"])
        df["year_of_stop"] = t.apply(lambda t: t.year)
        df["month_of_stop"] = t.apply(lambda t: t.month)
        df["weeknum_of_stop"] = t.apply(lambda t: t.week)
        df["weekday_of_stop"] = t.apply(lambda t: t.dayofweek)
        df.drop(columns="date_of_stop", inplace=True)

        print("Converting binaries")
        for col in YN_BINS:
            df[col] = df[col].apply(lambda s: 0 if s == "No" else 1)
        for col in TF_BINS:
            # don't use .loc here, incompatible dtypes due to bool
            df[col] = df[col].apply(lambda s: 0 if s is True else 1)

        # print("Cleaning descriptions")
        # df["description"] = df["description"].apply(clean)
        # df = reduce_descriptions(df)
        df.drop(columns=DROPS, errors="ignore", inplace=True)

        df = states_to_distances(df)
        df = clean_vehicle_info(df)
        df.to_parquet(MIN_CLEAN)

    print()
