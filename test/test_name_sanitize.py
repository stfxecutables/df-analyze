from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import re

import numpy as np
import pytest
from pandas import DataFrame

from src.preprocessing.cleaning import sanitize_names
from src.testing.datasets import (
    FAST_INSPECTION,
    MEDIUM_INSPECTION,
    SLOW_INSPECTION,
    TestDataset,
    fast_ds,
)

# fmt: off
CUPID_COLS = [
    "height_NAN", "income_NAN", "age", "height", "income", "sex_m", "sex_nan", "body_type_a little extra", "body_type_athletic", "body_type_average", "body_type_curvy", "body_type_fit", "body_type_full figured", "body_type_jacked", "body_type_overweight", "body_type_rather not say", "body_type_skinny", "body_type_thin", "body_type_used up", "body_type_nan", "diet_anything", "diet_mostly anything", "diet_mostly halal", "diet_mostly kosher", "diet_mostly other", "diet_mostly vegan", "diet_mostly vegetarian", "diet_other", "diet_strictly anything", "diet_strictly other", "diet_strictly vegan", "diet_strictly vegetarian", "diet_vegan", "diet_vegetarian", "diet_nan", "drinks_desperately", "drinks_not at all", "drinks_often", "drinks_rarely", "drinks_socially", "drinks_very often", "drinks_nan", "drugs_never", "drugs_often", "drugs_sometimes", "drugs_nan", "education_college/university", "education_dropped out of college/university", "education_dropped out of high school", "education_dropped out of masters program", "education_dropped out of ph.d program", "education_dropped out of space camp", "education_dropped out of two-year college", "education_graduated from college/university", "education_graduated from high school", "education_graduated from law school", "education_graduated from masters program", "education_graduated from med school", "education_graduated from ph.d program", "education_graduated from space camp", "education_graduated from two-year college", "education_high school", "education_masters program", "education_ph.d program", "education_space camp", "education_two-year college", "education_working on college/university", "education_working on high school", "education_working on law school", "education_working on masters program", "education_working on med school", "education_working on ph.d program", "education_working on space camp", "education_working on two-year college", "education_nan", "ethnicity_asian", "ethnicity_asian black", "ethnicity_asian hispanic / latin", "ethnicity_asian hispanic / latin white", "ethnicity_asian indian", "ethnicity_asian middle eastern black native american indian pacific islander hispanic / latin white other", "ethnicity_asian other", "ethnicity_asian pacific islander", "ethnicity_asian pacific islander other", "ethnicity_asian pacific islander white", "ethnicity_asian white", "ethnicity_asian white other", "ethnicity_black", "ethnicity_black hispanic / latin", "ethnicity_black hispanic / latin white", "ethnicity_black native american", "ethnicity_black native american hispanic / latin white", "ethnicity_black native american other", "ethnicity_black native american white", "ethnicity_black native american white other", "ethnicity_black other", "ethnicity_black white", "ethnicity_black white other", "ethnicity_hispanic / latin", "ethnicity_hispanic / latin other", "ethnicity_hispanic / latin white", "ethnicity_hispanic / latin white other", "ethnicity_indian", "ethnicity_indian other", "ethnicity_indian white", "ethnicity_middle eastern", "ethnicity_middle eastern hispanic / latin", "ethnicity_middle eastern other", "ethnicity_middle eastern white", "ethnicity_middle eastern white other", "ethnicity_native american", "ethnicity_native american hispanic / latin", "ethnicity_native american hispanic / latin white", "ethnicity_native american hispanic / latin white other", "ethnicity_native american white", "ethnicity_native american white other", "ethnicity_other", "ethnicity_pacific islander", "ethnicity_pacific islander hispanic / latin", "ethnicity_pacific islander hispanic / latin white", "ethnicity_pacific islander white", "ethnicity_white", "ethnicity_white other", "ethnicity_nan", "location_alameda california", "location_albany california", "location_atherton california", "location_belmont california", "location_belvedere tiburon california", "location_benicia california", "location_berkeley california", "location_brisbane california", "location_burlingame california", "location_castro valley california", "location_corte madera california", "location_crockett california", "location_daly city california", "location_el cerrito california", "location_el granada california", "location_el sobrante california", "location_emeryville california", "location_fairfax california", "location_foster city california", "location_fremont california", "location_green brae california", "location_half moon bay california", "location_hayward california", "location_hercules california", "location_lafayette california", "location_larkspur california", "location_martinez california", "location_menlo park california", "location_mill valley california", "location_millbrae california", "location_moraga california", "location_mountain view california", "location_novato california", "location_oakland california", "location_orinda california", "location_pacifica california", "location_palo alto california", "location_pinole california", "location_pleasant hill california", "location_redwood city california", "location_richmond california", "location_rodeo california", "location_san anselmo california", "location_san bruno california", "location_san carlos california", "location_san francisco california", "location_san leandro california", "location_san lorenzo california", "location_san mateo california", "location_san pablo california", "location_san rafael california", "location_sausalito california", "location_south san francisco california", "location_stanford california", "location_vallejo california", "location_walnut creek california", "location_nan", "offspring_doesn&rsquo;t have kids", "offspring_doesn&rsquo;t have kids and doesn&rsquo;t want any", "offspring_doesn&rsquo;t have kids but might want them", "offspring_doesn&rsquo;t have kids but wants them", "offspring_doesn&rsquo;t want kids", "offspring_has a kid", "offspring_has a kid and might want more", "offspring_has a kid and wants more", "offspring_has a kid but doesn&rsquo;t want more", "offspring_has kids", "offspring_has kids and might want more", "offspring_has kids but doesn&rsquo;t want more", "offspring_might want kids", "offspring_wants kids", "offspring_nan", "orientation_bisexual", "orientation_gay", "orientation_straight", "orientation_nan", "pets_dislikes cats", "pets_dislikes dogs", "pets_dislikes dogs and dislikes cats", "pets_dislikes dogs and has cats", "pets_dislikes dogs and likes cats", "pets_has cats", "pets_has dogs", "pets_has dogs and dislikes cats", "pets_has dogs and has cats", "pets_has dogs and likes cats", "pets_likes cats", "pets_likes dogs", "pets_likes dogs and dislikes cats", "pets_likes dogs and has cats", "pets_likes dogs and likes cats", "pets_nan", "religion_agnosticism", "religion_agnosticism and laughing about it", "religion_agnosticism and somewhat serious about it", "religion_agnosticism and very serious about it", "religion_agnosticism but not too serious about it", "religion_atheism", "religion_atheism and laughing about it", "religion_atheism and somewhat serious about it", "religion_atheism and very serious about it", "religion_atheism but not too serious about it", "religion_buddhism", "religion_buddhism and laughing about it", "religion_buddhism and somewhat serious about it", "religion_buddhism and very serious about it", "religion_buddhism but not too serious about it", "religion_catholicism", "religion_catholicism and laughing about it", "religion_catholicism and somewhat serious about it", "religion_catholicism and very serious about it", "religion_catholicism but not too serious about it", "religion_christianity", "religion_christianity and laughing about it", "religion_christianity and somewhat serious about it", "religion_christianity and very serious about it", "religion_christianity but not too serious about it", "religion_hinduism", "religion_hinduism and laughing about it", "religion_hinduism and somewhat serious about it", "religion_hinduism but not too serious about it", "religion_islam", "religion_islam and somewhat serious about it", "religion_islam but not too serious about it", "religion_judaism", "religion_judaism and laughing about it", "religion_judaism and somewhat serious about it", "religion_judaism but not too serious about it", "religion_other", "religion_other and laughing about it", "religion_other and somewhat serious about it", "religion_other and very serious about it", "religion_other but not too serious about it", "religion_nan", "sign_aquarius", "sign_aquarius and it matters a lot", "sign_aquarius and it&rsquo;s fun to think about", "sign_aquarius but it doesn&rsquo;t matter", "sign_aries", "sign_aries and it matters a lot", "sign_aries and it&rsquo;s fun to think about", "sign_aries but it doesn&rsquo;t matter", "sign_cancer", "sign_cancer and it matters a lot", "sign_cancer and it&rsquo;s fun to think about", "sign_cancer but it doesn&rsquo;t matter", "sign_capricorn", "sign_capricorn and it matters a lot", "sign_capricorn and it&rsquo;s fun to think about", "sign_capricorn but it doesn&rsquo;t matter", "sign_gemini", "sign_gemini and it matters a lot", "sign_gemini and it&rsquo;s fun to think about", "sign_gemini but it doesn&rsquo;t matter", "sign_leo", "sign_leo and it matters a lot", "sign_leo and it&rsquo;s fun to think about", "sign_leo but it doesn&rsquo;t matter", "sign_libra", "sign_libra and it matters a lot", "sign_libra and it&rsquo;s fun to think about", "sign_libra but it doesn&rsquo;t matter", "sign_pisces", "sign_pisces and it matters a lot", "sign_pisces and it&rsquo;s fun to think about", "sign_pisces but it doesn&rsquo;t matter", "sign_sagittarius", "sign_sagittarius and it matters a lot", "sign_sagittarius and it&rsquo;s fun to think about", "sign_sagittarius but it doesn&rsquo;t matter", "sign_scorpio", "sign_scorpio and it matters a lot", "sign_scorpio and it&rsquo;s fun to think about", "sign_scorpio but it doesn&rsquo;t matter", "sign_taurus", "sign_taurus and it matters a lot", "sign_taurus and it&rsquo;s fun to think about", "sign_taurus but it doesn&rsquo;t matter", "sign_virgo", "sign_virgo and it matters a lot", "sign_virgo and it&rsquo;s fun to think about", "sign_virgo but it doesn&rsquo;t matter", "sign_nan", "smokes_no", "smokes_sometimes", "smokes_trying to quit", "smokes_when drinking", "smokes_yes", "smokes_nan", "speaks_english", "speaks_english (fluently)", "speaks_english (fluently) arabic (okay)", "speaks_english (fluently) c++ (fluently)", "speaks_english (fluently) c++ (okay)", "speaks_english (fluently) c++ (poorly)", "speaks_english (fluently) chinese (fluently)", "speaks_english (fluently) chinese (fluently) french (poorly)", "speaks_english (fluently) chinese (fluently) japanese (okay)", "speaks_english (fluently) chinese (fluently) japanese (poorly)", "speaks_english (fluently) chinese (fluently) spanish (okay)", "speaks_english (fluently) chinese (fluently) spanish (poorly)", "speaks_english (fluently) chinese (okay)", "speaks_english (fluently) chinese (okay) french (poorly)", "speaks_english (fluently) chinese (okay) japanese (poorly)", "speaks_english (fluently) chinese (okay) spanish (okay)", "speaks_english (fluently) chinese (okay) spanish (poorly)", "speaks_english (fluently) chinese (poorly)", "speaks_english (fluently) chinese (poorly) spanish (poorly)", "speaks_english (fluently) english", "speaks_english (fluently) farsi (fluently)", "speaks_english (fluently) french (fluently)", "speaks_english (fluently) french (fluently) spanish (okay)", "speaks_english (fluently) french (fluently) spanish (poorly)", "speaks_english (fluently) french (okay)", "speaks_english (fluently) french (okay) german (poorly)", "speaks_english (fluently) french (okay) spanish (okay)", "speaks_english (fluently) french (okay) spanish (poorly)", "speaks_english (fluently) french (poorly)", "speaks_english (fluently) french (poorly) german (poorly)", "speaks_english (fluently) french (poorly) italian (poorly)", "speaks_english (fluently) french (poorly) spanish (okay)", "speaks_english (fluently) french (poorly) spanish (poorly)", "speaks_english (fluently) german (fluently)", "speaks_english (fluently) german (fluently) french (poorly)", "speaks_english (fluently) german (fluently) spanish (poorly)", "speaks_english (fluently) german (okay)", "speaks_english (fluently) german (okay) spanish (poorly)", "speaks_english (fluently) german (poorly)", "speaks_english (fluently) german (poorly) spanish (poorly)", "speaks_english (fluently) hebrew (fluently)", "speaks_english (fluently) hebrew (okay)", "speaks_english (fluently) hebrew (poorly)", "speaks_english (fluently) hindi (fluently)", "speaks_english (fluently) hindi (okay)", "speaks_english (fluently) italian (fluently)", "speaks_english (fluently) italian (okay)", "speaks_english (fluently) italian (poorly)", "speaks_english (fluently) italian (poorly) spanish (poorly)", "speaks_english (fluently) japanese (fluently)", "speaks_english (fluently) japanese (okay)", "speaks_english (fluently) japanese (okay) spanish (poorly)", "speaks_english (fluently) japanese (poorly)", "speaks_english (fluently) japanese (poorly) spanish (poorly)", "speaks_english (fluently) korean (fluently)", "speaks_english (fluently) korean (okay)", "speaks_english (fluently) korean (poorly)", "speaks_english (fluently) latin (poorly)", "speaks_english (fluently) other (fluently)", "speaks_english (fluently) other (okay)", "speaks_english (fluently) other (poorly)", "speaks_english (fluently) portuguese (fluently)", "speaks_english (fluently) portuguese (fluently) spanish (okay)", "speaks_english (fluently) portuguese (poorly)", "speaks_english (fluently) russian (fluently)", "speaks_english (fluently) russian (fluently) spanish (poorly)", "speaks_english (fluently) russian (okay)", "speaks_english (fluently) russian (poorly)", "speaks_english (fluently) sign language (fluently)", "speaks_english (fluently) sign language (okay)", "speaks_english (fluently) sign language (poorly)", "speaks_english (fluently) spanish", "speaks_english (fluently) spanish (fluently)", "speaks_english (fluently) spanish (fluently) french (okay)", "speaks_english (fluently) spanish (fluently) french (poorly)", "speaks_english (fluently) spanish (fluently) italian (okay)", "speaks_english (fluently) spanish (fluently) italian (poorly)", "speaks_english (fluently) spanish (fluently) japanese (poorly)", "speaks_english (fluently) spanish (fluently) portuguese (okay)", "speaks_english (fluently) spanish (fluently) portuguese (poorly)", "speaks_english (fluently) spanish (okay)", "speaks_english (fluently) spanish (okay) chinese (poorly)", "speaks_english (fluently) spanish (okay) french (okay)", "speaks_english (fluently) spanish (okay) french (poorly)", "speaks_english (fluently) spanish (okay) german (poorly)", "speaks_english (fluently) spanish (okay) italian (poorly)", "speaks_english (fluently) spanish (okay) japanese (poorly)", "speaks_english (fluently) spanish (okay) portuguese (poorly)", "speaks_english (fluently) spanish (okay) russian (poorly)", "speaks_english (fluently) spanish (poorly)", "speaks_english (fluently) spanish (poorly) c++ (fluently)", "speaks_english (fluently) spanish (poorly) c++ (okay)", "speaks_english (fluently) spanish (poorly) c++ (poorly)", "speaks_english (fluently) spanish (poorly) chinese (poorly)", "speaks_english (fluently) spanish (poorly) french (poorly)", "speaks_english (fluently) spanish (poorly) german (poorly)", "speaks_english (fluently) spanish (poorly) italian (poorly)", "speaks_english (fluently) spanish (poorly) japanese (poorly)", "speaks_english (fluently) spanish (poorly) sign language (poorly)", "speaks_english (fluently) tagalog (fluently)", "speaks_english (fluently) tagalog (okay)", "speaks_english (fluently) tagalog (poorly)", "speaks_english (fluently) vietnamese (fluently)", "speaks_english (fluently) vietnamese (okay)", "speaks_english (okay)", "speaks_english (okay) chinese (fluently)", "speaks_english (okay) french (poorly)", "speaks_english (okay) spanish (fluently)", "speaks_english (okay) spanish (poorly)", "speaks_english (poorly)", "speaks_english (poorly) spanish (poorly)", "speaks_english arabic", "speaks_english c++", "speaks_english chinese", "speaks_english chinese (fluently)", "speaks_english chinese (okay)", "speaks_english chinese (poorly)", "speaks_english chinese spanish", "speaks_english english", "speaks_english english (fluently)", "speaks_english farsi", "speaks_english french", "speaks_english french (fluently)", "speaks_english french (okay)", "speaks_english french (okay) spanish (poorly)", "speaks_english french (poorly)", "speaks_english french (poorly) spanish (poorly)", "speaks_english french spanish", "speaks_english german", "speaks_english german (okay)", "speaks_english german (poorly)", "speaks_english hebrew", "speaks_english hindi", "speaks_english italian", "speaks_english italian (okay)", "speaks_english italian (poorly)", "speaks_english japanese", "speaks_english japanese (okay)", "speaks_english japanese (poorly)", "speaks_english korean", "speaks_english other", "speaks_english other (fluently)", "speaks_english russian", "speaks_english spanish", "speaks_english spanish (fluently)", "speaks_english spanish (okay)", "speaks_english spanish (okay) french (poorly)", "speaks_english spanish (okay) italian (poorly)", "speaks_english spanish (poorly)", "speaks_english spanish (poorly) french (poorly)", "speaks_english spanish french", "speaks_english tagalog", "speaks_english tagalog (okay)", "speaks_english vietnamese", "speaks_nan", "status_available", "status_married", "status_seeing someone", "status_single", "status_nan"]
# fmt: on


def test_target_error() -> None:
    target = "target"
    cols = ["a", "a", "b"]
    df = DataFrame(columns=cols, index=[0])
    with pytest.raises(RuntimeError):
        dfr = sanitize_names(df, target)[0]

    target = "a"
    cols = ["a", "a", "b"]
    df = DataFrame(columns=cols, index=[0])
    with pytest.raises(RuntimeError):
        dfr = sanitize_names(df, target)[0]

    target = "b"
    cols = ["a", "a", "b"]
    df = DataFrame(columns=cols, index=[0])
    dfr, renames = sanitize_names(df, target)


def test_dedup() -> None:
    target = "target"
    cols = ["a", "a", "b"]
    tcols = cols + [target]
    df = DataFrame(columns=tcols, index=[0])
    dfr = sanitize_names(df, target)[0]
    assert not dfr.columns.has_duplicates

    names = [chr(i) for i in range(97, 123)]  # a-z
    for _ in range(1000):
        cols = np.random.choice(names, replace=True, size=100).tolist()
        tcols = cols + [target]
        df = DataFrame(columns=tcols, index=[0])
        dfr, renames = sanitize_names(df, target)
        originals, news = list(zip(*renames.renames))

        assert not dfr.columns.has_duplicates
        assert dfr.drop(columns=target).columns.to_list() == list(news)
        assert len(renames.renames) == len(cols)
        assert list(originals) == cols
        # this check works only in this simple case
        for orig, new in renames.renames:
            assert orig in new


def test_ok_cupid() -> None:
    target = "target"
    df = DataFrame(columns=CUPID_COLS + [target], index=[0])

    dfr, renames = sanitize_names(df, target)
    news = [new for old, new in renames.renames]
    for _ in range(500):
        n = np.random.randint(2, len(news))
        cols = np.random.choice(news, size=n, replace=False).tolist()
        regex = "|".join(cols)
        try:
            re.compile(regex)
        except Exception as e:
            message = str(e)
            mstart = "multiple repeat at position "
            start = message.find(mstart) + len(mstart)
            matches = re.findall(r" ?(\d+)", message[start : start + 20])
            if len(matches) == 0:
                raise ValueError(
                    f"Failed to compile regex `{regex}` for columns: {cols}"
                ) from e
            idx = int(matches[0].strip())
            ctx_start = max(0, idx - 30)
            ctx_end = min(len(regex), idx + 30)
            context = regex[ctx_start:ctx_end]

            raise ValueError(
                f"Failed to compile regex. Context: {context}.\n"
                f"Full regex: `{regex}`.\n"
                f"Columns: {cols}"
            ) from e


def test_null_report() -> None:
    target = "target"
    cols = ["a", "b", "c", target]
    df = DataFrame(columns=cols, index=[0])
    dfr, renames = sanitize_names(df, target)
    assert renames.to_markdown() is None


def test_simple_report() -> None:
    target = "target"
    cols = ["a", "a", "b", target]
    df = DataFrame(columns=cols, index=[0])
    dfr, renames = sanitize_names(df, target)
    report = renames.to_markdown()
    assert report is not None
    lines = report.split("\n")
    for i, line in enumerate(lines):
        if "---" not in line:
            continue
        table_line = lines[i + 1].replace("|", "").strip()
        clean = re.sub(r" +", " ", table_line)
        try:
            renamed, orig = clean.split(" ")
        except ValueError as e:
            raise ValueError(f"Could not split line: `{table_line}`") from e
        assert orig == "a"
        assert renamed == "a_1"
