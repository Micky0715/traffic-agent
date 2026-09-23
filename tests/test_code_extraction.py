"""Regulation / drawing code extraction across Chinese-adjacent contexts.

The bug these cover: Python's `re` treats CJK characters as `\\w`, so a `\\b`
placed at either end of a code pattern fails whenever a Chinese character sits
against the code — which in Chinese queries is nearly always. It did not only
miss matches: on "TB10621-2014对…" the trailing boundary failed, the regex
backtracked, and normalization rewrote the code as "TB 1-0621". A missed match
leaves a gap; a backtracked one corrupts the value everything downstream then
trusts.

Nothing here keys on a case id, a file name or a blacklist of specific strings.
"""
from __future__ import annotations

import pytest

from src.extractors import (
    ASSET_PATTERN, extract_slots, find_code_mentions, fold_for_match,
)
from src.normalize import normalize_regulation_code, normalize_text


def only(mentions, kind):
    return [m for m in mentions if m.kind == kind]


# --------------------------------------------------------------------------
# 1-3. Chinese adjacency, brackets, punctuation
# --------------------------------------------------------------------------

@pytest.mark.parametrize("query", [
    "查JTG D81-2017中的要求",          # Chinese immediately before AND after
    "依据（JTG D81-2017）查询",         # full-width brackets
    "规范：JTG D81-2017",              # full-width colon, code at end of string
    "JTG D81-2017对应急照明有什么要求",  # Chinese only after
    "根据 JTG D81-2017 的规定",         # spaces on both sides
])
def test_regulation_is_found_regardless_of_what_surrounds_it(query):
    found = only(find_code_mentions(query), "regulation")
    assert len(found) == 1, query
    assert found[0].normalized_value == "JTG D81-2017"


# --------------------------------------------------------------------------
# 4-5. spacing and dash variants
# --------------------------------------------------------------------------

def test_missing_space_is_canonicalized_not_rejected():
    found = only(find_code_mentions("jtgd81-2017里应急照明供电要求是什么"), "regulation")
    assert found[0].raw_value == "jtgd81-2017"       # evidence of what was written
    assert found[0].normalized_value == "JTG D81-2017"  # what matching needs


@pytest.mark.parametrize("dash", ["-", "‐", "–", "—", "−", "－"])
def test_dash_variants_all_normalize_to_one_code(dash):
    found = only(find_code_mentions(f"查JTG D81{dash}2017要求"), "regulation")
    assert found[0].normalized_value == "JTG D81-2017"


def test_full_width_letters_and_digits_are_folded():
    found = only(find_code_mentions("查询ＪＴＧ Ｄ８１－２０１７中风机检修要求"), "regulation")
    assert found[0].normalized_value == "JTG D81-2017"
    assert found[0].raw_value == "ＪＴＧ Ｄ８１－２０１７"  # original preserved


def test_backtracking_no_longer_corrupts_a_code():
    """Regression for the worst form of the bug: the old pattern rewrote
    TB10621-2014 as "TB 1-0621", a value that was never on the page."""
    assert normalize_regulation_code("TB10621—2014对轨道扣件间距有什么规定").startswith(
        "TB 10621-2014")
    found = only(find_code_mentions("TB10621—2014对轨道扣件间距有什么规定"), "regulation")
    assert found[0].normalized_value == "TB 10621-2014"


# --------------------------------------------------------------------------
# 6. part of a longer token must not match
# --------------------------------------------------------------------------

@pytest.mark.parametrize("query", [
    "SUBGB50157-2013",      # GB preceded by a letter
    "XJTG D81-2017",        # JTG preceded by a letter
    "AAFAN-A13-02",         # FAN preceded by letters
    "9GB50157-2013",        # preceded by a digit
])
def test_a_code_embedded_in_a_longer_token_is_not_matched(query):
    assert find_code_mentions(query) == []


def test_a_bare_prefix_without_a_number_is_not_a_code():
    assert only(find_code_mentions("GBK编码问题"), "regulation") == []


# --------------------------------------------------------------------------
# 7-9. span-scoped exclusion, not a blacklist
# --------------------------------------------------------------------------

def test_device_id_that_looks_like_a_code_fragment_still_works_on_its_own():
    """D81 inside "JTG D81-2017" is a fragment; D81风机 in a sentence of its own
    is a real device. A global blacklist could not tell them apart."""
    assert extract_slots("查D81风机的状态")["asset_id"] == "D81风机"


def test_regulation_and_a_real_device_coexist_in_one_sentence():
    slots = extract_slots(normalize_text("查JTG D81-2017应急照明要求，再查A12风机现在温度"))
    assert slots["regulation_code"] == "JTG D81-2017"
    assert slots["asset_id"] == "A12风机"


def test_the_same_characters_are_excluded_only_inside_the_code_span():
    """One sentence, two occurrences of D81: one a fragment of the regulation
    number, one a genuine device. Span-scoped exclusion keeps the second."""
    slots = extract_slots("查JTG D81-2017，再看D81风机状态")
    assert slots["regulation_code"] == "JTG D81-2017"
    assert slots["asset_id"] == "D81风机"


def test_drawing_number_fragment_is_not_reported_as_a_device():
    slots = extract_slots("看看FAN-A13-02这张图纸")
    assert slots["drawing_no"] == "FAN-A13-02"
    assert "asset_id" not in slots


# --------------------------------------------------------------------------
# 10. spans point into the text that was handed in
# --------------------------------------------------------------------------

@pytest.mark.parametrize("query", [
    "查JTG D81-2017中的应急照明要求",
    "依据（JTG D81-2017）查询",
    "查询ＪＴＧ Ｄ８１－２０１７中风机检修要求",
    "看看FAN-A13-02这张图纸",
])
def test_span_indexes_the_original_query_not_a_normalized_rewrite(query):
    for mention in find_code_mentions(query):
        assert query[mention.start:mention.end] == mention.raw_value


def test_folding_is_length_preserving():
    """The property the whole span guarantee rests on. NFKC would fold more but
    can change length, which would make offsets unmappable."""
    for text in ["ＪＴＧ Ｄ８１－２０１７", "查JTG D81—2017中", "ＡＢＣ０１２", "普通中文"]:
        assert len(fold_for_match(text)) == len(text)


def test_two_codes_in_one_query_keep_separate_spans():
    query = "对照JTG D81-2017和图纸FAN-A13-02"
    mentions = find_code_mentions(query)
    assert [m.kind for m in mentions] == ["regulation", "drawing"]
    assert [query[m.start:m.end] for m in mentions] == ["JTG D81-2017", "FAN-A13-02"]
    assert mentions[0].end <= mentions[1].start   # non-overlapping, in order


def test_asset_pattern_itself_is_unchanged_by_this_fix():
    """Scope guard: only the code boundaries and span exclusion moved. If the
    device pattern itself had been touched, every routing metric would shift for
    reasons unrelated to this bug."""
    assert ASSET_PATTERN.search("A12风机").group(0) == "A12风机"
    assert ASSET_PATTERN.search("3号水泵").group(0) == "3号水泵"
    assert ASSET_PATTERN.search("B07屏蔽门").group(0) == "B07屏蔽门"
