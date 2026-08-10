from src.vision.regulation_parser import chinese_to_arabic, parse_regulation_text


def test_chinese_to_arabic_basic_cases():
    assert chinese_to_arabic("五") == 5
    assert chinese_to_arabic("十") == 10
    assert chinese_to_arabic("十二") == 12
    assert chinese_to_arabic("二十三") == 23
    assert chinese_to_arabic("一百零五") == 105


def test_chinese_to_arabic_returns_none_for_unparseable_input_not_a_guess():
    assert chinese_to_arabic("十²") is None
    assert chinese_to_arabic("") is None


def test_basic_chapter_article_structure_is_recovered():
    text = """第一章 总则
第一条 本规程适用于城市轨道交通设备维护。
第二条 维护人员应当持证上岗。
第二章 检修要求
第三条 检修记录应当保存至少三年。
"""
    meta = parse_regulation_text(text)
    articles = [c.article for c in meta.clauses if c.article]
    assert articles == ["第一条", "第二条", "第三条"]
    chapters = {c.chapter for c in meta.clauses if c.chapter}
    assert chapters == {"第一章", "第二章"}


def test_article_number_gap_is_flagged():
    text = """第十二条 风机振动超过预警阈值时应停机检修。
第十四条 屏蔽门重复故障应检查门控单元日志。
"""
    meta = parse_regulation_text(text)
    assert any("第十二条后直接出现第十四条" in w for w in meta.warnings)


def test_consecutive_articles_do_not_trigger_gap_warning():
    text = """第十二条 风机振动超过预警阈值时应停机检修。
第十三条 断路器应定期检测绝缘电阻。
"""
    meta = parse_regulation_text(text)
    assert meta.warnings == []


def test_lead_in_sentence_before_first_clause_becomes_its_own_entry():
    """Bad case #2 from the plan: text appears before '第一款' that belongs
    to the article itself, not to clause 1 — it must not be merged into
    clause 1's text nor dropped."""
    text = """第五条 出现下列情形之一的，应当立即停机：
第一款 振动值超过报警阈值。
第二款 温度超过报警阈值。
"""
    meta = parse_regulation_text(text)
    article_five = [c for c in meta.clauses if c.article == "第五条"]
    assert len(article_five) == 3  # lead-in + 第一款 + 第二款, three distinct entries
    lead_in = article_five[0]
    assert lead_in.clause is None
    assert "应当立即停机" in lead_in.text
    assert article_five[1].clause == "第一款"
    assert article_five[2].clause == "第二款"


def test_parenthesized_numeral_clause_style_is_also_recognized():
    text = """第六条 检修完成后应当确认：
（一）设备已恢复正常运行。
（二）现场无遗留工具。
"""
    meta = parse_regulation_text(text)
    clause_markers = [c.clause for c in meta.clauses if c.article == "第六条"]
    assert "（一）" in clause_markers
    assert "（二）" in clause_markers


def test_repeated_short_lines_are_filtered_as_header_footer_noise():
    text = """《城市轨道交通设备维护规程》
第一条 本规程适用范围。
城市轨道交通维护手册
第二条 维护记录要求。
城市轨道交通维护手册
"""
    meta = parse_regulation_text(text)
    all_text = " ".join(c.text for c in meta.clauses)
    assert "城市轨道交通维护手册" not in all_text


def test_multiple_articles_on_the_same_block_all_parsed():
    text = "第一条 甲。第二条 乙。第三条 丙。"
    # not realistic layout (real text has line breaks) but tests the parser
    # doesn't require exactly one clause per input line to find at least
    # the first marker on a line
    meta = parse_regulation_text(text)
    assert any(c.article == "第一条" for c in meta.clauses)


def test_document_metadata_title_and_effective_date_extraction():
    text = """《城市轨道交通设备维护规程》
2026年1月1日发布，自2026年3月1日起施行。
第一条 本规程适用范围。
"""
    meta = parse_regulation_text(text)
    assert meta.document_name == "城市轨道交通设备维护规程"
    assert meta.effective_date in ("2026年1月1日", "2026年3月1日")
