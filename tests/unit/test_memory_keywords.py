"""The in-memory keyword index of facts and summaries (R-MEM-008)."""

from __future__ import annotations

from twin.memory.keywords import KeywordIndex, tokens_of


def test_chinese_text_is_indexed_by_character_pairs_and_latin_by_words() -> None:
    assert tokens_of("喜欢火锅") == {"喜欢", "欢火", "火锅"}
    # "和" is a particle on its own; single digits are too short to index; "号" stands alone
    assert tokens_of("看了 Python 和 3 号线 2号") == {"看了", "python", "号线", "号"}
    assert tokens_of("下载 Python3 版本 12") == {"下载", "python", "版本", "12"}


def test_a_lone_chinese_character_counts_unless_it_is_a_particle() -> None:
    assert tokens_of("猫") == {"猫"}
    assert tokens_of("的") == frozenset()
    assert tokens_of("我 猫 你") == {"猫"}


def test_search_ranks_by_the_share_of_the_query_weight() -> None:
    index = KeywordIndex()
    index.add("a", "她喜欢吃火锅")
    index.add("b", "他每天跑步")
    index.add("c", "周末想吃火锅和烧烤")
    hits = index.search("想吃火锅", 5)
    assert [h.doc_id for h in hits][:2] == ["c", "a"]
    assert all(0 < h.score <= 1 for h in hits)
    assert hits[0].score > hits[1].score
    assert index.search("完全无关的话题xyz", 5) == []


def test_documents_can_be_replaced_removed_and_filtered() -> None:
    index = KeywordIndex()
    index.add("a", "养了一只猫")
    index.add("b", "养了一条狗")
    assert {h.doc_id for h in index.search("只猫", 5)} == {"a"}
    index.add("a", "养了一只兔子")  # replaced: the old words are gone
    assert index.search("只猫", 5) == []
    assert {h.doc_id for h in index.search("兔子", 5)} == {"a"}
    assert "a" in index and len(index) == 2
    index.remove("a")
    index.remove("missing")
    assert "a" not in index and len(index) == 1
    # the filter is applied before the limit, so a hidden best hit does not take the only place
    index.add("c", "养了一条金鱼")
    only_c = index.search("养了一条", 1, allowed=lambda doc_id: doc_id == "c")
    assert [h.doc_id for h in only_c] == ["c"]
    assert index.search("养了一条", 0) == []
    index.clear()
    assert len(index) == 0 and index.search("养", 3) == []
