from datetime import date
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from agents import workflow
from tools import announcement_search as search
from ui import announcement_search as view


PAGE = Path(__file__).resolve().parents[1] / "streamlit_app.py"
QUESTION = "2023 年度分红的股权登记日是哪一天？"
PDF_URL = "https://static.cninfo.com.cn/finalpage/2024-06-12/1220324741.PDF"


def test_search_matches_stock_and_report_year_even_when_disclosed_next_year(monkeypatch):
    official = "http://www.cninfo.com.cn/new/disclosure/detail?announcementId=1220324741"
    source = pd.DataFrame([
        ["600519", "2023年年度<em>权益</em>分派实施公告", "2024-06-12", official],
        ["600519", "2023年年度权益分派实施公告", "2024-06-12", official],
        ["600519", "关于2023年权益分派的提示", "2024-06-13", "https://www.cninfo.com.cn/other"],
        ["600519", "2024年中期权益分派实施公告", "2024-12-14", "https://www.cninfo.com.cn/next"],
        ["000001", "2023年年度权益分派实施公告", "2024-06-12", "https://www.cninfo.com.cn/wrong-stock"],
        ["600519", "2023年年度权益分派公告", "2024-06-12", "https://www.cninfo.com.cn.example.org/bad"],
        ["600519", "2023年年度权益分派公告", "2025-02-01", "https://www.cninfo.com.cn/late"],
    ], columns=["代码", "公告标题", "公告时间", "公告链接"])
    loader = Mock(return_value=source)
    monkeypatch.setattr(search.ak, "stock_zh_a_disclosure_report_cninfo", loader)

    result = search.find_announcements("600519", "贵州茅台 2023 年度分红的股权登记日是哪一天？")

    loader.assert_called_once_with(symbol="600519", market="沪深京", keyword="权益分派",
                                   start_date="20230101", end_date="20241231")
    assert [row["title"] for row in result["records"]] == ["2023年年度权益分派实施公告", "关于2023年权益分派的提示"]
    assert result["records"][0]["url"] == official.replace("http:", "https:")
    assert result["records"][0]["date"].isoformat() == "2024-06-12"


def test_find_button_works_without_running_stock_analysis(monkeypatch):
    loader = Mock(side_effect=[pd.DataFrame([
        ["600519", "2023年年度报告", "2024-04-03", "https://www.cninfo.com.cn/report"],
    ], columns=["代码", "公告标题", "公告时间", "公告链接"]), RuntimeError("provider connection failed")])
    analyze = Mock(side_effect=AssertionError("公告检索不应提交分析"))
    importer = Mock(return_value={"duplicate": False, "chunks_added": 2})
    monkeypatch.setattr(view, "import_announcement", importer)
    monkeypatch.setattr(search.ak, "stock_zh_a_disclosure_report_cninfo", loader)
    monkeypatch.setattr(workflow, "run_analysis", analyze)
    page = Path(__file__).resolve().parents[1] / "streamlit_app.py"
    at = AppTest.from_file(str(page), default_timeout=30).run()
    loader.assert_not_called()
    at.text_area(key="question").set_value("2023 年报")
    next(button for button in at.sidebar.button if button.label == "查找公告").click().run()
    assert not at.exception
    assert any("2023年年度报告" in item.value for item in at.tabs[2].text)
    assert len(at.tabs[2].get("link_button")) == 1
    assert any("已下载并导入" in item.value for item in at.tabs[2].success)
    at.run()
    loader.assert_called_once()
    importer.assert_called_once()

    next(button for button in at.sidebar.button if button.label == "查找公告").click().run()
    assert any("暂时无法从巨潮资讯读取公告" in item.value for item in at.tabs[2].warning)
    assert not at.tabs[2].get("link_button")
    analyze.assert_not_called()


def test_meeting_registration_date_and_quoted_buyback_use_disclosure_year():
    today = date(2026, 11, 24)
    for question, keyword in [("2024 年股东大会的股权登记日", "股东大会"), ("找 2024 年的《回购公司股份》", "回购公司股份")]:
        actual, _, year, start, end = search._search_parameters(question, today)
        assert actual == keyword and year is None
        assert (start, end) == (date(2024, 1, 1), date(2024, 12, 31))


def test_multiple_announcements_require_selection_and_clear_only_outdated_rag_cache(monkeypatch):
    loader = Mock(return_value=pd.DataFrame([
        ["600519", "2023年年度报告", "2024-04-03", "https://www.cninfo.com.cn/report"],
        ["600519", "2023年年度报告摘要", "2024-04-03", "https://www.cninfo.com.cn/summary"],
    ], columns=["代码", "公告标题", "公告时间", "公告链接"]))
    importer = Mock(return_value={"duplicate": False, "chunks_added": 2})
    monkeypatch.setattr(search.ak, "stock_zh_a_disclosure_report_cninfo", loader)
    monkeypatch.setattr(view, "import_announcement", importer)
    page = Path(__file__).resolve().parents[1] / "streamlit_app.py"
    at = AppTest.from_file(str(page), default_timeout=30).run()
    old_rag = ("600519", "2024-01-01", "2024-12-31", "本地规则", False, "2023 年报")
    ordinary = (*old_rag[:-1], "")
    other_stock = ("000001", *old_rag[1:])
    at.session_state["completed"] = {key: {} for key in (old_rag, ordinary, other_stock)}
    at.text_area(key="question").set_value("2023 年报")
    next(button for button in at.sidebar.button if button.label == "查找公告").click().run()

    importer.assert_not_called()
    at.button(key="import_announcement_2").click().run()
    assert not at.exception
    assert importer.call_args.args[1]["title"] == "2023年年度报告摘要"
    assert set(at.session_state["completed"]) == {ordinary, other_stock}


def submit_original_query(page):
    page.text_area(key="question").set_value(QUESTION)
    return next(button for button in page.sidebar.button if button.label == "查询公告原文").click().run()


def original_facts():
    return {"question": QUESTION, "status": "ok", "records": [
        {"ref_id": "R1", "document_name": "2023年年度权益分派实施公告.pdf", "page": 1,
         "text": "股权登记日为2024年6月18日。", "official_pdf_url": PDF_URL, "document_id": "a" * 64},
    ]}


@pytest.mark.parametrize("status", ["ok", "index_error"])
def test_original_query_uses_local_results_or_reports_index_error_without_network(monkeypatch, status):
    facts = original_facts() if status == "ok" else {"question": QUESTION, "status": status, "message": "本地索引损坏。"}
    retrieval = Mock(return_value=facts)
    find = Mock()
    analyze = Mock()
    monkeypatch.setattr(view, "retrieve_document_facts", retrieval)
    monkeypatch.setattr(view, "find_announcements", find)
    monkeypatch.setattr(view.service, "analyze_once", analyze)
    page = AppTest.from_file(str(PAGE), default_timeout=30).run()
    page.selectbox(key="mode").select("DeepSeek")
    page.checkbox(key="research").check()
    submit_original_query(page)

    assert not page.exception
    rag = page.tabs[2]
    assert rag.label == "RAG原文"
    if status == "ok":
        assert any(PDF_URL + "#page=1" in item.value for item in rag.markdown)
        assert any("2024-06-18 [R1]" in item.value for item in rag.text)
    else:
        assert rag.warning[0].value == "本地索引损坏。"
    page.run()
    retrieval.assert_called_once_with(QUESTION, symbol="600519", report_year=2023, choose_document=True)
    find.assert_not_called()
    analyze.assert_not_called()


@pytest.mark.parametrize("count", [1, 2])
def test_original_query_imports_and_searches_only_the_selected_notice(monkeypatch, count):
    records = [
        {"title": f"权益分派公告 {i}", "date": date(2024, 6, 12), "url": f"https://www.cninfo.com.cn/{i}"}
        for i in range(count)
    ]
    retrieval = Mock(side_effect=[{"status": "no_match"}, original_facts()])
    find = Mock(return_value={"keyword": "权益分派", "year": 2023, "start": date(2023, 1, 1),
                              "end": date(2024, 12, 31), "records": records})
    importer = Mock(return_value={"document_id": "a" * 64, "duplicate": True, "chunks_added": 0, "pdf_url": PDF_URL})
    monkeypatch.setattr(view, "find_announcements", find)
    monkeypatch.setattr(view, "import_announcement", importer)
    monkeypatch.setattr(view, "retrieve_document_facts", retrieval)
    page = AppTest.from_file(str(PAGE), default_timeout=30).run()
    submit_original_query(page)
    if count == 2:
        importer.assert_not_called()
        assert retrieval.call_count == 1
        page.button(key="import_announcement_2").click().run()

    assert not page.exception
    find.assert_called_once_with("600519", QUESTION)
    importer.assert_called_once_with("600519", records[-1], 2023)
    assert retrieval.call_count == 2
    assert retrieval.call_args.args == (QUESTION,)
    assert retrieval.call_args.kwargs == {"symbol": "600519", "document_id": "a" * 64}
    assert any(PDF_URL + "#page=1" in item.value for item in page.tabs[2].markdown)
    page.run()
    assert retrieval.call_count == 2


def test_announcement_deepseek_uses_only_cited_snippets_after_explicit_click(monkeypatch):
    facts = original_facts()
    facts["filters"] = {"symbol": "600519", "document_id": "a" * 64}
    monkeypatch.setattr(view, "retrieve_document_facts", Mock(return_value=facts))
    client = Mock()
    client.return_value.describe.return_value = "登记日见公告安排 [R1]。"
    monkeypatch.setattr(view, "DeepSeekClient", client)
    page = AppTest.from_file(str(PAGE), default_timeout=30).run()
    page.selectbox(key="mode").select("DeepSeek")
    submit_original_query(page)
    client.assert_not_called()

    next(button for button in page.button if button.label == "用 DeepSeek 解读这份原文").click().run()
    assert any("登记日见公告安排 [R1]" in item.value for item in page.tabs[2].text)
    client.return_value.describe.assert_called_once_with("公告原文助手", {"rag": {
        "question": QUESTION, "status": "ok",
        "records": [{"ref_id": "R1", "page": 1, "text": "股权登记日为2024年6月18日。"}],
    }})
    page.run()
    client.assert_called_once()


@pytest.mark.parametrize("multiple", [False, True])
def test_original_query_is_bound_to_the_local_pdf_for_the_requested_year(monkeypatch, tmp_path, multiple):
    from rag import index as rag_index

    class FixedEmbedder:
        model_id = "TEST-ONLY"

        def encode(self, texts, *, query=False):
            return np.tile([1.0, 0.5], (len(texts), 1))

    embedder = FixedEmbedder()
    index = rag_index.DocumentIndex(tmp_path / "index", embedder=embedder)
    fixture = (Path(__file__).parent / "fixtures" / "rag_test_document.pdf").read_bytes()
    names = ["2022-implementation.pdf", "2023-implementation.pdf"]
    if multiple:
        names.append("2023-summary.pdf")
    ids = {}
    for name in names:
        path = tmp_path / name
        path.write_bytes(fixture + f"\n% TEST ONLY {name}".encode())
        year = int(name[:4])
        metadata = {"symbol": "600519", "report_year": year, "disclosure_date": f"{year + 1}-06-12"}
        ids[name] = index.import_pdf(path, metadata=metadata)["document_id"]
    monkeypatch.setattr(rag_index, "DEFAULT_INDEX_DIR", index.path)
    monkeypatch.setattr(rag_index, "LocalEmbedder", lambda: embedder)
    find = Mock(side_effect=AssertionError("本地文档查询不应访问巨潮资讯"))
    monkeypatch.setattr(view, "find_announcements", find)
    page = AppTest.from_file(str(PAGE), default_timeout=30).run()
    submit_original_query(page)

    if multiple:
        picker = page.selectbox(key="announcement_local_document")
        assert all("2023" in label for label in picker.options)
        assert len(picker.options) == 2
        assert not page.tabs[2].expander
        picker.set_value(ids["2023-implementation.pdf"]).run()
        next(button for button in page.button if button.label == "查询所选 PDF").click().run()

    assert not page.exception
    rendered = page.tabs[2].expander
    assert rendered and all("2023-implementation.pdf" in section.label for section in rendered)
    facts = page.session_state["announcement_search"]["facts"]
    assert {hit["document_id"] for hit in facts["records"]} == {ids["2023-implementation.pdf"]}
    find.assert_not_called()


def test_selected_local_pdf_with_no_match_waits_for_explicit_official_lookup(monkeypatch):
    ids = ["a" * 64, "b" * 64]
    documents = [{"document_id": identity, "document_name": "公告.pdf",
                  "metadata": {"report_year": 2023, "disclosure_date": "2024-06-12"}} for identity in ids]
    retrieval = Mock(side_effect=[{"status": "choose_document", "documents": documents},
                                  {"status": "no_match", "question": QUESTION}])
    find = Mock(side_effect=ValueError("没有找到对应标题。"))
    monkeypatch.setattr(view, "retrieve_document_facts", retrieval)
    monkeypatch.setattr(view, "find_announcements", find)
    page = AppTest.from_file(str(PAGE), default_timeout=30).run()
    submit_original_query(page)
    picker = page.selectbox(key="announcement_local_document")
    assert len(set(picker.options)) == 2
    picker.set_value(ids[1]).run()
    next(button for button in page.button if button.label == "查询所选 PDF").click().run()
    assert not page.exception
    assert "所选 PDF 没有找到相关片段" in page.tabs[2].warning[0].value
    assert retrieval.call_args.kwargs["document_id"] == ids[1]
    find.assert_not_called()

    next(button for button in page.button if button.label == "去巨潮资讯查找").click().run()
    assert page.tabs[2].warning[0].value == "没有找到对应标题。"
    assert page.selectbox(key="announcement_local_document")
    assert retrieval.call_count == 2
    find.assert_called_once_with("600519", QUESTION)
