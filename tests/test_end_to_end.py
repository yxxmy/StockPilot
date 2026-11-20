"""阶段七完整业务链路：真实BGE/FAISS/页面/Agent/SQLite，仅外部API替身。

样本身份和数值只用于测试。模型必须已在本地缓存，缺失时失败，不下载或用假向量替代。
"""

from datetime import date
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pyarrow  # noqa: F401；Windows主线程预载，避免AppTest线程首次加载原生库。
import pytest
import requests
from streamlit.testing.v1 import AppTest

from memory import ReportStore
from rag.embedding import LocalEmbedder
from rag.index import DocumentIndex
from tools import research_data as rd
import rag.index as index_module


ROOT = Path(__file__).resolve().parents[1]
QUESTION = "测试样本中经营活动现金流量如何反映现金流入与流出？"
NODES = ["market", "financial", "news", "risk", "rag_retrieval", "report"]


def formal_hashes():
    """包括索引所有旧快照、原PDF及SQLite/附属文件；不读取认证配置。"""
    paths = list((ROOT / "data/rag").rglob("*")) + list((ROOT / "data/documents").rglob("*"))
    paths += list((ROOT / "data").glob("*.sqlite3*"))
    return {str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}


@pytest.fixture(scope="module")
def real_corpus(tmp_path_factory):
    before = formal_hashes()
    cases = []
    with pytest.MonkeyPatch.context() as config:
        config.setenv("HF_HUB_OFFLINE", "1")
        config.setenv("TRANSFORMERS_OFFLINE", "1")
        embedder = LocalEmbedder()
        # 主线程加载真实CPU BGE权重；模型缺失必须阻塞测试，不能悄悄跳过。
        embedder.load()
        work = tmp_path_factory.mktemp("real_bge_e2e")
        index = DocumentIndex(work / "index", embedder=embedder)
        known = {}
        for symbol in ["600519", "000001", None]:
            name = f"TEST_ONLY_{symbol or 'UNKNOWN'}.pdf"
            path = work / name
            path.write_bytes((ROOT / "tests/fixtures/rag_test_document.pdf").read_bytes()
                             + f"\n% TEST ONLY IDENTITY {symbol}".encode())
            metadata = None if symbol is None else {"symbol": symbol, "company_name": f"仅测试公司{symbol}",
                "report_year": 2023, "document_type": "announcement", "disclosure_date": "2024-04-02"}
            document = index.import_pdf(path, metadata=metadata)
            known[symbol] = document["document_id"]
        yield SimpleNamespace(path=index.path, chunks=index.chunks, ids=known, cases=cases)
    after = formal_hashes()
    unchanged = before == after
    destination = os.environ.get("STOCKPILOT_E2E_AUDIT")
    if destination:
        Path(destination).write_text(json.dumps({"model": embedder.model_id, "real_local_bge": True,
            "network_downloads": False, "paid_api_calls": 0, "scenarios": cases,
            "formal_files_unchanged": unchanged, "formal_hashes_before": before,
            "formal_hashes_after": after}, ensure_ascii=False, indent=2), encoding="utf-8")
    assert unchanged, "测试前后正式PDF/FAISS/SQLite文件集合或哈希发生变化"


@pytest.fixture
def system(real_corpus, tmp_path, monkeypatch):
    path = tmp_path / "index"
    shutil.copytree(real_corpus.path, path)
    # 仅切换默认目录，不替换检索、Embedding、Agent或工作流函数。
    monkeypatch.setattr(index_module, "DEFAULT_INDEX_DIR", path)
    database = Path(os.environ["STOCKPILOT_DB_PATH"])
    assert database.resolve().is_relative_to(tmp_path.resolve())
    assert path.resolve().is_relative_to(tmp_path.resolve())
    monkeypatch.setenv("DEEPSEEK_API_KEY", "TEST_ONLY_E2E_NOT_A_REAL_KEY")
    history = pd.DataFrame({"日期": pd.date_range("2024-01-02", periods=40, freq="B"),
        "开盘": [10.] * 40, "最高": [12.] * 40, "最低": [9.] * 40,
        "收盘": [11. + i * .01 for i in range(40)], "成交量": [100.] * 40})
    market = Mock(return_value=history)
    finance = Mock(side_effect=lambda symbol, **kw: pd.DataFrame({"SECURITY_CODE": [symbol.split('.')[0]],
        "REPORT_DATE": ["2024-03-31"], "NOTICE_DATE": ["2024-04-28"], "CURRENCY": ["CNY"],
        **{field: [10.] for field in rd.FINANCIAL_FIELDS}}))
    news = Mock(side_effect=lambda symbol: pd.DataFrame({"关键词": [symbol], "新闻标题": ["仅测试新闻"],
        "新闻内容": ["明确标记的测试材料，不是真实公司新闻"], "发布时间": ["2024-05-02 10:00:00"],
        "文章来源": ["TEST ONLY"], "新闻链接": ["https://example.invalid/test-news"]}))
    notices = Mock(side_effect=lambda symbol, **kw: pd.DataFrame({"代码": [symbol], "公告标题": ["仅测试公告标题"],
        "公告时间": ["2024-05-01"], "公告链接": ["https://example.invalid/test-notice"]}))
    for name, api in [("stock_zh_a_hist", market), ("stock_financial_analysis_indicator_em", finance),
                      ("stock_news_em", news), ("stock_zh_a_disclosure_report_cninfo", notices)]:
        monkeypatch.setattr(rd.ak, name, api)
    payloads = []

    def respond(url, **kwargs):
        assert url == "https://api.deepseek.com/chat/completions"
        assert kwargs["headers"]["Authorization"] == "Bearer TEST_ONLY_E2E_NOT_A_REAL_KEY"
        messages = kwargs["json"]["messages"]
        facts = json.loads(messages[1]["content"])
        payloads.append({"facts": facts, "system": messages[0]["content"]})
        output = {"summary": "仅测试模型响应，不是真实分析结论。", "evidence": [next(iter(facts))]}
        if "rag" in facts:
            refs = [item["ref_id"] for item in facts["rag"]["records"]]
            output.update(summary="仅依据测试样本回答" + "".join(f"[{ref}]" for ref in refs), evidence=["rag"], citations=refs)
        return Mock(status_code=200, json=Mock(return_value={"choices": [{"finish_reason": "stop",
                    "message": {"content": json.dumps(output, ensure_ascii=False)}}]}))

    post = Mock(side_effect=respond)
    monkeypatch.setattr(requests, "post", post)
    return SimpleNamespace(corpus=real_corpus, index=path, db=database, market=market, finance=finance,
                           news=news, notices=notices, post=post, payloads=payloads)


def app():
    return AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=120).run()


def submit(at, symbol="600519"):
    at.text_input(key="symbol").set_value(symbol)
    at.date_input(key="start").set_value(date(2024, 1, 1))
    at.date_input(key="end").set_value(date(2024, 6, 30))
    at.selectbox(key="mode").select("DeepSeek")
    at.checkbox(key="research").check()
    at.text_area(key="question").set_value(QUESTION)
    return at.button[0].click().run()


def check_saved(at, system, symbol="600519"):
    assert not at.exception and not at.error, [item.value for item in at.error]
    state = at.session_state["result"]
    expected = []
    for name in NODES:
        expected += [f"supervisor -> {name}", name]
    assert state["trace"] == expected + ["supervisor -> end"]
    assert state["symbol"] == symbol and state["rag_facts"]["filters"] == {"symbol": symbol}
    record = ReportStore(system.db, read_only=True).get_report(state["report_id"])
    assert record["report"] == state["report"]
    assert record["sources"]["rag"]["status"] == state["rag_facts"]["status"]
    assert record["sources"]["rag"]["references"] == state["rag_facts"]["sources"]
    return state, record


def check_evidence(state, system):
    rag = state["rag_facts"]
    assert rag["status"] == "ok" and rag["records"]
    source = {c["chunk_id"]: c for c in system.corpus.chunks}
    for hit in rag["records"]:
        assert hit["document_id"] == system.corpus.ids[state["symbol"]]
        assert hit["metadata"]["symbol"] == state["symbol"]
        original = source[hit["chunk_id"]]
        for key in ["text", "metadata", "page", "document_name", "document_id"]:
            assert hit[key] == original[key]
    sent = system.payloads[-1]["facts"]
    assert sent["rag"] == rag
    assert all("rag" not in p["facts"] for p in system.payloads[-5:-1])
    assert "TEST_ONLY_" not in system.payloads[-1]["system"]
    assert all(f"[{item['ref_id']}]" in state["report"] for item in rag["records"])
    evidence, _ = json.JSONDecoder().raw_decode(state["report"].split("事实数据：\n")[1])
    assert evidence["rag"] == rag


def test_complete_ui_to_history_with_real_bge(system):
    at = submit(app())
    state, record = check_saved(at, system)
    check_evidence(state, system)
    assert system.post.call_count == 5 and system.market.call_count == 1
    assert len(at.get("plotly_chart")) == 1
    at.run()
    at.button[0].click().run()
    assert system.post.call_count == 5 and len(ReportStore(system.db).list_reports()) == 1
    # 新会话从SQLite读回，不能只证明页面显示了内存缓存。
    history = app()
    history.text_input(key="history_symbol").set_value("600519").run()
    history.selectbox(key="report_id").select(record["id"]).run()
    assert any(item.value == record["report"] for item in history.text)
    assert system.market.call_count == 1 and system.post.call_count == 5
    system.corpus.cases.append({"scenario": "complete_ui_history", "rag_status": "ok", "mock_model_calls": 5,
        "hit_document_ids": sorted({h["document_id"] for h in state["rag_facts"]["records"]}), "report_roundtrip": True})


def test_two_stocks_no_document_or_history_crosstalk(system):
    at = app()
    ids = []
    for symbol in ["600519", "000001"]:
        submit(at, symbol)
        state, record = check_saved(at, system, symbol)
        check_evidence(state, system)
        ids.append(record["id"])
        assert system.corpus.ids[None] not in {h["document_id"] for h in state["rag_facts"]["records"]}
    assert ids[0] != ids[1] and system.post.call_count == 10
    history = app()
    for symbol, report_id in zip(["600519", "000001"], ids):
        history.text_input(key="history_symbol").set_value(symbol).run()
        options = history.selectbox(key="report_id").options
        assert len(options) == 1
        history.selectbox(key="report_id").select(report_id).run()
        saved = ReportStore(system.db).get_report(report_id)
        assert any(item.value == saved["report"] for item in history.text)
    assert system.post.call_count == 10 and system.market.call_count == 2
    system.corpus.cases.append({"scenario": "two_stock_isolation", "symbols": ["600519", "000001"],
                               "distinct_reports": True, "mock_model_calls": 10})


@pytest.mark.parametrize("failure", ["no_matching_stock", "corrupt_snapshot"])
def test_unavailable_rag_full_chain_no_fabricated_body(system, failure):
    symbol = "000858" if failure == "no_matching_stock" else "600519"
    expected = "empty" if failure == "no_matching_stock" else "index_error"
    if failure == "corrupt_snapshot":
        manifest = json.loads((system.index / "CURRENT.json").read_text())
        # 只损坏隔离副本，触发现有清单校验，不模拟检索函数抛异常。
        (system.index / (manifest["generation"] + ".faiss")).write_bytes(b"TEST ONLY CORRUPT INDEX")
    at = submit(app(), symbol)
    state, record = check_saved(at, system, symbol)
    assert state["rag_facts"]["status"] == expected
    assert state["rag_facts"]["records"] == record["sources"]["rag"]["references"] == []
    assert system.post.call_count == 4 and all("rag" not in p["facts"] for p in system.payloads)
    assert "正文问题没有可用原文，无法回答" in record["report"] and at.warning
    system.corpus.cases.append({"scenario": failure, "rag_status": expected, "mock_model_calls": 4,
                               "degraded_report_saved": True})


def test_research_api_failure_degrades_with_real_rag(system):
    for api in [system.finance, system.news, system.notices]:
        api.side_effect = requests.ConnectionError("TEST ONLY external source failure")
    at = submit(app())
    state, record = check_saved(at, system)
    assert state["financial_facts"]["status"] == "error"
    assert state["news_facts"]["news"]["status"] == state["news_facts"]["notices"]["status"] == "error"
    assert record["sources"]["financial"]["status"] == record["sources"]["news"]["status"] == record["sources"]["notices"]["status"] == "error"
    assert system.payloads[-1]["facts"]["rag"] == state["rag_facts"]
    assert state["rag_facts"]["status"] == "ok" and system.post.call_count == 3
    assert len(at.warning) >= 3
    system.corpus.cases.append({"scenario": "research_sources_failed", "rag_status": "ok", "mock_model_calls": 3,
                               "source_errors_saved": True})


def test_market_api_failure_stops_without_save_or_retry(system):
    system.market.side_effect = requests.ConnectionError("TEST ONLY market failure")
    at = submit(app())
    assert not at.exception and at.error and not at.success
    assert "result" not in at.session_state
    assert not system.db.exists()
    system.post.assert_not_called()
    for api in [system.finance, system.news, system.notices]:
        api.assert_not_called()
    at.run()
    assert system.market.call_count == 1
    assert "result" not in at.session_state and not system.db.exists()
    system.corpus.cases.append({"scenario": "market_failed", "mock_model_calls": 0, "no_report_saved": True,
                               "no_automatic_retry": True})
