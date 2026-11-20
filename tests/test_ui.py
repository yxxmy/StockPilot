"""真实Streamlit表单→原LangGraph→临时SQLite集成；所有样本仅为离线测试。"""

from datetime import date
from pathlib import Path
from unittest.mock import Mock
import os

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from agents import workflow, retrieval
from agents.deepseek import AgentError
from memory import MemoryStoreError, ReportStore
from ui import service
from ui.charts import market_chart


@pytest.fixture
def graph(monkeypatch):
    data = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=40, freq="B"),
        "open": [10.0] * 40, "high": [12.0] * 40, "low": [9.0] * 40,
        "close": [11.0] * 40, "volume": [100.0] * 40})
    loader = Mock(return_value=data)
    original = workflow.build_graph
    def build(client, **kwargs):
        return original(client, history_loader=loader,
            financial_loader=lambda *a: {"status": "empty", "records": [], "coverage": "仅测试缺失财务"},
            news_loader=lambda *a: {"status": "empty", "records": [], "coverage": "仅测试缺失新闻"},
            notice_loader=lambda *a: {"status": "empty", "records": [], "coverage": "仅测试缺失公告"}, **kwargs)
    monkeypatch.setattr(workflow, "build_graph", build)
    return loader


def app():
    return AppTest.from_file(str(Path(__file__).resolve().parents[1] / "streamlit_app.py"), default_timeout=30).run()


def submit(at):
    at.date_input(key="start").set_value(date(2024, 1, 1))
    at.date_input(key="end").set_value(date(2024, 6, 30))
    return at.button[0].click().run()


def test_initial_page_no_analysis_or_db(graph):
    at = app()
    assert not at.exception and not at.error
    graph.assert_not_called()
    assert not Path(os.environ["STOCKPILOT_DB_PATH"]).exists()
    assert len(at.tabs) == 4


def test_local_submit_rerun_and_duplicate_click_single_save(graph):
    at = submit(app())
    assert not at.exception and not at.error
    graph.assert_called_once()
    graph.assert_called_with("600519", "20230101", "2024-06-30")
    saved = at.session_state["result"]
    assert saved["trace"] == ["supervisor -> market", "market", "supervisor -> risk", "risk",
                               "supervisor -> report", "report", "supervisor -> end"]
    assert len(at.get("plotly_chart")) == 1
    assert len(at.metric) == 3
    at.run()
    at.text_input(key="history_symbol").set_value("600519").run()
    at.button[0].click().run()
    assert graph.call_count == 1
    assert len(ReportStore().list_reports()) == 1
    assert at.session_state["result"]["report_id"] == saved["report_id"]


def test_deepseek_only_explicit_submit_and_cache(graph, monkeypatch):
    client = Mock()
    client.describe.return_value = "仅测试模型解读，不是真实分析"
    factory = Mock(return_value=client)
    monkeypatch.setattr(workflow, "DeepSeekClient", factory)
    at = app()
    at.selectbox(key="mode").select("DeepSeek").run()
    factory.assert_not_called()
    at.selectbox(key="mode").select("DeepSeek")
    submit(at)
    assert not at.exception and not at.error, [item.value for item in at.error]
    assert client.describe.call_count == 3
    at.run()
    at.button[0].click().run()
    assert client.describe.call_count == 3 and factory.call_count == 1
    assert len(ReportStore().list_reports()) == 1


@pytest.mark.parametrize("status", ["ok", "empty", "model_error", "index_error"])
def test_rag_states_and_research_real_route(graph, monkeypatch, status):
    from rag.errors import EmbeddingError, IndexCorruptionError
    hit = {"document_id": "a" * 64, "chunk_id": "b" * 64, "document_name": "仅测试.pdf",
           "page": 2, "text": "仅测试PDF原文，不是正式财报。", "score": 0.8, "metadata": {"symbol": "600519"}}
    search = Mock(return_value=[hit] if status == "ok" else [])
    if status in ("model_error", "index_error"):
        search.side_effect = EmbeddingError("测试") if status == "model_error" else IndexCorruptionError("测试")
    monkeypatch.setattr(retrieval, "search_documents", search)
    at = app()
    at.checkbox(key="research").check()
    at.text_area(key="question").set_value("仅测试正文问题")
    submit(at)
    assert not at.exception and not at.error
    state = at.session_state["result"]
    assert state["rag_facts"]["status"] == status
    assert [t for t in state["trace"] if " -> " not in t] == ["market", "financial", "news", "risk", "rag_retrieval", "report"]
    if status == "ok":
        assert any("仅测试PDF原文" in item.value for item in at.text)
        assert any("物理第2页" in item.label for item in at.expander)
    else:
        assert at.warning
    at.run()
    search.assert_called_once()


@pytest.mark.parametrize("error", [AgentError("sk-test-secret-not-for-display"), MemoryStoreError("私密数据库路径"), RuntimeError("私密异常")])
def test_errors_are_friendly_no_automatic_retry_or_raw_config(monkeypatch, error):
    runner = Mock(side_effect=error)
    monkeypatch.setattr(workflow, "run_analysis", runner)
    at = submit(app())
    assert not at.exception and at.error
    assert "私密" not in at.error[0].value and "sk-test" not in at.error[0].value
    at.run()
    runner.assert_called_once()
    assert not at.session_state["running"]


@pytest.mark.parametrize("symbol,start,end", [("123", "20240101", "20240630"),
    ("600519", "20250101", "20240101"), ("600519", "20000101", "20240101")])
def test_invalid_request_no_workflow(monkeypatch, symbol, start, end):
    runner = Mock()
    monkeypatch.setattr(workflow, "run_analysis", runner)
    with pytest.raises(ValueError):
        service.analyze_once({}, symbol, start, end, "本地规则")
    runner.assert_not_called()


def test_chart_exact_values_and_missing_warmup(graph):
    session = {}
    service.analyze_once(session, "600519", "20240101", "20240630", "本地规则")
    data = session["result"]["history"]
    fig = market_chart(data)
    assert list(fig.data[0].close) == data["close"].tolist()
    assert list(fig.data[4].y) == data["volume"].tolist()
    assert pd.isna(fig.data[1].y[0])
    assert fig.data[0].type == "candlestick" and len(fig.data) == 9


def test_history_read_only_and_no_analysis(graph):
    state = {}
    service.analyze_once(state, "600519", "20240101", "20240630", "本地规则")
    path = Path(os.environ["STOCKPILOT_DB_PATH"])
    original = path.read_bytes()
    at = app()
    assert not at.exception and any(state["result"]["report"] == item.value for item in at.text)
    at.selectbox(key="report_id").select(state["result"]["report_id"]).run()
    graph.assert_called_once()
    assert path.read_bytes() == original


def test_history_corrupt_db_friendly(monkeypatch):
    Path(os.environ["STOCKPILOT_DB_PATH"]).write_bytes(b"not sqlite")
    at = app()
    assert not at.exception and any("历史报告暂时无法读取" in item.value for item in at.error)


def test_secret_not_in_frontend_or_inputs(monkeypatch):
    secret = "test-only-secret-not-real"
    monkeypatch.setenv("DEEPSEEK_API_KEY", secret)
    monkeypatch.setattr(service, "list_reports", lambda *a: [{"id": "test-id", "symbol": "600519", "generated_at": "测试", "market_end_date": "2024-01-01", "mode": "local"}])
    monkeypatch.setattr(service, "read_report", lambda *a: {"report": "仅测试：" + secret})
    at = app()
    assert not at.exception
    assert all(secret not in item.value for item in at.text)
    assert not any("key" in item.label.lower() for item in at.text_input)


def test_form_to_actual_history_adapter_date_contract(monkeypatch):
    from tools import stock_data
    # 仅替换真实HTTP边界，保留原行情参数、字段校验以及整张图，防止替身掩盖日期格式错误。
    raw = pd.DataFrame({"日期": pd.date_range("2024-01-02", periods=40, freq="B"),
        "开盘": [10.0] * 40, "最高": [12.0] * 40, "最低": [9.0] * 40,
        "收盘": [11.0] * 40, "成交量": [100.0] * 40})
    before = raw.iloc[:25].copy()
    before["日期"] = pd.bdate_range(end="2023-12-29", periods=25)
    before["收盘"] = 10.0
    api = Mock(return_value=pd.concat([before, raw], ignore_index=True))
    monkeypatch.setattr(stock_data.ak, "stock_zh_a_hist", api)
    at = submit(app())
    assert not at.error and not at.exception
    displayed = at.session_state["result"]["history"]
    assert len(displayed) == 40 and displayed.iloc[0]["date"] == pd.Timestamp("2024-01-02")
    assert displayed.iloc[0]["ma20"] == pytest.approx(10.05)
    assert displayed.iloc[0]["pct_change"] == pytest.approx(10)
    api.assert_called_once_with(symbol="600519", period="daily", start_date="20230101", end_date="20240630", adjust="")
    assert len(ReportStore().list_reports()) == 1
