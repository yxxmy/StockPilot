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
    return next(button for button in at.sidebar.button if button.label == "开始分析").click().run()


def test_initial_page_no_analysis_or_db(graph):
    at = app()
    assert not at.exception and not at.error
    assert "**StockPilot**" in [item.value for item in at.markdown]
    assert "<h1>个股分析</h1>" in [item.proto.body for item in at.get("html")]
    graph.assert_not_called()
    assert not Path(os.environ["STOCKPILOT_DB_PATH"]).exists()
    assert len(at.tabs) == 4


def test_submit_stock_name_then_code_reuses_same_analysis(graph, monkeypatch):
    directory = Mock(return_value=[("平安银行", "000001")])
    monkeypatch.setattr(service, "load_a_share_names", directory)
    at = app()
    directory.assert_not_called()
    at.text_input(key="symbol").set_value(" 平安银行 ")
    submit(at)
    assert not at.error and not at.exception
    first = at.session_state["result"]
    assert first["symbol"] == "000001" and "平安银行（000001）" == at.session_state["last_attempt"]["symbol"]
    at.text_input(key="symbol").set_value("000001")
    submit(at)
    assert at.session_state["result"]["report_id"] == first["report_id"]
    assert len(ReportStore().list_reports()) == 1
    directory.assert_called_once()
    graph.assert_called_once()


@pytest.mark.parametrize(("query", "message"), [("相同简称", "候选"), ("相同", "候选"), ("不存在", "未找到")])
def test_unresolved_name_stops_before_analysis(graph, monkeypatch, query, message):
    directory = Mock(return_value=[("相同简称", "000001"), ("相同简称", "600001")])
    monkeypatch.setattr(service, "load_a_share_names", directory)
    at = app()
    at.text_input(key="symbol").set_value(query)
    at.selectbox(key="mode").select("DeepSeek")
    submit(at)
    assert not at.exception and message in at.error[0].value
    if message == "候选":
        assert all(code in at.error[0].value for code in ("000001", "600001"))
    at.run()
    directory.assert_called_once()
    graph.assert_not_called()
    assert not Path(os.environ["STOCKPILOT_DB_PATH"]).exists()


def test_directory_failure_allows_direct_code(graph, monkeypatch):
    import requests

    get = Mock(side_effect=requests.Timeout("private technical details"))
    monkeypatch.setattr("tools.stock_names.requests.get", get)
    at = app()
    at.text_input(key="symbol").set_value("平安银行")
    submit(at)
    assert "巨潮资讯" in at.error[0].value and "private" not in at.error[0].value
    graph.assert_not_called()
    at.text_input(key="symbol").set_value("000001")
    submit(at)
    assert not at.error and at.session_state["result"]["symbol"] == "000001"
    assert "<h1>000001</h1>" in [item.proto.body for item in at.get("html")]
    graph.assert_called_once()
    get.assert_called_once()


def test_directory_keeps_distinct_a_shares_and_leading_zeros(monkeypatch):
    from tools.stock_names import load_a_share_names

    stock = {"category": "A股", "zwjc": "仅测试", "code": "000001"}
    response = Mock()
    response.json.return_value = {"stockList": [stock, stock, dict(stock, category="B股", code="200001"),
                                               dict(stock, code="1"), dict(stock, code="600001")]}
    monkeypatch.setattr("tools.stock_names.requests.get", Mock(return_value=response))
    assert load_a_share_names() == [("仅测试", "000001"), ("仅测试", "600001")]


def test_local_submit_rerun_and_duplicate_click_single_save(graph):
    at = submit(app())
    assert not at.exception and not at.error
    graph.assert_called_once()
    graph.assert_called_with("600519", "20230101", "2024-06-30")
    saved = at.session_state["result"]
    assert "<h1>贵州茅台（600519）</h1>" in [item.proto.body for item in at.get("html")]
    assert saved["trace"] == ["supervisor -> market", "market", "supervisor -> risk", "risk",
                               "supervisor -> report", "report", "supervisor -> end"]
    market_tab, report_tab = at.tabs[:2]
    assert len(market_tab.get("plotly_chart")) == len(report_tab.get("plotly_chart")) == 1
    assert len(market_tab.metric) == 3
    assert [item.value for item in report_tab.metric] == ["11.00 元", "100 手", "0.00%"]
    trace = next(item for item in report_tab.expander if item.label == "查看实际执行轨迹")
    assert not trace.proto.expanded and "supervisor -> market" in trace.text[0].value
    saved.pop("stock_name")
    at.run()
    assert "<h1>贵州茅台（600519）</h1>" in [item.proto.body for item in at.get("html")]
    at.text_input(key="history_symbol").set_value("600519").run()
    submit(at)
    assert graph.call_count == 1
    assert len(ReportStore().list_reports()) == 1
    assert at.session_state["result"]["report_id"] == saved["report_id"]


def test_title_tracks_successful_stock_when_a_new_request_fails(graph):
    at = submit(app())
    at.text_input(key="symbol").set_value("000001")
    submit(at)
    assert "<h1>平安银行（000001）</h1>" in [item.proto.body for item in at.get("html")]
    at.text_input(key="symbol").set_value("999")
    submit(at)
    assert at.error and "<h1>平安银行（000001）</h1>" in [item.proto.body for item in at.get("html")]
    assert graph.call_count == 2


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
    submit(at)
    assert client.describe.call_count == 3 and factory.call_count == 1
    assert len(ReportStore().list_reports()) == 1


@pytest.mark.parametrize("status", ["ok", "empty", "model_error", "index_error"])
def test_rag_states_and_research_real_route(graph, monkeypatch, status):
    from rag.errors import EmbeddingError, IndexCorruptionError
    official_url = "https://static.cninfo.com.cn/finalpage/2024-06-12/1220324741.PDF"
    hit = {"document_id": "a" * 64, "chunk_id": "b" * 64, "document_name": "仅测试.pdf",
           "page": 2, "text": "仅测试PDF原文，不是正式财报。", "score": 0.8, "metadata": {"symbol": "600519"},
           "official_pdf_url": official_url}
    second_hit = {**hit, "chunk_id": "c" * 64}
    search = Mock(return_value=[hit, second_hit] if status == "ok" else [])
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
        for tab in (at.tabs[1], at.tabs[2], at.tabs[3]):
            assert sum(f"{official_url}#page=2" in item.value for item in tab.markdown) == 2
        assert ReportStore().get_report(state["report_id"])["sources"]["rag"]["references"][0]["official_pdf_url"] == official_url
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
    from agents.report_text import EVIDENCE_DIVIDER
    body, evidence = state["result"]["report"].split(EVIDENCE_DIVIDER)
    assert not at.exception and any(body == item.value for item in at.text)
    history = at.tabs[3]
    assert [item.value for item in history.metric] == ["11.00 元", "100 手", "0.00%"]
    assert not history.get("plotly_chart")
    appendix = next(item for item in history.expander if item.label == "查看原始数据（核对用）")
    assert not appendix.proto.expanded
    assert any(evidence == item.value for item in appendix.text)
    at.selectbox(key="report_id").select(state["result"]["report_id"]).run()
    graph.assert_called_once()
    assert path.read_bytes() == original


def test_history_corrupt_db_friendly(monkeypatch):
    Path(os.environ["STOCKPILOT_DB_PATH"]).write_bytes(b"not sqlite")
    at = app()
    assert not at.exception and any("历史报告暂时无法读取" in item.value for item in at.error)


def test_legacy_report_visible_in_cached_result_and_history(graph, monkeypatch):
    import json
    from agents.report_text import EVIDENCE_DIVIDER

    at = submit(app())
    saved = at.session_state["result"]
    legacy_facts = json.loads(saved["report"].split(EVIDENCE_DIVIDER)[1])
    legacy_facts["market"].pop("source")
    evidence = json.dumps(legacy_facts, ensure_ascii=False)
    legacy = ("StockPilot 600519\n本地规则输出（未调用大模型）\n\n事实数据：\n" + evidence
              + "\n\nMarket Agent：\n本地规则：行情和指标已计算，数值见 market_facts。"
              + "\n\nReport Agent：\n测试说明 [R1]")
    saved["report"] = legacy
    monkeypatch.setattr(service, "read_report", lambda report_id: {"report": legacy})
    at.run()
    assert not at.exception
    for tab in (at.tabs[1], at.tabs[3]):
        assert tab.metric[0].value == "11.00 元"
        full_report = next(item for item in tab.expander if item.label == "查看完整报告")
        assert not full_report.proto.expanded
        visible_body = full_report.text[0].value
        assert "11.00 元" in visible_body and "测试说明 [R1]" in visible_body
        assert "报告未记录行情来源" in visible_body
        assert '"market"' not in visible_body and "market_facts" not in visible_body
        appendix = next(item for item in tab.expander if item.label == "查看原始数据（核对用）")
        assert not appendix.proto.expanded and appendix.text[0].value == evidence
    graph.assert_called_once()


def test_incomplete_legacy_report_preserves_original_in_details():
    from agents.report_text import report_for_display

    value = 'StockPilot 600519\n\n事实数据：\n{"market": {}}'
    body, details = report_for_display(value)
    assert "数据不完整" in body and "{" not in body and details == value


def test_report_with_single_session_keeps_unavailable_data_visible(graph):
    graph.return_value = graph.return_value.iloc[:1].copy()
    at = submit(app())
    assert not at.error and not at.exception
    summary = at.tabs[1]
    assert [item.value for item in summary.metric] == ["11.00 元", "100 手", "暂无数据"]
    assert any("20 日均线暂缺" in item.value for item in summary.caption)
    assert any("前 5 个交易日均量" in item.value for item in summary.caption)


def test_report_loads_industry_only_on_click_and_offers_independent_downloads(graph, monkeypatch):
    from ui import report_view
    from tools.market_hotspots import PERIOD_RETURNS

    end = pd.Timestamp(graph.return_value["date"].iloc[-1])
    dates = list(pd.bdate_range(end=end, periods=21).date)
    quotes = pd.DataFrame({"行业": ["白酒", "银行"], PERIOD_RETURNS[5]: [2.0, 1.0], PERIOD_RETURNS[20]: [3.0, 4.0]})
    load = Mock(return_value={"dates": dates, "quotes": quotes})
    monkeypatch.setattr(report_view, "industry_snapshot", load)
    monkeypatch.setattr(report_view, "match_current_industry", lambda *_: None)
    at = submit(app())
    load.assert_not_called()
    assert len(at.get("download_button")) == 2
    at.button(key="load_report_industry_current").click().run()
    select = next(item for item in at.selectbox if item.label == "对照行业")
    at.selectbox(key=select.key).select("白酒").run()
    assert not at.exception
    assert any("+2.00% · 第 1 / 2 名" in item.value for item in at.tabs[1].text)
    assert all("第 1 / 2 名" not in item.value for item in at.tabs[3].text)
    load.assert_called_once_with(end.date(), {})
    graph.assert_called_once()


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
