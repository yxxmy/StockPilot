"""主备行情均为明确离线样本；不访问真实接口。"""

from unittest.mock import Mock

import pandas as pd
import pytest
import requests

from tools import stock_data
from agents.workflow import run_analysis
from agents.diagnostics import diagnostic
from memory import ReportStore
from test_ui import graph, app, submit  # noqa: F401；复用真实页面/工作流夹具。


def backup_sample():
    return pd.DataFrame({"date": ["2024-01-03", "2024-01-02"], "open": [10., 10.],
        "high": [12., 12.], "low": [9., 9.], "close": [11., 11.], "volume": [30000., 20000.]})


def test_network_failure_uses_real_adapter_once_and_preserves_source(monkeypatch):
    first = Mock(side_effect=requests.ConnectionError("TEST ONLY"))
    backup = Mock(return_value=backup_sample())
    monkeypatch.setattr(stock_data.ak, "stock_zh_a_hist", first)
    monkeypatch.setattr(stock_data.ak, "stock_zh_a_hist_tx", backup)
    result = run_analysis("600519", "20240101", "20240131", local=True)
    data = result["history"]
    assert data.volume.tolist() == [200., 300.]
    assert data.attrs["source"] == "AKShare / 腾讯证券"
    assert result["market_facts"]["source"] == data.attrs["source"]
    assert "AKShare / 腾讯证券" in result["report"]
    assert "来源 AKShare / 东方财富" not in result["report"]
    stored = ReportStore().get_report(result["report_id"])
    assert stored["sources"]["market"]["source"] == data.attrs["source"]
    assert stored["sources"]["market"]["source_warning"]
    first.assert_called_once()
    backup.assert_called_once_with(symbol="600519", start_date="20240101", end_date="20240131", adjust="")


@pytest.mark.parametrize("failure,category", [
    (requests.Timeout("secret"), "market_timeout"),
    (requests.ConnectionError("secret"), "market_connection"),
])
def test_both_sources_fail_without_retry(monkeypatch, failure, category):
    primary, backup = Mock(side_effect=failure), Mock(side_effect=requests.ConnectionError("secret"))
    monkeypatch.setattr(stock_data.ak, "stock_zh_a_hist", primary)
    monkeypatch.setattr(stock_data.ak, "stock_zh_a_hist_tx", backup)
    with pytest.raises(stock_data.StockDataError) as caught:
        stock_data.get_stock_history("600519", "20240101", "20240131")
    assert caught.value.__cause__ is failure
    assert caught.value.category == category
    assert "secret" not in str(caught.value)
    primary.assert_called_once()
    backup.assert_called_once()


def test_suspected_rate_limit_is_not_retried(monkeypatch):
    from akshare.utils.http import Push2RateLimitError
    monkeypatch.setattr(stock_data.ak, "stock_zh_a_hist", Mock(side_effect=Push2RateLimitError("TEST ONLY")))
    with pytest.raises(stock_data.StockDataError) as caught:
        stock_data.get_stock_history("600519", "20240101", "20240131")
    assert caught.value.category == "market_rate_limit"
    assert "疑似限流" in diagnostic("market", caught.value.category, [])["message"]


@pytest.mark.parametrize("bad", [pd.DataFrame(), backup_sample().drop(columns="volume")])
def test_backup_missing_data_does_not_save(monkeypatch, bad):
    monkeypatch.setattr(stock_data.ak, "stock_zh_a_hist", Mock(side_effect=requests.Timeout()))
    monkeypatch.setattr(stock_data.ak, "stock_zh_a_hist_tx", Mock(return_value=bad))
    from agents.deepseek import AgentError
    with pytest.raises(AgentError):
        run_analysis("600519", "20240101", "20240131", local=True)
    assert ReportStore().list_reports() == []


def test_primary_validation_failure_does_not_silently_switch_source(monkeypatch):
    primary = backup_sample().rename(columns={v:k for k,v in stock_data.FIELD_MAP.items()}).drop(columns="最高")
    monkeypatch.setattr(stock_data.ak, "stock_zh_a_hist", Mock(return_value=primary))
    backup = Mock(return_value=backup_sample())
    monkeypatch.setattr(stock_data.ak, "stock_zh_a_hist_tx", backup)
    with pytest.raises(stock_data.DataValidationError):
        stock_data.get_stock_history("600519", "20240101", "20240131")
    backup.assert_not_called()


def test_page_attempt_is_distinct_and_rerun_does_not_resubmit(graph):
    at = submit(app())
    first = dict(at.session_state["last_attempt"])
    assert first["status"] == "完成" and first["elapsed"] >= 0
    count = graph.call_count
    at.run()
    assert dict(at.session_state["last_attempt"]) == first and graph.call_count == count
    submit(at)
    assert at.session_state["last_attempt"]["id"] != first["id"]
    assert at.session_state["last_attempt"]["status"] == "复用成功结果"
    assert graph.call_count == count


def test_failed_attempt_keeps_time_and_never_retries_on_rerun(graph):
    graph.side_effect = stock_data.StockDataError("TEST ONLY")
    at = submit(app())
    first = dict(at.session_state["last_attempt"])
    assert first["status"] == "失败" and first["elapsed"] >= 0 and first["time"]
    assert at.error and not at.exception
    assert "result" not in at.session_state
    count = graph.call_count
    at.run()
    assert at.session_state["last_attempt"] == first and graph.call_count == count
    assert ReportStore().list_reports() == []
