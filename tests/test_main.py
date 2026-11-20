import pandas as pd
import pytest

import main
from tools.stock_data import EmptyDataError, StockDataError


def test_cli_success_and_defaults(monkeypatch, capsys):
    def fake_history(symbol, start, end):
        assert symbol == "600519"
        return pd.DataFrame({
            "date": pd.to_datetime(["2024-01-02"]),
            "open": [10.0], "high": [12.0], "low": [9.0], "close": [11.0], "volume": [100.0],
        })
    monkeypatch.setattr(main, "get_stock_history", fake_history)
    assert main.main([]) == 0
    output = capsys.readouterr().out
    for expected in ["前五行", "后五行", "2024-01-02 至 2024-01-02", "记录数：1"]:
        assert expected in output


@pytest.mark.parametrize("error", [ValueError("参数错误"), StockDataError("接口故障"), EmptyDataError("无数据")])
def test_cli_failure(monkeypatch, capsys, error):
    def fail(*args):
        raise error
    monkeypatch.setattr(main, "get_stock_history", fail)
    assert main.main(["600519", "--start-date", "20240101", "--end-date", "20240131"]) == 1
    captured = capsys.readouterr()
    assert "错误：" in captured.err
    assert captured.out == ""


def test_cli_indicators(monkeypatch, capsys):
    source = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=40, freq="B"),
        "open": [10.0] * 40, "high": [12.0] * 40, "low": [9.0] * 40,
        "close": [11.0] * 40, "volume": [100.0] * 40,
    })
    monkeypatch.setattr(main, "get_stock_history", lambda *args: source)
    assert main.main(["600519", "--indicators"]) == 0
    output = capsys.readouterr().out
    for column in ["ma5", "macd_dif", "macd_dea", "macd_hist", "rsi14", "pct_change", "volume_ratio"]:
        assert column in output
    assert "预热期为 NaN" in output
    assert "记录数：40" in output


def test_cli_indicator_validation_error(monkeypatch, capsys):
    monkeypatch.setattr(main, "get_stock_history", lambda *args: pd.DataFrame({"close": [10]}))
    assert main.main(["--indicators"]) == 1
    assert "错误：" in capsys.readouterr().err


@pytest.mark.parametrize("flag,local", [("--agent", False), ("--agent-local", True)])
def test_cli_agent_mode(monkeypatch, capsys, flag, local):
    from agents import workflow
    from unittest.mock import Mock
    run = Mock(return_value={"report": "测试报告", "trace": ["market", "risk", "report"]})
    monkeypatch.setattr(workflow, "run_analysis", run)
    assert main.main(["600519", "--start-date", "20240101", "--end-date", "20240630", flag]) == 0
    run.assert_called_once_with("600519", "20240101", "20240630", local=local)
    assert "测试报告" in capsys.readouterr().out


def test_cli_agent_failure(monkeypatch, capsys):
    from agents import workflow
    from agents.deepseek import AgentError
    from unittest.mock import Mock
    monkeypatch.setattr(workflow, "run_analysis", Mock(side_effect=AgentError("测试失败")))
    assert main.main(["--agent"]) == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert "测试失败" in output.err
