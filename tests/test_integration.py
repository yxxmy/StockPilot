"""端到端离线测试：真实入口、行情转换、指标、LangGraph 和 HTTP 客户端。

只在外部边界替换 AKShare 返回值和 HTTP 响应，测试行情不是真实数据。
"""

import json
from unittest.mock import Mock

import pandas as pd
import pytest
import requests

import main
from tools import stock_data


@pytest.fixture
def external_services(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "integration-test-key")
    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)
    # 原始中文列走完整转换流程，不绕过生产环境的数据校验。
    frame = pd.DataFrame({
        "日期": pd.date_range("2024-01-01", periods=40, freq="B").date,
        "开盘": ["10"] * 40, "最高": ["12"] * 40, "最低": ["9"] * 40,
        "收盘": ["11"] * 40, "成交量": ["100"] * 40,
    })
    api = Mock(return_value=frame)
    monkeypatch.setattr(stock_data.ak, "stock_zh_a_hist", api)

    def fake_post(url, **kwargs):
        facts = json.loads(kwargs["json"]["messages"][1]["content"])
        # 只是明确的测试响应，不提供或模拟投资分析。
        content = json.dumps({"summary": "离线集成测试响应", "evidence": [next(iter(facts))]})
        return Mock(status_code=200, json=Mock(return_value={"choices": [
            {"finish_reason": "stop", "message": {"content": content}},
        ]}))

    post = Mock(side_effect=fake_post)
    monkeypatch.setattr(requests, "post", post)
    return api, post


def command(mode):
    return ["600519", "--start-date", "20240101", "--end-date", "20240630", mode]


@pytest.mark.parametrize("mode,expected_calls", [("--agent", 3), ("--agent-local", 0)])
def test_full_cli_graph_and_client(external_services, capsys, mode, expected_calls):
    api, post = external_services
    assert main.main(command(mode)) == 0
    output = capsys.readouterr()
    assert output.err == ""
    assert '"record_count": 40' in output.out
    assert '"close": 11.0' in output.out
    assert '"max_drawdown_pct": 0.0' in output.out
    assert "supervisor -> market | market | supervisor -> risk | risk | supervisor -> report | report | supervisor -> end" in output.out
    assert "integration-test-key" not in output.out
    api.assert_called_once()
    assert post.call_count == expected_calls
    if mode == "--agent":
        for call in post.call_args_list:
            assert call.args[0] == "https://api.deepseek.com/chat/completions"
        assert "离线集成测试响应" in output.out
    else:
        assert "未调用大模型" in output.out
        assert "离线集成测试响应" not in output.out


@pytest.mark.parametrize("failure_index", [0, 1, 2])
def test_full_cli_stops_after_model_failure(external_services, capsys, failure_index):
    _, post = external_services
    normal_response = post.side_effect
    calls = 0

    def fail_at_node(*args, **kwargs):
        nonlocal calls
        current = calls
        calls += 1
        if current == failure_index:
            return Mock(status_code=429)
        return normal_response(*args, **kwargs)

    post.side_effect = fail_at_node
    assert main.main(command("--agent")) == 1
    output = capsys.readouterr()
    assert output.out == ""  # 不能把部分节点完成当作完整分析成功。
    assert "HTTP 429" in output.err
    assert "integration-test-key" not in output.err
    assert post.call_count == failure_index + 1


def test_full_cli_empty_market_never_calls_model(external_services, capsys):
    api, post = external_services
    api.return_value = pd.DataFrame()
    assert main.main(command("--agent")) == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert "没有行情" in output.err
    post.assert_not_called()


@pytest.mark.parametrize("symbol,start,end", [
    ("sh600519", "20240101", "20240630"),
    ("600519", "20240230", "20240630"),
    ("600519", "20240630", "20240101"),
])
def test_full_cli_invalid_arguments_never_call_external_services(external_services, capsys, symbol, start, end):
    api, post = external_services
    assert main.main([symbol, "--start-date", start, "--end-date", end, "--agent"]) == 1
    assert capsys.readouterr().out == ""
    api.assert_not_called()
    post.assert_not_called()


def test_agent_modes_are_mutually_exclusive(external_services, capsys):
    api, post = external_services
    with pytest.raises(SystemExit) as error:
        main.main(["--agent", "--agent-local"])
    assert error.value.code == 2
    api.assert_not_called()
    post.assert_not_called()
    assert capsys.readouterr().out == ""
