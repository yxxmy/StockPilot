"""注入行情和模型替身，离线验证实际 LangGraph 的执行与失败分支。"""

from unittest.mock import Mock

import pandas as pd
import pytest

from agents.deepseek import AgentError
from agents import workflow
from agents.workflow import build_graph, risk_facts
from tools.indicators import calculate_indicators
from tools.stock_data import EmptyDataError


def history(closes=None):
    closes = closes if closes is not None else [10 + i % 7 for i in range(40)]
    return pd.DataFrame({"date": pd.date_range("2024-01-01", periods=len(closes), freq="B"),
                         "close": closes, "volume": [100.0] * len(closes)})


def initial():
    return {"symbol": "600519", "start_date": "20240101", "end_date": "20240630", "trace": []}


def test_local_graph_order_and_shared_state():
    loader = Mock(return_value=history())
    result = build_graph(history_loader=loader).invoke(initial())
    loader.assert_called_once_with("600519", "20240101", "20240630")
    assert result["trace"] == ["supervisor -> market", "market", "supervisor -> risk", "risk",
                               "supervisor -> report", "report", "supervisor -> end"]
    assert result["market_facts"]["record_count"] == 40
    assert result["market_facts"]["actual_range"] == ["2024-01-01", "2024-02-23"]
    assert "未调用大模型" in result["report"]
    assert "事实数据" in result["report"]
    assert "error" not in result


def test_model_agents_receive_preceding_results():
    client = Mock()
    client.describe.side_effect = ["测试行情解读", "测试风险解读", "测试综合解读"]
    result = build_graph(client, Mock(return_value=history())).invoke(initial())
    calls = client.describe.call_args_list
    assert [call.args[0] for call in calls] == ["Market Agent", "Risk Agent", "Report Agent"]
    assert calls[2].args[1]["market_commentary"] == "测试行情解读"
    assert calls[2].args[1]["risk_commentary"] == "测试风险解读"
    assert calls[2].args[1]["market"] == result["market_facts"]
    assert "DeepSeek 模型解读" in result["report"]


@pytest.mark.parametrize("error", [EmptyDataError("无数据"), ValueError("数据非法")])
def test_market_data_failure_stops_graph(error):
    client = Mock()
    result = build_graph(client, Mock(side_effect=error)).invoke(initial())
    assert result["trace"] == ["supervisor -> market", "market: error", "supervisor -> end"]
    assert "Market Agent 失败" in result["error"]
    assert "report" not in result
    client.describe.assert_not_called()


@pytest.mark.parametrize("failure_index,node", [(0, "Market"), (1, "Risk"), (2, "Report")])
def test_model_error_stops_at_current_node(failure_index, node):
    client = Mock()
    client.describe.side_effect = ["测试解读"] * failure_index + [AgentError("测试 API 故障")]
    result = build_graph(client, Mock(return_value=history())).invoke(initial())
    assert f"{node} Agent 失败" in result["error"]
    assert "report" not in result
    assert client.describe.call_count == failure_index + 1
    assert result["trace"][-1] == "supervisor -> end"


def test_risk_formulas_and_small_sample():
    data = calculate_indicators(history([100, 80, 120, 90]))
    facts = risk_facts(data)
    assert facts["sample_return_pct"] == pytest.approx(-10)
    assert facts["max_drawdown_pct"] == pytest.approx(-25)
    # 收益为 -0.2、0.5、-0.25，样本标准差独立计算。
    mean = (-0.2 + 0.5 - 0.25) / 3
    std = (sum((x - mean) ** 2 for x in [-0.2, 0.5, -0.25]) / 2) ** 0.5
    assert facts["annualized_volatility_pct"] == pytest.approx(std * 252 ** 0.5 * 100)
    one = risk_facts(calculate_indicators(history([100])))
    assert one["annualized_volatility_pct"] is None
    assert one["max_drawdown_pct"] == 0
    assert one["flags"]


def test_short_sample_keeps_null_and_zero_volume_flag():
    data = history([10])
    data["volume"] = 0
    result = build_graph(history_loader=Mock(return_value=data)).invoke(initial())
    assert result["market_facts"]["latest"]["rsi14"] is None
    assert "rsi14" in result["market_facts"]["missing_indicators"]
    assert any("零成交量" in flag for flag in result["risk_facts"]["flags"])
    assert "NaN" not in result["report"]


def test_missing_key_fails_before_loading_market(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    graph = Mock()
    monkeypatch.setattr(workflow, "build_graph", graph)
    with pytest.raises(AgentError, match="DEEPSEEK_API_KEY"):
        workflow.run_analysis("600519", "20240101", "20240630")
    graph.assert_not_called()


def test_run_analysis_propagates_workflow_error(monkeypatch):
    graph = Mock()
    graph.invoke.return_value = {"error": "Market Agent 失败：无数据"}
    monkeypatch.setattr(workflow, "build_graph", Mock(return_value=graph))
    with pytest.raises(AgentError, match="无数据"):
        workflow.run_analysis("600519", "20240101", "20240630", local=True)
