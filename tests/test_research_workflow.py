"""研究节点的协作、缺失源和失败路由；所有材料为离线测试数据。"""

import json
from unittest.mock import Mock

import pandas as pd
import pytest

import main
from agents.report_text import report_for_display, EVIDENCE_DIVIDER
from agents.workflow import build_graph
from agents.deepseek import AgentError
from tools import research_data as rd
from tools.stock_data import StockDataError


def state():
    return {"symbol": "600519", "start_date": "20240101", "end_date": "20240630", "trace": []}


def graph(client=None, *, finance=None, news=None, notices=None):
    data = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=40, freq="B"),
                         "close": [10.0] * 40, "volume": [100.0] * 40})
    good = {"status": "ok", "records": [{"title": "离线测试材料"}]}
    return build_graph(client, Mock(return_value=data), include_research=True,
                       financial_loader=finance or Mock(return_value=good),
                       news_loader=news or Mock(return_value=good),
                       notice_loader=notices or Mock(return_value=good))


def test_research_local_order_and_state():
    result = graph().invoke(state())
    assert [step for step in result["trace"] if " -> " not in step] == ["market", "financial", "news", "risk", "report"]
    assert result["risk_facts"]["research_context"]["financial"] == result["financial_facts"]
    assert result["risk_facts"]["research_context"]["news"] == result["news_facts"]
    text = result["report"]
    assert "Financial Agent" not in text and "News Agent" not in text
    assert "未生成模型解读" not in text
    assert "未获取财务、公告、新闻" not in text
    parsed, _ = json.JSONDecoder().raw_decode(text.split("事实数据：\n")[1])
    assert parsed["financial"] == result["financial_facts"]
    assert parsed["news"] == result["news_facts"]


@pytest.mark.parametrize("legacy", [True, False])
def test_saved_research_status_hidden_without_losing_commentary_or_evidence(legacy):
    new_text = graph().invoke(state())["report"]
    body, evidence = new_text.split(EVIDENCE_DIVIDER)
    tail = ("\n\nFinancial Agent：\n财务数据状态：ok；4条记录，未生成模型解读。"
            "\n\nNews Agent：\n新闻状态：ok；公告状态：ok；未生成模型解读，空结果不代表无事件。"
            "\n\nNews Agent：\n测试公告提及股东大会 [R1]")
    saved = ("StockPilot 600519\n\n事实数据：\n" + evidence + tail
             if legacy else body + tail + EVIDENCE_DIVIDER + evidence)
    visible, details = report_for_display(saved)
    assert "未生成模型解读" not in visible and "Financial Agent" not in visible
    assert "测试公告提及股东大会 [R1]" in visible and details == evidence


def test_five_agents_share_research_with_report():
    client = Mock()
    client.describe.side_effect = ["行情测试", "财务测试", "新闻测试", "风险测试", "报告测试"]
    result = graph(client).invoke(state())
    calls = client.describe.call_args_list
    assert [c.args[0] for c in calls] == ["Market Agent", "Financial Agent", "News Agent", "Risk Agent", "Report Agent"]
    facts = calls[-1].args[1]
    assert facts["financial_commentary"] == "财务测试"
    assert facts["news_commentary"] == "新闻测试"
    assert "财务解读\n财务测试" in result["report"] and "新闻与公告解读\n新闻测试" in result["report"]
    assert facts["financial"] == result["financial_facts"]
    assert facts["news"] == result["news_facts"]


def test_all_research_unavailable_skips_two_model_calls():
    client = Mock()
    client.describe.return_value = "测试解读"
    result = graph(client, finance=Mock(side_effect=StockDataError("上游失败")),
                   news=Mock(return_value={"status": "empty", "records": []}),
                   notices=Mock(side_effect=StockDataError("公告失败"))).invoke(state())
    assert result["financial_facts"]["status"] == "error"
    assert result["news_facts"]["news"]["status"] == "empty"
    assert result["news_facts"]["notices"]["status"] == "error"
    assert [call.args[0] for call in client.describe.call_args_list] == ["Market Agent", "Risk Agent", "Report Agent"]
    assert '"status": "error"' in result["report"]
    assert "未生成模型解读" not in result["report"]
    assert "财务解读\n" not in result["report"] and "新闻与公告解读\n" not in result["report"]


def test_news_failure_keeps_real_notice_material():
    client = Mock()
    client.describe.return_value = "测试解读"
    result = graph(client, news=Mock(side_effect=StockDataError("新闻失败")) ).invoke(state())
    assert result["news_facts"]["news"]["status"] == "error"
    assert result["news_facts"]["notices"]["status"] == "ok"
    assert client.describe.call_count == 5


@pytest.mark.parametrize("index,role", [(1, "Financial"), (2, "News")])
def test_research_model_failure_stops_graph(index, role):
    client = Mock()
    client.describe.side_effect = ["测试解读"] * index + [AgentError("模型失败")]
    result = graph(client).invoke(state())
    assert result["error"].startswith(role + " Agent 失败")
    assert "report" not in result
    assert client.describe.call_count == index + 1
    assert result["trace"][-1] == "supervisor -> end"


@pytest.mark.parametrize("mode,calls", [("--agent", 5), ("--agent-local", 0)])
def test_full_research_cli(monkeypatch, capsys, mode, calls):
    import requests
    monkeypatch.setenv("DEEPSEEK_API_KEY", "research-test-key")
    history = pd.DataFrame({"日期": pd.date_range("2024-01-01", periods=40, freq="B"),
                            "开盘": [10] * 40, "最高": [12] * 40, "最低": [9] * 40,
                            "收盘": [11] * 40, "成交量": [100] * 40})
    monkeypatch.setattr(rd.ak, "stock_zh_a_hist", Mock(return_value=history))
    finance = pd.DataFrame({"SECURITY_CODE": ["600519"], "REPORT_DATE": ["2024-03-31"],
                            "NOTICE_DATE": ["2024-04-28"], "CURRENCY": ["CNY"],
                            **{field: [10.0] for field in rd.FINANCIAL_FIELDS}})
    monkeypatch.setattr(rd.ak, "stock_financial_analysis_indicator_em", Mock(return_value=finance))
    monkeypatch.setattr(rd.ak, "stock_news_em", Mock(return_value=pd.DataFrame()))
    notices = pd.DataFrame({"代码": ["600519"], "公告标题": ["测试公告"], "公告时间": ["2024-05-01"], "公告链接": ["https://example.com/1"]})
    monkeypatch.setattr(rd.ak, "stock_zh_a_disclosure_report_cninfo", Mock(return_value=notices))

    def respond(*args, **kwargs):
        facts = json.loads(kwargs["json"]["messages"][1]["content"])
        content = json.dumps({"summary": "离线测试解读", "evidence": [next(iter(facts))]})
        return Mock(status_code=200, json=Mock(return_value={"choices": [
            {"finish_reason": "stop", "message": {"content": content}},
        ]}))

    post = Mock(side_effect=respond)
    monkeypatch.setattr(requests, "post", post)
    assert main.main(["600519", "--start-date", "20240101", "--end-date", "20240630", mode, "--research"]) == 0
    output = capsys.readouterr()
    assert output.err == ""
    assert ("财务解读\n" in output.out) == bool(calls)
    assert ("新闻与公告解读\n" in output.out) == bool(calls)
    assert "未生成模型解读" not in output.out
    assert "research-test-key" not in output.out
    assert post.call_count == calls


def test_research_requires_agent_mode():
    with pytest.raises(SystemExit) as exc:
        main.main(["--research"])
    assert exc.value.code == 2
