"""验证持久化只接在成功出口，历史查询不访问网络或模型。"""

import os
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest

import main
from agents import workflow
from agents.deepseek import AgentError
from memory import MemoryStoreError, ReportStore


def inject_graph(monkeypatch, *, failure=False):
    frame = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=40, freq="B"),
                          "close": [11.0] * 40, "volume": [100.0] * 40})
    loader = Mock(side_effect=ValueError("测试失败")) if failure else Mock(return_value=frame)
    original = workflow.build_graph
    monkeypatch.setattr(workflow, "build_graph", lambda client, **kwargs: original(client, history_loader=loader, **kwargs))


def test_run_analysis_persists_each_success(monkeypatch):
    inject_graph(monkeypatch)
    first = workflow.run_analysis("600519", "20240101", "20240630", local=True)
    second = workflow.run_analysis("600519", "20240101", "20240630", local=True)
    assert first["report_id"] != second["report_id"]
    rows = ReportStore().list_reports("600519")
    assert len(rows) == 2
    assert ReportStore().get_report(first["report_id"])["report"] == first["report"]


def test_failed_workflow_never_initializes_database(monkeypatch):
    inject_graph(monkeypatch, failure=True)
    with pytest.raises(AgentError):
        workflow.run_analysis("600519", "20240101", "20240630", local=True)
    assert not Path(os.environ["STOCKPILOT_DB_PATH"]).exists()


def test_missing_key_never_initializes_database(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(AgentError):
        workflow.run_analysis("600519", "20240101", "20240630")
    assert not Path(os.environ["STOCKPILOT_DB_PATH"]).exists()


def test_cli_save_and_history_round_trip(monkeypatch, capsys, tmp_path):
    inject_graph(monkeypatch)
    path = tmp_path / "chosen.sqlite3"
    assert main.main(["600519", "--agent-local", "--start-date", "20240101", "--end-date", "20240630", "--db", str(path)]) == 0
    assert "报告已保存" in capsys.readouterr().out
    row = ReportStore(path).list_reports()[0]
    # 禁止再运行图，证明查询路径不依赖模型/行情。
    monkeypatch.setattr(workflow, "run_analysis", Mock(side_effect=AssertionError("不能运行分析")))
    assert main.main(["--history", "--db", str(path)]) == 0
    assert row["id"] in capsys.readouterr().out
    assert main.main(["600519", "--history", "--db", str(path)]) == 0
    assert row["id"] in capsys.readouterr().out
    assert main.main(["000001", "--history", "--db", str(path)]) == 0
    assert "暂无历史报告" in capsys.readouterr().out
    assert main.main(["--report-id", row["id"], "--db", str(path)]) == 0
    assert ReportStore(path).get_report(row["id"])["report"] in capsys.readouterr().out


def test_cli_empty_and_unknown_id(capsys):
    assert main.main(["--history"]) == 0
    assert "暂无历史报告" in capsys.readouterr().out
    assert main.main(["--report-id", "missing"]) == 1
    assert "未找到" in capsys.readouterr().err


def test_cli_database_error_does_not_claim_success(monkeypatch, capsys, tmp_path):
    inject_graph(monkeypatch)
    bad = tmp_path / "directory"
    bad.mkdir()
    assert main.main(["--agent-local", "--db", str(bad)]) == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert "SQLite 操作失败" in output.err
    assert main.main(["--history", "--db", str(bad)]) == 1
    assert "SQLite 操作失败" in capsys.readouterr().err


@pytest.mark.parametrize("args", [["--history", "--agent"], ["--history", "--indicators"], ["--history", "--research"], ["--history", "--report-id", "id"], ["600519", "--report-id", "id"]])
def test_cli_rejects_conflicting_modes(args):
    with pytest.raises(SystemExit) as error:
        main.main(args)
    assert error.value.code == 2


def test_legacy_history_and_indicator_commands_do_not_create_database(monkeypatch, capsys):
    frame = pd.DataFrame({"date": pd.to_datetime(["2024-01-01"]), "close": [10.0], "volume": [100.0]})
    monkeypatch.setattr(main, "get_stock_history", Mock(return_value=frame))
    assert main.main([]) == 0
    assert main.main(["--indicators"]) == 0
    assert not Path(os.environ["STOCKPILOT_DB_PATH"]).exists()


@pytest.mark.parametrize("failure_index", [0, 1, 2, 3, 4])
def test_each_research_model_failure_never_saves(monkeypatch, failure_index):
    inject_graph(monkeypatch)
    valid = {"status": "ok", "records": [{"title": "离线测试材料"}]}
    for name in ["get_stock_financials", "get_stock_news", "get_stock_notices"]:
        monkeypatch.setattr(workflow, name, Mock(return_value=valid))
    client = Mock()
    client.describe.side_effect = ["测试解读"] * failure_index + [AgentError("测试模型失败")]
    monkeypatch.setattr(workflow, "DeepSeekClient", Mock(return_value=client))
    with pytest.raises(AgentError, match="测试模型失败"):
        workflow.run_analysis("600519", "20240101", "20240630", include_research=True)
    assert client.describe.call_count == failure_index + 1
    assert not Path(os.environ["STOCKPILOT_DB_PATH"]).exists()
