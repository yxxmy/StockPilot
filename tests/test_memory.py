"""SQLite离线测试：报告数据仅为测试用途，数据库均位于tmp_path。"""

from copy import deepcopy
from datetime import datetime
import json
import sqlite3
from unittest.mock import Mock
from uuid import UUID

import pandas as pd
import pytest

from agents.workflow import build_graph
from memory import MemoryStoreError, ReportStore


@pytest.fixture
def completed():
    frame = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=40, freq="B"),
                          "close": [11.0] * 40, "volume": [100.0] * 40})
    return build_graph(history_loader=Mock(return_value=frame)).invoke({
        "symbol": "600519", "start_date": "20240101", "end_date": "20240630", "trace": [],
    })


def test_persist_and_reopen(tmp_path, completed):
    path = tmp_path / "nested" / "reports.sqlite3"
    store = ReportStore(path)
    report_id = store.save_report(completed, mode="local")
    assert UUID(report_id).version == 4
    row = ReportStore(path).get_report(report_id)
    assert row["report"] == completed["report"]
    assert row["symbol"] == "600519"
    assert row["market_end_date"] == "2024-02-23"  # 实际日期，不是查询截止日。
    assert datetime.fromisoformat(row["generated_at"]).utcoffset().total_seconds() == 8 * 3600
    assert row["mode"] == "local" and row["include_research"] is False
    assert row["sources"]["market"]["actual_range"] == ["2024-01-01", "2024-02-23"]
    assert "report" not in store.list_reports()[0]
    assert "history" not in json.dumps(row, ensure_ascii=False)


def test_repeated_runs_and_symbol_queries(tmp_path, completed):
    store = ReportStore(tmp_path / "reports.sqlite3")
    first = store.save_report(completed, mode="local")
    second = store.save_report(completed, mode="deepseek")
    another = deepcopy(completed)
    another["symbol"] = "000001"
    third = store.save_report(another, mode="local")
    assert len({first, second, third}) == 3
    assert [row["id"] for row in store.list_reports(limit=2)] == [third, second]
    assert [row["id"] for row in store.list_reports("600519")] == [second, first]
    assert store.list_reports("000001")[0]["id"] == third


def test_empty_results(tmp_path):
    store = ReportStore(tmp_path / "empty.sqlite3")
    assert store.list_reports() == []
    assert store.list_reports("600519") == []
    assert store.get_report("not-found") is None


def test_source_whitelist_and_partial_research_status(tmp_path, completed):
    completed["api_key"] = "not-for-storage"
    completed["financial_facts"] = {
        "status": "ok", "source": "测试财务来源", "fetched_at": "2026-11-06T10:00:00+08:00",
        "api_key": "not-for-storage", "records": [{"report_date": "2023-12-31", "published_date": "2024-04-01", "api_key": "not-for-storage"}],
    }
    completed["news_facts"] = {"news": {"status": "empty", "records": [], "coverage": "覆盖不足"},
                               "notices": {"status": "error", "records": [], "error": "测试源失败"}}
    store = ReportStore(tmp_path / "reports.sqlite3")
    row = store.get_report(store.save_report(completed, mode="local", include_research=True))
    assert row["include_research"] is True
    assert row["sources"]["financial"]["references"][0]["published_date"] == "2024-04-01"
    assert row["sources"]["notices"]["status"] == "error"
    assert "not-for-storage" not in json.dumps(row)


@pytest.mark.parametrize("change", [
    {"error": "模型失败"}, {"report": ""}, {"report": "   "}, {"report": None},
    {"trace": []}, {"trace": ["report"]}, {"trace": ["market", "supervisor -> end"]},
    {"market_facts": {}}, {"market_facts": {"actual_range": ["bad", "2024-01-01"]}},
    {"market_facts": {"actual_range": ["2024-02-01", "2024-01-01"]}},
    {"symbol": "600519'; DROP TABLE analysis_reports; --"},
])
def test_invalid_or_failed_state_never_saved(tmp_path, completed, change):
    store = ReportStore(tmp_path / "reports.sqlite3")
    completed.update(change)
    with pytest.raises(MemoryStoreError):
        store.save_report(completed, mode="local")
    assert store.list_reports() == []


@pytest.mark.parametrize("field", ["report", "sources"])
def test_configured_secret_rejected_before_insert(tmp_path, completed, monkeypatch, field):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "private-unit-test-key")
    if field == "report":
        completed["report"] += "private-unit-test-key"
    else:
        completed["market_facts"]["limitations"] = "private-unit-test-key"
    store = ReportStore(tmp_path / "reports.sqlite3")
    with pytest.raises(MemoryStoreError, match="认证信息") as error:
        store.save_report(completed, mode="local")
    assert "private-unit-test-key" not in str(error.value)
    assert store.list_reports() == []


def test_parameterized_id_and_report_text(tmp_path, completed):
    store = ReportStore(tmp_path / "reports.sqlite3")
    completed["report"] += "\n' ; DROP TABLE analysis_reports; --\n中文原文"
    report_id = store.save_report(completed, mode="local")
    assert store.get_report("' OR 1=1 --") is None
    assert store.get_report(report_id)["report"] == completed["report"]
    assert len(store.list_reports()) == 1


@pytest.mark.parametrize("limit", [0, -1, 101, True, 1.5, "10"])
def test_invalid_limit(tmp_path, limit):
    with pytest.raises(ValueError):
        ReportStore(tmp_path / "reports.sqlite3").list_reports(limit=limit)


def test_invalid_query_and_mode(tmp_path, completed):
    store = ReportStore(tmp_path / "reports.sqlite3")
    with pytest.raises(ValueError):
        store.list_reports("bad")
    with pytest.raises(ValueError):
        store.get_report("")
    with pytest.raises(ValueError):
        store.save_report(completed, mode="unknown")


@pytest.mark.parametrize("kind", ["directory", "corrupt", "parent_file"])
def test_database_creation_errors(tmp_path, kind):
    path = tmp_path / "bad"
    if kind == "directory":
        path.mkdir()
    else:
        path.write_text("not a SQLite database", encoding="utf-8")
    with pytest.raises(MemoryStoreError, match="SQLite 操作失败"):
        ReportStore(path / "db.sqlite3" if kind == "parent_file" else path)


def test_transaction_rollback_on_statement_error(tmp_path, completed):
    path = tmp_path / "reports.sqlite3"
    store = ReportStore(path)
    original_id = store.save_report(completed, mode="local")
    with sqlite3.connect(path) as connection:
        # FAIL 保留语句中的修改，由 ReportStore 的事务上下文负责回滚。
        connection.execute("CREATE TRIGGER reject_report AFTER INSERT ON analysis_reports BEGIN SELECT RAISE(FAIL, 'test failure'); END")
    with pytest.raises(MemoryStoreError):
        store.save_report(completed, mode="local")
    assert [row["id"] for row in store.list_reports()] == [original_id]


def test_transaction_rollback_on_commit_error(tmp_path, completed, monkeypatch):
    path = tmp_path / "reports.sqlite3"
    store = ReportStore(path)
    # 延迟外键直到 COMMIT 才检查，用真实SQLite证明提交失败不会留下成功记录。
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE parent(id INTEGER PRIMARY KEY)")
        connection.execute("CREATE TABLE child(parent_id INTEGER REFERENCES parent(id) DEFERRABLE INITIALLY DEFERRED)")
        connection.execute("CREATE TRIGGER bad_commit AFTER INSERT ON analysis_reports BEGIN INSERT INTO child VALUES (1); END")
    real_connect = sqlite3.connect

    def connect_with_foreign_keys(*args, **kwargs):
        connection = real_connect(*args, **kwargs)
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    monkeypatch.setattr(sqlite3, "connect", connect_with_foreign_keys)
    with pytest.raises(MemoryStoreError):
        store.save_report(completed, mode="local")
    assert store.list_reports() == []
    with real_connect(path) as connection:
        assert connection.execute("SELECT count(*) FROM child").fetchone()[0] == 0


def test_corrupt_sources_are_reported(tmp_path, completed):
    path = tmp_path / "reports.sqlite3"
    store = ReportStore(path)
    report_id = store.save_report(completed, mode="local")
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE analysis_reports SET sources_json=? WHERE id=?", ("bad json", report_id))
    with pytest.raises(MemoryStoreError, match="无法解析"):
        store.get_report(report_id)


@pytest.mark.parametrize("payload", ["null", "[]", '"wrong type"', "{}", '{"market": null}'])
def test_valid_json_with_invalid_source_structure(tmp_path, completed, payload):
    path = tmp_path / "reports.sqlite3"
    store = ReportStore(path)
    report_id = store.save_report(completed, mode="local")
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE analysis_reports SET sources_json=? WHERE id=?", (payload, report_id))
    with pytest.raises(MemoryStoreError, match="来源信息"):
        store.get_report(report_id)


def test_locked_database_leaves_no_new_success_record(tmp_path, completed, monkeypatch):
    path = tmp_path / "reports.sqlite3"
    store = ReportStore(path)
    original_id = store.save_report(completed, mode="local")
    real_connect = sqlite3.connect

    def quick_connect(*args, **kwargs):
        kwargs["timeout"] = 0.01  # 测试缩短等待，生产环境仍为5秒。
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", quick_connect)
    lock = real_connect(path)
    try:
        lock.execute("BEGIN EXCLUSIVE")
        with pytest.raises(MemoryStoreError, match="锁占用"):
            store.save_report(completed, mode="local")
    finally:
        lock.rollback()
        lock.close()
    assert [row["id"] for row in store.list_reports()] == [original_id]
