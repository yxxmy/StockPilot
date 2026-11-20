"""真实页面/工作流失败链路；沿用隔离BGE夹具，付费HTTP仅使用替身。"""

import json
import sqlite3
from unittest.mock import Mock

import pytest
import requests

from agents import workflow
from agents.deepseek import AgentError
from agents.diagnostics import diagnostic
from memory import MemoryStoreError, ReportStore
from test_end_to_end import real_corpus, system, app, submit  # noqa: F401；复用隔离夹具，不重复场景实现。


@pytest.mark.parametrize("kind,node,category", [
    ("market", "market", "market_connection"),
    ("http", "risk", "model_http"),
    ("citation", "report", "model_citation"),
    ("sqlite", "persistence", "storage_error"),
])
def test_failed_page_keeps_real_trace_and_never_saves(system, monkeypatch, kind, node, category):
    captured = []
    original = workflow.run_analysis
    def observe(*args, **kwargs):
        try:
            return original(*args, **kwargs)
        except (AgentError, MemoryStoreError) as exc:
            captured.append(exc)
            raise
    monkeypatch.setattr(workflow, "run_analysis", observe)
    connection = None
    if kind == "market":
        system.market.side_effect = requests.ConnectionError("SECRET_FAILED C:\\private\\config sk-test-private-credential")
    elif kind in ("http", "citation"):
        respond = system.post.side_effect
        def failure(url, **kwargs):
            facts = json.loads(kwargs["json"]["messages"][1]["content"])
            if kind == "http" and "flags" in facts:
                return Mock(status_code=503)
            response = respond(url, **kwargs)
            if kind == "citation" and "rag" in facts:
                payload = response.json()
                payload["choices"][0]["message"]["content"] = json.dumps({
                    "summary": "测试伪造引用[R99]", "evidence": ["rag"], "citations": ["R99"]})
                return Mock(status_code=200, json=Mock(return_value=payload))
            return response
        system.post.side_effect = failure
    else:
        ReportStore(system.db)
        connection = sqlite3.connect(system.db)
        connection.execute("BEGIN EXCLUSIVE")  # 真实SQLite锁，拒绝初始化/写入；不是模拟保存函数。
    try:
        at = submit(app())
    finally:
        if connection:
            connection.rollback()
            connection.close()
    assert not at.exception and at.error and not at.success
    assert len(captured) == 1
    assert isinstance(captured[0], MemoryStoreError if kind == "sqlite" else AgentError)
    assert "sk-test-private" not in str(captured[0]) and "C:\\private" not in str(captured[0])
    info = captured[0].diagnostic
    assert info == at.session_state["failure_diagnostic"]
    assert info["failed_node"] == node and info["category"] == category
    route = []
    executed = ["market", "financial", "news", "risk", "rag_retrieval", "report"]
    if kind != "sqlite":
        executed = executed[:executed.index(node) + 1]
    for step in executed:
        route += [f"supervisor -> {step}", f"{step}: error" if step == node else step]
    assert info["trace"] == route + ["supervisor -> end"]
    assert any("失败位置：" + node in text.value for text in at.text)
    assert any("\n".join(info["trace"]) in text.value for text in at.text)
    displayed = " ".join(item.value for item in list(at.error) + list(at.text))
    assert all(secret not in displayed for secret in ["SECRET_FAILED", "C:\\private", "sk-test-private", "TEST_ONLY_E2E_NOT_A_REAL_KEY"])
    assert "result" not in at.session_state
    assert not system.db.exists() or ReportStore(system.db).list_reports() == []
    counts = (system.market.call_count, system.post.call_count)
    at.run()
    assert len(captured) == 1 and (system.market.call_count, system.post.call_count) == counts
    assert at.session_state["failure_diagnostic"] == info
    assert not at.session_state["running"]
    # 失败后的下一次显式成功提交应清除旧诊断，不把失败缓存为成功。
    if kind == "http":
        system.post.side_effect = respond
        submit(at)
        assert not at.error and not at.exception
        assert "failure_diagnostic" not in at.session_state
        assert len(ReportStore(system.db).list_reports()) == 1


def test_diagnostic_whitelist_discards_untrusted_content():
    value = diagnostic("C:\\private", "sk-secret", ["market", "SECRET_FAILED", "supervisor -> end"])
    assert value["failed_node"] == "unknown" and value["category"] == "analysis_error"
    assert value["trace"] == ["market", "supervisor -> end"]


def test_missing_key_has_no_fabricated_node_trace(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(AgentError) as caught:
        workflow.run_analysis("600519", "20240101", "20240630")
    assert caught.value.diagnostic == diagnostic("configuration", "configuration", [])
