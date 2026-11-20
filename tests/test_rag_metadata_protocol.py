"""真实MCP与DeepSeek HTTP替身验证元数据传递，不请求付费API。"""

import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import requests
from mcp.shared.memory import create_connected_server_and_client_session

from agents.deepseek import DeepSeekClient
from agents import workflow
from mcp_server.server import create_server
from mcp_server.privacy import ServerConfig
from rag.index import DocumentIndex


INFO = {"symbol": "600519", "company_name": "仅测试公司", "report_year": 2023,
        "document_type": "annual_report", "disclosure_date": "2024-04-02"}


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_actual_mcp_filter_and_legacy_query_compatibility(tmp_path, monkeypatch):
    class OfflineVectors:
        model_id = "TEST-ONLY-MCP-metadata"

        def encode(self, texts, *, query=False):
            return np.array([[1., 0.1] for _ in texts], dtype="float32")

    monkeypatch.setattr("rag.index.LocalEmbedder", OfflineVectors)
    index_path = tmp_path / "index"
    DocumentIndex(index_path).import_pdf(Path(__file__).parent / "fixtures" / "rag_test_document.pdf", metadata=INFO)
    server = create_server(ServerConfig(tmp_path / "unused.sqlite3", index_path, allow_documents=True))
    async with create_connected_server_and_client_session(server) as client:
        tools = (await client.list_tools()).tools
        assert len(tools) == 5
        selected = await client.call_tool("search_documents", {"request": {"question": "测试", **{
            "symbol": "600519", "report_year": 2023, "document_type": "annual_report"}}})
        assert not selected.isError
        assert selected.structuredContent["records"][0]["metadata"] == INFO
        for filters in [{"symbol": "000001"}, {"report_year": 2022}, {"document_type": "announcement"}]:
            empty = await client.call_tool("search_documents", {"request": {"question": "测试", **filters}})
            assert not empty.isError and empty.structuredContent["status"] == "empty"
        legacy = await client.call_tool("search_documents", {"request": {"question": "测试"}})
        assert not legacy.isError and legacy.structuredContent["records"]
        for filters in [{"symbol": "bad"}, {"report_year": True}, {"document_type": " "}]:
            assert (await client.call_tool("search_documents", {"request": {"question": "测试", **filters}})).isError


def test_deepseek_receives_metadata_as_data_not_system_instruction(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-only-key")
    response = Mock(status_code=200, json=Mock(return_value={"choices": [{"finish_reason": "stop", "message": {
        "content": json.dumps({"summary": "仅测试引用[R1]", "evidence": ["rag"], "citations": ["R1"]})}}]}))
    post = Mock(return_value=response)
    monkeypatch.setattr(requests, "post", post)
    facts = {"rag": {"status": "ok", "filters": {"symbol": "600519", "report_year": 2023},
                     "records": [{"ref_id": "R1", "text": "仅测试原文", "metadata": INFO}]}}
    DeepSeekClient().describe("Report Agent", facts)
    messages = post.call_args.kwargs["json"]["messages"]
    assert json.loads(messages[1]["content"])["rag"]["records"][0]["metadata"] == INFO
    assert "仅测试公司" not in messages[0]["content"]
    assert "report_year" in messages[0]["content"] and "test-only-key" not in json.dumps(messages)


@pytest.mark.parametrize("options", [{"rag_report_year": 2023}, {"rag_document_type": "annual_report"},
    {"rag_question": "测试", "rag_report_year": True}, {"rag_question": "测试", "rag_document_type": " "}])
def test_bad_graph_filters_rejected_before_model_or_data(monkeypatch, options):
    build = Mock(side_effect=AssertionError("不应建图"))
    model = Mock(side_effect=AssertionError("不应调用模型"))
    monkeypatch.setattr(workflow, "build_graph", build)
    monkeypatch.setattr(workflow, "DeepSeekClient", model)
    with pytest.raises(ValueError):
        workflow.run_analysis("600519", "20240101", "20240630", **options)
    build.assert_not_called()
    model.assert_not_called()
