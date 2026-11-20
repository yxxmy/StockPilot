"""真实MCP ClientSession协议测试；行情/PDF替身仅为明确的离线测试样本。"""

import json
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest
from mcp.shared.memory import create_connected_server_and_client_session as connect

from memory import MemoryStoreError, ReportStore
from mcp_server.privacy import ServerConfig
from mcp_server.server import create_server
from tools.indicators import calculate_indicators
from tools.stock_data import EmptyDataError
from rag.errors import EmbeddingError, IndexCorruptionError


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def setup_server(tmp_path):
    config = ServerConfig(tmp_path / "reports.sqlite3", tmp_path / "index", True, True)
    # 仅测试样本，不代表600519的真实行情或正式公告。
    frame = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=40, freq="B"),
                          "open": [10.0] * 40, "high": [12.0] * 40, "low": [9.0] * 40,
                          "close": [10.0 + i / 10 for i in range(40)], "volume": [100.0] * 40})
    hit = {"document_id": "test-only-doc", "document_name": "测试样本.pdf", "page": 2,
           "chunk_id": "test-only-chunk", "score": 0.8, "text": "仅测试原文：现金流与风险"}
    loader, search = Mock(return_value=frame), Mock(return_value=[hit])
    return config, frame, loader, search


def seed(config, text="仅测试报告，不是真实分析。", symbol="600519"):
    state = {"symbol": symbol, "report": text, "trace": ["report", "supervisor -> end"],
             "market_facts": {"actual_range": ["2024-01-01", "2024-02-23"]}}
    return ReportStore(config.db_path).save_report(state, mode="local")


async def call(client, name, request):
    return await client.call_tool(name, {"request": request})


@pytest.mark.anyio
async def test_protocol_tools_and_structured_schemas(setup_server):
    config, _, loader, search = setup_server
    async with connect(create_server(config, history_loader=loader, document_search=search)) as client:
        # 官方helper通过JSON-RPC initialize握手后才返回ClientSession。
        tools = (await client.list_tools()).tools
        assert {tool.name for tool in tools} == {"get_stock_history", "get_stock_indicators",
            "list_analysis_reports", "get_analysis_report", "search_documents"}
        for tool in tools:
            assert tool.description and tool.outputSchema
            assert "request" in tool.inputSchema["properties"]
            assert tool.annotations.readOnlyHint and not tool.annotations.destructiveHint
        result = await call(client, "get_stock_history", {
            "symbol": "600519", "start_date": "20240101", "end_date": "20240630", "limit": 3})
        assert not result.isError
        body = result.structuredContent
        assert body["total_records"] == 40 and body["returned_records"] == 3
        assert body["truncated"] and body["returned_range"] == ["2024-02-21", "2024-02-23"]
        assert body["records"][-1]["close"] == 13.9
        assert body["volume_unit"] == "手" and body["source"] == "AKShare / 东方财富"
    loader.assert_called_once_with("600519", "20240101", "20240630")
    search.assert_not_called()


@pytest.mark.anyio
async def test_indicators_reuse_and_null_warmup(setup_server):
    config, frame, loader, _ = setup_server
    async with connect(create_server(config, history_loader=loader)) as client:
        result = await call(client, "get_stock_indicators", {
            "symbol": "600519", "start_date": "20240101", "end_date": "20240630", "limit": 40})
        assert not result.isError
        records = result.structuredContent["records"]
        expected = calculate_indicators(frame)
        for name in expected.columns:
            if name != "date":
                assert records[-1][name] == pytest.approx(expected.iloc[-1][name])
        assert records[0]["ma5"] is None
        assert "NaN" not in json.dumps(result.structuredContent, allow_nan=False)


@pytest.mark.anyio
@pytest.mark.parametrize("change", [
    {"symbol": "600519;DROP TABLE"}, {"symbol": 600519}, {"symbol": "123"},
    {"start_date": "20240230"}, {"start_date": "20250101"},
    {"start_date": "20000101"}, {"limit": 0}, {"limit": 251}, {"limit": True},
    {"limit": "3"}, {"db_path": "arbitrary.sqlite3"}, {"sql": "SELECT 1"},
])
async def test_invalid_history_inputs_do_not_call_backend(setup_server, change):
    config, _, loader, _ = setup_server
    async with connect(create_server(config, history_loader=loader)) as client:
        args = {"symbol": "600519", "start_date": "20240101", "end_date": "20240630", **change}
        assert (await call(client, "get_stock_history", args)).isError
    loader.assert_not_called()


@pytest.mark.anyio
@pytest.mark.parametrize("error", [EmptyDataError("empty"), RuntimeError("C:\\private\\api-key=secret")])
async def test_market_empty_and_backend_errors(setup_server, error):
    config, _, loader, _ = setup_server
    loader.side_effect = error
    async with connect(create_server(config, history_loader=loader)) as client:
        result = await call(client, "get_stock_history", {
            "symbol": "600519", "start_date": "20240101", "end_date": "20240630"})
        if isinstance(error, EmptyDataError):
            assert result.structuredContent["status"] == "empty"
            assert result.structuredContent["records"] == []
        else:
            assert result.isError and "private" not in str(result) and "secret" not in str(result)


@pytest.mark.anyio
async def test_permissions_default_deny(setup_server):
    config, _, _, search = setup_server
    config = ServerConfig(config.db_path, config.index_dir)
    async with connect(create_server(config, document_search=search)) as client:
        for name, request in [("list_analysis_reports", {}),
                              ("get_analysis_report", {"report_id": "00000000-0000-0000-0000-000000000000"}),
                              ("search_documents", {"question": "测试"})]:
            assert (await call(client, name, request)).isError
    search.assert_not_called()
    assert not config.db_path.exists()


@pytest.mark.anyio
async def test_reports_persist_filter_page_and_read_only(setup_server):
    config, _, _, _ = setup_server
    first = seed(config, "仅测试：" + "正文" * 20)
    second = seed(config, symbol="000001")
    original = config.db_path.read_bytes()
    async with connect(create_server(config)) as client:
        latest = (await call(client, "list_analysis_reports", {"limit": 1})).structuredContent
        assert latest["records"][0]["id"] == second
        filtered = (await call(client, "list_analysis_reports", {"symbol": "600519"})).structuredContent
        assert filtered["records"][0]["id"] == first
        assert "report" not in filtered["records"][0]
        result = (await call(client, "get_analysis_report", {"report_id": first, "max_chars": 7})).structuredContent
        assert len(result["report"]) == 7 and result["next_offset"] == 7
        rest = (await call(client, "get_analysis_report", {"report_id": first, "offset": 7})).structuredContent
        assert result["report"] + rest["report"] == "仅测试：" + "正文" * 20
        assert rest["next_offset"] is None
        assert "sources" not in rest and str(config.db_path) not in json.dumps(rest)
    assert config.db_path.read_bytes() == original


@pytest.mark.anyio
async def test_absent_reports_no_creation(setup_server):
    config, _, _, _ = setup_server
    async with connect(create_server(config)) as client:
        assert (await call(client, "list_analysis_reports", {})).structuredContent["status"] == "empty"
        missing = await call(client, "get_analysis_report", {"report_id": "00000000-0000-0000-0000-000000000000"})
        assert missing.structuredContent["status"] == "not_found"
    assert not config.db_path.exists()


@pytest.mark.anyio
async def test_corrupt_db_errors_do_not_leak(setup_server):
    config, _, _, _ = setup_server
    config.db_path.write_bytes(b"not sqlite")
    async with connect(create_server(config)) as client:
        assert (await call(client, "list_analysis_reports", {})).isError
        result = await call(client, "get_analysis_report", {"report_id": "00000000-0000-0000-0000-000000000000"})
        assert result.isError and str(config.db_path) not in str(result)


@pytest.mark.anyio
async def test_redact_before_pagination_and_pdf_output(setup_server, monkeypatch):
    config, _, _, search = setup_server
    fake_secret = "test-only-credential-not-a-real-key"
    report_id = seed(config, "仅测试：" + fake_secret + "\nC:\\private\\report.sqlite3\nBearer test-only-auth")
    monkeypatch.setenv("DEEPSEEK_API_KEY", fake_secret)
    search.return_value[0]["text"] = "仅测试：" + fake_secret + "\nC:\\private\\data.pdf\n" + "片段" * 1000
    async with connect(create_server(config, document_search=search)) as client:
        chunks, offset = [], 0
        while True:
            result = (await call(client, "get_analysis_report", {"report_id": report_id, "offset": offset, "max_chars": 5})).structuredContent
            assert result["redacted"]
            chunks.append(result["report"])
            if result["next_offset"] is None:
                break
            offset = result["next_offset"]
        whole = "".join(chunks)
        assert fake_secret not in whole and "private" not in whole and "test-only-auth" not in whole
        hits = (await call(client, "search_documents", {"question": "测试", "top_k": 1})).structuredContent
        assert hits["redacted"] and hits["records"][0]["text_truncated"]
        assert fake_secret not in str(hits) and "private" not in str(hits)
        assert hits["records"][0]["page"] == 2 and "外部原文" in hits["limitations"]
    search.assert_called_once_with("测试", 1, index_dir=config.index_dir)


@pytest.mark.anyio
@pytest.mark.parametrize("query", [{"question": ""}, {"question": "  "}, {"question": "长" * 501},
    {"question": "测试", "top_k": 0}, {"question": "测试", "top_k": 6},
    {"question": "测试", "index_dir": "arbitrary"}])
async def test_invalid_search_no_backend(setup_server, query):
    config, _, _, search = setup_server
    async with connect(create_server(config, document_search=search)) as client:
        assert (await call(client, "search_documents", query)).isError
    search.assert_not_called()


@pytest.mark.anyio
@pytest.mark.parametrize("error", [None, RuntimeError("secret C:\\private\\index"),
    ValueError("模型加载失败"), OSError("索引损坏"),
    EmbeddingError("private model config"), IndexCorruptionError("private index config")])
async def test_search_empty_errors(setup_server, error):
    config, _, _, search = setup_server
    search.return_value = []
    search.side_effect = error
    async with connect(create_server(config, document_search=search)) as client:
        result = await call(client, "search_documents", {"question": "仅测试"})
        if error:
            assert result.isError and "private" not in str(result) and "secret" not in str(result)
        else:
            assert result.structuredContent["status"] == "empty"


@pytest.mark.anyio
@pytest.mark.parametrize("name,query", [
    ("list_analysis_reports", {"limit": 51}),
    ("list_analysis_reports", {"symbol": "' OR 1=1 --"}),
    ("list_analysis_reports", {"sql": "DELETE FROM analysis_reports"}),
    ("get_analysis_report", {"report_id": "../../data/private"}),
    ("get_analysis_report", {"report_id": "00000000-0000-0000-0000-000000000000", "max_chars": 20001}),
    ("get_analysis_report", {"report_id": "00000000-0000-0000-0000-000000000000", "offset": -1}),
])
async def test_invalid_report_parameters_no_db_creation(setup_server, name, query):
    config, _, _, _ = setup_server
    async with connect(create_server(config)) as client:
        assert (await call(client, name, query)).isError
    assert not config.db_path.exists()


@pytest.mark.anyio
async def test_pdf_faiss_reload_via_real_protocol(setup_server, monkeypatch):
    from rag.index import DocumentIndex
    import numpy as np
    from rag.pdf import extract_pdf

    class OfflineEmbedder:
        # 只替换模型推理边界，其余pypdf/FAISS/快照/检索/MCP均实际运行。
        model_id = "OFFLINE-TEST-ONLY-vectors"

        def encode(self, texts, *, query=False):
            return np.array([[1 + text.count("现金流") * 4, 1 + text.count("风险") * 4, 0.1]
                             for text in texts], dtype="float32")

    config, _, _, _ = setup_server
    sample = Path(__file__).parent / "fixtures" / "rag_test_document.pdf"
    monkeypatch.setattr("rag.index.LocalEmbedder", OfflineEmbedder)
    DocumentIndex(config.index_dir).import_pdf(sample)
    before = {path.name: path.read_bytes() for path in config.index_dir.iterdir()}
    expected = {chunk["chunk_id"]: chunk for chunk in extract_pdf(sample)["chunks"]}
    async with connect(create_server(config)) as client:
        result = await call(client, "search_documents", {"question": "现金流", "top_k": 1})
        assert not result.isError
        hit = result.structuredContent["records"][0]
        assert hit["page"] == 1 and hit["document_name"] == "rag_test_document.pdf"
        assert hit["text"] == expected[hit["chunk_id"]]["text"]
        assert hit["document_id"] == expected[hit["chunk_id"]]["document_id"]
    assert {path.name: path.read_bytes() for path in config.index_dir.iterdir()} == before


@pytest.mark.anyio
async def test_payload_size_and_unknown_unsafe_tools(setup_server, monkeypatch):
    config, _, loader, _ = setup_server
    monkeypatch.setattr("mcp_server.privacy.MAX_PAYLOAD_BYTES", 100)
    async with connect(create_server(config, history_loader=loader)) as client:
        result = await call(client, "get_stock_history", {
            "symbol": "600519", "start_date": "20240101", "end_date": "20240630"})
        assert result.isError and "大小限制" in str(result)
        for name in ["execute_sql", "read_file", "write_file", "run_command", "trade"]:
            assert (await call(client, name, {})).isError


def test_read_only_store_missing_and_prohibits_insert(setup_server):
    config, _, _, _ = setup_server
    with pytest.raises(MemoryStoreError):
        ReportStore(config.db_path, read_only=True)
    assert not config.db_path.exists()
    report_id = seed(config)
    original = config.db_path.read_bytes()
    store = ReportStore(config.db_path, read_only=True)
    assert store.get_report(report_id)["report"] == "仅测试报告，不是真实分析。"
    with pytest.raises(MemoryStoreError):
        with store._connection() as connection:
            connection.execute("DELETE FROM analysis_reports")
    assert config.db_path.read_bytes() == original


def test_config_owner_flags_only(monkeypatch, tmp_path):
    monkeypatch.setenv("STOCKPILOT_MCP_DB_PATH", str(tmp_path / "owner.sqlite3"))
    monkeypatch.setenv("STOCKPILOT_MCP_INDEX_DIR", str(tmp_path / "owner-index"))
    monkeypatch.setenv("STOCKPILOT_MCP_ALLOW_REPORTS", "true")
    monkeypatch.setenv("STOCKPILOT_MCP_ALLOW_DOCUMENTS", "1")
    config = ServerConfig.from_env()
    assert not config.allow_reports and config.allow_documents
    assert config.db_path == tmp_path / "owner.sqlite3" and config.index_dir == tmp_path / "owner-index"
