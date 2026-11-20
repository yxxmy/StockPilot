"""启动真正的stdio子进程；只查询临时数据库/空索引，不访问网络。"""

from datetime import timedelta
import os
from pathlib import Path
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import pytest

from memory import ReportStore


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
@pytest.mark.parametrize("entry_kind", ["module", "absolute_script"])
async def test_real_stdio_initialize_list_and_call_tools(tmp_path, entry_kind):
    path = tmp_path / "test-only.sqlite3"
    report_id = ReportStore(path).save_report({"symbol": "600519", "report": "仅测试stdio报告，不是真实分析。",
        "trace": ["report", "supervisor -> end"],
        "market_facts": {"actual_range": ["2024-01-01", "2024-01-02"]}}, mode="local")
    original = path.read_bytes()
    # 不向子进程传递DEEPSEEK_API_KEY等认证配置；只传系统运行所需值。
    env = {key: value for key, value in os.environ.items() if key.upper() in {
        "PATH", "SYSTEMROOT", "TEMP", "TMP", "USERPROFILE", "APPDATA", "LOCALAPPDATA"}}
    env.update({"STOCKPILOT_MCP_DB_PATH": str(path), "STOCKPILOT_MCP_INDEX_DIR": str(tmp_path / "empty-index"),
                "STOCKPILOT_MCP_ALLOW_REPORTS": "1", "STOCKPILOT_MCP_ALLOW_DOCUMENTS": "1",
                "PYTHONIOENCODING": "utf-8"})
    root = Path(__file__).resolve().parents[1]
    args = ["-m", "mcp_server"] if entry_kind == "module" else [str(root / "mcp_stdio.py")]
    params = StdioServerParameters(command=sys.executable, args=args,
        cwd=str(root if entry_kind == "module" else tmp_path), env=env)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write, read_timeout_seconds=timedelta(seconds=30)) as client:
            initialized = await client.initialize()
            assert initialized.serverInfo.name == "StockPilot Read-Only" and initialized.capabilities.tools
            assert initialized.protocolVersion
            tools = (await client.list_tools()).tools
            assert len(tools) == 5 and all(tool.outputSchema for tool in tools)
            listed = await client.call_tool("list_analysis_reports", {"request": {}})
            assert not listed.isError and listed.structuredContent["records"][0]["id"] == report_id
            report = await client.call_tool("get_analysis_report", {"request": {"report_id": report_id}})
            assert report.structuredContent["report"] == "仅测试stdio报告，不是真实分析。"
            searched = await client.call_tool("search_documents", {"request": {"question": "测试"}})
            assert searched.structuredContent["status"] == "empty"
            bad = await client.call_tool("get_stock_history", {"request": {
                "symbol": "bad", "start_date": "20240101", "end_date": "20240131"}})
            assert bad.isError
            assert (await client.list_resources()).resources == []
            assert (await client.list_prompts()).prompts == []
    assert path.read_bytes() == original
    assert not (tmp_path / "empty-index").exists()
