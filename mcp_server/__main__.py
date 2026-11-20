"""只运行stdio，不监听端口。stdout严格保留给MCP协议。"""

from contextlib import redirect_stdout
from io import TextIOWrapper
import os
import sys

import anyio
from mcp.server.stdio import stdio_server

from mcp_server.server import create_server
from mcp_server.privacy import ServerConfig


async def serve_stdio(server):
    # 捕获协议输出后把普通print转向stderr，第三方库的进度不会污染JSON-RPC。
    protocol_stdout = anyio.wrap_file(TextIOWrapper(sys.stdout.buffer, encoding="utf-8", write_through=True))
    with redirect_stdout(sys.stderr):
        async with stdio_server(stdout=protocol_stdout) as (read_stream, write_stream):
            low_level = server._mcp_server  # 官方SDK的FastMCP stdio入口同样调用此底层Server。
            await low_level.run(read_stream, write_stream, low_level.create_initialization_options())


def main():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    config = ServerConfig.from_env()
    if config.allow_documents:
        # Windows下先加载原生FAISS库，再启动stdio读线程，避免DLL初始化阻塞。
        with redirect_stdout(sys.stderr):
            try:
                import rag.index  # noqa: F401；不加载Embedding权重、不读PDF或建立索引。
                if (config.index_dir / "CURRENT.json").is_file():
                    # 同理预加载PyTorch原生依赖；权重仍由现有检索函数按需加载。
                    import sentence_transformers  # noqa: F401
            except Exception:
                pass  # 依赖问题由具体检索调用返回，其他只读工具仍可使用。
    anyio.run(serve_stdio, create_server(config))


if __name__ == "__main__":
    main()
