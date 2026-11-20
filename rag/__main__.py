"""独立命令行：先显式下载模型，再导入本地PDF并检索。"""

import argparse
import json
import sys

from rag.embedding import DEFAULT_MODEL, LocalEmbedder
from rag.errors import RagError


def main(argv=None):
    parser = argparse.ArgumentParser(description="StockPilot 5B：本地PDF语义检索，不生成分析答案")
    parser.add_argument("--index-dir", help="索引目录，默认data/rag")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="中文Embedding模型名或完整本地目录")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("download-model", help="显式下载本地模型，需要网络，无付费API")
    importing = commands.add_parser("import-pdf", help="导入一份本地可提取文字的PDF")
    importing.add_argument("path")
    importing.add_argument("--chunk-size", type=int, default=350)
    importing.add_argument("--overlap", type=int, default=60)
    updating = commands.add_parser("set-metadata", help="核验后显式补录文档元数据，保留原文及旧快照")
    updating.add_argument("document_id")
    for command in (importing, updating):
        command.add_argument("--symbol", help="经人工核验的六位股票代码；不填则未知")
        command.add_argument("--company-name")
        command.add_argument("--report-year", type=int)
        command.add_argument("--document-type", help="精确类型，如annual_report或announcement")
        command.add_argument("--disclosure-date", help="披露日YYYY-MM-DD，与报告年份分开")
    searching = commands.add_parser("search", help="返回原文、来源和余弦相似度")
    searching.add_argument("question")
    searching.add_argument("--top-k", type=int, default=5)
    searching.add_argument("--symbol")
    searching.add_argument("--report-year", type=int)
    searching.add_argument("--document-type")
    args = parser.parse_args(argv)
    try:
        embedder = LocalEmbedder(args.model, allow_download=args.command == "download-model")
        if args.command == "download-model":
            embedder.load()
            print("模型已准备，可进行离线导入与查询")
            return 0
        from rag.index import DocumentIndex
        index = DocumentIndex(args.index_dir, embedder=embedder,
                              chunk_size=getattr(args, "chunk_size", 350), overlap=getattr(args, "overlap", 60))
        if args.command in ("import-pdf", "set-metadata"):
            metadata = {key: getattr(args, key) for key in ["symbol", "company_name", "report_year", "document_type", "disclosure_date"]
                        if getattr(args, key) is not None}
            result = index.import_pdf(args.path, metadata=metadata or None) if args.command == "import-pdf" else index.update_metadata(args.document_id, metadata)
        else:
            result = index.search_documents(args.question, args.top_k, symbol=args.symbol,
                report_year=args.report_year, document_type=args.document_type)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if result == []:
            print("索引为空或没有符合元数据过滤条件的PDF；未知身份不属于指定股票")
        return 0
    except (RagError, ValueError, OSError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
