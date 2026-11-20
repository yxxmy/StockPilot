"""StockPilot 行情与技术指标命令行入口。"""

import argparse
from datetime import date, timedelta
import sys

from tools.stock_data import StockDataError, get_stock_history
from tools.indicators import calculate_indicators
from memory import MemoryStoreError, ReportStore


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="StockPilot：获取真实 A 股历史日线（不复权）")
    parser.add_argument("symbol", nargs="?", default=None, help="六位股票代码，分析默认600519；历史查询时为可选筛选")
    parser.add_argument("--start-date", default=(date.today() - timedelta(days=365)).isoformat())
    parser.add_argument("--end-date", default=date.today().isoformat())
    parser.add_argument("--indicators", action="store_true", help="增加 MA、MACD、RSI 等技术指标")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--agent", action="store_true", help="使用 LangGraph + DeepSeek 生成分析（需要密钥，会产生 DeepSeek 费用）")
    mode.add_argument("--agent-local", action="store_true", help="使用真实行情验证 LangGraph 本地规则流程，不调用模型")
    parser.add_argument("--research", action="store_true", help="增加 Financial 和 News Agent，获取财务、公告及新闻；需配合 Agent 模式")
    queries = parser.add_mutually_exclusive_group()
    queries.add_argument("--history", action="store_true", help="查询最近报告，可用位置参数按股票筛选")
    queries.add_argument("--report-id", help="按ID读取完整报告")
    parser.add_argument("--limit", type=int, default=10, help="历史报告条数，1到100，默认10")
    parser.add_argument("--db", help="SQLite文件路径，默认项目data/stockpilot.sqlite3")
    parser.add_argument("--rag-question", help="明确要求查询财报/公告正文才填写，启用本地PDF检索")
    parser.add_argument("--rag-index-dir", help="正文检索索引目录，需配合--rag-question")
    args = parser.parse_args(argv)
    if args.history or args.report_id is not None:
        if args.agent or args.agent_local or args.research or args.indicators or args.rag_question is not None or args.rag_index_dir is not None:
            parser.error("历史查询不能与分析模式混用")
        if args.report_id is not None and args.symbol is not None:
            parser.error("按ID读取时不需要股票代码")
        try:
            store = ReportStore(args.db)
            if args.history:
                rows = store.list_reports(args.symbol, limit=args.limit)
                if not rows:
                    print("暂无历史报告")
                for row in rows:
                    print(f"{row['id']} | {row['symbol']} | {row['generated_at']} | 行情截止 {row['market_end_date']} | {row['mode']} | research={row['include_research']}")
            else:
                row = store.get_report(args.report_id)
                if row is None:
                    print("未找到该ID的报告", file=sys.stderr)
                    return 1
                print(f"报告ID：{row['id']} | 股票：{row['symbol']} | 生成时间：{row['generated_at']} | 行情截止：{row['market_end_date']} | {row['mode']} | research={row['include_research']}")
                print(row["report"])
        except (MemoryStoreError, ValueError) as exc:
            print(f"错误：{exc}", file=sys.stderr)
            return 1
        return 0
    args.symbol = args.symbol or "600519"
    if args.rag_question is not None and not (args.agent or args.agent_local):
        parser.error("--rag-question需要配合--agent或--agent-local")
    if args.rag_index_dir is not None and args.rag_question is None:
        parser.error("--rag-index-dir需要配合--rag-question")
    if args.research and not (args.agent or args.agent_local):
        parser.error("--research 需要配合 --agent 或 --agent-local")
    if args.agent or args.agent_local:
        # 延迟导入，阶段一、二不初始化任何模型客户端。
        from agents.deepseek import AgentError
        from agents.workflow import run_analysis
        try:
            options = {"include_research": True} if args.research else {}
            if args.db is not None:
                options["memory_path"] = args.db
            if args.rag_question is not None:
                options["rag_question"] = args.rag_question
            if args.rag_index_dir is not None:
                options["rag_index_dir"] = args.rag_index_dir
            result = run_analysis(args.symbol, args.start_date, args.end_date, local=args.agent_local, **options)
        except (AgentError, MemoryStoreError, ValueError) as exc:
            print(f"错误：{exc}", file=sys.stderr)
            return 1
        print(result["report"])
        print("\n执行轨迹：" + " | ".join(result["trace"]))
        if result.get("report_id"):
            print("报告已保存，ID：" + result["report_id"])
        return 0
    try:
        data = get_stock_history(args.symbol, args.start_date, args.end_date)
        if args.indicators:
            data = calculate_indicators(data)
    except (ValueError, StockDataError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1
    print(f"股票：{args.symbol} | 来源：{data.attrs.get('source', 'AKShare / 东方财富')} | 不复权 | 成交量单位：手")
    if data.attrs.get("source_warning"):
        print(data.attrs["source_warning"])
    if args.indicators:
        print("技术指标仅基于查询范围内行情；预热期为 NaN，不复权指标可能受除权除息影响。")
    print("\n前五行：")
    print(data.head().to_string(index=False))
    print("\n后五行：")
    print(data.tail().to_string(index=False))
    print(f"\n数据日期范围：{data['date'].min():%Y-%m-%d} 至 {data['date'].max():%Y-%m-%d}")
    print(f"记录数：{len(data)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
