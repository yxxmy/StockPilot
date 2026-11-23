"""共享 State、四个 Node 和条件路由组成的最小多 Agent 工作流。"""

import json
import operator
from datetime import timedelta
from typing import Annotated, TypedDict

import numpy as np
import pandas as pd
from langgraph.graph import END, START, StateGraph

from agents.deepseek import AgentError, DeepSeekClient
from agents.report_text import EVIDENCE_DIVIDER, market_and_risk_text
from tools.indicators import calculate_indicators
from tools.stock_data import EmptyDataError, StockDataError, get_stock_history
from tools.research_data import get_stock_financials, get_stock_news, get_stock_notices
from memory import ReportStore, MemoryStoreError
from agents.diagnostics import diagnostic, safe_error_text


class StockState(TypedDict, total=False):
    symbol: str
    start_date: str
    end_date: str
    history: pd.DataFrame
    market_facts: dict
    market_summary: str
    risk_facts: dict
    risk_summary: str
    financial_facts: dict
    financial_summary: str
    news_facts: dict
    news_summary: str
    rag_question: str
    rag_report_year: int
    rag_document_type: str
    rag_facts: dict
    report: str
    report_id: str
    error: str
    failure: dict
    next_node: str
    # reducer：每个节点返回本次轨迹，LangGraph 自动追加，不覆盖旧记录。
    trace: Annotated[list[str], operator.add]


LIMITATIONS = (
    "仅分析查询范围内的历史日线，来源 AKShare / 东方财富，不复权，成交量单位为手。"
    "历史数据不足时，相关指标留空；除权除息可能造成价格跳空。"
    "未获取财务、公告、新闻或持仓，不能评估完整投资风险；不是收益预测或买卖建议。"
)

RESEARCH_LIMITATIONS = (
    "仅分析查询范围内的不复权历史日线，成交量单位为手；历史数据不足时，相关指标留空。"
    "财务只选截止日已披露记录，但上游可能事后修订，不是严格时点回测数据。"
    "新闻仅覆盖近期有限搜索结果，公告列表仅含标题及链接；缺失或失败见各数据源说明。"
    "未获取持仓，不能评估完整投资风险；不是收益预测或买卖建议。"
)


def _number(value):
    return None if pd.isna(value) else float(value)


def market_facts(data: pd.DataFrame, symbol: str, start: str, end: str) -> dict:
    latest = data.iloc[-1]
    columns = ["close", "volume", "ma5", "ma10", "ma20", "macd_dif", "macd_dea",
               "macd_hist", "rsi14", "pct_change", "volume_ma5", "volume_ratio"]
    return {
        "symbol": symbol,
        "source": data.attrs.get("source", "AKShare / 东方财富"),
        "source_warning": data.attrs.get("source_warning", ""),
        "requested_range": [start, end],
        "actual_range": [data["date"].min().strftime("%Y-%m-%d"), latest["date"].strftime("%Y-%m-%d")],
        "record_count": len(data),
        "latest": {column: _number(latest[column]) for column in columns},
        "missing_indicators": [column for column in columns if pd.isna(latest[column])],
        "limitations": LIMITATIONS.replace("AKShare / 东方财富", data.attrs.get("source", "AKShare / 东方财富")),
    }


def risk_facts(data: pd.DataFrame) -> dict:
    close = data["close"]
    # 统计仅使用查询区间；年化波动率采用 252 个交易日、样本标准差。
    returns = close.pct_change(fill_method=None).dropna()
    volatility = _number(returns.std(ddof=1) * np.sqrt(252) * 100) if len(returns) >= 2 else None
    drawdown = _number((close / close.cummax() - 1).min() * 100)
    latest = data.iloc[-1]
    flags = []
    if pd.isna(latest["macd_hist"]):
        flags.append("样本少于34条，MACD尚未完成预热")
    if volatility is None:
        flags.append("有效涨跌幅少于2条，无法估计样本波动率")
    if data["rsi14"].notna().iloc[-1]:
        if latest["rsi14"] >= 70:
            flags.append("RSI14达到70或以上（教学阈值，不构成卖出建议）")
        elif latest["rsi14"] <= 30:
            flags.append("RSI14达到30或以下（教学阈值，不构成买入建议）")
    if data["volume"].eq(0).any():
        flags.append("样本中存在零成交量记录")
    return {
        "sample_return_pct": _number((close.iloc[-1] / close.iloc[0] - 1) * 100),
        "max_drawdown_pct": drawdown,
        "annualized_volatility_pct": volatility,
        "flags": flags,
        "method": "不复权区间价格变化；最大回撤为非正数，单条样本为0；波动率=日收益样本标准差×sqrt(252)×100",
        "limitations": LIMITATIONS.replace("AKShare / 东方财富", data.attrs.get("source", "AKShare / 东方财富")),
    }


def build_graph(client=None, history_loader=get_stock_history, *, include_research=False,
                financial_loader=None, news_loader=None, notice_loader=None, rag_index_dir=None, warmup=False):
    """client=None 为真实数据上的本地规则模式；测试可注入模型替身。"""
    def failed(node, label, exc):
        category = exc.category if isinstance(exc, (AgentError, StockDataError)) else "market_data" if node == "market" else "analysis_error"
        return {"error": f"{label} Agent 失败：{safe_error_text(exc)}", "trace": [f"{node}: error"],
                "failure": diagnostic(node, category, [])}

    def supervisor(state: StockState):
        if state.get("error") or state.get("report"):
            target = "end"
        elif "market_facts" not in state:
            target = "market"
        elif include_research and "financial_facts" not in state:
            target = "financial"
        elif include_research and "news_facts" not in state:
            target = "news"
        elif "risk_facts" not in state:
            target = "risk"
        elif state.get("rag_question") and "rag_facts" not in state:
            target = "rag_retrieval"
        else:
            target = "report"
        return {"next_node": target, "trace": [f"supervisor -> {target}"]}

    def market(state: StockState):
        try:
            start, end = state["start_date"], state["end_date"]
            # 网页按所选区间展示；提前取得真实历史，让首日也能计算均线和昨收涨跌幅。
            fetch_start = (pd.Timestamp(start).date() - timedelta(days=365)).strftime("%Y%m%d") if warmup else start
            data = calculate_indicators(history_loader(state["symbol"], fetch_start, end))
            if warmup:
                data = data.loc[data["date"].between(pd.Timestamp(start), pd.Timestamp(end))].reset_index(drop=True)
                if data.empty:
                    raise EmptyDataError("所选区间没有交易记录")
            facts = market_facts(data, state["symbol"], state["start_date"], state["end_date"])
            facts["limitations"] = facts["limitations"].replace("AKShare / 东方财富", facts["source"])
            if include_research:
                facts["limitations"] = RESEARCH_LIMITATIONS
            summary = client.describe("Market Agent", facts) if client else "本地规则：行情和指标已计算，数值见 market_facts。"
            return {"history": data, "market_facts": facts, "market_summary": summary, "trace": ["market"]}
        except (ValueError, StockDataError, AgentError) as exc:
            return failed("market", "Market", exc)

    def load_optional(loader, *args):
        # 研究源失败不伪造数据；仍把明确的错误状态传给后续节点。
        try:
            return loader(*args)
        except (ValueError, StockDataError) as exc:
            return {"status": "error", "records": [], "error": str(exc),
                    "symbol": args[0], "coverage": "获取失败，不能据此推断财务状况或事件是否存在"}

    def financial(state: StockState):
        facts = load_optional(financial_loader or get_stock_financials, state["symbol"], state["end_date"])
        try:
            summary = client.describe("Financial Agent", facts) if client and facts["records"] else ""
            return {"financial_facts": facts, "financial_summary": summary, "trace": ["financial"]}
        except AgentError as exc:
            return failed("financial", "Financial", exc)

    def news(state: StockState):
        args = (state["symbol"], state["start_date"], state["end_date"])
        facts = {"symbol": state["symbol"], "requested_range": [state["start_date"], state["end_date"]],
                 "news": load_optional(news_loader or get_stock_news, *args),
                 "notices": load_optional(notice_loader or get_stock_notices, *args),
                 "limitations": RESEARCH_LIMITATIONS}
        try:
            has_records = facts["news"]["records"] or facts["notices"]["records"]
            summary = client.describe("News Agent", facts) if client and has_records else ""
            return {"news_facts": facts, "news_summary": summary, "trace": ["news"]}
        except AgentError as exc:
            return failed("news", "News", exc)

    def risk(state: StockState):
        try:
            facts = risk_facts(state["history"])
            if include_research:
                facts["limitations"] = RESEARCH_LIMITATIONS
                facts["research_context"] = {"financial": state["financial_facts"], "news": state["news_facts"]}
            summary = client.describe("Risk Agent", facts) if client else "本地规则：历史风险统计已计算，风险提示见 risk_facts.flags。"
            return {"risk_facts": facts, "risk_summary": summary, "trace": ["risk"]}
        except (ValueError, AgentError) as exc:
            return failed("risk", "Risk", exc)

    def rag_retrieval(state: StockState):
        # 显式正文问题才走此节点，普通分析不导入RAG/模型依赖。
        from agents.retrieval import retrieve_document_facts
        options = {key: state[field] for key, field in [("report_year", "rag_report_year"), ("document_type", "rag_document_type")] if field in state}
        facts = retrieve_document_facts(state["rag_question"], rag_index_dir, symbol=state["symbol"], **options)
        return {"rag_facts": facts, "trace": ["rag_retrieval"]}

    def report(state: StockState):
        limitations = RESEARCH_LIMITATIONS if include_research else LIMITATIONS
        limitations = limitations.replace("AKShare / 东方财富", state["market_facts"].get("source", "AKShare / 东方财富"))
        if "rag_facts" in state:
            limitations = ("行情、结构化财务及新闻的原有计算口径不变；本地PDF正文片段单独列示，"
                           "不表示完整财报已读取或其适用于行情时点。" + state["rag_facts"]["limitations"])
        facts = {"market": state["market_facts"], "risk": state["risk_facts"],
                 "market_commentary": state["market_summary"], "risk_commentary": state["risk_summary"],
                 "limitations": limitations}
        evidence = {"market": facts["market"], "risk": facts["risk"]}
        research_text = ""
        retrieval_text = ""
        if "rag_facts" in state:
            facts["rag"] = state["rag_facts"]
            evidence["rag"] = state["rag_facts"]
            retrieval_text = f"\n\n正文检索状态：{state['rag_facts']['status']}\n{state['rag_facts']['message']}"
            for source in state["rag_facts"]["sources"]:
                retrieval_text += f"\n[{source['ref_id']}] {source['document_name']}，物理第{source['page']}页，文档ID {source['document_id']}，片段ID {source['chunk_id']}"
        if include_research:
            facts.update(financial=state["financial_facts"], news=state["news_facts"],
                         financial_commentary=state["financial_summary"], news_commentary=state["news_summary"])
            evidence.update(financial=facts["financial"], news=facts["news"])
            for title, summary in (("财务解读", state["financial_summary"]), ("新闻与公告解读", state["news_summary"])):
                if summary:
                    research_text += f"\n\n{title}\n{summary}"
        try:
            if "rag_facts" in state and state["rag_facts"]["status"] != "ok":
                # 没有可用原文时不让模型凭知识补答正文，保留已有行情/风险结果。
                summary = "正文问题没有可用原文，无法回答。" + state["rag_facts"]["message"]
            else:
                summary = client.describe("Report Agent", facts) if client else ""
            # 数值证据由代码直接输出；模型解读独立标注，不能覆盖事实。
            label = "DeepSeek 模型解读（需核对事实）" if client else "本地规则输出（未调用大模型）"
            text = (f"{state['symbol']} 行情分析｜截至 {facts['market']['actual_range'][1]}\n{label}\n\n"
                    + market_and_risk_text(facts["market"], facts["risk"]))
            if client:
                text += f"\n\n模型行情解读\n{state['market_summary']}\n\n模型风险解读\n{state['risk_summary']}"
            text += research_text + retrieval_text
            if summary:
                text += "\n\n综合说明\n" + summary
            text += "\n\n数据使用说明\n" + limitations
            # 完整证据随报告保存，页面默认只呈现上面的中文正文。
            text += EVIDENCE_DIVIDER + json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False)
            return {"report": text, "trace": ["report"]}
        except (ValueError, AgentError) as exc:
            return failed("report", "Report", exc)

    graph = StateGraph(StockState)
    nodes = [("supervisor", supervisor), ("market", market), ("risk", risk), ("rag_retrieval", rag_retrieval), ("report", report)]
    if include_research:
        nodes += [("financial", financial), ("news", news)]
    for name, node in nodes:
        graph.add_node(name, node)
    graph.add_edge(START, "supervisor")
    routes = {name: name for name, _ in nodes if name != "supervisor"}
    routes["end"] = END
    graph.add_conditional_edges("supervisor", lambda state: state["next_node"], routes)
    for name in routes:
        if name == "end":
            continue
        graph.add_edge(name, "supervisor")
    return graph.compile()


def run_analysis(symbol: str, start_date: str, end_date: str, *, local=False, include_research=False,
                 memory_path=None, rag_question=None, rag_index_dir=None, rag_report_year=None, rag_document_type=None, warmup=False) -> StockState:
    if rag_question is not None and (not isinstance(rag_question, str) or not rag_question.strip()):
        raise ValueError("正文检索问题不能为空")
    if rag_index_dir is not None and rag_question is None:
        raise ValueError("指定RAG索引目录时必须提供正文问题")
    if rag_report_year is not None or rag_document_type is not None:
        if rag_question is None:
            raise ValueError("指定RAG元数据过滤时必须提供正文问题")
        from rag.metadata import search_filters
        search_filters(symbol, rag_report_year, rag_document_type)
    # 先检查模型配置，缺少密钥时不请求行情，不产生不完整分析。
    try:
        client = None if local else DeepSeekClient()
    except AgentError as exc:
        exc.diagnostic = diagnostic("configuration", "configuration", [])
        raise
    initial = {"symbol": symbol, "start_date": start_date, "end_date": end_date, "trace": []}
    options = {"rag_index_dir": rag_index_dir} if rag_index_dir is not None else {}
    if warmup:
        options["warmup"] = True
    if rag_question is not None:
        initial["rag_question"] = rag_question.strip()
    if rag_report_year is not None:
        initial["rag_report_year"] = rag_report_year
    if rag_document_type is not None:
        initial["rag_document_type"] = rag_document_type
    result = build_graph(client, include_research=include_research, **options).invoke(
        initial,
        config={"recursion_limit": 20},
    )
    if result.get("error"):
        failure = result.get("failure", {})
        info = diagnostic(failure.get("failed_node"), failure.get("category"), result.get("trace", []))
        error = AgentError(result["error"], category=info["category"])
        error.diagnostic = info
        raise error
    # 在所有 Agent 成功结束及已有模型校验通过后保存，不改动图内部节点。
    try:
        result["report_id"] = ReportStore(memory_path).save_report(
            result, mode="local" if local else "deepseek", include_research=include_research,
        )
    except MemoryStoreError as exc:
        # 存储在图外执行；保留真实图trace，不伪造SQLite节点已经进入LangGraph。
        exc.diagnostic = diagnostic("persistence", "storage_error", result.get("trace", []))
        exc.args = tuple(safe_error_text(arg) if isinstance(arg, str) else arg for arg in exc.args)
        raise
    return result
