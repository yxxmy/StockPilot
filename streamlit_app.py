"""运行：python -m streamlit run streamlit_app.py --server.address 127.0.0.1"""

from datetime import date, timedelta, datetime, timezone
from time import monotonic
from uuid import uuid4
import streamlit as st

from agents.deepseek import AgentError
from ui.report_view import show_report
from memory import MemoryStoreError
from ui import service
from ui.charts import market_chart
from agents.diagnostics import diagnostic


def external_text(value):
    # 外部正文用纯文本显示，避免把PDF里的HTML、图片或指令当作页面行为。
    st.text(service.public_text(value))


def render():
    st.set_page_config(page_title="StockPilot · 股票分析", page_icon="📈", layout="wide")
    st.title("StockPilot")
    st.caption("A股研究工作台 · 真实数据 · 可追溯分析")
    stock_from_market = st.session_state.pop("stock_from_market", None)
    if stock_from_market:
        st.session_state["symbol"] = stock_from_market
    st.session_state.setdefault("symbol", "600519")
    with st.sidebar:
        st.subheader("分析设置")
        with st.form("analysis", enter_to_submit=False):
            stock_query = st.text_input("股票代码或名称", value=None, key="symbol", max_chars=60,
                help="输入六位 A 股代码或股票简称，例如 600519 或 贵州茅台。输入部分简称可查看候选。")
            start = st.date_input("开始日期", date.today() - timedelta(days=365), key="start")
            end = st.date_input("结束日期", date.today(), key="end")
            mode = st.selectbox("分析模式", ["本地规则", "DeepSeek"], key="mode")
            research = st.checkbox("启用财务新闻研究", key="research")
            question = st.text_area("财报或公告正文问题（可选）", key="question", max_chars=500,
                help="只有填写问题才启用RAG；使用已导入的本地PDF索引。")
            submitted = st.form_submit_button("开始分析", type="primary")
        st.caption("DeepSeek读取本机环境变量，可能产生费用。相同参数在当前会话复用成功结果；普通页面重跑不发起分析。")
    if submitted:
        # 每次显式提交有独立标识；rerun只展示，不创建新请求。
        attempt = {"id": uuid4().hex[:8], "time": datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S"),
            "symbol": stock_query, "mode": mode, "status": "执行中"}
        st.session_state["last_attempt"] = attempt
        started = monotonic()
        st.session_state.pop("analysis_error", None)
        st.session_state.pop("failure_diagnostic", None)
        try:
            with st.spinner(f"任务 {attempt['id']}：正在查询股票并准备分析，请等待…"):
                symbol, name = service.resolve_stock(st.session_state, stock_query)
                attempt["symbol"] = f"{name}（{symbol}）" if name else symbol
                fresh = service.analyze_once(st.session_state, symbol, start.isoformat(), end.isoformat(), mode, research, question)
            if not fresh:
                st.info("已复用本会话相同参数的成功结果，没有重复请求或保存。")
        except ValueError as exc:
            st.session_state.analysis_error = service.public_text(exc)
        except MemoryStoreError as exc:
            st.session_state.analysis_error = "报告保存失败，请检查本地数据库及历史记录。不会自动重试分析。"
            st.session_state.failure_diagnostic = getattr(exc, "diagnostic", None)
        except AgentError as exc:
            st.session_state.analysis_error = "分析未成功完成，不会自动重试。"
            st.session_state.failure_diagnostic = getattr(exc, "diagnostic", None)
        except Exception:
            st.session_state.analysis_error = "分析服务暂时不可用，请检查本机依赖及网络后手动重试。"
        finally:
            attempt["elapsed"] = round(monotonic() - started, 2)
            attempt["status"] = "失败" if st.session_state.get("analysis_error") else "完成" if fresh else "复用成功结果"
    attempt = st.session_state.get("last_attempt")
    if attempt:
        st.caption(f"最近提交 {attempt['time']}（北京时间） · 任务 {attempt['id']} · {service.public_text(attempt['symbol'])} · {attempt['mode']} · {attempt['status']} · 耗时 {attempt.get('elapsed', 0):.2f} 秒")
    if st.session_state.get("analysis_error"):
        st.error(st.session_state.analysis_error)
        failure = st.session_state.get("failure_diagnostic")
        if isinstance(failure, dict):
            safe = diagnostic(failure.get("failed_node"), failure.get("category"), failure.get("trace", []))
            st.text(f"失败位置：{safe['failed_node']} · 错误类别：{safe['category']}\n{safe['message']}")
            st.text("本次失败的实际执行轨迹：\n" + ("\n".join(safe["trace"]) or "尚未进入工作流，无节点执行记录。"))
            if safe["failed_node"] == "persistence":
                st.caption("工作流已结束，失败发生于图外SQLite保存阶段；上述trace不是新的存储节点。")
        if st.session_state.get("result"):
            st.info("下方保留的是上一次成功分析，本次失败没有替换该结果。")
    result = st.session_state.get("result")
    if result:
        st.caption(f"当前展示已完成分析：{result['symbol']} · 请求 {result['start_date']} 至 {result['end_date']}；与尚未提交的表单独立。")
    market, report, rag, history = st.tabs(["行情与指标", "报告与执行轨迹", "RAG原文", "历史报告"])
    with market:
        if result:
            data = result["history"]
            cols = st.columns(3)
            cols[0].metric("最新收盘价（元）", f"{data.iloc[-1]['close']:.2f}")
            cols[1].metric("实际行情截止日期", f"{data['date'].max():%Y-%m-%d}")
            cols[2].metric("真实交易记录", len(data))
            st.caption(f"来源：{result['market_facts'].get('source', 'AKShare / 东方财富')} · 不复权 · 成交量单位手；指标使用开始日前最多365个自然日的真实历史预热，仅展示所选区间；历史不足留空。")
            if result["market_facts"].get("source_warning"):
                st.warning(result["market_facts"]["source_warning"])
            st.plotly_chart(market_chart(data), width="stretch")
            with st.expander("查看行情与指标明细"):
                st.dataframe(data, hide_index=True, width="stretch")
        else:
            st.info("请填写参数并点击“开始分析”，当前没有加载行情。")
    with report:
        if result:
            show_report(result["report"], result=result, scope="current")
            with st.expander("查看实际执行轨迹"):
                st.caption("已保存报告 · " + result["report_id"])
                external_text("\n".join(f"{i}. {step}" for i, step in enumerate(result.get("trace", []), 1)))
            sources = [("财务", result.get("financial_facts"))]
            sources += [(label, result.get("news_facts", {}).get(key)) for label, key in [("新闻", "news"), ("公告", "notices")]]
            for label, source in sources:
                if source is not None and source.get("status") != "ok":
                    st.warning(service.public_text(f"{label}来源状态：{source.get('status', '未知')}。{source.get('coverage', '')} {source.get('error', '')}"))
            for name in ["financial_facts", "news_facts", "rag_facts"]:
                if name in result:
                    with st.expander(name + " · 来源、状态与缺失说明"):
                        external_text(result[name])
        else:
            st.info("分析完成后展示报告与真实执行轨迹。")
    with rag:
        facts = result.get("rag_facts") if result else None
        if not facts:
            st.info("本次未启用正文检索。填写财报或公告正文问题后提交分析即可启用。")
        else:
            external_text("问题：" + facts["question"])
            external_text("检索状态：" + facts["status"] + " · " + facts.get("message", ""))
            if facts["status"] != "ok":
                st.warning("没有取得可用原文，不编造财报或公告内容。")
            for hit in facts.get("records", []):
                with st.expander(f"[{hit['ref_id']}] {service.public_text(hit['document_name'])} · 物理第{hit['page']}页 · {hit['score']:.3f}", expanded=True):
                    external_text(hit["text"])
                    st.caption("文档标识：" + hit["document_id"] + " · 片段：" + hit["chunk_id"])
            external_text(facts.get("limitations", ""))
    with history:
        st.subheader("已保存的分析报告")
        filter_symbol = st.text_input("历史股票筛选（留空为全部）", key="history_symbol", max_chars=6)
        try:
            if filter_symbol and (len(filter_symbol) != 6 or not filter_symbol.isascii() or not filter_symbol.isdigit()):
                st.info("历史筛选请输入六位数字。")
            else:
                rows = service.list_reports(filter_symbol or None)
                if not rows:
                    st.info("暂无匹配的历史报告。")
                else:
                    st.dataframe(rows, hide_index=True, width="stretch")
                    labels = {row["id"]: f"{row['symbol']} · {row['generated_at']} · {row['mode']} · 截止{row['market_end_date']}" for row in rows}
                    chosen = st.selectbox("查看完整历史报告", list(labels), format_func=labels.get, key="report_id")
                    saved = service.read_report(chosen)
                    if saved:
                        show_report(saved["report"], scope="history")
                    else:
                        st.warning("该报告已不存在，请刷新历史列表。")
        except Exception:
            st.error("历史报告暂时无法读取，请检查本机数据库。历史查询不会调用行情接口或模型。")


if __name__ == "__main__":
    render()
