"""在网页上呈现简报，直接使用工作流或已保存报告的事实数据。"""

import json
import re

import streamlit as st

from agents.report_text import report_for_display
from tools.report_industry import RANKING_SOURCE, industry_context, industry_snapshot, match_current_industry, report_end
from ui.charts import report_trend_chart
from ui.report_events import event_highlights, show_event_highlights
from ui.report_share import (
    INDUSTRY_MANUAL, INDUSTRY_MEMBERSHIP, INDUSTRY_METHOD, industry_details, share_html, summary_metrics,
)
from ui.service import public_text


def _render_summary(market, metrics):
    dates = market["actual_range"]
    st.subheader(f"{public_text(market.get('symbol', ''))} · 个股研究简报")
    source = public_text(market.get("source") or "来源未记录")
    st.caption(f"行情区间 {public_text(dates[0])} 至 {public_text(dates[-1])} · {source} · 不复权")
    if warning := market.get("source_warning"):
        st.warning(public_text(warning))

    for column, (label, value, delta, note) in zip(st.columns(len(metrics)), metrics):
        with column.container(border=True):
            st.metric(label, value, delta=delta)
            st.caption(note)


def _render_industry(market, scope):
    st.subheader("行业背景")
    symbol = str(market.get("symbol", ""))
    end = report_end(market)
    if end is None or not re.fullmatch(r"[0-9]{6}", symbol):
        st.caption("报告缺少股票代码或有效的行情截止日。")
        return None
    report_key = (symbol, end)
    state_key = f"report_industry_{scope}"
    state = st.session_state.get(state_key, {})
    if state.get("report_key") != report_key:
        state = {}
    st.caption(f"比较截至 {end:%Y-%m-%d} 的近 5 和 20 个交易日。首次加载可能需要约一分钟。")
    if not state.get("snapshot") and st.button("加载行业排名", key=f"load_{state_key}"):
        state = {"report_key": report_key}
        try:
            with st.spinner("正在读取对应交易日的行业数据…"):
                snapshot = industry_snapshot(end, st.session_state.get("market_rankings", {}))
                if snapshot:
                    names = set(snapshot["quotes"]["行业"])
                    state.update(snapshot=snapshot, membership=match_current_industry(symbol, names))
                else:
                    state["message"] = "该截止日没有足够的已收盘交易数据用于排名。"
        except Exception:
            state["message"] = "暂时无法读取该日期的行业排名。可以稍后重新加载。"
        st.session_state[state_key] = state

    snapshot = state.get("snapshot")
    if not snapshot:
        if state.get("message"):
            st.warning(state["message"])
        return None

    membership = state.get("membership")
    names = sorted(snapshot["quotes"]["行业"].tolist())
    preferred = membership["name"] if membership else None
    if preferred is None:
        st.caption("当前公开资料未能核实所属行业，可以手动选择一个对照行业。")
    name = st.selectbox("对照行业", names, index=names.index(preferred) if preferred else None,
                        placeholder="请选择行业", key=f"{state_key}_{symbol}_{end.isoformat()}")
    if name is None:
        return None
    context = industry_context(snapshot, name, membership)
    if context is None:
        st.info("所选行业缺少该周期的完整数据。")
        return None
    for detail in industry_details(context):
        st.text(detail)
    st.caption(INDUSTRY_METHOD)
    st.link_button("查看同花顺行业", RANKING_SOURCE)
    if context["membership"]:
        st.caption(INDUSTRY_MEMBERSHIP)
        st.link_button("核对当前成分股名单", context["membership"]["source"])
    else:
        st.caption(INDUSTRY_MANUAL)
    return context


def show_report(value, *, scope, result=None):
    body, evidence = report_for_display(value)
    # 两种展示都复用报告附录中的研究材料。
    try:
        facts = json.loads(evidence) if evidence is not None else {}
    except ValueError:
        facts = {}
    facts = facts if isinstance(facts, dict) else {}
    if result:
        market, risk = result.get("market_facts", {}), result.get("risk_facts", {})
    else:
        # 历史报告的附录已经保存了用于展示的收盘、成交和风险数字。
        market, risk = facts.get("market"), facts.get("risk")
    metrics = summary_metrics(market, risk if isinstance(risk, dict) else {}) if isinstance(market, dict) else []
    if metrics:
        _render_summary(market, metrics)
        history = result.get("history") if result else None
        if history is not None and not history.empty:
            st.markdown("**收盘价与均线**")
            st.plotly_chart(report_trend_chart(history), width="stretch")
        else:
            st.caption("这份已保存的报告包含摘要和指标。可重新分析该区间查看逐日走势图。")
        industry = _render_industry(market, scope)
        events = event_highlights(facts, market)
        show_event_highlights(facts, market, items=events)
        symbol = str(market.get("symbol", ""))
        filename = f"StockPilot_{symbol}_{report_end(market)}.html" if re.fullmatch(r"[0-9]{6}", symbol) else "StockPilot.html"
        st.download_button("下载分享版 HTML", share_html(market, metrics, events, industry).encode("utf-8"),
                           file_name=filename, mime="text/html", key=f"share_report_{scope}", on_click="ignore")
        with st.expander("查看完整报告"):
            st.text(public_text(body))
    else:
        st.text(public_text(body))
    if evidence is not None:
        with st.expander("查看原始数据（核对用）"):
            st.text(public_text(evidence))
