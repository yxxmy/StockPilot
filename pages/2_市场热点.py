"""StockPilot 市场热点页。"""

import plotly.graph_objects as go
import streamlit as st

from tools import market_hotspots


_RETURN = "涨跌幅（%）"
_AMOUNT = "成交额（亿元）"
_PERIODS = {1: "单日", 5: "近 5 个交易日", 20: "近 20 个交易日"}


def _caption(snapshot):
    reference = snapshot["reference_date"]
    fetched = snapshot["fetched_at"].strftime("%Y-%m-%d %H:%M:%S")
    st.caption(f"{snapshot['source']} · 最近已收盘参考日 {reference or '无法获取'} · 抓取时间 {fetched}（北京时间）")


def _table(rows, **selection_options):
    return st.dataframe(
        rows, hide_index=True, width="stretch",
        column_config={
            label: st.column_config.NumberColumn(format="%.2f")
            for label in (
                _RETURN, _AMOUNT, "换手率（%）", "领涨股涨跌幅（%）",
                *market_hotspots.PERIOD_RETURNS.values(), *market_hotspots.PERIOD_AMOUNTS.values(),
            )
        },
        **selection_options,
    )


def render():
    st.set_page_config(page_title="StockPilot · 市场热点", layout="wide")
    st.title("市场热点")
    days = st.selectbox("榜单周期", list(_PERIODS), format_func=_PERIODS.get, key="ranking_period")
    multi_day = days != 1
    industry_key = market_hotspots.HISTORICAL_INDUSTRY if multi_day else "行业"
    return_column = market_hotspots.PERIOD_RETURNS[days] if multi_day else _RETURN
    amount_column = market_hotspots.PERIOD_AMOUNTS[days] if multi_day else _AMOUNT
    if st.button("刷新榜单", type="primary"):
        message = "正在获取个股快照和行业日线，多日行业榜加载时间较长…" if multi_day else "正在获取个股与行业行情…"
        with st.spinner(message):
            saved = st.session_state.get("market_rankings", {})
            snapshots, failed = market_hotspots.refresh_rankings(saved, multi_day=multi_day)
            st.session_state["market_rankings"] = snapshots
            previous_failures = set(st.session_state.get("market_rankings_failed", []))
            st.session_state["market_rankings_failed"] = previous_failures.difference({"个股", industry_key}).union(failed)

    snapshots = st.session_state.get("market_rankings", {})
    failed = st.session_state.get("market_rankings_failed", [])
    for label in ("个股", industry_key):
        if label not in failed:
            continue
        message = "当前显示本会话上次成功的快照。" if label in snapshots else "暂时没有可用数据。"
        st.warning(f"{label}榜刷新失败。{message}")
    if not snapshots:
        if not failed:
            st.info("点击「刷新榜单」获取个股和行业行情。")
        return

    if "个股" in snapshots or not multi_day:
        st.caption("行情日期未确认，快照接口未提供所属交易日。已收盘参考日来自上证指数日线，盘中刷新可能返回当日未收盘行情。")
    if stock := snapshots.get("个股"):
        st.subheader("沪深京 A 股领涨前 10")
        _caption(stock)
        columns = ["代码", "名称", return_column] if multi_day else ["代码", "名称", _RETURN, _AMOUNT, "换手率（%）"]
        quotes = stock["quotes"].reindex(columns=columns)
        available = quotes[return_column].count()
        st.caption(f"涨幅可用 {available:,} / {len(quotes):,} 只 A 股。")
        if multi_day:
            st.caption("多日涨幅取自腾讯证券快照，复权口径由数据源决定。")
        leaders = market_hotspots.ranking(quotes[quotes[return_column].gt(0)], by=return_column)
        if not available:
            st.info("本次快照未提供所选周期的涨跌幅。")
        elif leaders.empty:
            st.info("本次快照中没有录得上涨的股票。")
        else:
            # 每次成功刷新重置行选择，防止同一行位置在新榜单指向另一只股票。
            selection = _table(
                leaders, key=f"market_leaders_{days}_{stock['fetched_at'].isoformat()}",
                on_select="rerun", selection_mode="single-row",
            ).selection.rows
            has_selection = bool(selection) and 0 <= selection[0] < len(leaders)
            if st.button("打开个股分析", key="open_market_stock", disabled=not has_selection):
                st.session_state["stock_from_market"] = leaders.iloc[selection[0]]["代码"]
                st.switch_page("streamlit_app.py")

    if industry := snapshots.get(industry_key):
        st.subheader("行业热度前 10")
        _caption(industry)
        quotes = industry["quotes"]
        if multi_day:
            dates = industry["dates"]
            if len(dates) <= days:
                st.info("行业日线交易日不足，暂时无法计算所选周期。")
                return
            st.caption(f"统计区间 {dates[-days]} 至 {dates[-1]} · 涨幅基于 {dates[-days - 1]} 收盘价")
            quotes = quotes[["行业", return_column, amount_column]]
        order = st.radio("行业排序", [return_column, amount_column], horizontal=True, key=f"industry_order_{days}")
        ranked = market_hotspots.ranking(quotes, by=order)
        st.caption(f"所选指标可用 {quotes[order].count():,} / {len(quotes):,} 个同花顺行业。")
        if ranked.empty:
            st.info("暂无可按该指标排序的行业。")
            return
        values = ranked[order]
        colors = ["#dc2626" if value > 0 else "#059669" if value < 0 else "#64748b" for value in values]
        chart = go.Figure(go.Bar(
            x=values, y=ranked["行业"], orientation="h",
            marker_color=colors if order == return_column else "#0f766e",
            hovertemplate="%{y} %{x:.2f}" + ("%" if order == return_column else " 亿元") + "<extra></extra>",
        ))
        chart.update_yaxes(autorange="reversed")
        chart.update_layout(height=350, margin={"l": 12, "r": 12, "t": 12, "b": 24},
                            xaxis_title=order)
        st.plotly_chart(chart, width="stretch")
        _table(ranked)
    elif industry_key not in failed:
        st.info("点击「刷新榜单」获取所选周期的行业数据。")


if __name__ == "__main__":
    render()
