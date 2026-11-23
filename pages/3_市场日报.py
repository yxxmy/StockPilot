"""StockPilot 独立市场日报页。"""

import streamlit as st

from tools.market_daily import fetch_daily_report
from ui.market_daily import show_daily


def render():
    st.set_page_config(page_title="StockPilot · 市场日报", page_icon="📊", layout="wide")
    st.title("市场日报")
    st.caption("查看最近交易日的指数与行业表现，以及腾讯证券最新个股快照。")
    if st.button("生成市场日报", type="primary"):
        with st.spinner("正在读取指数、行业日线和个股快照，首次加载可能需要约一分钟…"):
            report = fetch_daily_report(st.session_state.get("market_rankings", {}))
        if report is None:
            st.session_state["daily_refresh_failed"] = True
        else:
            st.session_state["market_daily"] = report
            st.session_state.pop("daily_refresh_failed", None)

    report = st.session_state.get("market_daily")
    if st.session_state.get("daily_refresh_failed"):
        message = "当前展示本次会话上一份成功生成的日报。" if report else "可以稍后重新生成。"
        st.warning("行情暂时无法获取。" + message)
    if report:
        show_daily(report)


if __name__ == "__main__":
    render()
