"""个股波动预测页面。"""

from datetime import datetime, timedelta, timezone
import json

import pandas as pd
import streamlit as st

from forecasting.evaluation import experiment_figure
from forecasting.models import GarchVariance, RollingVariance
from ui import forecast_service, service


_MODELS = {RollingVariance.name: "20 日历史法", GarchVariance.name: "GARCH 波动模型"}
_SOURCES = {"auto": "优先用已保存行情", "tencent": "联网获取 · 腾讯证券", "eastmoney": "联网获取 · 东方财富"}


def _metadata(snapshot):
    return snapshot["body"]["metadata"]


def _saved_order(snapshot):
    metadata = _metadata(snapshot)
    return metadata["data_kind"] == "live", metadata["actual_range"][-1], metadata["fetched_at"]


def _show_results(experiment):
    metadata = experiment["data_metadata"]
    start, end = metadata["actual_range"]
    st.subheader(f"{metadata['symbol']} · 下一交易日的波动估计")
    st.caption(f"根据 {start} 至 {end} 的行情计算 · {metadata['source_name']}")
    if metadata["data_kind"] == "test_sample":
        st.warning("当前使用示例数据。")

    prediction = experiment.get("next_observation")
    if prediction:
        for column, (name, label) in zip(st.columns(2), _MODELS.items()):
            model = prediction["predictions"][name]
            value = f"{model['stddev']:.2f}%" if model["status"] == "ok" else "暂无法估计"
            column.metric(label, value)
        st.caption(f"预测对应 {prediction['origin_date']} 收盘之后的下一条交易日行情。数值越高，表示预计波动越大。")
    else:
        st.info("点击「开始预测」可用这份行情计算下一交易日的估计。")

    st.subheader("过去预测得怎么样")
    test = experiment["summary"]["test"]
    paired = test["paired_count"]
    if paired:
        st.caption(f"最近 {test['target_count']} 个历史测试日里，有 {paired} 天两种方法都算出了结果。")
        baseline = test["models"][RollingVariance.name]["paired_metrics"]
        garch = test["models"][GarchVariance.name]["paired_metrics"]
        if all(baseline[score] < garch[score] for score in ("mse", "qlike")):
            st.write("在这些日期上，20 日历史法的两项误差都更低。")
        elif all(garch[score] < baseline[score] for score in ("mse", "qlike")):
            st.write("在这些日期上，GARCH 波动模型的两项误差都更低。")
        else:
            st.write("两项误差没有给出一致的优胜方法。")
    else:
        st.info("历史测试中，两种方法都成功算出结果的日期为 0，暂时无法比较。")

    with st.expander("查看历史对照图"):
        st.caption("灰线是后来实际发生的涨跌幅大小，彩色线是当天之前算出的波动估计。")
        st.plotly_chart(experiment_figure(experiment), width="stretch")
    with st.expander("查看误差、数据来源和计算明细"):
        for split, label in (("validation", "验证段"), ("test", "测试段")):
            st.markdown(f"**{label}**")
            metrics = []
            for name, stats in experiment["summary"][split]["models"].items():
                scores = stats["paired_metrics"] or {}
                metrics.append({
                    "方法": _MODELS[name], "成功数": stats["success_count"], "失败数": stats["failure_count"],
                    "共同日期 MSE": scores.get("mse"), "共同日期 QLIKE": scores.get("qlike"),
                })
            st.dataframe(pd.DataFrame(metrics), hide_index=True)
        st.caption(f"行情获取时间 {metadata['fetched_at']} · 前复权 · {metadata['record_count']} 条价格")
        st.text(metadata["limitation"] + "\n" + experiment["limitation"])
        st.json({"配置": experiment["config"], "逐日记录": experiment["rows"]}, expanded=False)
        st.download_button(
            "下载完整计算记录", json.dumps(experiment, ensure_ascii=False, indent=2, allow_nan=False),
            file_name=f"forecast_{experiment['run_id']}.json", mime="application/json",
        )


def render():
    st.set_page_config(page_title="StockPilot · 个股波动预测", page_icon="📉", layout="wide")
    st.title("个股波动预测")
    st.caption("估计下一交易日股价可能波动多大，并查看用过去行情检验的表现。")

    entries, errors = forecast_service.snapshots()
    saved = sorted((snapshot for _, snapshot in entries), key=_saved_order, reverse=True)
    default = _metadata(saved[0])["symbol"] if saved else "600519"
    query = st.text_input("股票代码或简称", default, key="forecast_stock", max_chars=60,
                          placeholder="例如 600519 或 贵州茅台")
    matching = [snapshot for snapshot in saved if _metadata(snapshot)["symbol"] == query.strip()]
    if matching:
        st.caption(f"已有此股票的行情，最新保存版本截至 {_metadata(matching[0])['actual_range'][-1]}。")
    else:
        st.caption("首次预测会联网获取历史行情，通常需要约 3 年以上的交易记录。")
    if errors:
        st.warning(f"有 {errors} 份已保存行情无法读取。")

    end = datetime.now(timezone(timedelta(hours=8))).date() - timedelta(days=1)
    start = end - timedelta(days=4 * 365)
    selected = None
    refresh = False
    with st.expander("更多设置"):
        source = st.selectbox("行情来源", list(_SOURCES), format_func=_SOURCES.get)
        if source == "auto" and len(matching) > 1:
            versions = {snapshot["snapshot_id"]: snapshot for snapshot in matching}
            version = st.selectbox(
                "已保存的行情版本", list(versions),
                format_func=lambda key: " 至 ".join(_metadata(versions[key])["actual_range"])
                + " · 保存于 " + _metadata(versions[key])["fetched_at"][:19].replace("T", " "),
            )
            selected = versions[version]
        if source != "auto":
            dates = st.columns(2)
            start = dates[0].date_input("开始日期", start)
            end = dates[1].date_input("结束日期", end)
            refresh = st.checkbox("重新下载行情")
        force = st.checkbox("重新计算已有结果")

    if st.button("开始预测", type="primary"):
        st.session_state.pop("forecast_error", None)
        try:
            with st.spinner("正在准备行情并检查过去的预测表现…", show_time=True):
                symbol, _ = service.resolve_stock(st.session_state, query)
                chosen = selected
                if source == "auto" and chosen is None:
                    chosen = next((item for item in saved if _metadata(item)["symbol"] == symbol), None)
                if chosen is None:
                    provider = "tencent" if source == "auto" else source
                    chosen = forecast_service.acquire(st.session_state, symbol, start.isoformat(), end.isoformat(),
                                                      provider, refresh)
                forecast_service.run_once(st.session_state, chosen, force)
        except ValueError as exc:
            st.session_state["forecast_error"] = service.public_text(exc)
        except Exception:
            st.session_state["forecast_error"] = "暂时无法完成计算。请检查行情来源或运行环境后重试。"

    bundle = st.session_state.get("forecast_result")
    if error := st.session_state.get("forecast_error"):
        st.error(error)
        if bundle:
            st.info("下方显示的是上一次完成的预测。")
    if bundle:
        _show_results(bundle["experiment"])


render()
