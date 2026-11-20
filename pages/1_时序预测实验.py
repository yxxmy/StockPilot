"""8D独立页面，原有股票分析入口不变。"""
from datetime import date,timedelta
from time import monotonic
import json

import pandas as pd
import streamlit as st

from forecasting.data import ForecastDataError
from forecasting.evaluation import experiment_figure
from ui import forecast_service as service


def render():
    st.set_page_config(page_title="StockPilot · 时序预测实验",page_icon="📉",layout="wide")
    st.title("时序预测实验")
    st.caption("固定配置：20日历史基线与500日GARCH · 验证60日、测试60日 · 不调用DeepSeek")
    mode=st.radio("数据方式",["已保存快照（离线）","联网创建快照"])
    chosen=None
    if mode.startswith("已保存"):
        entries,errors=service.snapshots()
        if errors: st.warning(f"有{errors}份快照无法校验，已排除；未猜测或修复内容。")
        if entries:
            labels={s["snapshot_id"]:f"{s['body']['metadata']['symbol']} · {s['body']['metadata']['actual_range'][0]} 至 {s['body']['metadata']['actual_range'][1]} · {s['snapshot_id'][:12]}" for _,s in entries}
            identifier=st.selectbox("选择股票及日期快照",list(labels),format_func=labels.get)
            chosen=next(s for _,s in entries if s["snapshot_id"]==identifier)
            meta=chosen["body"]["metadata"]
            st.caption(f"来源：{meta['source_name']} · 前复权 · {meta['record_count']}条价格 · {meta['data_kind']}")
        else: st.info("没有可用快照。请先按8A准备，或选择联网创建。至少需要621条有效价格。")
    with st.form("forecast_run"):
        if mode.startswith("联网"):
            symbol=st.text_input("股票代码","600519",max_chars=6)
            start=st.date_input("开始日期",date(2023,1,1))
            end=st.date_input("结束日期",date.today()-timedelta(days=1))
            source=st.selectbox("单一数据源",["tencent","eastmoney"])
            refresh=st.checkbox("重新获取行情快照（额外联网请求）")
        force=st.checkbox("强制新建实验（相同快照也重新训练）")
        submitted=st.form_submit_button("运行滚动实验",type="primary",disabled=mode.startswith("已保存") and chosen is None)
    if submitted:
        st.session_state.pop("forecast_error",None)
        begin=monotonic()
        try:
            with st.spinner("正在准备数据并逐日拟合，请等待；不会调用付费API…"):
                if mode.startswith("联网"):
                    chosen=service.acquire(st.session_state,symbol.strip(),start.isoformat(),end.isoformat(),source,refresh)
                fresh=service.run_once(st.session_state,chosen,force)
            st.session_state["forecast_notice"]="已生成独立实验记录。" if fresh else "已复用本会话相同快照与配置的结果，没有重复训练或保存。"
        except ForecastDataError as exc:
            st.session_state["forecast_error"]=str(exc)
        except Exception:
            st.session_state["forecast_error"]="实验执行或保存失败，请检查模型依赖、快照及输出目录；不会自动重试。"
        finally:
            st.session_state["forecast_elapsed"]=round(monotonic()-begin,2)
    if st.session_state.get("forecast_error"):
        st.error(st.session_state["forecast_error"])
        if st.session_state.get("forecast_result"): st.info("下方保留的是上一次已保存实验，本次失败未替换旧结果。")
    elif st.session_state.get("forecast_notice"): st.info(st.session_state["forecast_notice"])
    bundle=st.session_state.get("forecast_result")
    if not bundle: return
    exp=bundle["experiment"]
    meta=exp["data_metadata"]
    st.caption(f"当前结果：{meta['symbol']} · {meta['actual_range']} · {meta['source_name']} · {meta['adjustment']} · 获取于 {meta['fetched_at']}")
    st.text(f"实验ID：{exp['run_id']}\n快照ID：{exp['snapshot_id']}\n状态：{exp['status']} · 最近提交耗时 {st.session_state.get('forecast_elapsed',0):.2f} 秒")
    if exp["status"]!="completed": st.warning("存在失败预测，请同时查看失败数和共同样本数。")
    for split,label in [("validation","验证段"),("test","测试段")]:
        rows=[r for r in exp["rows"] if r["split"]==split]
        st.subheader(label)
        st.caption(f"目标区间 {rows[0]['target_date']} 至 {rows[-1]['target_date']}；共同成功日期 {exp['summary'][split]['paired_count']} / {len(rows)}。指标越低越好，但不等于交易收益。")
        display=[]
        for name,stats in exp["summary"][split]["models"].items():
            metric=stats["paired_metrics"] or {}
            display.append({"模型":name,"成功数":stats["success_count"],"失败数":stats["failure_count"],"共同日期MSE":metric.get("mse"),"共同日期QLIKE":metric.get("qlike")})
        st.dataframe(pd.DataFrame(display),hide_index=True)
    st.plotly_chart(experiment_figure(exp),width="stretch")
    with st.expander("逐日预测、训练范围与失败记录"):
        st.json(exp["rows"])
    with st.expander("固定配置及口径限制"):
        st.json(exp["config"])
        st.text(meta["limitation"]+"\n"+exp["limitation"])
    st.download_button("下载完整实验JSON",data=json.dumps(exp,ensure_ascii=False,indent=2,allow_nan=False),file_name=f"forecast_{exp['run_id']}.json",mime="application/json")


render()
