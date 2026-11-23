"""8C：固定设定、按日期滚动拟合，保留失败日期和可审计实验。"""

from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import tempfile
from time import monotonic
from uuid import uuid4

import numpy as np

from forecasting.models import RollingVariance, GarchVariance, ForecastModelError
from forecasting.targets import prepare_targets, history_at


CONFIG = {"garch_window":500, "baseline_window":20, "validation_size":60, "test_size":60,
    "mean":"Zero", "vol":"GARCH(1,1)", "distribution":"normal", "rescale":False,
    "maxiter":500, "horizon":1, "tuning":False}
ERROR_CATEGORIES = {"model_error", "invalid_input", "insufficient_data", "degenerate_data", "dependency_error",
    "not_converged", "invalid_parameters", "invalid_forecast", "fit_error", "not_fitted", "invalid_target"}


def _json(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)


def scores(actual, predicted):
    """actual为收益平方代理；QLIKE允许actual为0，但预测必须严格正。"""
    actual, predicted = np.asarray(actual,dtype=float), np.asarray(predicted,dtype=float)
    if actual.ndim != 1 or predicted.shape != actual.shape or not len(actual):
        raise ValueError("评估样本须为同长非空一维数组")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all() or (actual<0).any() or (predicted<=0).any():
        raise ValueError("评估输入必须有限，实际方差非负、预测方差严格正")
    with np.errstate(over="ignore",invalid="ignore",divide="ignore"):
        mse=float(np.mean(np.square(predicted-actual)))
        qlike=float(np.mean(np.log(predicted)+actual/predicted))
    if not np.isfinite([mse,qlike]).all():
        raise ValueError("误差计算溢出，未生成无穷指标")
    return {"n":len(actual),"mse":mse,"qlike":qlike}


def _summarise(rows):
    summary={}
    for split in ["validation","test"]:
        subset=[r for r in rows if r["split"]==split]
        both=[r for r in subset if all(p["status"]=="ok" for p in r["predictions"].values())]
        item={"target_count":len(subset),"paired_count":len(both),"models":{}}
        for name in [RollingVariance.name,GarchVariance.name]:
            valid=[r for r in subset if r["predictions"][name]["status"]=="ok"]
            def metric(part):
                if not part: return None
                return scores([r["variance_proxy"] for r in part],[r["predictions"][name]["variance"] for r in part])
            item["models"][name]={"success_count":len(valid),"failure_count":len(subset)-len(valid),
                "all_success_metrics":metric(valid),"paired_metrics":metric(both)}
        summary[split]=item
    return summary


def _estimate_next(returns, factories):
    origin = returns.date.iloc[-1]
    history = history_at(returns, origin)
    predictions = {}
    for name, factory in factories.items():
        try:
            predictions[name] = factory().fit(history.copy(), origin).estimate()
        except ForecastModelError as exc:
            category = exc.category if exc.category in ERROR_CATEGORIES else "model_error"
            predictions[name] = {"status": "error", "category": category}
        except Exception:
            predictions[name] = {"status": "error", "category": "model_error"}
    return {"origin_date": origin.strftime("%Y-%m-%d"), "predictions": predictions}


def evaluate_snapshot(snapshot, *, model_factories=None, include_latest=False):
    """固定模型、不调参；测试可替换模型边界验证失败和时间约束。"""
    started=monotonic()
    prepared=prepare_targets(snapshot)
    factories=model_factories or {RollingVariance.name:RollingVariance,GarchVariance.name:GarchVariance}
    if set(factories)!={RollingVariance.name,GarchVariance.name}:
        raise ValueError("须同时评估预定基线与GARCH")
    config=json.loads(_json(CONFIG))
    locked_hash=sha256(_json(config).encode()).hexdigest()
    rows=[]
    # 先全部验证，再运行测试；无测试结果反馈到配置或模型选择。
    for split in ["validation","test"]:
        for target in prepared["targets"].loc[lambda f:f.split==split].itertuples():
            origin=target.origin_date.strftime("%Y-%m-%d")
            date=target.target_date.strftime("%Y-%m-%d")
            history=history_at(prepared["returns"],target.origin_date)
            if len(history)!=500 or history.date.max()>=target.target_date:
                raise ValueError("滚动窗口不足或包含目标日")
            predictions={}
            for name,factory in factories.items():
                try:
                    # 每次独立拟合；输入只含历史收益，不传入未来标签。
                    result=factory().fit_predict(history.copy(),target.origin_date,target.target_date)
                    window=20 if name==RollingVariance.name else 500
                    expected=history.tail(window)
                    expected_hash=sha256(json.dumps([[d.strftime("%Y-%m-%d"),float(r)] for d,r in zip(expected.date,expected.return_pct)],allow_nan=False).encode()).hexdigest()
                    if (result.get("status")!="ok" or result.get("model")!=name or result.get("origin_date")!=origin or result.get("target_date")!=date
                        or result.get("training_records")!=window or result.get("training_hash")!=expected_hash
                        or result.get("training_range")!=[expected.date.iloc[0].strftime("%Y-%m-%d"),origin]
                        or result.get("convergence")!=("not_applicable" if name==RollingVariance.name else "converged")):
                        raise ForecastModelError("预测与窗口契约不一致","invalid_forecast")
                    scores([0.],[result["variance"]])
                    if not np.isclose(result["stddev"]**2,result["variance"]):
                        raise ForecastModelError("标准差与方差不一致","invalid_forecast")
                    predictions[name]=result
                except ForecastModelError as exc:
                    category=exc.category if exc.category in ERROR_CATEGORIES else "model_error"
                    predictions[name]={"status":"error","category":category,"message":"本日模型未产生有效预测，已保留失败记录，未使用替代值。"}
                except Exception:
                    predictions[name]={"status":"error","category":"model_error","message":"本日模型执行或输出校验失败，未生成替代预测。"}
            rows.append({"split":split,"origin_date":origin,"target_date":date,
                "realized_return_pct":float(target.realized_return_pct),"variance_proxy":float(target.variance_proxy),"predictions":predictions})
    metadata=json.loads(_json(snapshot["body"]["metadata"]))
    successes=sum(p["status"]=="ok" for r in rows for p in r["predictions"].values())
    status="completed" if successes==2*len(rows) else "completed_with_failures" if successes else "failed"
    experiment = {"schema_version":1,"status":status,"run_id":str(uuid4()),"created_at":datetime.now(timezone.utc).isoformat(),
        "snapshot_id":snapshot["snapshot_id"],"data_metadata":metadata,"config":config,"locked_config_hash":locked_hash,
        "protocol":"验证60条后按相同固定设定评估测试60条；未调参，测试误差不参与模型选择。后续窗口可使用当时已实现的先前收益。",
        "versions":{p:version(p) for p in ["arch","numpy","pandas","scipy"]},
        "rows":rows,"summary":_summarise(rows),"elapsed_seconds":round(monotonic()-started,3),
        "limitation":prepared["limitation"]+" 固定供应商快照不等于历史时点数据；60个测试点不足以证明稳定优势，不评估交易收益。所有成功日期指标与共同成功日期指标分开，不隐藏失败。"}
    if include_latest:
        experiment["next_observation"] = _estimate_next(prepared["returns"], factories)
        experiment["elapsed_seconds"] = monotonic() - started
    return experiment


def experiment_figure(experiment):
    """独立HTML图表，不接Streamlit。失败日期留空，不连线补值。"""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    fig=make_subplots(rows=2,cols=1,subplot_titles=["验证段（固定设定）","测试段（固定设定）"])
    for i,split in enumerate(["validation","test"],1):
        rows=[r for r in experiment["rows"] if r["split"]==split]
        dates=[r["target_date"] for r in rows]
        fig.add_trace(go.Scatter(x=dates,y=[abs(r["realized_return_pct"]) for r in rows],name="事后绝对收益（高噪声代理）",line_color="#64748b",legendgroup="actual",showlegend=i==1),row=i,col=1)
        for name,label,color in [(RollingVariance.name,"历史20日基线","#d97706"),(GarchVariance.name,"GARCH","#2563eb")]:
            values=[r["predictions"][name].get("stddev") for r in rows]
            fig.add_trace(go.Scatter(x=dates,y=values,name=label,line_color=color,legendgroup=name,showlegend=i==1,connectgaps=False),row=i,col=1)
        fig.update_yaxes(title_text="百分比（未年化）",row=i,col=1)
    fig.update_layout(height=800,title="StockPilot 波动率实验：预测标准差与事后绝对收益",template="plotly_white")
    return fig


def save_experiment(experiment, directory):
    """唯一目录内原子发布结果、离线HTML与文件哈希；不覆盖原实验。"""
    from uuid import UUID
    if str(UUID(experiment["run_id"]))!=experiment["run_id"]:
        raise ValueError("实验ID无效")
    root=Path(directory)
    root.mkdir(parents=True,exist_ok=True)
    destination=root/experiment["run_id"]
    if destination.exists():
        raise ValueError("实验ID已存在，拒绝覆盖")
    with tempfile.TemporaryDirectory(prefix=".experiment-",dir=root) as temp:
        staging=Path(temp)/"result"
        staging.mkdir()
        (staging/"experiment.json").write_text(_json(experiment),encoding="utf-8")
        experiment_figure(experiment).write_html(str(staging/"comparison.html"),include_plotlyjs=True,auto_open=False)
        hashes={name:sha256((staging/name).read_bytes()).hexdigest() for name in ["experiment.json","comparison.html"]}
        (staging/"manifest.json").write_text(_json(hashes),encoding="utf-8")
        staging.rename(destination)
    return destination
