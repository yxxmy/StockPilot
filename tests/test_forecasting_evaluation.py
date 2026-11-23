"""8C离线评估；模型调度替身只用于检查时间约束/失败，真实拟合另测。"""

from hashlib import sha256
import json
from math import log

import numpy as np
import pytest

from forecasting.data import create_snapshot
from forecasting.evaluation import evaluate_snapshot,scores,save_experiment,experiment_figure
from forecasting.models import RollingVariance,GarchVariance,ForecastModelError
from test_forecasting_data import frames  # noqa:F401


def make_snapshot(frames):
    return create_snapshot(*frames,symbol="600519",source="tencent",start_date="20210101",end_date="20240101")


def doubles(calls, fail=False, corrupt=False):
    class Baseline(RollingVariance):
        def fit_predict(self,history,origin,target):
            calls.append(("baseline",origin,target,history.copy()))
            return super().fit_predict(history,origin,target)
    class FakeGarch:
        def fit_predict(self,history,origin,target):
            calls.append(("garch",origin,target,history.copy()))
            if fail:
                raise ForecastModelError("SECRET 不应显示","not_converged")
            result=Baseline().fit_predict(history,origin,target)
            records=[[d.strftime("%Y-%m-%d"),float(r)] for d,r in zip(history.date,history.return_pct)]
            result.update(model=GarchVariance.name,convergence="converged",training_records=500,
                training_range=[history.date.iloc[0].strftime("%Y-%m-%d"),origin.strftime("%Y-%m-%d")],
                training_hash=sha256(json.dumps(records,allow_nan=False).encode()).hexdigest())
            if corrupt: result["training_range"][1]=target.strftime("%Y-%m-%d")
            return result
    return {RollingVariance.name:Baseline,GarchVariance.name:FakeGarch}


def test_metric_formulas_allow_zero_actual():
    result=scores([0,4],[1,2])
    assert result["mse"]==2.5
    assert result["qlike"]==pytest.approx((log(1)+0+log(2)+2)/2)
    assert result["n"]==2


@pytest.mark.parametrize("actual,predicted",[([],[]),([1],[0]),([-1],[1]),([1],[np.nan]),([1],[np.inf]),([1,2],[1]),([1e308],[1e-308])])
def test_invalid_metrics_are_not_clipped(actual,predicted):
    with pytest.raises(ValueError): scores(actual,predicted)


def test_temporal_windows_fixed_config_and_reproducible_predictions(frames):
    calls=[]
    snap=make_snapshot(frames)
    a=evaluate_snapshot(snap,model_factories=doubles(calls))
    b=evaluate_snapshot(snap,model_factories=doubles([]))
    assert a["rows"]==b["rows"] and a["summary"]==b["summary"]
    assert a["run_id"]!=b["run_id"] and a["locked_config_hash"]==b["locked_config_hash"]
    assert [r["split"] for r in a["rows"]]==["validation"]*60+["test"]*60
    for _,origin,target,history in calls:
        assert len(history)==500 and history.date.max()==origin and origin<target
    for item in a["summary"].values():
        assert item["paired_count"]==60
        assert all(m["failure_count"]==0 for m in item["models"].values())


def test_future_perturbation_does_not_change_earlier_predictions(frames):
    first=evaluate_snapshot(make_snapshot(frames),model_factories=doubles([]))
    cut=first["rows"][5]["origin_date"]
    changed=tuple(f.copy() for f in frames)
    for f in changed:
        f.loc[f.date>cut,["open","high","low","close"]]*=1.3
    second=evaluate_snapshot(make_snapshot(changed),model_factories=doubles([]))
    assert [r["predictions"] for r in first["rows"][:6]]==[r["predictions"] for r in second["rows"][:6]]
    assert first["rows"][5]["variance_proxy"]!=second["rows"][5]["variance_proxy"]


@pytest.mark.parametrize("corrupt",[False,True])
def test_failures_are_retained_and_paired_metrics_not_misleading(frames,corrupt):
    exp=evaluate_snapshot(make_snapshot(frames),model_factories=doubles([],fail=not corrupt,corrupt=corrupt))
    assert exp["status"]=="completed_with_failures"
    assert len(exp["rows"])==120 and "SECRET" not in json.dumps(exp)
    for item in exp["summary"].values():
        assert item["paired_count"]==0
        assert item["models"][GarchVariance.name]["failure_count"]==60
        assert item["models"][GarchVariance.name]["all_success_metrics"] is None
        assert item["models"][RollingVariance.name]["paired_metrics"] is None
        assert item["models"][RollingVariance.name]["success_count"]==60
    # 真正模型失败的日期在图中留空，不连接补值。
    fig=experiment_figure(exp)
    assert all(v is None for v in fig.data[2].y)


def test_unique_persistence_manifest_and_no_overwrite(frames,tmp_path):
    exp=evaluate_snapshot(make_snapshot(frames),model_factories=doubles([]))
    path=save_experiment(exp,tmp_path)
    assert json.loads((path/"experiment.json").read_text(encoding="utf-8"))==exp
    manifest=json.loads((path/"manifest.json").read_text(encoding="utf-8"))
    for name,digest in manifest.items(): assert sha256((path/name).read_bytes()).hexdigest()==digest
    with pytest.raises(ValueError,match="覆盖"): save_experiment(exp,tmp_path)
    assert not list(tmp_path.glob(".experiment-*"))


def test_real_models_complete_rolling_pipeline(frames):
    # 实际调用原基线与arch，不要求每个窗口都收敛，更不预设哪种模型更优。
    exp=evaluate_snapshot(make_snapshot(frames), include_latest=True)
    latest = exp["next_observation"]
    end = exp["data_metadata"]["actual_range"][-1]
    baseline = latest["predictions"][RollingVariance.name]
    assert latest["origin_date"] == end
    assert baseline["training_range"][-1] == end and baseline["training_records"] == 20
    assert "target_date" not in baseline and baseline["stddev"] > 0
    assert len(exp["rows"])==120 and exp["status"] in ["completed","completed_with_failures"]
    for row in exp["rows"]:
        assert row["predictions"][RollingVariance.name]["status"]=="ok"
        garch=row["predictions"][GarchVariance.name]
        assert garch["status"] in ["ok","error"]
        if garch["status"]=="ok": assert garch["convergence"]=="converged"


def test_cli_records_all_failures_and_returns_failure(frames,tmp_path,monkeypatch,capsys):
    import sys
    from forecasting import evaluation
    from forecasting.__main__ import main
    from forecasting.data import save_snapshot
    path=save_snapshot(make_snapshot(frames),tmp_path/"snapshots")
    class Failed:
        def fit_predict(self,*args): raise ForecastModelError("SECRET","not_converged")
    original=evaluation.evaluate_snapshot
    monkeypatch.setattr(evaluation,"evaluate_snapshot",lambda snap:original(snap,model_factories={RollingVariance.name:Failed,GarchVariance.name:Failed}))
    monkeypatch.setattr(sys,"argv",["forecasting","evaluate",str(path),"--output-dir",str(tmp_path/"experiments")])
    assert main()==1
    output=json.loads(capsys.readouterr().out)
    assert output["status"]=="failed" and "SECRET" not in json.dumps(output)
    assert all(p["paired_count"]==0 for p in output["summary"].values())
    from pathlib import Path
    assert (Path(output["path"])/"experiment.json").is_file()
