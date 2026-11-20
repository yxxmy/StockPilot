"""明确模拟的时序样本；真实arch拟合不代表真实股票预测效果。"""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from forecasting import models
from forecasting.models import RollingVariance, GarchVariance, ForecastModelError


@pytest.fixture
def returns():
    rng = np.random.default_rng(42)
    # 固定随机种子模拟GARCH，用于可重复验证拟合与预测合法性。
    values = np.zeros(950)
    v = 1.
    for i in range(1,len(values)):
        v = .05 + .1*values[i-1]**2 + .85*v
        values[i] = np.sqrt(v)*rng.normal()
    return pd.DataFrame({"date":pd.bdate_range("2020-01-01",periods=650),"return_pct":values[300:]})


def test_baseline_matches_hand_calculated_second_moment(returns):
    result=RollingVariance().fit_predict(returns,returns.date.iloc[499],returns.date.iloc[500])
    expected=sum(float(v)**2 for v in returns.return_pct.iloc[480:500])/20
    assert result["variance"] == pytest.approx(expected)
    assert result["stddev"] == pytest.approx(np.sqrt(expected))
    assert result["training_records"] == 20 and result["convergence"] == "not_applicable"
    assert result["training_range"] == [str(returns.date.iloc[480].date()),str(returns.date.iloc[499].date())]


def test_real_garch_fit_forecast_and_recursion(returns):
    result=GarchVariance().fit_predict(returns,returns.date.iloc[499],returns.date.iloc[500])
    assert result["training_records"] == 500 and result["convergence"] == "converged"
    assert result["variance"] > 0 and np.isfinite(result["stddev"])
    assert result["variance_unit"] == "百分比平方" and result["stddev_unit"] == "百分比"
    # 用arch的拟合条件方差，独立代入单步GARCH递推核对。
    from arch import arch_model
    fitted=arch_model(returns.return_pct.iloc[:500].to_numpy(),mean="Zero",vol="GARCH",p=1,q=1,dist="normal",rescale=False).fit(disp="off")
    params=result["parameters"]
    expected=params["omega"]+params["alpha[1]"]*returns.return_pct.iloc[499]**2+params["beta[1]"]*fitted.conditional_volatility[-1]**2
    assert result["variance"] == pytest.approx(expected,rel=1e-8)


@pytest.mark.parametrize("model",[RollingVariance,GarchVariance])
def test_future_values_do_not_enter_training_and_output_is_detached(returns,model):
    origin,target=returns.date.iloc[520],returns.date.iloc[521]
    first=model().fit_predict(returns,origin,target)
    changed=returns.copy()
    changed.loc[521:,"return_pct"]=np.nan
    trained=model().fit(changed,origin)
    second=trained.predict(target)
    assert first == second
    second["training_range"][0]="tampered"
    second["parameters"].clear()
    assert trained.predict(target) == first


@pytest.mark.parametrize("model,count",[(RollingVariance,19),(GarchVariance,499)])
def test_short_training_is_an_error(returns,model,count):
    with pytest.raises(ForecastModelError) as caught:
        model().fit(returns.iloc[:count],returns.date.iloc[count-1])
    assert caught.value.category == "insufficient_data"


@pytest.mark.parametrize("value",[np.nan,np.inf,True,"bad"])
def test_invalid_training_values(returns,value):
    returns["return_pct"]=returns.return_pct.astype(object)
    returns.loc[490,"return_pct"]=value
    with pytest.raises(ForecastModelError) as caught:
        RollingVariance().fit(returns,returns.date.iloc[499])
    assert caught.value.category == "invalid_input"


@pytest.mark.parametrize("kind",["duplicate","reverse","missing_origin","intraday"])
def test_bad_dates_are_not_silently_sorted(returns,kind):
    origin=returns.date.iloc[499]
    if kind=="duplicate": returns.loc[1,"date"]=returns.date.iloc[0]
    if kind=="reverse": returns=returns.iloc[::-1]
    if kind=="missing_origin": origin=pd.Timestamp("2099-01-01")
    if kind=="intraday": returns.loc[1,"date"]=pd.Timestamp("2020-01-02 01:00:00")
    with pytest.raises(ForecastModelError): RollingVariance().fit(returns,origin)


def test_not_fitted_and_invalid_target(returns):
    model=RollingVariance()
    with pytest.raises(ForecastModelError,match="尚未"):
        model.predict("20240101")
    origin=returns.date.iloc[499]
    model.fit(returns,origin)
    with pytest.raises(ForecastModelError,match="晚于"):
        model.predict(origin)


def test_zero_variance_and_degenerate_garch_are_failures(returns):
    returns["return_pct"]=0.
    with pytest.raises(ForecastModelError) as caught:
        RollingVariance().fit(returns,returns.date.iloc[499])
    assert caught.value.category=="invalid_forecast"
    with pytest.raises(ForecastModelError) as caught:
        GarchVariance().fit(returns,returns.date.iloc[499])
    assert caught.value.category=="degenerate_data"


def test_failed_refit_never_reuses_previous_success(returns,monkeypatch):
    model=GarchVariance().fit(returns,returns.date.iloc[499])
    factory=Mock(return_value=SimpleNamespace(fit=Mock(return_value=SimpleNamespace(convergence_flag=9))))
    monkeypatch.setattr(models,"_arch_model",factory)
    with pytest.raises(ForecastModelError) as caught:
        model.fit(returns,returns.date.iloc[499])
    assert caught.value.category=="not_converged"
    with pytest.raises(ForecastModelError,match="尚未"):
        model.predict(returns.date.iloc[500])
    factory.assert_called_once()


@pytest.mark.parametrize("omega,alpha,beta,variance,category",[
    (-1,.1,.8,1,"invalid_parameters"),
    (.1,.8,.8,1,"invalid_parameters"),
    (np.nan,.1,.8,1,"invalid_parameters"),
    (.1,.1,.8,-1,"invalid_forecast"),
    (.1,.1,.8,np.inf,"invalid_forecast"),
])
def test_invalid_parameters_or_forecast_never_succeed(returns,monkeypatch,omega,alpha,beta,variance,category):
    fitted=SimpleNamespace(convergence_flag=0,params={"omega":omega,"alpha[1]":alpha,"beta[1]":beta},
        forecast=lambda **kw:SimpleNamespace(variance=pd.DataFrame([[variance]])))
    monkeypatch.setattr(models,"_arch_model",lambda values:SimpleNamespace(fit=lambda **kw:fitted))
    with pytest.raises(ForecastModelError) as caught:
        GarchVariance().fit(returns,returns.date.iloc[499])
    assert caught.value.category==category


def test_dependency_and_unknown_error_are_explicit(returns,monkeypatch):
    def fail(values): raise ForecastModelError("依赖缺失","dependency_error")
    monkeypatch.setattr(models,"_arch_model",fail)
    with pytest.raises(ForecastModelError) as caught: GarchVariance().fit(returns,returns.date.iloc[499])
    assert caught.value.category=="dependency_error"
    monkeypatch.setattr(models,"_arch_model",Mock(side_effect=RuntimeError("SECRET_PATH")))
    with pytest.raises(ForecastModelError) as caught: GarchVariance().fit(returns,returns.date.iloc[499])
    assert caught.value.category=="fit_error" and "SECRET_PATH" not in str(caught.value)
