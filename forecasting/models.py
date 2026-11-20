"""8B：历史二阶矩基线与真实GARCH拟合，不做滚动评估或交易。"""

from hashlib import sha256
from copy import deepcopy
import json
import warnings

import numpy as np
import pandas as pd


class ForecastModelError(RuntimeError):
    def __init__(self, message, category="model_error"):
        super().__init__(message)
        self.category = category


def _date(value):
    try:
        result = pd.Timestamp(value)
        if pd.isna(result) or result.tzinfo is not None or result != result.normalize():
            raise ValueError()
        return result
    except (ValueError, TypeError, OverflowError):
        raise ForecastModelError("日期必须是有效的无时区日历日期", "invalid_input") from None


def _training_data(returns, origin_date, window):
    origin = _date(origin_date)
    if not isinstance(returns, pd.DataFrame) or not {"date", "return_pct"}.issubset(returns.columns) or returns.columns.duplicated().any():
        raise ForecastModelError("缺少唯一的date/return_pct列", "invalid_input")
    try:
        dates = pd.to_datetime(returns.date, errors="raise")
        if dates.isna().any() or dates.dt.tz is not None or not dates.eq(dates.dt.normalize()).all():
            raise ValueError()
        if dates.duplicated().any() or not dates.is_monotonic_increasing or origin not in dates.to_numpy():
            raise ValueError()
        # 先按起点截断，再检查数值和计算统计；不读取未来收益或目标标签。
        frame = returns.loc[dates <= origin, ["date", "return_pct"]].tail(window).copy()
        frame["date"] = dates.loc[frame.index]
        if len(frame) < window:
            raise ForecastModelError(f"训练样本不足：需要{window}条，实际{len(frame)}条", "insufficient_data")
        if frame.return_pct.map(lambda x: isinstance(x, (bool, np.bool_))).any():
            raise ValueError()
        frame["return_pct"] = pd.to_numeric(frame.return_pct, errors="raise").astype(float)
        if not np.isfinite(frame.return_pct).all():
            raise ValueError()
    except ForecastModelError:
        raise
    except (ValueError, TypeError, AttributeError):
        raise ForecastModelError("训练日期须升序唯一、包含起点，收益率须为有限数值", "invalid_input") from None
    return frame.reset_index(drop=True), origin


def _arch_model(values):
    # 延迟导入，可选依赖缺失时基线及前七阶段仍可运行。
    try:
        from arch import arch_model
    except ImportError:
        raise ForecastModelError("请先安装requirements-forecast.txt", "dependency_error") from None
    return arch_model(values, mean="Zero", vol="GARCH", p=1, o=0, q=1, dist="normal", rescale=False)


class _OneStepModel:
    name = ""
    window = 0

    def __init__(self):
        self._state = None

    def predict(self, target_date):
        if self._state is None:
            raise ForecastModelError("模型尚未成功拟合", "not_fitted")
        target = _date(target_date)
        if target <= pd.Timestamp(self._state["origin_date"]):
            raise ForecastModelError("目标日期必须晚于拟合起点", "invalid_target")
        # target_date只是调用者提供的下一有效观测标签，不自动推断交易日历。
        return {**deepcopy(self._state), "target_date": target.strftime("%Y-%m-%d")}

    def fit_predict(self, returns, origin_date, target_date):
        return self.fit(returns, origin_date).predict(target_date)

    def _success(self, frame, origin, variance, parameters, convergence):
        if not np.isfinite(variance) or variance <= 0:
            raise ForecastModelError("预测方差必须是严格正的有限数值", "invalid_forecast")
        records = [[d.strftime("%Y-%m-%d"), float(r)] for d,r in zip(frame.date, frame.return_pct)]
        self._state = {"status": "ok", "model": self.name, "origin_date": origin.strftime("%Y-%m-%d"),
            "training_range": [frame.date.iloc[0].strftime("%Y-%m-%d"), frame.date.iloc[-1].strftime("%Y-%m-%d")],
            "training_records": len(frame), "training_hash": sha256(json.dumps(records,allow_nan=False).encode()).hexdigest(),
            "variance": float(variance), "stddev": float(np.sqrt(variance)), "variance_unit": "百分比平方",
            "stddev_unit": "百分比", "parameters": parameters, "convergence": convergence,
            "limitation": "单步零均值条件方差预测；未年化，不预测价格涨跌或交易收益；目标日为调用者提供的下一观测日期。"}
        return self


class RollingVariance(_OneStepModel):
    """过去20条百分比收益平方的均值，不减均值、不做ddof校正。"""
    name = "rolling_second_moment_20"
    window = 20

    def fit(self, returns, origin_date):
        self._state = None
        frame, origin = _training_data(returns, origin_date, self.window)
        with np.errstate(over="ignore", invalid="ignore"):
            variance = np.square(frame.return_pct.to_numpy()).mean()
        return self._success(frame, origin, variance, {"window":20,"assumed_mean":0.}, "not_applicable")


class GarchVariance(_OneStepModel):
    """固定最近500条，零均值/正态误差GARCH(1,1)，不调参。"""
    name = "garch_1_1_zero_normal"
    window = 500

    def fit(self, returns, origin_date):
        self._state = None  # 重拟合失败后不能继续输出上次结果。
        frame, origin = _training_data(returns, origin_date, self.window)
        if np.ptp(frame.return_pct.to_numpy()) <= 1e-12:
            raise ForecastModelError("收益序列退化，不能可靠拟合GARCH", "degenerate_data")
        try:
            model = _arch_model(frame.return_pct.to_numpy())
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                result = model.fit(disp="off", update_freq=0, show_warning=False, options={"maxiter":500})
            if result.convergence_flag != 0:
                raise ForecastModelError("GARCH优化未收敛，未产生成功预测", "not_converged")
            params = {key:float(result.params[key]) for key in ["omega", "alpha[1]", "beta[1]"]}
            if not all(np.isfinite(list(params.values()))) or params["omega"] <= 0 or min(params["alpha[1]"], params["beta[1]"]) < 0 or params["alpha[1]"]+params["beta[1]"] > 1+1e-8:
                raise ForecastModelError("GARCH参数未通过有效性校验", "invalid_parameters")
            forecast = result.forecast(horizon=1, method="analytic", reindex=False)
            variance = float(forecast.variance.iloc[-1,0])
            return self._success(frame, origin, variance, params, "converged")
        except ForecastModelError:
            raise
        except Exception:
            raise ForecastModelError("GARCH拟合或预测失败，未使用其他模型替代", "fit_error") from None
