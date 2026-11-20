"""在历史行情上计算技术指标，不联网、不调用大模型。"""

import numpy as np
import pandas as pd


def _rsi(close: pd.Series, period: int) -> pd.Series:
    """Wilder RSI：先用 period 次涨跌的均值初始化，再递推平滑。"""
    changes = close.diff()
    gains = changes.clip(lower=0)
    losses = -changes.clip(upper=0)
    result = pd.Series(np.nan, index=close.index, dtype="float64")
    if len(close) <= period:
        return result
    avg_gain = gains.iloc[1:period + 1].mean()
    avg_loss = losses.iloc[1:period + 1].mean()
    for i in range(period, len(close)):
        if i > period:
            avg_gain = (avg_gain * (period - 1) + gains.iloc[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses.iloc[i]) / period
        # 无涨跌取中性值 50；只有上涨取 100，只有下跌取 0。
        if avg_gain == 0 and avg_loss == 0:
            result.iloc[i] = 50.0
        elif avg_loss == 0:
            result.iloc[i] = 100.0
        else:
            result.iloc[i] = 100 - 100 / (1 + avg_gain / avg_loss)
    return result


def calculate_indicators(history: pd.DataFrame) -> pd.DataFrame:
    """复制行情并增加指标；要求 date 升序且唯一，close/volume 有效。

    所有周期均按输入记录（交易日）计算。历史不足保留 NaN，不回填。
    返回值保留原始列，指标同名列会被重新计算。
    """
    if not isinstance(history, pd.DataFrame) or history.empty:
        raise ValueError("指标输入必须是非空 pandas DataFrame")
    if history.columns.duplicated().any():
        raise ValueError("指标输入包含重复列名")
    missing = {"date", "close", "volume"} - set(history.columns)
    if missing:
        raise ValueError(f"指标输入缺失字段：{', '.join(sorted(missing))}")
    dates = history["date"]
    if not pd.api.types.is_datetime64_ns_dtype(dates.dtype) or dates.dt.tz is not None:
        raise ValueError("date 必须是无时区 datetime64[ns] 类型")
    if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("date 不能缺失或重复，必须按日期升序排列")
    if not dates.eq(dates.dt.normalize()).all():
        raise ValueError("date 只能包含日线日期")

    data = history.copy()
    for column in ["close", "volume"]:
        values = data[column]
        if not pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values):
            raise ValueError(f"{column} 必须是数值类型")
        if pd.api.types.is_complex_dtype(values):
            raise ValueError(f"{column} 不能是复数")
        values = values.astype("float64")
        if not np.isfinite(values.to_numpy()).all():
            raise ValueError(f"{column} 不能包含空值或无穷值")
        invalid = values <= 0 if column == "close" else values < 0
        if invalid.any():
            raise ValueError("收盘价必须为正数，成交量不能为负数")
        data[column] = values

    close = data["close"]
    for window in (5, 10, 20):
        data[f"ma{window}"] = close.rolling(window, min_periods=window).mean()

    # EMA 从首个收盘价初始化；DEA 在 DIF 有 9 个有效值后输出。
    fast = close.ewm(span=12, adjust=False, min_periods=12).mean()
    slow = close.ewm(span=26, adjust=False, min_periods=26).mean()
    data["macd_dif"] = fast - slow
    data["macd_dea"] = data["macd_dif"].ewm(span=9, adjust=False, min_periods=9).mean()
    data["macd_hist"] = 2 * (data["macd_dif"] - data["macd_dea"])
    data["rsi14"] = _rsi(close, 14)
    data["pct_change"] = close.pct_change(fill_method=None) * 100
    data["volume_ma5"] = data["volume"].rolling(5, min_periods=5).mean()
    # 相对成交量：当天成交量 / 前 5 个交易日平均成交量，不是盘中量比。
    previous_average = data["volume_ma5"].shift(1)
    data["volume_ratio"] = data["volume"] / previous_average.replace(0, np.nan)
    return data
