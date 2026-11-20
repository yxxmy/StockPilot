"""通过 AKShare 获取真实 A 股历史日线，不生成或填补行情。"""

from datetime import datetime
import re

import akshare as ak
import numpy as np
import pandas as pd
import requests

# AKShare的公开配置：每次HTTP连接5秒、读取15秒；不是整个分析的硬性总时限。
# 设置一次，不在并发调用中反复修改/恢复全局配置。
ak.set_timeout((5, 15))


class StockDataError(RuntimeError):
    """行情获取或校验失败。"""

    category = "market_data"


def _network_category(exc):
    from akshare.utils.http import Push2RateLimitError
    if isinstance(exc, Push2RateLimitError):
        return "market_rate_limit"
    if isinstance(exc, (requests.Timeout, TimeoutError)):
        return "market_timeout"
    return "market_connection"


class EmptyDataError(StockDataError):
    """查询范围内没有行情。"""
    category = "market_empty"


class DataValidationError(StockDataError):
    """接口返回的字段或数据不符合约定。"""
    category = "market_validation"


FIELD_MAP = {
    "日期": "date", "开盘": "open", "最高": "high",
    "最低": "low", "收盘": "close", "成交量": "volume",
}


def _parse_date(value: str, name: str) -> pd.Timestamp:
    pattern = r"[0-9]{8}|[0-9]{4}-[0-9]{2}-[0-9]{2}"
    if not isinstance(value, str) or not re.fullmatch(pattern, value):
        raise ValueError(f"{name} 必须是 YYYYMMDD 或 YYYY-MM-DD 格式的日期")
    try:
        return pd.Timestamp(datetime.strptime(value.replace("-", ""), "%Y%m%d"))
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{name} 不是有效日期：{value}") from exc


def get_stock_history(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """返回指定范围（含首尾日期）的不复权日线。

    symbol 为六位数字字符串；日期支持 YYYYMMDD 和 YYYY-MM-DD。
    date 类型为 datetime64[ns]，其余列为 float64；volume 单位为手。
    参数错误抛出 ValueError，接口和数据错误抛出 StockDataError 的子类。
    """
    if not isinstance(symbol, str) or not re.fullmatch(r"[0-9]{6}", symbol):
        raise ValueError("股票代码必须是六位数字字符串，例如 '600519' 或 '000001'")
    start = _parse_date(start_date, "start_date")
    end = _parse_date(end_date, "end_date")
    if start > end:
        raise ValueError("start_date 不能晚于 end_date")

    source = "AKShare / 东方财富"
    warning = ""
    try:
        raw = ak.stock_zh_a_hist(
            symbol=symbol, period="daily", start_date=start.strftime("%Y%m%d"),
            end_date=end.strftime("%Y%m%d"), adjust="",
        )
    except Exception as exc:
        # 仅网络失败尝试另一真实数据源一次，不重试已限流的东方财富。
        if isinstance(exc, (requests.RequestException, TimeoutError)):
            try:
                raw = ak.stock_zh_a_hist_tx(symbol=symbol, start_date=start.strftime("%Y%m%d"),
                    end_date=end.strftime("%Y%m%d"), adjust="")
                if not isinstance(raw, pd.DataFrame):
                    raise DataValidationError("备用行情返回值不是DataFrame")
                if raw.empty:
                    raise EmptyDataError("备用行情源没有返回指定区间的数据")
                missing = set(FIELD_MAP.values()) - set(raw.columns)
                if missing:
                    raise DataValidationError("备用行情缺失必要字段")
                raw = raw.copy()
                if raw["volume"].map(lambda value: isinstance(value, (bool, np.bool_))).any():
                    raise DataValidationError("备用行情成交量含布尔值")
                # AKShare>=1.20.35腾讯接口统一volume为股；项目统一输出手。
                raw["volume"] = pd.to_numeric(raw["volume"], errors="raise") / 100
                raw = raw.rename(columns={value: key for key, value in FIELD_MAP.items()})
                source = "AKShare / 腾讯证券"
                warning = "东方财富连接失败，已切换腾讯真实日线；两源均为不复权，成交量统一为手。"
            except Exception as fallback_exc:
                error = StockDataError(f"获取 {symbol} 历史行情失败：主源及备用源均不可用")
                error.category = _network_category(exc)
                if isinstance(fallback_exc, (EmptyDataError, DataValidationError)):
                    error.category = "market_empty" if isinstance(fallback_exc, EmptyDataError) else "market_validation"
                raise error from exc
        else:
            raise StockDataError(f"获取 {symbol} 历史行情失败") from exc

    data = normalize_history(raw, symbol, start_date, end_date)
    data.attrs.update(source=source, source_warning=warning)
    return data


def normalize_history(raw, symbol, start_date, end_date):
    """复用纯校验流程，不请求网络；不改变既有日线清洗口径。"""
    start, end = _parse_date(start_date, "start_date"), _parse_date(end_date, "end_date")
    if start > end:
        raise ValueError("start_date 不能晚于 end_date")
    if raw is None or (isinstance(raw, pd.DataFrame) and raw.empty):
        raise EmptyDataError(f"{symbol} 在指定日期范围内没有行情，请检查代码、交易日或上市日期")
    if not isinstance(raw, pd.DataFrame):
        raise DataValidationError("接口返回值不是 pandas DataFrame")
    missing = set(FIELD_MAP) - set(raw.columns)
    if missing:
        raise DataValidationError(f"行情缺失字段：{', '.join(sorted(missing))}")
    if raw.columns.duplicated().any():
        raise DataValidationError("行情包含重复列名")

    # 只保留约定的六列，复制后处理，避免修改接口原始结果。
    data = raw[list(FIELD_MAP)].rename(columns=FIELD_MAP).copy()
    try:
        data["date"] = pd.to_datetime(data["date"], format="ISO8601", errors="raise").astype("datetime64[ns]")
        if data["date"].isna().any() or not data["date"].eq(data["date"].dt.normalize()).all():
            raise ValueError("日期缺失或含有非零时分秒")
        for column in ["open", "high", "low", "close", "volume"]:
            if data[column].map(lambda value: isinstance(value, (bool, np.bool_))).any():
                raise ValueError(f"{column} 含布尔值")
            data[column] = pd.to_numeric(data[column], errors="raise").astype("float64")
        numeric = data.drop(columns="date")
        if not np.isfinite(numeric.to_numpy()).all():
            raise ValueError("价格或成交量含空值、无穷值")
        prices = data[["open", "high", "low", "close"]]
        if (prices <= 0).any().any() or (data["volume"] < 0).any():
            raise ValueError("价格必须为正数，成交量不能为负数")
        invalid_high = data["high"] < prices.max(axis=1)
        invalid_low = data["low"] > prices.min(axis=1)
        if invalid_high.any() or invalid_low.any():
            raise ValueError("最高价或最低价与其他价格不一致")
    except (ValueError, TypeError, OverflowError) as exc:
        raise DataValidationError(f"行情数据校验失败：{exc}") from exc

    # 同一天重复返回时保留接口最后一条；不补齐周末、节假日或停牌日。
    data = data[data["date"].between(start, end)]
    data = data.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
    if data.empty:
        raise EmptyDataError(f"{symbol} 在指定日期范围内没有行情")
    return data
