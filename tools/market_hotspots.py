"""取得市场榜单快照，整理字段并记录参考交易日和抓取时间。"""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, time, timedelta, timezone
from functools import partial
from threading import Lock

import pandas as pd

from tools.stock_data import ak


BEIJING = timezone(timedelta(hours=8))
STOCK_SOURCE = "AKShare / 腾讯证券"
INDUSTRY_SOURCE = "AKShare / 同花顺"
PERIOD_RETURNS = {5: "近5日涨跌幅（%）", 20: "近20日涨跌幅（%）"}
PERIOD_AMOUNTS = {5: "近5日成交额（亿元）", 20: "近20日成交额（亿元）"}
DAILY_RETURN = "涨跌幅（%）"
DAILY_AMOUNT = "成交额（亿元）"
_INDEX_RETURNS = {1: DAILY_RETURN, **PERIOD_RETURNS}
_INDEX_AMOUNTS = {1: DAILY_AMOUNT, **PERIOD_AMOUNTS}
HISTORICAL_INDUSTRY = "多日行业"
_INDUSTRY_LOCK = Lock()


def _numbers(frame, columns):
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").replace(
            [float("inf"), float("-inf")], float("nan")
        )


def stock_quotes():
    raw = ak.stock_zh_a_spot_tx()
    kind = raw["stock_type"]
    # 腾讯分别标记主板、科创板和创业板，北交所在 A 股榜中使用 GP。
    mainland_a = kind.isin(["GP-A", "GP-A-KCB", "GP-A-CYB"])
    beijing_a = kind.eq("GP") & raw["code"].astype("string").str.startswith("bj", na=False)
    fields = {"code": "代码", "name": "名称", "zdf": "涨跌幅（%）", "turnover": "成交额（亿元）", "hsl": "换手率（%）"}
    quotes = raw.loc[mainland_a | beijing_a, list(fields)].rename(columns=fields)
    for days, column in PERIOD_RETURNS.items():
        quotes[column] = raw[f"zdf_d{days}"] if f"zdf_d{days}" in raw else float("nan")
    quotes["代码"] = quotes["代码"].astype("string").str.extract(r"^(?:sh|sz|bj)([0-9]{6})$", expand=False)
    quotes["名称"] = quotes["名称"].astype("string").str.strip().replace("", pd.NA)
    _numbers(quotes, ["涨跌幅（%）", "成交额（亿元）", "换手率（%）", *PERIOD_RETURNS.values()])
    # 腾讯快照成交额为万元，统一转为亿元。
    quotes["成交额（亿元）"] /= 10_000
    quotes = quotes.dropna(subset=["代码", "名称"]).drop_duplicates(subset=["代码"])
    if not quotes[["涨跌幅（%）", *PERIOD_RETURNS.values()]].notna().any().any():
        raise ValueError("个股榜没有可用涨幅")
    return quotes


def industry_quotes():
    # 同花顺 SDK 会初始化 JS 运行时，同一进程中的行业请求顺序执行。
    with _INDUSTRY_LOCK:
        raw = ak.stock_board_industry_summary_ths()
    columns = ["板块", "涨跌幅", "总成交额", "上涨家数", "下跌家数", "领涨股", "领涨股-涨跌幅"]
    quotes = raw[columns].rename(columns={
        "板块": "行业", "涨跌幅": "涨跌幅（%）", "总成交额": "成交额（亿元）",
        "领涨股-涨跌幅": "领涨股涨跌幅（%）",
    })
    quotes["行业"] = quotes["行业"].astype("string").str.strip().replace("", pd.NA)
    _numbers(quotes, ["涨跌幅（%）", "成交额（亿元）", "上涨家数", "下跌家数", "领涨股涨跌幅（%）"])
    quotes = quotes.dropna(subset=["行业"]).drop_duplicates(subset=["行业"])
    if quotes["涨跌幅（%）"].count() == 0:
        raise ValueError("行业榜没有可用涨幅")
    return quotes


def completed_dates(now):
    """以带日期的上证指数日线取得已收盘交易日。"""
    end = now.date()
    if now.time() < time(15):
        end -= timedelta(days=1)
    return trading_dates_through(end)


def trading_dates_through(end):
    start = end - timedelta(days=65)
    index = index_daily("sh000001", start, end)
    dates = pd.to_datetime(index["date"], errors="coerce").dropna().dt.date
    return sorted(dates[dates.between(start, end)].unique())


def index_daily(symbol, start, end):
    return ak.stock_zh_index_daily_tx(
        symbol=symbol, start_date=start.strftime("%Y%m%d"), end_date=end.strftime("%Y%m%d")
    )


def latest_closed_date(now):
    dates = completed_dates(now)
    return dates[-1] if dates else None


def _industry_periods(name, dates, periods):
    record = {"行业": name}
    try:
        history = ak.stock_board_industry_index_ths(
            symbol=name, start_date=dates[0].strftime("%Y%m%d"), end_date=dates[-1].strftime("%Y%m%d")
        )
        history = history.reindex(columns=["日期", "收盘价", "成交额"])
        history["日期"] = pd.to_datetime(history["日期"], errors="coerce").dt.date
        history = history.dropna(subset=["日期"]).drop_duplicates("日期", keep=False)
        _numbers(history, ["收盘价", "成交额"])
        history = history.set_index("日期").reindex(dates)
    except Exception:
        return record

    for days in periods:
        if len(dates) <= days:
            continue
        closes = history["收盘价"].iloc[-days - 1:]
        if closes.gt(0).fillna(False).all():
            record[_INDEX_RETURNS[days]] = (closes.iloc[-1] / closes.iloc[0] - 1) * 100
        amounts = history["成交额"].iloc[-days:]
        if amounts.ge(0).fillna(False).all():
            record[_INDEX_AMOUNTS[days]] = amounts.sum() / 100_000_000
    return record


def industry_period_quotes(dates, *, periods=(5, 20)):
    if len(dates) <= min(periods):
        raise ValueError("交易日不足，无法计算行业涨幅")
    # 同批行业日线同时计算单日数据，供市场日报复用。
    windows = tuple(dict.fromkeys((1, *periods)))
    # 先在单线程加载行业名录并初始化 AKShare 的 JS 运行时，再有限并发拉取日线。
    with _INDUSTRY_LOCK:
        catalog = ak.stock_board_industry_name_ths()
        names = catalog["name"].astype("string").str.strip().replace("", pd.NA).dropna().unique()
        with ThreadPoolExecutor(max_workers=4) as workers:
            records = list(workers.map(partial(_industry_periods, dates=dates[-max(windows) - 1:], periods=windows), names))
    returns = [_INDEX_RETURNS[days] for days in windows]
    amounts = [_INDEX_AMOUNTS[days] for days in windows]
    quotes = pd.DataFrame(records, columns=["行业", *returns, *amounts])
    required = [column for days in periods for column in (_INDEX_RETURNS[days], _INDEX_AMOUNTS[days])]
    if not quotes[required].notna().any().any():
        raise ValueError("行业日线没有可用的涨幅或成交额")
    return quotes


def refresh_rankings(saved, *, multi_day=False):
    """各榜单独立更新，失败时继续使用该榜已有的快照。"""
    now = datetime.now(BEIJING)
    dates = []
    try:
        if multi_day:
            dates = completed_dates(now)
            reference_date = dates[-1] if dates else None
        else:
            reference_date = latest_closed_date(now)
    except Exception:
        reference_date = None

    snapshots, failed = dict(saved), []
    industry_loader = (HISTORICAL_INDUSTRY, partial(industry_period_quotes, dates), INDUSTRY_SOURCE) if multi_day else (
        "行业", industry_quotes, INDUSTRY_SOURCE
    )
    for label, loader, source in (("个股", stock_quotes, STOCK_SOURCE), industry_loader):
        try:
            quotes = loader()
        except Exception:
            failed.append(label)
        else:
            snapshots[label] = {
                "quotes": quotes,
                "fetched_at": datetime.now(BEIJING),
                "reference_date": reference_date,
                "source": source,
            }
            if label == HISTORICAL_INDUSTRY:
                snapshots[label]["dates"] = dates[-21:]
    return snapshots, failed


def ranking(quotes, by="涨跌幅（%）"):
    """使用表中同一列排序，缺失该项的记录不参与排名。"""
    return quotes.dropna(subset=[by]).sort_values(by, ascending=False, kind="stable").head(10)
