"""基于已收盘行业日线与注明时点的个股快照制作市场日报。"""

from datetime import datetime, time, timedelta
from math import isfinite

import pandas as pd

from tools import market_hotspots as market


BENCHMARKS = {"sh000001": "上证指数", "sz399001": "深证成指", "sz399006": "创业板指"}
RETURN = market.DAILY_RETURN
AMOUNT = market.DAILY_AMOUNT


def _dated_closes(symbol, start, end):
    quotes = market.index_daily(symbol, start, end)
    dates = pd.to_datetime(quotes["date"], errors="coerce").dt.date
    values = pd.to_numeric(quotes["close"], errors="coerce")
    included = dates.between(start, end)
    valid = ~dates.duplicated(keep=False) & values.map(isfinite) & values.gt(0)
    return dict(zip(dates[included], values.where(valid)[included]))


def _benchmarks(now):
    cutoff = now.date() if now.time() >= time(15) else now.date() - timedelta(days=1)
    try:
        shanghai = _dated_closes("sh000001", cutoff - timedelta(days=35), cutoff)
        previous, day = sorted(shanghai)[-2:]
    except Exception:
        return None, None, []

    indices = []
    for symbol, name in BENCHMARKS.items():
        try:
            closes = shanghai if symbol == "sh000001" else _dated_closes(symbol, previous, day)
            change = (closes[day] / closes[previous] - 1) * 100
            if isfinite(change):
                indices.append({"name": name, "close": closes[day], "change": change})
        except Exception:
            continue
    return previous, day, indices


def _rankings(quotes, names):
    totals = quotes[RETURN]
    leads = market.ranking(quotes[totals.gt(0)], by=RETURN)[names + [RETURN]]
    active = market.ranking(quotes[quotes[AMOUNT].ge(0)], by=AMOUNT)[names + [AMOUNT]]
    return {
        "total": len(quotes), "return_count": int(totals.count()), "amount_count": int(quotes[AMOUNT].ge(0).sum()),
        "up": int(totals.gt(0).sum()), "down": int(totals.lt(0).sum()), "flat": int(totals.eq(0).sum()),
        "leaders": leads.head(5).to_dict("records"), "active": active.head(5).to_dict("records"),
    }


def fetch_daily_report(saved):
    """独立获取各部分，只有数据日期一致的指数才计入当日概况。"""
    previous, day, indices = _benchmarks(datetime.now(market.BEIJING))
    industry, stocks, stock_time = None, None, None
    if day is not None:
        try:
            cached = saved.get(market.HISTORICAL_INDUSTRY) or {}
            dates = cached.get("dates", [])
            quotes = cached.get("quotes")
            if (dates[-2:] != [previous, day] or quotes is None or not {RETURN, AMOUNT} <= set(quotes)
                    or not quotes[[RETURN, AMOUNT]].notna().any().any()):
                quotes = market.industry_period_quotes([previous, day], periods=(1,))
            industry = _rankings(quotes, ["行业"])
        except Exception:
            pass
    try:
        snapshot = _rankings(market.stock_quotes(), ["代码", "名称"])
        if snapshot["return_count"] or snapshot["amount_count"]:
            stocks, stock_time = snapshot, datetime.now(market.BEIJING)
    except Exception:
        pass

    if not indices and industry is None and stocks is None:
        return None
    return {"day": day, "previous": previous, "indices": indices, "industry": industry, "stocks": stocks,
            "stock_time": stock_time, "generated": datetime.now(market.BEIJING)}
