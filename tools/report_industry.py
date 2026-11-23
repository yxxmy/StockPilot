"""为个股简报匹配截至行情日的行业表现。"""

import re
from datetime import date, datetime, time
from math import isfinite
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from tools import market_hotspots as hotspots


RANKING_SOURCE = "https://q.10jqka.com.cn/thshy/"


def _read_page(url):
    response = requests.get(url, timeout=(4, 8))
    response.raise_for_status()
    return BeautifulSoup(response.content, "html.parser")


def match_current_industry(symbol, names):
    """用公司资料缩小查找范围，再用同花顺行业成分股名单核对股票代码。"""
    if not re.fullmatch(r"[0-9]{6}", symbol):
        return None
    try:
        profile = _read_page(f"https://basic.10jqka.com.cn/{symbol}/company.html")
        field = profile.find(string=re.compile(r"所属申万行业\s*[：:]"))
        cell = field.find_parent("td") if field else None
        if cell is None:
            return None
        text = re.sub(r"^\s*所属申万行业\s*[：:]\s*", "", cell.get_text(" ", strip=True))
        candidates = [part.strip().rstrip("ⅠⅡⅢⅣⅤ") for part in re.split(r"[—–]", text)]
        catalog = _read_page(RANKING_SOURCE)
        for name in reversed(candidates):
            if name not in names:
                continue
            links = {
                path for link in catalog.find_all("a", href=True)
                if link.get_text(strip=True) == name
                and re.fullmatch(r"/thshy/detail/code/[0-9]+/", path := urlparse(link["href"]).path)
            }
            if len(links) != 1:
                continue
            source = "https://q.10jqka.com.cn" + links.pop()
            members = _read_page(source)
            for link in members.select("table td a[href]"):
                target = urlparse(link["href"])
                if (target.hostname == "stockpage.10jqka.com.cn"
                        and target.path.rstrip("/") == f"/{symbol}" and link.get_text(strip=True) == symbol):
                    return {"name": name, "source": source}
    except (requests.RequestException, ValueError):
        pass
    return None


def industry_snapshot(end, saved):
    now = datetime.now(hotspots.BEIJING)
    if end > now.date() or (end == now.date() and now.time() < time(15)):
        return None
    cached = saved.get(hotspots.HISTORICAL_INDUSTRY)
    if cached and cached.get("dates") and cached["dates"][-1] == end:
        return cached
    dates = hotspots.trading_dates_through(end)[-21:]
    if len(dates) < 6 or dates[-1] != end:
        return None
    return {"dates": dates, "quotes": hotspots.industry_period_quotes(dates)}


def industry_context(snapshot, name, membership):
    dates, quotes = snapshot["dates"], snapshot["quotes"]
    metrics = []
    for days, column in hotspots.PERIOD_RETURNS.items():
        if len(dates) <= days:
            continue
        values = quotes.set_index("行业")[column].dropna()
        values = values[values.map(isfinite)]
        if name not in values:
            continue
        returns = values[name]
        metrics.append({"days": days, "start": dates[-days - 1], "value": returns,
                        "rank": int(values.gt(returns).sum()) + 1, "count": len(values)})
    if not metrics:
        return None
    return {"name": name, "end": dates[-1], "metrics": metrics,
            "membership": membership if membership and membership["name"] == name else None}


def report_end(market):
    try:
        return date.fromisoformat(market["actual_range"][-1])
    except (KeyError, TypeError, ValueError, IndexError):
        return None
