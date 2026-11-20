"""第四阶段真实财务、公告和新闻；不生成任何回退数据。"""

from datetime import datetime
from html import unescape
import math
import re
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import akshare as ak
import pandas as pd

from tools.stock_data import DataValidationError, StockDataError, _parse_date


FINANCIAL_FIELDS = {
    "TOTALOPERATEREVE": "revenue_yuan", "PARENTNETPROFIT": "parent_net_profit_yuan",
    "EPSJB": "basic_eps_yuan", "ROEJQ": "weighted_roe_pct",
    "XSMLL": "gross_margin_pct", "ZCFZL": "debt_asset_ratio_pct",
    "TOTALOPERATEREVETZ": "revenue_yoy_pct", "PARENTNETPROFITTZ": "profit_yoy_pct",
}


def _validate(symbol, start_date, end_date):
    if not isinstance(symbol, str) or not re.fullmatch(r"[0-9]{6}", symbol):
        raise ValueError("股票代码必须是六位数字字符串")
    start, end = _parse_date(start_date, "start_date"), _parse_date(end_date, "end_date")
    if start > end:
        raise ValueError("start_date 不能晚于 end_date")
    return start, end


def _fetch(api, **kwargs):
    try:
        result = api(**kwargs)
    except Exception as exc:
        raise StockDataError(f"研究数据接口失败（{type(exc).__name__}），请检查网络或上游服务") from None
    if not isinstance(result, pd.DataFrame):
        raise DataValidationError("研究数据接口返回值不是 DataFrame")
    if result.columns.duplicated().any():
        raise DataValidationError("研究数据包含重复列名")
    return result.copy()


def _require(data, columns):
    missing = set(columns) - set(data.columns)
    if missing:
        raise DataValidationError(f"研究数据缺失字段：{', '.join(sorted(missing))}")


def _text(value, limit=400):
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]*>", "", str(value)))).strip()[:limit]


def _url(value):
    value = str(value).strip()
    try:
        parsed = urlparse(value)
        valid = parsed.scheme in {"http", "https"} and parsed.hostname and not parsed.username and not parsed.password
    except ValueError:
        valid = False
    if not valid:
        raise DataValidationError("研究数据原文链接不是有效 HTTP(S) URL")
    return value


def _date_column(values):
    try:
        result = pd.to_datetime(values, format="ISO8601", errors="raise")
        if result.dt.tz is not None:
            raise ValueError("不接受未约定的时区")
    except (ValueError, TypeError, AttributeError):
        raise DataValidationError("研究数据日期格式不合法") from None
    return result


def _number(value):
    if pd.isna(value) or (isinstance(value, str) and value.strip() in {"", "--"}):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise DataValidationError("财务指标不是有效数值") from None
    if pd.api.types.is_bool(value) or not math.isfinite(number):
        raise DataValidationError("财务指标包含布尔值或无穷值")
    return number


def _envelope(symbol, start, end, source, coverage):
    return {
        "symbol": symbol, "requested_range": [str(start.date()), str(end.date())],
        "source": source, "fetched_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "status": "empty", "records": [], "coverage": coverage,
    }


def get_stock_financials(symbol: str, end_date: str) -> dict:
    """取截止日已披露的最近4个报告期；累计口径不转换为单季值。"""
    start, end = _validate(symbol, "19000101", end_date)
    suffix = "SH" if symbol.startswith("6") else ("BJ" if symbol.startswith(("4", "8", "92")) else "SZ")
    qualified = f"{symbol}.{suffix}"
    result = _envelope(symbol, start, end, "AKShare / 东方财富财务主要指标",
                       "仅返回上游可用且截止日已披露的最近4个报告期；累计指标不能当作单季度。当前接口可能包含事后修订，不是严格时点回测数据库。")
    result["source_url"] = f"https://emweb.securities.eastmoney.com/pc_hsf10/pages/index.html?code={suffix}{symbol}#/cwfx"
    result["units"] = "*_yuan 为元（basic_eps_yuan 为元/股）；*_pct 为百分数，数值1代表1%；缺失值为null，不是0"
    raw = _fetch(ak.stock_financial_analysis_indicator_em, symbol=qualified, indicator="按报告期")
    result["upstream_count"] = len(raw)
    if raw.empty:
        return result
    _require(raw, ["SECURITY_CODE", "REPORT_DATE", "NOTICE_DATE", "CURRENCY", *FINANCIAL_FIELDS])
    raw = raw[raw["SECURITY_CODE"].astype(str) == symbol].copy()
    raw["REPORT_DATE"] = _date_column(raw["REPORT_DATE"])
    raw["NOTICE_DATE"] = _date_column(raw["NOTICE_DATE"])
    result["excluded_undated_count"] = int(raw[["REPORT_DATE", "NOTICE_DATE"]].isna().any(axis=1).sum())
    raw = raw.dropna(subset=["REPORT_DATE", "NOTICE_DATE"])
    raw = raw[(raw["REPORT_DATE"].dt.normalize() <= end) & (raw["NOTICE_DATE"].dt.normalize() <= end)]
    if (raw["NOTICE_DATE"] < raw["REPORT_DATE"]).any():
        raise DataValidationError("财务披露日期早于报告期")
    raw = raw.sort_values(["REPORT_DATE", "NOTICE_DATE"]).drop_duplicates("REPORT_DATE", keep="last").tail(4)
    for _, row in raw.iterrows():
        if not isinstance(row["CURRENCY"], str) or row["CURRENCY"] != "CNY":
            raise DataValidationError("财务指标币种不是CNY，不能标记为人民币元")
        result["records"].append({
            "report_date": str(row["REPORT_DATE"].date()), "published_date": str(row["NOTICE_DATE"].date()),
            "currency": "CNY", **{name: _number(row[field]) for field, name in FINANCIAL_FIELDS.items()},
        })
    result["status"] = "ok" if result["records"] else "empty"
    return result


def _documents(symbol, start_date, end_date, kind):
    start, end = _validate(symbol, start_date, end_date)
    news = kind == "news"
    source = "AKShare / 东方财富新闻搜索" if news else "AKShare / 巨潮资讯公告"
    coverage = ("仅搜索上游当前返回的近期有限新闻，不支持完整历史检索；匹配为空不能解释为该期间无新闻。关键词命中不保证整篇仅讨论本公司。"
                if news else "查询区间内公告元数据，仅展示最新5条标题和链接；未读取PDF正文，不能推断公告详细内容。")
    result = _envelope(symbol, start, end, source, coverage)
    if news:
        raw = _fetch(ak.stock_news_em, symbol=symbol)
        fields = {"新闻标题": "title", "新闻内容": "excerpt", "发布时间": "published_at", "文章来源": "publisher", "新闻链接": "url"}
    else:
        raw = _fetch(ak.stock_zh_a_disclosure_report_cninfo, symbol=symbol, market="沪深京",
                     start_date=start.strftime("%Y%m%d"), end_date=end.strftime("%Y%m%d"))
        fields = {"公告标题": "title", "公告时间": "published_at", "公告链接": "url"}
    result["upstream_count"] = len(raw)
    if raw.empty:
        result["matched_count"] = 0
        return result
    _require(raw, [*fields, "关键词" if news else "代码"])
    raw = raw[raw["关键词" if news else "代码"].astype(str) == symbol]
    data = raw[list(fields)].rename(columns=fields).copy()
    data["published_at"] = _date_column(data["published_at"])
    if data["published_at"].isna().any():
        raise DataValidationError("新闻或公告缺失发布时间")
    data = data[data["published_at"].dt.normalize().between(start, end)]
    data["url"] = data["url"].map(_url)
    for field in ["title", "excerpt", "publisher"]:
        if field in data:
            data[field] = data[field].map(_text)
    if data["title"].eq("").any():
        raise DataValidationError("新闻或公告缺失标题")
    data = data.sort_values("published_at", ascending=False).drop_duplicates("url")
    result["matched_count"] = len(data)
    data = data.head(5).copy()
    data["published_at"] = data["published_at"].dt.strftime("%Y-%m-%d %H:%M:%S")
    result["records"] = data.to_dict("records")
    result["status"] = "ok" if len(data) else "empty"
    return result


def get_stock_news(symbol: str, start_date: str, end_date: str) -> dict:
    return _documents(symbol, start_date, end_date, "news")


def get_stock_notices(symbol: str, start_date: str, end_date: str) -> dict:
    return _documents(symbol, start_date, end_date, "notices")
