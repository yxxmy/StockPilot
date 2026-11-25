"""根据正文问题过滤巨潮资讯的公告标题和披露日期。"""

from datetime import date, datetime, timedelta
import re
from unicodedata import normalize
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

import pandas as pd

from tools.research_data import _fetch, _require, _text, _url, ak
from tools.stock_data import DataValidationError


_BEIJING = ZoneInfo("Asia/Shanghai")
_TOPICS = (
    ("权益分派", ("分红", "股息", "股利", "红利", "权益分派", "除权", "除息"), "权益分派", True),
    ("第三季度报告", ("第三季度", "三季报", "三季度"), "第三季度报告", True),
    ("第一季度报告", ("第一季度", "一季报", "一季度"), "第一季度报告", True),
    ("半年度报告", ("半年度报告", "半年报", "中报", "半年"), "半年度报告", True),
    ("年度报告", ("年报", "年度报告", "营业收入", "净利润", "经营活动现金流"), r"(?<!半)年度报告", True),
    ("回购", ("回购",), "回购", False),
    ("业绩预告", ("业绩预告", "预亏", "预盈"), "业绩预告", False),
    ("股东大会", ("股东大会",), "股东大会", False),
    ("减持", ("减持",), "减持", False),
)
_OFFICIAL_HOSTS = {"cninfo.com.cn", "www.cninfo.com.cn", "static.cninfo.com.cn"}


def _search_parameters(question, today):
    question = normalize("NFKC", question).strip()
    if not question:
        raise ValueError("请先填写财报或公告正文问题。")
    quoted = re.search(r"《([^《》]{2,60})》", question)
    if quoted:
        keyword = quoted[1].strip()
        if not keyword:
            raise ValueError("请在书名号里填写公告标题关键词。")
        pattern = re.escape(keyword)
        report_year = any(uses_report_year and re.search(title_pattern, keyword)
                          for _, _, title_pattern, uses_report_year in _TOPICS)
    else:
        topic = next((topic for topic in _TOPICS if any(term in question for term in topic[1])), None)
        if topic is None and "股权登记日" in question:
            topic = _TOPICS[0]
        if topic is None:
            raise ValueError("请在问题中写明公告类型，例如年报、分红、回购，或用《》写出标题关键词。")
        keyword, _, pattern, report_year = topic

    years = {int(year) for year in re.findall(r"(?<!\d)((?:19|20)\d{2})\s*年", question)}
    if len(years) > 1:
        raise ValueError("请每次指定一个报告或公告年份。")
    year = next(iter(years), None)
    if year is None:
        return keyword, pattern, None, today - timedelta(days=365), today
    if year > today.year:
        raise ValueError("问题中的年份晚于当前年份，请核对后再查找。")
    last_year = min(year + int(report_year), today.year)
    return keyword, pattern, year if report_year else None, date(year, 1, 1), min(today, date(last_year, 12, 31))


def report_year_hint(question):
    """已导入的 PDF 也能回答没有公告类型关键词的问题。"""
    try:
        return _search_parameters(question, datetime.now(_BEIJING).date())[2]
    except ValueError:
        return None


def find_announcements(symbol, question):
    if not isinstance(symbol, str) or not re.fullmatch(r"[0-9]{6}", symbol):
        raise ValueError("查找公告需要六位股票代码。")
    keyword, pattern, year, start, end = _search_parameters(question, datetime.now(_BEIJING).date())
    raw = _fetch(ak.stock_zh_a_disclosure_report_cninfo, symbol=symbol, market="沪深京", keyword=keyword,
                 start_date=start.strftime("%Y%m%d"), end_date=end.strftime("%Y%m%d"))
    records, seen = [], set()
    if not raw.empty:
        _require(raw, ["代码", "公告标题", "公告时间", "公告链接"])
        rows = raw.loc[raw["代码"].eq(symbol)]
        dates = pd.to_datetime(rows["公告时间"], format="ISO8601", errors="coerce").dt.date
        for (title, link), published in zip(rows[["公告标题", "公告链接"]].itertuples(index=False, name=None), dates):
            title = _text(title)
            if not re.search(pattern, title) or (year is not None and not re.search(rf"(?<!\d){year}\s*年", title)):
                continue
            if pd.isna(published) or not start <= published <= end:
                continue
            try:
                parsed = urlsplit(_url(link))
            except (DataValidationError, ValueError):
                continue
            if parsed.hostname not in _OFFICIAL_HOSTS:
                continue
            url = parsed._replace(scheme="https").geturl()
            if url not in seen:
                seen.add(url)
                records.append({"title": title, "date": published, "url": url})

    if keyword == "权益分派":
        records.sort(key=lambda item: ("权益分派实施公告" in item["title"], item["date"]), reverse=True)
    else:
        records.sort(key=lambda item: ("摘要" not in item["title"], item["date"]), reverse=True)
    return {"symbol": symbol, "keyword": keyword, "year": year, "start": start, "end": end, "records": records[:5]}
