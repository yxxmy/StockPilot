"""从报告内的材料提取近期事项和可核对的观察项。"""

from datetime import datetime, timedelta, timezone
from math import isfinite

import streamlit as st

from tools.research_data import _url
from tools.stock_data import DataValidationError
from ui.service import public_text


_BEIJING = timezone(timedelta(hours=8))
_CHECKS = (
    (("权益分派", "分红", "派息"), "股东现金分配安排", "核对每股金额、股权登记日和实施日期。"),
    (("回购",), "回购安排及执行情况", "核对公告处于计划、进展还是完成阶段，以及涉及的金额和期限。"),
    (("业绩", "年报", "年度报告", "季报", "季度报告", "半年度报告"),
     "公司经营表现", "核对对应报告期、收入和利润变化，以及公司说明的原因。"),
    (("风险提示", "诉讼", "处罚", "减持"), "公司披露的风险或股东变动", "核对事项涉及的主体、规模和后续进展。"),
)


def _stamp(value):
    if not isinstance(value, str):
        return None
    try:
        stamp = datetime.fromisoformat(value)
    except ValueError:
        return None
    return stamp.astimezone(_BEIJING).replace(tzinfo=None) if stamp.tzinfo is not None else stamp


def _dict(value):
    return value if isinstance(value, dict) else {}


def _records(envelope, symbol):
    records = envelope.get("records")
    if envelope.get("status") == "ok" and envelope.get("symbol") == symbol and isinstance(records, list):
        return [record for record in records if isinstance(record, dict)]
    return []


def _source_url(value):
    if not isinstance(value, str) or public_text(value) != value:
        return None
    try:
        return _url(value)
    except DataValidationError:
        return None


def _document_items(envelope, symbol, kind, start, end):
    items = []
    for entry in _records(envelope, symbol):
        stamp = _stamp(entry.get("published_at"))
        title = entry.get("title")
        url = _source_url(entry.get("url"))
        if not (stamp and start <= stamp.date() <= end and isinstance(title, str) and title.strip() and url):
            continue
        focus, check = next(
            ((focus, check) for terms, focus, check in _CHECKS if any(word in title for word in terms)),
            ("事项进度及影响范围", "打开原文核对涉及主体、发生时间和后续披露。"),
        )
        items.append({
            "kind": kind, "stamp": stamp, "title": title.strip(), "focus": focus, "check": check, "url": url,
            "source": entry.get("publisher") or envelope.get("source") or url,
            "excerpt": entry.get("excerpt"),
        })
    return sorted(items, key=lambda item: item["stamp"], reverse=True)


def _financial_item(envelope, symbol, start, end):
    url = _source_url(envelope.get("source_url"))
    if not url:
        return []
    candidates = []
    for row in _records(envelope, symbol):
        disclosed = _stamp(row.get("published_date"))
        period = _stamp(row.get("report_date"))
        if disclosed and period and period <= disclosed and start <= disclosed.date() <= end:
            candidates.append((disclosed, period, row))
    if not candidates:
        return []
    disclosed, period, latest = max(candidates, key=lambda entry: entry[:2])
    figures = []
    if latest.get("currency") == "CNY":
        for key, label in (("revenue_yuan", "营业收入"), ("parent_net_profit_yuan", "归母净利润")):
            value = latest.get(key)
            if type(value) in (int, float) and isfinite(value):
                figures.append(f"{label} {value / 100_000_000:,.2f} 亿元")
    return [{
        "kind": "财务披露", "stamp": disclosed, "title": f"截至 {period:%Y-%m-%d} 报告期的财务指标",
        "source": envelope.get("source") or "报告保存的财务数据",
        "url": url, "excerpt": "，".join(figures),
        "focus": "该报告期的经营表现",
        "check": "回看财报原文，核对相同报告期的同比数据、经营现金流和非经常性损益。",
    }]


def event_date_range(market):
    span = market.get("requested_range") or market.get("actual_range")
    if not isinstance(span, list) or len(span) != 2:
        return None
    first, last = map(_stamp, span)
    if first is None or last is None or first > last:
        return None
    return first.date(), last.date()


def event_highlights(facts, market):
    """按报告请求的日期和股票取可追溯事项，优先各取一条公告、财务和新闻。"""
    symbol = market.get("symbol")
    span = event_date_range(market)
    if not symbol or span is None:
        return []
    start, end = span
    news = _dict(facts.get("news"))
    groups = (
        _document_items(_dict(news.get("notices")), symbol, "公告", start, end),
        _financial_item(_dict(facts.get("financial")), symbol, start, end),
        _document_items(_dict(news.get("news")), symbol, "新闻", start, end),
    )
    ranked = []
    seen_titles, seen_links = set(), set()

    def add_once(item):
        title = " ".join(item["title"].split()).casefold()
        url = item["url"]
        if title in seen_titles or url in seen_links:
            return False
        ranked.append(item)
        seen_titles.add(title)
        seen_links.add(url)
        return True

    for group in groups:
        for item in group:
            if add_once(item):
                break
    for item in sorted((item for group in groups for item in group), key=lambda item: item["stamp"], reverse=True):
        if len(ranked) == 3:
            break
        add_once(item)
    return ranked


def _show_document_references(rag, symbol):
    records = rag.get("records")
    if not symbol or rag.get("status") != "ok" or not isinstance(records, list):
        return
    matches = [
        hit for hit in records
        if isinstance(hit, dict) and _dict(hit.get("metadata")).get("symbol") == symbol
        and type(hit.get("page")) is int and hit["page"] > 0
        and isinstance(hit.get("text"), str) and hit["text"].strip()
    ]
    if matches:
        with st.expander("核对本地 PDF 原文"):
            st.caption("按报告的正文问题检索，披露日期见各条标注。")
            for hit in matches[:3]:
                date = _stamp(hit["metadata"].get("disclosure_date"))
                disclosed = f" · 披露日 {date:%Y-%m-%d}" if date else ""
                st.text(public_text(f"[{hit.get('ref_id', '')}] {hit.get('document_name', '')} · 物理第{hit['page']}页{disclosed}"))
                st.text(public_text(hit["text"]))


def show_event_highlights(facts, market, *, items):
    if any(name in facts for name in ("news", "financial")):
        st.subheader("近期事件")
        if span := event_date_range(market):
            st.caption(f"事件筛选区间 {span[0]} 至 {span[1]}")
        if not items:
            st.info("报告所选区间内暂无可追溯的事件条目。可展开原始数据查看覆盖情况。")
        for item in items:
            with st.container(border=True):
                label = "披露" if item["kind"] == "财务披露" else "发布"
                stamp = item["stamp"].isoformat(sep=" ", timespec="seconds").removesuffix(" 00:00:00")
                st.caption(f"{item['kind']} · {stamp} {label}")
                st.text(public_text(item["title"]))
                if item["excerpt"]:
                    st.text(public_text(item["excerpt"]))
                st.text("关注 · " + item["focus"])
                st.text("接下来核对 · " + item["check"])
                st.text("来源 · " + public_text(item["source"]))
                label = "查看财务数据来源" if item["kind"] == "财务披露" else "打开来源"
                st.link_button(label, item["url"])
    _show_document_references(_dict(facts.get("rag")), market.get("symbol"))
