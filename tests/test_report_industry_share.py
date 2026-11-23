from datetime import date
from unittest.mock import Mock

import pandas as pd
from bs4 import BeautifulSoup

from tools import report_industry as industry
from ui.report_events import event_highlights
from ui.report_share import share_html, summary_metrics


def test_membership_requires_exact_code_in_sector_members(monkeypatch):
    pages = [
        '<table><tr><td><span>所属申万行业：</span>食品饮料 — 白酒Ⅱ</td></tr></table>',
        '<a href="/thshy/detail/code/881273/">白酒</a>',
        '<table><tbody><tr><td><a href="https://stockpage.10jqka.com.cn/600519/">600519</a></td></tr></tbody></table>',
    ]
    monkeypatch.setattr(industry, "_read_page", Mock(side_effect=[BeautifulSoup(page, "html.parser") for page in pages]))
    assert industry.match_current_industry("600519", {"白酒"}) == {
        "name": "白酒", "source": "https://q.10jqka.com.cn/thshy/detail/code/881273/",
    }
    monkeypatch.setattr(industry, "_read_page", Mock(side_effect=[BeautifulSoup(page, "html.parser") for page in pages]))
    assert industry.match_current_industry("000001", {"白酒"}) is None


def test_rank_snapshot_requires_same_trade_date_and_only_counts_valid_rows(monkeypatch):
    dates = list(pd.bdate_range(end="2024-08-30", periods=21).date)
    five, twenty = industry.hotspots.PERIOD_RETURNS.values()
    data = pd.DataFrame({"行业": ["白酒", "银行", "保险", "半导体"], five: [2, 6, 2, None], twenty: [None, 1, 2, 3]})
    snapshot = {"dates": dates, "quotes": data}
    load = Mock(return_value=dates)
    monkeypatch.setattr(industry.hotspots, "trading_dates_through", load)
    saved = {industry.hotspots.HISTORICAL_INDUSTRY: snapshot}
    assert industry.industry_snapshot(dates[-1], saved) is snapshot
    load.assert_not_called()
    assert industry.industry_snapshot(date(2024, 8, 31), saved) is None
    context = industry.industry_context(snapshot, "白酒", {"name": "银行"})
    assert context["membership"] is None
    assert context["metrics"] == [{"days": 5, "start": dates[-6], "value": 2, "rank": 2, "count": 3}]


def test_share_has_readable_figures_and_escapes_untrusted_source_text():
    market = {
        "symbol": "600519", "source": "Tencent <img src=evil>", "record_count": 2,
        "actual_range": ["2024-08-01", "2024-08-30"], "requested_range": ["2024-08-01", "2024-09-01"],
        "latest": {"close": 11, "volume": 100},
    }
    facts = {
        "rag": {"records": [{"text": "LOCAL PRIVATE PDF"}]},
        "news": {"news": {"status": "ok", "symbol": "600519", "source": "发布方", "records": [
            {"title": "<script>bad</script>", "published_at": "2024-08-30", "url": "https://example.org/item"},
            {"title": "unsafe link", "published_at": "2024-08-30", "url": "javascript:bad"},
        ]}},
        "financial": {"status": "ok", "symbol": "600519", "source_url": "https://example.org/financials", "records": [
            {"published_date": "2024-08-30", "report_date": "2024-06-30", "currency": "CNY", "revenue_yuan": 100_000_000},
        ]},
    }
    context = {
        "name": "白酒", "end": date(2024, 8, 30), "membership": None,
        "metrics": [{"days": 5, "start": date(2024, 8, 23), "value": 2.1, "rank": 3, "count": 90}],
    }
    body = share_html(market, summary_metrics(market, {"max_drawdown_pct": -5.1}), event_highlights(facts, market), context)
    document = BeautifulSoup(body, "html.parser")
    content = document.get_text(" ", strip=True)
    assert all(value in content for value in ["11.00 元", "100 手", "5.10%", "+2.10%", "第 3 / 90 名", "行业由用户选作对照"])
    assert "近 20 个交易日暂无可用排名" in content
    assert "营业收入 1.00 亿元" in content
    assert "事件筛选区间 2024-08-01 至 2024-09-01" in content
    assert document.find(["script", "img"]) is None
    assert "LOCAL PRIVATE PDF" not in body and "javascript:" not in body
    assert document.find("a", href="https://example.org/item")
