"""简报事项只使用同股票、同区间且来源可核对的记录。"""

from ui.report_events import event_highlights


def test_event_highlights_balances_sources_and_skips_unusable_records():
    def item(title, date, url):
        return {"title": title, "published_at": date, "url": url}

    def source(records, symbol="600519", **details):
        return {"symbol": symbol, "status": "ok", "records": records, **details}

    market = {"symbol": "600519", "requested_range": ["2026-11-01", "2026-11-20"]}
    facts = {
        "news": {
            "notices": source([
                item("后续事项", "2026-11-21", "https://example.invalid/future"),
                item("回购进展公告", "2026-11-20", "https://example.invalid/buyback"),
            ], source="披露来源"),
            "news": source([
                item("回购进展公告", "2026-11-20", "https://example.invalid/reprint"),
                item("异常链接", "2026-11-19", "javascript:alert(1)"),
                item("有效新闻", "2026-11-18 09:00:00", "https://example.invalid/news"),
            ]),
        },
        "financial": source([
            {"published_date": "2026-11-19", "report_date": "2026-09-30", "currency": "CNY", "revenue_yuan": 900_000_000},
        ], source_url="https://example.invalid/financials"),
    }
    entries = event_highlights(facts, market)
    assert [entry["kind"] for entry in entries] == ["公告", "财务披露", "新闻"]
    assert [entry["url"] for entry in entries] == [
        "https://example.invalid/buyback", "https://example.invalid/financials", "https://example.invalid/news",
    ]
    assert "金额和期限" in entries[0]["check"]
    assert "营业收入 9.00 亿元" == entries[1]["excerpt"]

    facts["news"]["notices"]["symbol"] = "000001"
    facts["news"]["news"]["status"] = "error"
    facts["financial"]["source_url"] = "ftp://example.invalid/financials"
    assert event_highlights(facts, market) == []


def test_only_selected_notices_remove_duplicate_headlines_from_news():
    market = {"symbol": "600519", "requested_range": ["2024-04-01", "2024-04-30"]}
    notice = {"symbol": "600519", "status": "ok", "records": [
        {"title": "最新公告", "published_at": "2024-04-26", "url": "https://example.invalid/latest-notice"},
        {"title": "旧公告", "published_at": "2024-04-01", "url": "https://example.invalid/older-notice"},
    ]}
    news = {"symbol": "600519", "status": "ok", "records": [
        {"title": "旧公告", "published_at": "2024-04-25", "url": "https://example.invalid/latest-news"},
        {"title": "其他新闻", "published_at": "2024-04-24", "url": "https://example.invalid/other-news"},
    ]}
    selected = event_highlights({"news": {"notices": notice, "news": news}}, market)
    assert [item["url"] for item in selected] == [notice["records"][0]["url"], *[item["url"] for item in news["records"]]]
