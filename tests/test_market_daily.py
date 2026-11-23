from datetime import date, datetime
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
from bs4 import BeautifulSoup
from streamlit.testing.v1 import AppTest

from tools import market_daily as daily
from ui.market_daily import daily_html


def test_indices_require_matching_close_dates_and_do_not_use_intraday_rows(monkeypatch):
    rows = {
        "sh000001": [("2026-11-19", 100), ("2026-11-20", 105), ("2026-11-23", 200)],
        "sz399001": [("2026-11-19", 100), ("2026-11-20", 98)],
        "sz399006": [("2026-11-19", 100), ("2026-11-23", 103)],
    }
    loader = Mock(side_effect=lambda symbol, start, end: pd.DataFrame(rows[symbol], columns=["date", "close"]))
    monkeypatch.setattr(daily.market, "index_daily", loader)
    now = datetime(2026, 11, 23, 14, tzinfo=daily.market.BEIJING)
    previous, day, indices = daily._benchmarks(now)
    assert (previous, day) == (date(2026, 11, 19), date(2026, 11, 20))
    assert [item["name"] for item in indices] == ["上证指数", "深证成指"]
    assert [item["close"] for item in indices] == [105, 98]
    assert loader.call_args_list[0].args[-1] == date(2026, 11, 22)


def test_daily_uses_dated_industries_and_labels_undated_stock_snapshot(monkeypatch):
    previous, day = date(2026, 11, 19), date(2026, 11, 20)
    monkeypatch.setattr(daily, "_benchmarks", lambda now: (previous, day, [{"name": "上证指数", "close": 100.0, "change": 1.0}]))
    industry = Mock(return_value=pd.DataFrame({"行业": ["行业甲", "<script>", "行业丙"], daily.RETURN: [-1.0, 2.0, None],
                                              daily.AMOUNT: [100.0, 2.0, -1.0]}))
    stocks = Mock(return_value=pd.DataFrame({"代码": ["000001", "600519"], "名称": ["甲", "乙"],
                                            daily.RETURN: [0.0, -1.0], daily.AMOUNT: [1.0, None]}))
    monkeypatch.setattr(daily.market, "industry_period_quotes", industry)
    monkeypatch.setattr(daily.market, "stock_quotes", stocks)
    report = daily.fetch_daily_report({daily.market.HISTORICAL_INDUSTRY: {"dates": [previous, previous]}})
    industry.assert_called_once_with([previous, day], periods=(1,))
    assert [row["行业"] for row in report["industry"]["leaders"]] == ["<script>"]
    assert [row["行业"] for row in report["industry"]["active"]] == ["行业甲", "<script>"]
    assert report["stocks"]["leaders"] == []
    assert report["stocks"]["flat"] == 1
    html = BeautifulSoup(daily_html(report), "html.parser")
    contents = html.get_text(" ", strip=True)
    assert "2026-11-20 市场日报" in contents and "行情所属交易日未核实" in contents
    assert "000001" in contents and "深证成指 暂无数据" in contents
    assert html.find("script") is None


def test_page_builds_on_request_and_retains_previous_report_on_failure(monkeypatch):
    stamp = datetime(2026, 11, 20, 17, tzinfo=daily.market.BEIJING)
    snapshot = {"day": stamp.date(), "previous": date(2026, 11, 19), "generated": stamp, "stock_time": None,
                "industry": None, "stocks": None, "indices": [{"name": "上证指数", "close": 100, "change": -1.0}]}
    fetch = Mock(side_effect=[snapshot, None])
    monkeypatch.setattr(daily, "fetch_daily_report", fetch)
    page = Path(__file__).resolve().parents[1] / "pages/3_市场日报.py"
    at = AppTest.from_file(str(page), default_timeout=30).run()
    fetch.assert_not_called()
    at.button[0].click().run()
    assert not at.exception
    assert [metric.value for metric in at.metric] == ["100.00 点", "暂无数据", "暂无数据"]
    assert len(at.get("download_button")) == 1
    at.button[0].click().run()
    assert "上一份成功生成的日报" in at.warning[0].value
    assert at.metric[0].value == "100.00 点"


def test_missing_industry_close_preserves_turnover_without_filling_missing_amount(monkeypatch):
    previous, day = date(2026, 11, 19), date(2026, 11, 20)
    histories = {
        "仅成交额": pd.DataFrame({"日期": [previous, day], "成交额": [100_000_000, 400_000_000]}),
        "缺成交额": pd.DataFrame({"日期": [previous, day], "收盘价": [100, pd.NA],
                                "成交额": pd.array([100_000_000, pd.NA], dtype="Float64")}),
    }
    monkeypatch.setattr(daily.market.ak, "stock_board_industry_name_ths", lambda: pd.DataFrame({"name": list(histories)}))
    monkeypatch.setattr(daily.market.ak, "stock_board_industry_index_ths", lambda symbol, **kw: histories[symbol])
    monkeypatch.setattr(daily, "_benchmarks", lambda now: (previous, day, []))
    monkeypatch.setattr(daily.market, "stock_quotes", Mock(side_effect=ValueError("snapshot missing")))
    report = daily.fetch_daily_report({})
    assert report is not None and report["industry"]["return_count"] == 0
    assert report["industry"]["active"] == [{"行业": "仅成交额", daily.AMOUNT: 4.0}]
    assert "暂无可用涨跌幅" in BeautifulSoup(daily_html(report), "html.parser").get_text()

    prices = pd.DataFrame({"日期": [previous, day], "收盘价": [100, 105]})
    monkeypatch.setattr(daily.market.ak, "stock_board_industry_index_ths", lambda symbol, **kw: prices)
    report = daily.fetch_daily_report({})
    assert report["industry"]["return_count"] == 2
    assert report["industry"]["amount_count"] == 0
