from datetime import date, datetime
from functools import partial
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from agents import workflow
from memory import ReportStore
from tools import market_hotspots as market


def stock_fixture():
    quotes = pd.DataFrame([
        ["sz000001", "测试银行", "1.75", "979631", "0.54", "GP-A"],
        ["bj920001", "测试科技", "9.10", "10000", "-", "GP"],
        ["sh688256", "测试科创股", "0.90", "5000", "1.00", "GP-A-KCB"],
        ["sz300308", "测试创业股", "0.80", "1000", "2.00", "GP-A-CYB"],
        ["sh600519", "测试酒业", "-", "-", "-", "GP-A"],
        ["sh510000", "测试基金", "19.00", "10000", "3.60", "ETF"],
        ["sh900000", "测试其他股票", "11.00", "10000", "3.60", "GP"],
    ], columns=["code", "name", "zdf", "turnover", "hsl", "stock_type"])
    quotes["zdf_d5"] = ["1.2", "9.8", "22", "4", "-", "50", "99"]
    quotes["zdf_d20"] = ["10.5", "3", "4", "30", "-", "50", "99"]
    return quotes


def industry_fixture():
    return pd.DataFrame([
        ["银行", "0.21", "451.18", "23", "19", "测试银行", "1.75"],
        ["保险", "0.39", "97.22", "3", "2", "测试保险", "2.30"],
    ], columns=["板块", "涨跌幅", "总成交额", "上涨家数", "下跌家数", "领涨股", "领涨股-涨跌幅"])


def test_stock_quotes_require_a_share_and_preserve_code(monkeypatch):
    monkeypatch.setattr(market.ak, "stock_zh_a_spot_tx", stock_fixture)
    quotes = market.stock_quotes()
    assert len(quotes) == 5
    assert quotes["涨跌幅（%）"].isna().sum() == 1
    leaders = market.ranking(quotes)
    assert leaders["代码"].tolist() == ["920001", "000001", "688256", "300308"]
    assert leaders["成交额（亿元）"].tolist() == pytest.approx([1, 97.9631, 0.5, 0.1])
    assert pd.isna(leaders.iloc[0]["换手率（%）"])


@pytest.mark.parametrize(("hour", "day"), [(14, 20), (15, 23)])
def test_reference_date_only_uses_completed_index_sessions(monkeypatch, hour, day):
    rows = pd.DataFrame({"date": ["2026-11-20", "2026-11-23", "2026-11-24", "invalid"]})
    load = Mock(return_value=rows)
    monkeypatch.setattr(market.ak, "stock_zh_index_daily_tx", load)
    assert market.latest_closed_date(datetime(2026, 11, 23, hour, tzinfo=market.BEIJING)) == date(2026, 11, day)
    assert load.call_args.kwargs["symbol"] == "sh000001"


def test_page_refresh_sort_and_keep_failed_snapshot(monkeypatch):
    stocks = Mock(side_effect=[stock_fixture(), RuntimeError("raw provider detail")])
    industries = Mock(return_value=industry_fixture())
    monkeypatch.setattr(market.ak, "stock_zh_a_spot_tx", stocks)
    monkeypatch.setattr(market.ak, "stock_board_industry_summary_ths", industries)
    monkeypatch.setattr(market, "latest_closed_date", lambda now: date(2026, 11, 20))
    page = Path(__file__).resolve().parents[1] / "pages/2_市场热点.py"
    at = AppTest.from_file(str(page)).run()
    stocks.assert_not_called()
    at.button[0].click().run()
    assert not at.exception and not at.warning
    assert at.dataframe[0].value["代码"].tolist() == ["920001", "000001", "688256", "300308"]
    assert at.dataframe[1].value["行业"].tolist() == ["保险", "银行"]
    assert any("行情日期未确认" in caption.value for caption in at.caption)
    assert any("最近已收盘参考日 2026-11-20" in caption.value for caption in at.caption)
    original = at.session_state["market_rankings"]["个股"]["fetched_at"]
    at.radio[0].set_value("成交额（亿元）").run()
    assert at.dataframe[1].value["行业"].tolist() == ["银行", "保险"]
    stocks.assert_called_once()
    industries.assert_called_once()
    at.button[0].click().run()
    assert not at.exception and "上次成功" in at.warning[0].value
    assert "raw provider detail" not in at.warning[0].value
    assert at.session_state["market_rankings"]["个股"]["fetched_at"] == original
    assert len(at.dataframe) == 2


def test_open_ranked_stock_prefills_form_and_waits_for_submission(monkeypatch):
    monkeypatch.setattr(market.ak, "stock_zh_a_spot_tx", stock_fixture)
    monkeypatch.setattr(market.ak, "stock_board_industry_summary_ths", industry_fixture)
    monkeypatch.setattr(market, "latest_closed_date", lambda now: date(2026, 11, 20))
    prices = pd.DataFrame({
        "date": pd.date_range("2024-01-02", periods=30, freq="B"),
        "open": 10.0, "high": 12.0, "low": 9.0, "close": 11.0, "volume": 100.0,
    })
    history = Mock(return_value=prices)
    monkeypatch.setattr(workflow, "build_graph", partial(workflow.build_graph, history_loader=history))
    main_page = Path(__file__).resolve().parents[1] / "streamlit_app.py"

    at = AppTest.from_file(str(main_page)).run()
    at.switch_page("pages/2_市场热点.py").run()
    at.button[0].click().run()
    assert at.button(key="open_market_stock").disabled
    at.session_state[at.dataframe[0].key] = {"selection": {"rows": [1], "columns": [], "cells": []}}
    at.run()
    at.button(key="open_market_stock").click().run()
    assert not at.exception
    assert at.text_input(key="symbol").value == "000001"
    history.assert_not_called()
    assert "report_id" not in at.session_state and "result" not in at.session_state

    at.switch_page("streamlit_app.py")
    at.date_input(key="start").set_value(date(2024, 1, 1))
    at.date_input(key="end").set_value(date(2024, 6, 30))
    at.button[0].click().run()
    assert not at.error and not at.exception
    history.assert_called_once_with("000001", "20230101", "2024-06-30")
    assert [row["symbol"] for row in ReportStore().list_reports()] == ["000001"]


def test_industry_period_uses_exact_sessions_and_requires_complete_windows(monkeypatch):
    dates = pd.bdate_range(end="2026-11-20", periods=21).date.tolist()
    full = pd.DataFrame({"日期": dates, "收盘价": range(100, 121), "成交额": 200_000_000})
    histories = {
        "完整": full,
        "缺中间日": full.drop(index=5),
        "旧数据": full.iloc[:-1],
        "成交额缺失": full.assign(成交额=[200_000_000] * 20 + [None]),
    }
    monkeypatch.setattr(market.ak, "stock_board_industry_name_ths",
                        lambda: pd.DataFrame({"name": list(histories)}))
    monkeypatch.setattr(market.ak, "stock_board_industry_index_ths", lambda symbol, **dates: histories[symbol])
    quotes = market.industry_period_quotes(dates).set_index("行业")
    assert quotes.loc["完整", market.PERIOD_RETURNS[5]] == pytest.approx((120 / 115 - 1) * 100)
    assert quotes.loc["完整", market.PERIOD_RETURNS[20]] == pytest.approx(20)
    assert quotes.loc["完整", market.PERIOD_AMOUNTS[5]] == 10
    assert quotes.loc["完整", market.PERIOD_AMOUNTS[20]] == 40
    assert quotes[market.PERIOD_RETURNS[5]].notna().sum() == 3
    assert quotes[market.PERIOD_RETURNS[20]].notna().sum() == 2
    assert quotes[market.PERIOD_AMOUNTS[20]].notna().sum() == 1


def test_period_switch_reuses_quotes_and_industry_daily_cache(monkeypatch):
    stocks = Mock(side_effect=stock_fixture)
    dates = pd.bdate_range(end="2026-11-20", periods=21).date.tolist()
    frame = pd.DataFrame({
        "行业": ["测试行业"],
        market.PERIOD_RETURNS[5]: [5.5], market.PERIOD_RETURNS[20]: [10.0],
        market.PERIOD_AMOUNTS[5]: [200], market.PERIOD_AMOUNTS[20]: [350],
    })
    history = Mock(return_value=frame)
    monkeypatch.setattr(market.ak, "stock_zh_a_spot_tx", stocks)
    monkeypatch.setattr(market, "industry_period_quotes", history)
    monkeypatch.setattr(market, "completed_dates", lambda now: dates)
    page = Path(__file__).resolve().parents[1] / "pages/2_市场热点.py"
    at = AppTest.from_file(str(page)).run()
    at.selectbox(key="ranking_period").select(5).run()
    stocks.assert_not_called()
    at.button[0].click().run()
    assert not at.exception and not at.warning
    assert at.dataframe[0].value["代码"].iloc[0] == "688256"
    assert at.dataframe[1].value[market.PERIOD_RETURNS[5]].tolist() == [5.5]
    assert any("统计区间 2026-11-16 至 2026-11-20" in caption.value for caption in at.caption)

    at.selectbox(key="ranking_period").select(20).run()
    assert at.dataframe[0].value["代码"].iloc[0] == "300308"
    assert at.dataframe[1].value[market.PERIOD_RETURNS[20]].tolist() == [10]
    assert any("涨幅可用 4 / 5 只 A 股" in caption.value for caption in at.caption)
    assert any("所选指标可用 1 / 1 个同花顺行业" in caption.value for caption in at.caption)
    stocks.assert_called_once()
    history.assert_called_once_with(dates)
