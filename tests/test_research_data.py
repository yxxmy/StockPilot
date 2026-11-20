"""人工财务及新闻记录仅用于离线测试，不代表任何公司真实信息。"""

from unittest.mock import Mock

import pandas as pd
import pytest

from tools import research_data as rd
from tools.stock_data import DataValidationError, StockDataError


@pytest.fixture
def finance():
    return pd.DataFrame({
        "SECURITY_CODE": ["600519"] * 3,
        "REPORT_DATE": ["2023-12-31", "2024-03-31", "2024-06-30"],
        "NOTICE_DATE": ["2024-04-01", "2024-04-28", "2024-08-01"],
        "CURRENCY": ["CNY"] * 3,
        **{field: ["10", "20", "30"] for field in rd.FINANCIAL_FIELDS},
    })


def finance_api(monkeypatch, frame):
    api = Mock(return_value=frame)
    monkeypatch.setattr(rd.ak, "stock_financial_analysis_indicator_em", api)
    return api


def test_financial_dates_values_and_source(monkeypatch, finance):
    original = finance.copy(deep=True)
    api = finance_api(monkeypatch, finance)
    result = rd.get_stock_financials("600519", "20240630")
    api.assert_called_once_with(symbol="600519.SH", indicator="按报告期")
    assert result["status"] == "ok"
    assert [row["report_date"] for row in result["records"]] == ["2023-12-31", "2024-03-31"]
    assert result["records"][-1]["revenue_yuan"] == 20.0
    assert result["source_url"].startswith("https://")
    assert "事后修订" in result["coverage"]
    assert result["fetched_at"].endswith("+08:00")
    pd.testing.assert_frame_equal(finance, original)


def test_missing_disclosure_excluded_not_assumed_available(monkeypatch, finance):
    finance.loc[1, "NOTICE_DATE"] = None
    finance_api(monkeypatch, finance)
    result = rd.get_stock_financials("600519", "20240630")
    assert result["excluded_undated_count"] == 1
    assert len(result["records"]) == 1


def test_missing_metric_remains_null_and_negative_allowed(monkeypatch, finance):
    finance.loc[1, "EPSJB"] = "--"
    finance.loc[1, "PARENTNETPROFIT"] = "-20"
    finance_api(monkeypatch, finance)
    row = rd.get_stock_financials("600519", "20240630")["records"][-1]
    assert row["basic_eps_yuan"] is None
    assert row["parent_net_profit_yuan"] == -20.0


@pytest.mark.parametrize("field,value", [("EPSJB", "bad"), ("EPSJB", float("inf")), ("EPSJB", True), ("CURRENCY", "USD"), ("REPORT_DATE", "bad"), ("NOTICE_DATE", "2022-01-01")])
def test_invalid_financial_values(monkeypatch, finance, field, value):
    finance.loc[1, field] = value
    finance_api(monkeypatch, finance)
    with pytest.raises(DataValidationError):
        rd.get_stock_financials("600519", "20240630")


@pytest.mark.parametrize("field", ["NOTICE_DATE", "TOTALOPERATEREVE", "SECURITY_CODE"])
def test_financial_missing_fields(monkeypatch, finance, field):
    finance_api(monkeypatch, finance.drop(columns=field))
    with pytest.raises(DataValidationError, match="缺失字段"):
        rd.get_stock_financials("600519", "20240630")


def test_financial_deduplicate_limit_and_no_other_stock(monkeypatch, finance):
    more = pd.concat([finance.iloc[[0]]] * 8, ignore_index=True)
    more["REPORT_DATE"] = [f"{year}-12-31" for year in range(2016, 2024)]
    more["NOTICE_DATE"] = [f"{year}-04-01" for year in range(2017, 2025)]
    duplicate = more.iloc[[-1]].copy()
    duplicate["EPSJB"] = "50"
    duplicate["NOTICE_DATE"] = "2024-04-02"
    other = more.iloc[[-1]].copy()
    other["SECURITY_CODE"] = "000001"
    finance_api(monkeypatch, pd.concat([more, duplicate, other]))
    records = rd.get_stock_financials("600519", "20240630")["records"]
    assert len(records) == 4
    assert records[-1]["basic_eps_yuan"] == 50


def test_future_financial_report_not_used(monkeypatch, finance):
    finance_api(monkeypatch, finance)
    assert rd.get_stock_financials("600519", "20240101")["status"] == "empty"


@pytest.fixture
def news():
    return pd.DataFrame({"关键词": ["600519"] * 4,
                         "新闻标题": ["<em>测试标题</em>"] * 4,
                         "新闻内容": ["<b>测试摘要</b>" + "长" * 500] * 4,
                         "发布时间": ["2024-06-01 10:00:00", "2024-06-02 20:00:00", "2024-07-01 10:00:00", "2024-06-02 20:00:00"],
                         "文章来源": ["测试媒体"] * 4,
                         "新闻链接": ["https://example.com/1", "https://example.com/2", "https://example.com/3", "https://example.com/2"]})


def test_news_filter_sort_deduplicate_and_sanitize(monkeypatch, news):
    monkeypatch.setattr(rd.ak, "stock_news_em", Mock(return_value=news))
    result = rd.get_stock_news("600519", "20240601", "20240602")
    assert result["matched_count"] == 2
    assert result["records"][0]["url"] == "https://example.com/2"
    assert result["records"][0]["title"] == "测试标题"
    assert len(result["records"][0]["excerpt"]) == 400
    assert "<" not in result["records"][0]["excerpt"]
    assert "不支持完整历史" in result["coverage"]


def test_news_empty_historical_match_is_not_absence_claim(monkeypatch, news):
    monkeypatch.setattr(rd.ak, "stock_news_em", Mock(return_value=news))
    result = rd.get_stock_news("600519", "20230101", "20231231")
    assert result["status"] == "empty" and result["upstream_count"] == 4
    assert "不能解释" in result["coverage"]


@pytest.mark.parametrize("field,value", [("新闻链接", "javascript:alert(1)"), ("新闻链接", "https://user:password@example.com"), ("发布时间", "bad"), ("发布时间", None), ("新闻标题", None)])
def test_invalid_news(monkeypatch, news, field, value):
    news.loc[0, field] = value
    monkeypatch.setattr(rd.ak, "stock_news_em", Mock(return_value=news))
    with pytest.raises(DataValidationError):
        rd.get_stock_news("600519", "20240101", "20240630")


def test_notice_parameters_and_stock_filter(monkeypatch):
    frame = pd.DataFrame({"代码": ["600519", "000001"], "公告标题": ["测试公告", "其他股票"],
                          "公告时间": ["2024-06-01"] * 2, "公告链接": ["https://example.com/1", "https://example.com/2"]})
    api = Mock(return_value=frame)
    monkeypatch.setattr(rd.ak, "stock_zh_a_disclosure_report_cninfo", api)
    result = rd.get_stock_notices("600519", "2024-01-01", "2024-06-30")
    api.assert_called_once_with(symbol="600519", market="沪深京", start_date="20240101", end_date="20240630")
    assert len(result["records"]) == 1
    assert "未读取PDF" in result["coverage"]


@pytest.mark.parametrize("kind", ["financial", "news", "notices"])
@pytest.mark.parametrize("mode", ["empty", "exception", "wrong_type", "duplicate_columns"])
def test_source_edge_cases(monkeypatch, kind, mode):
    name = {"financial": "stock_financial_analysis_indicator_em", "news": "stock_news_em", "notices": "stock_zh_a_disclosure_report_cninfo"}[kind]
    if mode == "exception":
        api = Mock(side_effect=TimeoutError("测试超时"))
    else:
        value = {"empty": pd.DataFrame(), "wrong_type": None,
                 "duplicate_columns": pd.DataFrame([[1, 2]], columns=["重复", "重复"])}[mode]
        api = Mock(return_value=value)
    monkeypatch.setattr(rd.ak, name, api)
    call = lambda: rd.get_stock_financials("600519", "20240630") if kind == "financial" else getattr(rd, f"get_stock_{kind}")("600519", "20240101", "20240630")
    if mode == "empty":
        assert call()["status"] == "empty"
    else:
        with pytest.raises(StockDataError):
            call()


@pytest.mark.parametrize("args", [("bad", "20240101", "20240630"), ("600519", "20240230", "20240630"), ("600519", "20240630", "20240101")])
def test_invalid_query_never_fetches(args):
    with pytest.raises(ValueError):
        rd.get_stock_news(*args)
