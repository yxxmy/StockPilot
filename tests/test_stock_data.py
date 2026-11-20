"""本文件中的小数值行情仅用于离线测试，不是真实行情。"""

from unittest.mock import Mock

import pandas as pd
import pytest

from tools import stock_data
from tools.stock_data import DataValidationError, EmptyDataError, StockDataError, get_stock_history


@pytest.fixture
def raw():
    return pd.DataFrame({
        "日期": ["2024-01-03", "2024-01-02", "2024-01-03", "2024-01-04"],
        "开盘": ["10", "10", "11", "10"], "最高": ["12"] * 4,
        "最低": ["9"] * 4, "收盘": ["11"] * 4,
        "成交量": ["100", "200", "300", "400"], "成交额": [1000] * 4,
    })


def mock_api(monkeypatch, value):
    api = Mock(return_value=value)
    monkeypatch.setattr(stock_data.ak, "stock_zh_a_hist", api)
    return api


def test_normalize_sort_deduplicate_filter_and_api_arguments(monkeypatch, raw):
    original = raw.copy(deep=True)
    api = mock_api(monkeypatch, raw)
    data = get_stock_history("000001", "2024-01-02", "20240103")
    assert list(data.columns) == ["date", "open", "high", "low", "close", "volume"]
    assert data["date"].dt.strftime("%Y-%m-%d").tolist() == ["2024-01-02", "2024-01-03"]
    assert data["volume"].tolist() == [200.0, 300.0]
    assert data["open"].tolist() == [10.0, 11.0]
    assert str(data["date"].dtype) == "datetime64[ns]"
    assert all(str(data[c].dtype) == "float64" for c in data.columns[1:])
    assert data.index.tolist() == [0, 1]
    pd.testing.assert_frame_equal(raw, original)
    api.assert_called_once_with(symbol="000001", period="daily", start_date="20240102",
                                end_date="20240103", adjust="")


@pytest.mark.parametrize("symbol", ["", "60051", "6005190", "sh600519", "ABCDEF", "６００５１９", 600519, None])
def test_invalid_symbol(symbol):
    with pytest.raises(ValueError, match="六位"):
        get_stock_history(symbol, "20240101", "20240131")


@pytest.mark.parametrize("start,end", [
    ("20240230", "20240301"), ("2024/01/01", "20240131"),
    ("20240102", "20240101"), (None, "20240131"),
    ("20240101", "bad"), ("2024-1-1", "20240131"),
])
def test_invalid_dates(start, end):
    with pytest.raises(ValueError):
        get_stock_history("600519", start, end)


def test_api_failure_preserves_cause(monkeypatch):
    cause = TimeoutError("测试超时")
    api = Mock(side_effect=cause)
    monkeypatch.setattr(stock_data.ak, "stock_zh_a_hist", api)
    with pytest.raises(StockDataError, match="获取 600519 历史行情失败") as error:
        get_stock_history("600519", "20240101", "20240131")
    assert error.value.__cause__ is cause
    api.assert_called_once()


@pytest.mark.parametrize("value", [None, pd.DataFrame()])
def test_empty_data(monkeypatch, value):
    mock_api(monkeypatch, value)
    with pytest.raises(EmptyDataError):
        get_stock_history("600519", "20240101", "20240131")


@pytest.mark.parametrize("field", list(stock_data.FIELD_MAP))
def test_missing_fields(monkeypatch, raw, field):
    mock_api(monkeypatch, raw.drop(columns=field))
    with pytest.raises(DataValidationError, match="缺失字段"):
        get_stock_history("600519", "20240101", "20240131")


@pytest.mark.parametrize("field,value", [
    ("日期", "not-a-date"), ("日期", None), ("日期", "2024-01-03 10:00:00"),
    ("开盘", "bad"), ("收盘", None), ("成交量", float("inf")),
    ("成交量", "-1"), ("最高", "1"), ("最低", "20"),
    ("开盘", "0"), ("成交量", True),
])
def test_invalid_values(monkeypatch, raw, field, value):
    raw.loc[0, field] = value
    mock_api(monkeypatch, raw)
    with pytest.raises(DataValidationError):
        get_stock_history("600519", "20240101", "20240131")


def test_no_rows_in_requested_range(monkeypatch, raw):
    mock_api(monkeypatch, raw)
    with pytest.raises(EmptyDataError):
        get_stock_history("600519", "20240201", "20240202")


def test_unexpected_response(monkeypatch):
    mock_api(monkeypatch, {"日期": []})
    with pytest.raises(DataValidationError, match="DataFrame"):
        get_stock_history("600519", "20240101", "20240131")


def test_duplicate_columns(monkeypatch, raw):
    mock_api(monkeypatch, pd.concat([raw, raw[["开盘"]]], axis=1))
    with pytest.raises(DataValidationError, match="重复列名"):
        get_stock_history("600519", "20240101", "20240131")


def test_single_day_and_zero_volume(monkeypatch, raw):
    raw.loc[1, "成交量"] = "0"
    mock_api(monkeypatch, raw)
    data = get_stock_history("600519", "20240102", "20240102")
    assert len(data) == 1
    assert data.iloc[0]["volume"] == 0
