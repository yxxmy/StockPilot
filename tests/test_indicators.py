"""人工序列只用于公式测试，不是真实行情或分析结果。"""

import numpy as np
import pandas as pd
import pytest

from tools.indicators import calculate_indicators


def history(closes, volumes=None):
    size = len(closes)
    return pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=size, freq="B"),
        "close": np.array(closes, dtype=float),
        "volume": np.array(volumes if volumes is not None else [100] * size, dtype=float),
    })


def test_ma_percent_change_volume_and_no_mutation():
    source = history(range(1, 41), range(10, 410, 10))
    source.index = range(100, 140)
    original = source.copy(deep=True)
    result = calculate_indicators(source)
    pd.testing.assert_frame_equal(source, original)
    assert result.index.equals(source.index)
    for window in (5, 10, 20):
        assert result[f"ma{window}"].iloc[:window - 1].isna().all()
        assert result[f"ma{window}"].iloc[window - 1] == (window + 1) / 2
    assert pd.isna(result["pct_change"].iloc[0])
    assert result["pct_change"].iloc[1] == pytest.approx(100)
    assert result["pct_change"].iloc[2] == pytest.approx(50)
    assert result["volume_ma5"].iloc[4] == 30
    assert result["volume_ratio"].iloc[:5].isna().all()
    assert result["volume_ratio"].iloc[5] == 2  # 60 / 前五日均量 30


def test_macd_against_closed_form_ema():
    result = calculate_indicators(history(range(1, 41)))

    # 对收盘价 1,2,...,n，EMA 的闭式解用于独立核对 pandas 输出。
    def ema(n, span):
        decay = 1 - 2 / (span + 1)
        return n - decay * (1 - decay ** (n - 1)) / (1 - decay)

    difs = [ema(n, 12) - ema(n, 26) for n in range(26, 41)]
    assert result["macd_dif"].iloc[:25].isna().all()
    np.testing.assert_allclose(result["macd_dif"].iloc[25:], difs)
    assert result["macd_dea"].iloc[:33].isna().all()
    assert result["macd_hist"].iloc[:33].isna().all()
    dea = difs[0]
    for i, dif in enumerate(difs):
        if i:
            dea = 0.2 * dif + 0.8 * dea
        if i >= 8:
            assert result["macd_dea"].iloc[25 + i] == pytest.approx(dea)
            assert result["macd_hist"].iloc[25 + i] == pytest.approx(2 * (dif - dea))


def test_wilder_rsi_initial_mean_and_next_step():
    result = calculate_indicators(history([10, 11] * 7 + [10, 12]))
    assert result["rsi14"].iloc[:14].isna().all()
    assert result["rsi14"].iloc[14] == pytest.approx(50)
    # 初始平均涨跌均为 0.5，下一日上涨 2：100 * 8.5 / (8.5 + 6.5)。
    assert result["rsi14"].iloc[15] == pytest.approx(100 * 8.5 / 15)


@pytest.mark.parametrize("closes,expected", [
    (list(range(1, 31)), 100), (list(range(30, 0, -1)), 0), ([10] * 30, 50),
])
def test_rsi_rising_falling_and_flat(closes, expected):
    result = calculate_indicators(history(closes))
    assert (result["rsi14"].iloc[14:] == expected).all()


def test_zero_volume_does_not_create_infinity():
    result = calculate_indicators(history([10] * 8, [0] * 5 + [100, 0, 100]))
    assert pd.isna(result["volume_ratio"].iloc[5])
    assert result["volume_ratio"].iloc[6] == 0
    assert not np.isinf(result.select_dtypes("number").to_numpy()).any()


def test_single_record_keeps_warmup_nan():
    result = calculate_indicators(history([10]))
    assert result.iloc[:, 3:].isna().all().all()


def test_no_future_data_and_recalculation():
    source = history([10 + (i % 7) for i in range(60)])
    prefix = calculate_indicators(source.iloc[:40])
    full = calculate_indicators(source)
    pd.testing.assert_frame_equal(prefix, full.iloc[:40])
    pd.testing.assert_frame_equal(full, calculate_indicators(full))


@pytest.mark.parametrize("value", [None, [], pd.DataFrame()])
def test_invalid_container(value):
    with pytest.raises(ValueError, match="非空"):
        calculate_indicators(value)


@pytest.mark.parametrize("column", ["date", "close", "volume"])
def test_missing_column(column):
    with pytest.raises(ValueError, match="缺失字段"):
        calculate_indicators(history([10, 11]).drop(columns=column))


@pytest.mark.parametrize("column,value", [
    ("close", 0), ("close", -1), ("close", np.nan), ("close", np.inf),
    ("volume", -1), ("volume", np.nan), ("volume", np.inf),
])
def test_invalid_numeric_values(column, value):
    source = history([10, 11])
    source.loc[0, column] = value
    with pytest.raises(ValueError):
        calculate_indicators(source)


@pytest.mark.parametrize("values", [["10", "11"], [True, False], [10 + 1j, 11 + 0j]])
def test_invalid_numeric_types(values):
    source = history([10, 11])
    source["close"] = values
    with pytest.raises(ValueError):
        calculate_indicators(source)


@pytest.mark.parametrize("dates", [
    ["2024-01-01", "2024-01-02"],
    pd.to_datetime(["2024-01-02", "2024-01-01"]),
    pd.to_datetime(["2024-01-01", "2024-01-01"]),
    pd.to_datetime(["2024-01-01", None]),
    pd.date_range("2024-01-01", periods=2, tz="Asia/Shanghai"),
    pd.to_datetime(["2024-01-01 10:00:00", "2024-01-02 10:00:00"]),
])
def test_invalid_dates(dates):
    source = history([10, 11])
    source["date"] = dates
    with pytest.raises(ValueError):
        calculate_indicators(source)


def test_duplicate_columns():
    source = history([10, 11])
    with pytest.raises(ValueError, match="重复列名"):
        calculate_indicators(pd.concat([source, source[["close"]]], axis=1))
