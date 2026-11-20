"""明确合成样本：不代表600519的真实价格或分红。"""

from hashlib import sha256
import json
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from forecasting import data
from forecasting.targets import prepare_targets, history_at


@pytest.fixture
def frames():
    dates = pd.bdate_range("2021-01-01", periods=650)
    adjusted = pd.DataFrame({"date": dates, "open": np.arange(650)*.01+10,
        "high": np.arange(650)*.01+12, "low": np.arange(650)*.01+9,
        "close": np.arange(650)*.01+11, "volume": 100.})
    raw = adjusted.copy()
    for name in ["open", "high", "low", "close"]:
        raw[name] = raw[name]*1.2
    return adjusted, raw


def snapshot(frames):
    return data.create_snapshot(*frames, symbol="600519", source="tencent", start_date="20210101", end_date="20240101")


def test_roundtrip_and_identity_includes_source(tmp_path, frames):
    value = snapshot(frames)
    path = data.save_snapshot(value, tmp_path)
    assert data.load_snapshot(path) == value
    assert data.save_snapshot(value, tmp_path) == path
    assert value["body"]["metadata"]["data_kind"] == "test_sample"
    assert "严格历史时点" in value["body"]["metadata"]["limitation"]
    other = data.create_snapshot(*frames, symbol="600519", source="eastmoney", start_date="20210101", end_date="20240101")
    assert value["snapshot_id"] != other["snapshot_id"]
    assert len(list(tmp_path.glob("*.json"))) == 1
    first = prepare_targets(value)
    second = prepare_targets(data.load_snapshot(path))
    pd.testing.assert_frame_equal(first["returns"], second["returns"])
    pd.testing.assert_frame_equal(first["targets"], second["targets"])


def test_returns_formula_splits_and_actual_next_observation(frames):
    result = prepare_targets(snapshot(frames))
    returns, targets = result["returns"], result["targets"]
    assert len(returns) == 649 and len(targets) == 120
    np.testing.assert_allclose(returns.return_pct, 100*np.diff(np.log(frames[0].close)))
    assert targets.split.value_counts().to_dict() == {"validation":60,"test":60}
    for row in targets.itertuples():
        history = history_at(returns, row.origin_date)
        assert len(history) == 500 and history.date.max() < row.target_date
        actual = returns.loc[returns.date == row.target_date, "return_pct"].item()
        assert row.realized_return_pct == actual and row.variance_proxy == actual**2
    assert result["initial_train_range"][0] == str(history_at(returns, targets.origin_date.iloc[0]).date.min().date())


def test_future_change_cannot_change_past_inputs(frames):
    before = prepare_targets(snapshot(frames))
    cut = before["targets"].origin_date.iloc[5]
    changed = tuple(frame.copy() for frame in frames)
    for frame in changed:
        frame.loc[frame.date > cut, ["open","high","low","close"]] *= 1.5
    after = prepare_targets(snapshot(changed))
    pd.testing.assert_frame_equal(history_at(before["returns"], cut), history_at(after["returns"], cut))
    assert not before["targets"].variance_proxy.equals(after["targets"].variance_proxy)


def test_minimum_and_no_zero_fill(frames):
    with pytest.raises(data.ForecastDataError, match="621"):
        snapshot(tuple(f.iloc[:620].copy() for f in frames))
    result = prepare_targets(snapshot(tuple(f.iloc[:621].copy() for f in frames)))
    assert len(result["returns"]) == 620
    assert result["returns"].return_pct.notna().all()


@pytest.mark.parametrize("field,value", [("close",0),("close",-1),("close",np.nan),("open",np.inf),("volume",True)])
def test_invalid_prices_rejected(frames, field, value):
    frames[0][field] = frames[0][field].astype(object)
    frames[0].loc[20,field] = value
    with pytest.raises(data.ForecastDataError):
        snapshot(frames)


def test_deduplication_sorting_and_quality_are_recorded(frames):
    duplicate = tuple(pd.concat([f.iloc[::-1],f.iloc[[0]]],ignore_index=True) for f in frames)
    value = snapshot(duplicate)
    assert value["body"]["metadata"]["quality"]["discarded_adjusted_rows"] == 1
    assert len(prepare_targets(value)["returns"]) == 649


def test_alignment_gap_and_unknown_adjustment_rejected(frames):
    with pytest.raises(data.ForecastDataError, match="日期"):
        snapshot((frames[0],frames[1].iloc[1:]))
    with pytest.raises(data.ForecastDataError, match="无法确认"):
        snapshot((frames[1],frames[1]))
    value = snapshot(frames)
    value["body"]["metadata"]["adjustment"] = "unknown"
    # 即使调用方重新计算哈希，也须执行口径校验。
    value["snapshot_id"] = sha256(data._canonical(value["body"])).hexdigest()
    with pytest.raises(data.ForecastDataError):
        data.load_snapshot_value(value)


def test_corrupt_files_and_no_overwrite(tmp_path, frames):
    value = snapshot(frames)
    path = data.save_snapshot(value,tmp_path)
    path.write_text("{broken",encoding="utf-8")
    with pytest.raises(data.ForecastDataError):
        data.load_snapshot(path)
    with pytest.raises(data.ForecastDataError, match="覆盖"):
        data.save_snapshot(value,tmp_path)
    value["body"]["prices"][0]["close"] = 2
    with pytest.raises(data.ForecastDataError, match="哈希"):
        data.load_snapshot_value(value)


def test_explicit_source_requests_both_adjustments_and_converts_volume(monkeypatch, frames):
    responses = []
    for frame in frames:
        frame = frame.copy()
        frame.volume *= 100
        responses.append(frame)
    tx = Mock(side_effect=responses)
    monkeypatch.setattr(data.ak,"stock_zh_a_hist_tx",tx)
    value = data.fetch_snapshot("600519","20210101","20240101",source="tencent")
    assert [c.kwargs["adjust"] for c in tx.call_args_list] == ["qfq", ""]
    assert value["body"]["prices"][0]["volume"] == 100
    assert value["body"]["metadata"]["data_kind"] == "live"


def test_fetch_failure_never_switches_or_saves(monkeypatch,tmp_path):
    tx=Mock(side_effect=RuntimeError("TEST ONLY private config"))
    other=Mock()
    monkeypatch.setattr(data.ak,"stock_zh_a_hist_tx",tx)
    monkeypatch.setattr(data.ak,"stock_zh_a_hist",other)
    with pytest.raises(data.ForecastDataError,match="未换源") as caught:
        data.fetch_snapshot("600519","20200101","20240101")
    assert "private" not in str(caught.value)
    tx.assert_called_once()
    other.assert_not_called()
    assert list(tmp_path.iterdir()) == []


def test_eastmoney_explicit_qfq_and_unadjusted_keep_lot_units(monkeypatch,frames):
    values=[f.rename(columns={v:k for k,v in data.FIELD_MAP.items()}) for f in frames]
    api=Mock(side_effect=values)
    monkeypatch.setattr(data.ak,"stock_zh_a_hist",api)
    value=data.fetch_snapshot("600519","20210101","20240101",source="eastmoney")
    assert [c.kwargs["adjust"] for c in api.call_args_list] == ["qfq", ""]
    assert value["body"]["prices"][0]["volume"] == 100


def test_offline_cli_reads_snapshot_without_network(tmp_path,frames,monkeypatch,capsys):
    import sys
    from forecasting.__main__ import main
    path=data.save_snapshot(snapshot(frames),tmp_path)
    api=Mock(side_effect=AssertionError("inspect must be offline"))
    monkeypatch.setattr(data.ak,"stock_zh_a_hist_tx",api)
    monkeypatch.setattr(sys,"argv",["forecasting","inspect",str(path)])
    assert main() == 0
    output=json.loads(capsys.readouterr().out)
    assert output["evaluation_targets"] == 120 and output["returns"] == 649
    api.assert_not_called()
