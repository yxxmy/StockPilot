"""真实时序页面与评估集成；合成快照仅供离线测试。"""
from pathlib import Path
from unittest.mock import Mock
import arch  # noqa:F401；Windows原生统计依赖在主线程预载，避免首次在AppTest线程加载。
import pytest
from streamlit.testing.v1 import AppTest

from ui import forecast_service as service
from forecasting.data import ForecastDataError, create_snapshot, save_snapshot
from test_forecasting_data import frames  # noqa:F401


def page():
    return AppTest.from_file(str(Path(__file__).resolve().parents[1]/"pages/1_个股波动预测.py"),default_timeout=60).run()


@pytest.fixture
def setup(monkeypatch,tmp_path,frames):
    monkeypatch.setenv("STOCKPILOT_FORECAST_DIR",str(tmp_path))
    snap=create_snapshot(*frames,symbol="600519",source="tencent",start_date="20210101",end_date="20240101")
    save_snapshot(snap,tmp_path)
    evaluate=Mock(wraps=service.evaluate_snapshot)
    save=Mock(wraps=service.save_experiment)
    monkeypatch.setattr(service,"evaluate_snapshot",evaluate)
    monkeypatch.setattr(service,"save_experiment",save)
    return tmp_path,snap,evaluate,save


def test_actual_page_runs_and_rerun_duplicate_do_not_train(setup):
    root,snap,evaluate,save=setup
    at=page()
    assert not at.exception and evaluate.call_count==0
    assert at.title[0].value == "个股波动预测"
    assert at.text_input(key="forecast_stock").value == "600519"
    assert at.button[0].label == "开始预测"
    at.button[0].click().run()
    assert not at.exception and not at.error
    first=at.session_state["forecast_result"]["experiment"]["run_id"]
    assert len(at.session_state["forecast_result"]["experiment"]["rows"])==120
    prediction = at.session_state["forecast_result"]["experiment"]["next_observation"]
    assert prediction["origin_date"] == snap["body"]["metadata"]["actual_range"][-1]
    assert at.metric[0].value.endswith("%") and len(at.metric) == 2
    assert any("过去预测得怎么样" in header.value for header in at.subheader)
    at.run()
    at.button[0].click().run()
    assert evaluate.call_count==save.call_count==1
    assert at.session_state["forecast_result"]["experiment"]["run_id"]==first
    assert len(list((root/"experiments").iterdir()))==1
    assert len(at.dataframe)==2


def test_empty_snapshot_can_fetch_on_submit_and_corruption_is_visible(monkeypatch,tmp_path):
    monkeypatch.setenv("STOCKPILOT_FORECAST_DIR",str(tmp_path))
    fetch = Mock(side_effect=ForecastDataError("测试行情源失败"))
    monkeypatch.setattr(service, "fetch_snapshot", fetch)
    at=page()
    assert not at.button[0].disabled and not at.exception
    (tmp_path/"bad.json").write_text("{}",encoding="utf-8")
    at.run()
    assert at.warning and not at.button[0].disabled
    fetch.assert_not_called()
    at.text_input(key="forecast_stock").set_value("000001").run()
    fetch.assert_not_called()
    at.button[0].click().run()
    assert at.error[0].value == "测试行情源失败"
    assert fetch.call_args.args[0] == "000001" and fetch.call_args.kwargs["source"] == "tencent"
    at.run()
    fetch.assert_called_once()


def test_save_error_not_cached_and_rerun_does_not_retry(setup,monkeypatch):
    _,_,evaluate,save=setup
    save.side_effect=OSError("SECRET_PATH")
    at=page().button[0].click().run()
    assert at.error and not at.exception
    assert "forecast_result" not in at.session_state
    assert not at.session_state["forecast_completed"]
    assert "SECRET_PATH" not in str([x.value for x in at.error])
    count=evaluate.call_count
    at.run()
    assert evaluate.call_count==count


def test_new_snapshot_identity_generates_new_experiment(setup,frames):
    root,snap,evaluate,save=setup
    session={}
    assert service.run_once(session,snap)
    for f in frames:
        f.loc[0,["open","high","low","close"]]*=.99
    second=create_snapshot(*frames,symbol="600519",source="tencent",start_date="20210101",end_date="20240101")
    assert service.run_once(session,second)
    assert evaluate.call_count==save.call_count==2
    assert len(list((root/"experiments").iterdir()))==2


def test_missing_model_dependency_is_friendly_and_not_retried(setup):
    _,_,evaluate,_=setup
    evaluate.side_effect=ImportError("SECRET_PATH")
    at=page().button[0].click().run()
    assert at.error and not at.exception
    assert "SECRET_PATH" not in str([x.value for x in at.error])
    at.run()
    assert evaluate.call_count==1


def test_acquire_requires_explicit_refresh(setup,monkeypatch):
    _,snap,_,_=setup
    fetch=Mock(return_value=snap)
    monkeypatch.setattr(service,"fetch_snapshot",fetch)
    session={}
    service.acquire(session,"600519","20210101","20240101","tencent")
    service.acquire(session,"600519","20210101","20240101","tencent")
    fetch.assert_called_once()
    service.acquire(session,"600519","20210101","20240101","tencent",refresh=True)
    assert fetch.call_count==2
