"""所有单元测试默认禁止真实接口调用。"""

from unittest.mock import Mock

import pytest
import requests

from tools import stock_data


@pytest.fixture(autouse=True)
def block_live_api(monkeypatch, tmp_path):
    # 每个测试使用独立临时数据库，测试报告绝不进入真实分析历史。
    monkeypatch.setenv("STOCKPILOT_DB_PATH", str(tmp_path / "stockpilot.sqlite3"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    def blocked(*args, **kwargs):
        pytest.fail("单元测试不允许调用真实行情接口")
    monkeypatch.setattr(stock_data.ak, "stock_zh_a_hist", blocked)
    def unavailable_backup(*args, **kwargs):
        raise requests.ConnectionError("TEST ONLY: backup unavailable")
    monkeypatch.setattr(stock_data.ak, "stock_zh_a_hist_tx", unavailable_backup)
    for name in ["stock_financial_analysis_indicator_em", "stock_news_em", "stock_zh_a_disclosure_report_cninfo"]:
        monkeypatch.setattr(stock_data.ak, name, blocked)
    def offline_request(session, method, url, **kwargs):
        if method.upper() == "GET" and url == "https://www.cninfo.com.cn/new/data/szse_stock.json":
            response = Mock()
            response.json.return_value = {"stockList": [
                {"category": "A股", "code": "600519", "zwjc": "贵州茅台"},
                {"category": "A股", "code": "000001", "zwjc": "平安银行"},
            ]}
            return response
        return blocked(session, method, url, **kwargs)
    monkeypatch.setattr(requests.sessions.Session, "request", offline_request)
