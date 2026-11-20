"""只构造历史收益和事后评估标签，不拟合、预测或计算收益策略。"""

import numpy as np
import pandas as pd

from forecasting.data import ForecastDataError, load_snapshot_value


TRAIN_WINDOW = 500
VALIDATION_SIZE = 60
TEST_SIZE = 60


def prepare_targets(snapshot):
    prices = load_snapshot_value(snapshot)
    # 对数之差避免直接除法溢出；第一条没有前置价格，不填零。
    r = 100 * np.log(prices.close).diff()
    returns = pd.DataFrame({"date": prices.date, "return_pct": r}).iloc[1:].reset_index(drop=True)
    if not np.isfinite(returns.return_pct).all():
        raise ForecastDataError("收益率不是有限数值")
    split = len(returns) - VALIDATION_SIZE - TEST_SIZE
    if split < TRAIN_WINDOW:
        raise ForecastDataError("至少需要500条训练收益、60条验证收益和60条测试收益")
    # 目标独立于输入表；未来r平方只能用于评估，不能拿来作为特征。
    targets = pd.DataFrame({
        "origin_date": returns.date.iloc[split-1:-1].to_numpy(),
        "target_date": returns.date.iloc[split:].to_numpy(),
        "realized_return_pct": returns.return_pct.iloc[split:].to_numpy(),
        "variance_proxy": np.square(returns.return_pct.iloc[split:].to_numpy()),
        "split": ["validation"]*VALIDATION_SIZE + ["test"]*TEST_SIZE})
    return {"snapshot_id": snapshot["snapshot_id"], "returns": returns, "targets": targets,
        "initial_train_range": [str(returns.date.iloc[split-TRAIN_WINDOW].date()), str(returns.date.iloc[split-1].date())],
        "train_window": TRAIN_WINDOW,
        "limitation": "未来单日收益平方是高噪声事后代理，不是真实条件方差；缺少交易日历，下一日指下一条有效观测，可能跨停牌。"}


def history_at(returns, origin_date):
    """8B可用的输入窗口；没有访问目标表或未来数据。"""
    return returns.loc[returns.date <= pd.Timestamp(origin_date)].tail(TRAIN_WINDOW).copy()
