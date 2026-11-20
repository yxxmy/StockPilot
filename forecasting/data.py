"""同源前复权/不复权对照快照；失败不写成功文件。"""

from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from tools.stock_data import ak, FIELD_MAP, normalize_history, _parse_date


MIN_PRICES = 621
SOURCES = {"tencent": "AKShare / 腾讯证券", "eastmoney": "AKShare / 东方财富"}
LIMITATION = "供应商当前版本前复权价格；仅验证请求口径与同源调整差异，未独立审计全部分红因子，不是总收益或严格历史时点回测数据。"


class ForecastDataError(ValueError):
    """数据不足、口径未确认或快照损坏。"""


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _records(frame):
    frame = frame.copy()
    frame["date"] = frame["date"].dt.strftime("%Y-%m-%d")
    return frame.to_dict(orient="records")


def _normalise(frame, symbol, start, end):
    """输入统一六列，复用原行情校验；异常记录不悄悄剔除。"""
    if not isinstance(frame, pd.DataFrame) or set(FIELD_MAP.values()) - set(frame.columns):
        raise ForecastDataError("时序数据缺少必要行情字段")
    try:
        return normalize_history(frame.rename(columns={v:k for k,v in FIELD_MAP.items()}), symbol, start, end)
    except (ValueError, RuntimeError, TypeError) as exc:
        raise ForecastDataError("时序行情字段、日期或数值校验失败") from exc


def _validate(body):
    """重载时重新核验数据契约，不只依赖文件哈希。"""
    try:
        meta = body["metadata"]
        if body["schema_version"] != 1 or meta["source"] not in SOURCES or meta["source_name"] != SOURCES[meta["source"]]:
            raise ValueError()
        if not re.fullmatch(r"[0-9]{6}", meta["symbol"]):
            raise ValueError()
        if meta["adjustment"] != "qfq" or meta["volume_unit"] != "手" or meta["data_kind"] not in ("live", "test_sample"):
            raise ValueError()
        stamp = datetime.fromisoformat(meta["fetched_at"])
        if stamp.tzinfo is None or meta["limitation"] != LIMITATION:
            raise ValueError()
        start, end = meta["requested_range"]
        adjusted = _normalise(pd.DataFrame(body["prices"]), meta["symbol"], start, end)
        raw = _normalise(pd.DataFrame(body["unadjusted_prices"]), meta["symbol"], start, end)
        # 序列化文件必须已经规范化，拒绝重复、乱序和范围外行。
        if _records(adjusted) != body["prices"] or _records(raw) != body["unadjusted_prices"]:
            raise ValueError()
        if len(adjusted) < MIN_PRICES:
            raise ForecastDataError(f"至少需要{MIN_PRICES}条有效价格，当前{len(adjusted)}条")
        if not adjusted.date.equals(raw.date) or not adjusted.volume.equals(raw.volume):
            raise ForecastDataError("复权/不复权的交易日期或成交量不一致，不能拼接或填补")
        different = ~np.isclose(adjusted.close.to_numpy(), raw.close.to_numpy(), rtol=1e-8, atol=1e-8)
        if not different.any():
            raise ForecastDataError("未观察到前复权与不复权差异，无法确认调整口径，停止实验")
        actual = [adjusted.date.min().strftime("%Y-%m-%d"), adjusted.date.max().strftime("%Y-%m-%d")]
        if meta["actual_range"] != actual or meta["record_count"] != len(adjusted) or meta["adjusted_close_count"] != int(different.sum()):
            raise ValueError()
        return adjusted
    except ForecastDataError:
        raise
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ForecastDataError("快照结构或元数据不合法") from exc


def create_snapshot(adjusted, unadjusted, *, symbol, source, start_date, end_date, data_kind="test_sample"):
    """本地样本默认标为test_sample；正式拉取入口显式标记live。"""
    if source not in SOURCES or not isinstance(symbol, str) or not re.fullmatch(r"[0-9]{6}", symbol):
        raise ForecastDataError("股票代码或来源无效")
    start = _parse_date(start_date, "start_date").strftime("%Y-%m-%d")
    end = _parse_date(end_date, "end_date").strftime("%Y-%m-%d")
    a = _normalise(adjusted, symbol, start, end)
    b = _normalise(unadjusted, symbol, start, end)
    same_dates = a.date.equals(b.date)
    changed = int((~np.isclose(a.close, b.close, rtol=1e-8, atol=1e-8)).sum()) if same_dates else 0
    body = {"schema_version": 1, "metadata": {
        "symbol": symbol, "source": source, "source_name": SOURCES[source], "adjustment": "qfq",
        "data_kind": data_kind, "fetched_at": datetime.now(timezone.utc).isoformat(),
        "requested_range": [start, end], "actual_range": [a.date.min().strftime("%Y-%m-%d"), a.date.max().strftime("%Y-%m-%d")],
        "record_count": len(a), "adjusted_close_count": changed, "volume_unit": "手",
        "quality": {"adjusted_input_rows": len(adjusted), "raw_input_rows": len(unadjusted),
            "discarded_adjusted_rows": len(adjusted)-len(a), "discarded_raw_rows": len(unadjusted)-len(b),
            "missing_policy": "异常数值拒绝；排序、重复日保留最后一条、仅保留请求区间；不补交易日、不填零"},
        "versions": {name: version(name) for name in ["akshare", "pandas", "numpy"]}, "limitation": LIMITATION},
        "prices": _records(a), "unadjusted_prices": _records(b)}
    _validate(body)
    return {"snapshot_id": sha256(_canonical(body)).hexdigest(), "body": body}


def save_snapshot(snapshot, directory):
    """内容寻址、只增不改；单文件写入后失败时清理半成品。"""
    load_snapshot_value(snapshot)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (snapshot["snapshot_id"] + ".json")
    encoded = _canonical(snapshot)
    if path.exists():
        if path.read_bytes() != encoded:
            raise ForecastDataError("同ID文件内容不一致，拒绝覆盖")
        return path
    try:
        with path.open("xb") as file:
            file.write(encoded)
    except FileExistsError:
        raise ForecastDataError("快照已由另一进程创建，请重新加载核验") from None
    except OSError:
        path.unlink(missing_ok=True)
        raise ForecastDataError("快照保存失败") from None
    return path


def load_snapshot_value(snapshot):
    try:
        if set(snapshot) != {"snapshot_id", "body"} or sha256(_canonical(snapshot["body"])).hexdigest() != snapshot["snapshot_id"]:
            raise ForecastDataError("快照哈希不一致")
        return _validate(snapshot["body"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ForecastDataError(str(exc) if isinstance(exc, ForecastDataError) else "快照格式无效") from exc


def load_snapshot(path):
    try:
        snapshot = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ForecastDataError("快照无法读取或JSON损坏") from exc
    load_snapshot_value(snapshot)
    return snapshot


def fetch_snapshot(symbol, start_date, end_date, *, source="tencent"):
    """明确选定单源；两份数据均成功并校验后才可保存，不自动换源。"""
    if source not in SOURCES or not isinstance(symbol, str) or not re.fullmatch(r"[0-9]{6}", symbol):
        raise ForecastDataError("股票代码或来源无效")
    start, end = _parse_date(start_date, "start_date"), _parse_date(end_date, "end_date")
    if start > end:
        raise ForecastDataError("开始日不能晚于结束日")
    frames = []
    try:
        for adjust in ["qfq", ""]:
            if source == "tencent":
                frame = ak.stock_zh_a_hist_tx(symbol=symbol, start_date=start.strftime("%Y%m%d"), end_date=end.strftime("%Y%m%d"), adjust=adjust)
                frame = frame.copy()
                if frame["volume"].map(lambda x: isinstance(x, (bool, np.bool_))).any():
                    raise ForecastDataError("成交量含布尔值")
                frame["volume"] = pd.to_numeric(frame["volume"], errors="raise") / 100
            else:
                frame = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start.strftime("%Y%m%d"), end_date=end.strftime("%Y%m%d"), adjust=adjust)
                frame = frame.rename(columns=FIELD_MAP)
            frames.append(frame)
    except Exception as exc:
        raise ForecastDataError("所选行情源获取失败，未保存快照、未换源或填补数据") from exc
    return create_snapshot(*frames, symbol=symbol, source=source, start_date=start_date, end_date=end_date, data_kind="live")
