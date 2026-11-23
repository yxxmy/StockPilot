"""会话级显式执行与已有服务的适配，不使用全局共享报告缓存。"""

from pathlib import Path
import os
import re
from time import monotonic
from unicodedata import normalize

from agents import workflow
from memory import ReportStore
from memory.store import DEFAULT_DB_PATH
from tools.stock_data import _parse_date
from tools.stock_names import load_a_share_names


def resolve_stock(session, query):
    """代码直接使用；名称必须唯一且完整匹配，避免把分析提交给相似股票。"""
    query = normalize("NFKC", query).strip()
    if re.fullmatch(r"[0-9]{6}", query):
        return query, ""
    if not query or query.isdecimal():
        raise ValueError("请输入六位股票代码或完整股票简称。")

    # 只在按名称提交时取名录，成功后在本会话复用一小时。
    cached = session.get("stock_name_directory")
    if cached is None or monotonic() - cached[0] >= 3600:
        directory = load_a_share_names()
        session["stock_name_directory"] = (monotonic(), directory)
    else:
        directory = cached[1]
    keyword = query.casefold()
    exact, partial = [], []
    for name, code in directory:
        normalized = normalize("NFKC", name).casefold()
        if normalized == keyword:
            exact.append((name, code))
        elif keyword in normalized:
            partial.append((name, code))
    if len(exact) == 1:
        name, code = exact[0]
        return code, name
    matches = exact or partial
    if matches:
        examples = "、".join(f"{name}（{code}）" for name, code in matches[:6])
        raise ValueError(f"找到 {len(matches)} 个候选。请填入对应的六位代码后再提交。候选包括：{examples}。")
    raise ValueError("在 A 股名录中未找到该简称，请核对股票简称或直接输入六位代码。")


def analyze_once(session, symbol, start, end, mode, research=False, question=""):
    symbol, question = symbol.strip(), question.strip()
    if not re.fullmatch(r"[0-9]{6}", symbol):
        raise ValueError("股票代码须为六位数字，例如600519。")
    first, last = _parse_date(start, "开始日期"), _parse_date(end, "结束日期")
    if first > last or (last - first).days > 3660:
        raise ValueError("日期范围无效，开始日须不晚于结束日，跨度最多3660天。")
    if mode not in ("本地规则", "DeepSeek") or len(question) > 500:
        raise ValueError("请选择支持的模式，正文问题不超过500字符。")
    start_text, end_text = first.strftime("%Y-%m-%d"), last.strftime("%Y-%m-%d")
    key = (symbol, start_text, end_text, mode, research, question)
    # 同一浏览器会话内相同参数直接复用成功结果，连点/重跑不再次计费或落库。
    completed = session.setdefault("completed", {})
    if key in completed:
        session["result"] = completed[key]
        return False
    if session.get("running"):
        raise ValueError("已有分析正在执行，请等待完成。")
    session["running"] = True
    try:
        options = {"rag_question": question} if question else {}
        result = workflow.run_analysis(symbol, start_text, end_text,
            local=mode == "本地规则", include_research=research, warmup=True, **options)
        completed[key] = result
        session["result"] = result
        return True
    finally:
        session["running"] = False


def list_reports(symbol=None):
    path = Path(os.environ.get("STOCKPILOT_DB_PATH", DEFAULT_DB_PATH)).resolve()
    return ReportStore(path, read_only=True).list_reports(symbol, limit=50) if path.is_file() else []


def read_report(report_id):
    path = Path(os.environ.get("STOCKPILOT_DB_PATH", DEFAULT_DB_PATH)).resolve()
    return ReportStore(path, read_only=True).get_report(report_id) if path.is_file() else None


def public_text(value):
    """展示外部文本前删除当前认证值；不把配置或原始异常传给前端。"""
    value = str(value)
    for key, secret in os.environ.items():
        if key.upper().endswith(("_API_KEY", "_TOKEN", "_SECRET")) and len(secret) >= 8:
            value = value.replace(secret, "[认证信息已隐藏]")
    return re.sub(r"\bsk-[A-Za-z0-9_-]{12,}\b", "[认证信息已隐藏]", value)
