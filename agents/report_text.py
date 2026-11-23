"""把工作流已经算好的行情和风险指标写成中文，不请求数据或模型。"""

import json
import re

EVIDENCE_DIVIDER = "\n\n附录 · 原始事实数据：\n"

# 精确匹配旧版自动生成的整段占位文字，避免删掉含引用或数字的实际解读。
_OLD_RESEARCH_STATUS = re.compile(
    r"Financial Agent：\n财务数据状态：[\w-]+；\d+条记录，未生成模型解读。"
    r"|News Agent：\n新闻状态：[\w-]+；公告状态：[\w-]+；未生成模型解读，空结果不代表无事件。"
)


def market_and_risk_text(market, risk):
    latest = market["latest"]
    start, end = market["actual_range"]
    close = latest["close"]
    change = latest["pct_change"]
    if change is None:
        daily = "缺少上一交易日收盘价，无法计算当日涨跌幅"
    elif change == 0:
        daily = "与上一交易日持平"
    else:
        direction = "上涨" if change > 0 else "下跌"
        daily = f"较上一交易日{direction} {abs(change):.2f}%"
    volume = f"{latest['volume']:,.2f}".rstrip("0").rstrip(".")
    overview = (f"{end} 收盘 {close:,.2f} 元，{daily}，成交 {volume} 手。\n"
                f"本次取得 {start} 至 {end} 的 {market['record_count']} 个交易日记录。")

    comparisons = []
    missing = []
    for days in (5, 10, 20):
        average = latest[f"ma{days}"]
        if average is None:
            missing.append(f"{days} 日均价")
            continue
        # 比较精度与展示精度一致，避免两者都显示同一价格却写成“高于”。
        rounded_average = round(average, 2)
        rounded_close = round(close, 2)
        relation = "高于" if rounded_close > rounded_average else "低于" if rounded_close < rounded_average else "等于"
        comparisons.append(f"{relation}近 {days} 个交易日均价（{average:,.2f} 元）")
    prices = []
    if comparisons:
        prices.append("收盘价" + "，".join(comparisons) + "。")
    if missing:
        prices.append("历史数据不足，暂无法计算" + "、".join(missing) + "。")
    ratio = latest["volume_ratio"]
    if ratio is not None:
        prices.append(f"当日成交量是此前 5 个交易日平均成交量的 {ratio:.2f} 倍。")

    if market["record_count"] == 1:
        risk_lines = ["区间仅有一个交易日，无法比较区间涨跌或观察回撤。"]
    else:
        risk_lines = [f"从区间首日收盘至末日收盘，价格变动 {risk['sample_return_pct']:+.2f}%。"
                     f"按每日收盘价计算，从此前高点回落的最大幅度为 {abs(risk['max_drawdown_pct']):.2f}%。"]
    volatility = risk["annualized_volatility_pct"]
    if volatility is None:
        risk_lines.append("有效日涨跌幅少于两条，暂无法估计年化波动率。")
    else:
        risk_lines.append(f"日涨跌幅折算的年化波动率为 {volatility:.2f}%，用于衡量这段时间价格波动的大小。")
    risk_lines.extend(flag + "。" for flag in risk["flags"] if not flag.startswith("有效涨跌幅"))
    provider = market.get("source")
    source = f"行情来自 {provider}，股价采用不复权口径。" if provider else "股价采用不复权口径，报告未记录行情来源。"
    if market.get("source_warning"):
        source += "\n" + market["source_warning"]
    return "\n\n".join(["收盘表现\n" + overview, "价格与成交量\n" + "".join(prices),
                        "区间表现与风险\n" + "".join(risk_lines), "数据来源\n" + source])


def report_for_display(value):
    """仅在阅读时整理存量报告，缓存结果和 SQLite 原文走同一条路径。"""
    header, legacy, rest = value.partition("\n\n事实数据：\n")
    if not (legacy and header.startswith("StockPilot ") and rest.lstrip().startswith("{")):
        body, divider, evidence = value.rpartition(EVIDENCE_DIVIDER)
        body = body if divider else value
        body = "\n\n".join(p for p in body.split("\n\n") if not _OLD_RESEARCH_STATUS.fullmatch(p))
        return body, evidence if divider else None

    try:
        rest = rest.lstrip()
        facts, end = json.JSONDecoder().raw_decode(rest)
        market = facts["market"]
        overview = market_and_risk_text(market, facts["risk"])
    except (ValueError, TypeError, KeyError, IndexError, AttributeError):
        return "这份旧报告的数据不完整，无法可靠地生成中文摘要。可展开查看保存的原文。", value

    body = [header, overview]
    # 保留旧版模型解读、研究材料、原文引用与限制说明；跳过只指向代码字段的占位段落。
    placeholders = {
        "Market Agent：\n本地规则：行情和指标已计算，数值见 market_facts。",
        "Risk Agent：\n本地规则：历史风险统计已计算，风险提示见 risk_facts.flags。",
        "Report Agent：\n本地规则报告：仅展示已计算事实，不生成模型解读。",
    }
    labels = {"Market Agent：": "行情解读", "Risk Agent：": "风险解读", "Financial Agent：": "财务研究",
              "News Agent：": "新闻与公告", "Report Agent：": "综合说明"}
    for paragraph in rest[end:].strip().split("\n\n"):
        if paragraph and paragraph not in placeholders and not _OLD_RESEARCH_STATUS.fullmatch(paragraph):
            label, newline, text = paragraph.partition("\n")
            body.append(labels.get(label, label) + newline + text)
    return "\n\n".join(body), rest[:end]
