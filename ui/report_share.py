"""制作可离线阅读的简报，网页和导出共用数值口径。"""

from html import escape
from math import isfinite

from tools.market_hotspots import PERIOD_RETURNS
from tools.report_industry import RANKING_SOURCE
from ui.report_events import event_date_range
from ui.service import public_text


def _number(value):
    return value if type(value) in (int, float) and isfinite(value) else None


def summary_metrics(market, risk):
    latest = market.get("latest")
    dates = market.get("actual_range")
    if not isinstance(latest, dict) or not isinstance(dates, list) or len(dates) != 2:
        return []
    close = _number(latest.get("close"))
    if close is None:
        return []

    change = _number(latest.get("pct_change"))
    average = _number(latest.get("ma20"))
    position = "20 日均线暂缺，待取得足够的历史数据。"
    if average is not None:
        difference = round(close, 2) - round(average, 2)
        comparison = "高于" if difference > 0 else "低于" if difference < 0 else "等于"
        position = f"收盘价{comparison} 20 日均线（{average:,.2f} 元）。"
    volume = _number(latest.get("volume"))
    volume_text = f"{volume:,.2f}".rstrip("0").rstrip(".") + " 手" if volume is not None else "暂无数据"
    ratio = _number(latest.get("volume_ratio"))
    volume_note = "暂缺可用的前 5 个交易日均量。" if ratio is None else f"是此前 5 个交易日平均成交量的 {ratio:.2f} 倍。"
    drawdown = _number(risk.get("max_drawdown_pct"))
    records = _number(market.get("record_count"))
    drawdown_text = "暂无数据"
    drawdown_note = "暂缺充分的收盘价或已计算的回撤数据。"
    if drawdown is not None and records is not None and records > 1:
        drawdown_text = f"{abs(drawdown):.2f}%"
        drawdown_note = "按每日收盘价计算，表示从此前高点到随后低点的最大回落。"
    return [
        ("收盘价", f"{close:,.2f} 元", None if change is None else f"{change:+.2f}% 较昨收", position),
        ("当日成交量", volume_text, None, volume_note),
        ("区间最大回撤", drawdown_text, None, drawdown_note),
    ]


def industry_details(context):
    metrics = {item["days"]: item for item in context["metrics"]}
    details = []
    for days in PERIOD_RETURNS:
        item = metrics.get(days)
        if item is None:
            details.append(f"近 {days} 个交易日暂无可用排名。")
        else:
            details.append(f"近 {days} 个交易日 {item['value']:+.2f}% · 第 {item['rank']} / {item['count']} 名"
                           f"（{item['start']:%Y-%m-%d} 收盘至 {context['end']:%Y-%m-%d} 收盘）")
    return details


INDUSTRY_METHOD = "按同花顺行业指数收盘价计算累计涨跌幅，仅对同一周期有完整数据的行业排名。"
INDUSTRY_MEMBERSHIP = "归属根据当前公开成分股名单核对。"
INDUSTRY_MANUAL = "行业由用户选作对照。"


def _safe(value):
    return escape(public_text(value))


def share_html(market, metrics, events, industry):
    title = _safe(f"{market.get('symbol', '')} · 个股研究简报")
    first, last = market["actual_range"]
    source = _safe(market.get("source") or "来源未记录")
    warning = f'<p class="note">{_safe(market["source_warning"])}</p>' if market.get("source_warning") else ""
    cards = "".join(
        f'<article><h3>{_safe(label)}</h3><strong>{_safe(value)}</strong>'
        f'<p>{_safe(delta + " · " if delta else "")}{_safe(note)}</p></article>'
        for label, value, delta, note in metrics
    )
    industry_body = "<p>本次导出未包含行业排名。</p>"
    if industry:
        membership = industry["membership"]
        ranks = "".join(f"<li>{_safe(line)}</li>" for line in industry_details(industry))
        attribution = (f'<a href="{_safe(membership["source"])}">成分股名单</a> · {INDUSTRY_MEMBERSHIP}'
                       if membership else INDUSTRY_MANUAL)
        industry_body = (f'<h3>{_safe(industry["name"])}</h3><ul>{ranks}</ul>'
                         f'<p>{INDUSTRY_METHOD} <a href="{RANKING_SOURCE}">同花顺行业</a> · {attribution}</p>')

    event_body = "".join(
        f'<article><p class="meta">{_safe(item["kind"])} · {item["stamp"]:%Y-%m-%d}</p>'
        f'<h3>{_safe(item["title"])}</h3>'
        + (f'<p>{_safe(item["excerpt"])}</p>' if item["kind"] == "财务披露" and item.get("excerpt") else "")
        + f'<p>{_safe(item["check"])}</p>'
        f'<p><a href="{_safe(item["url"])}">{_safe(item["source"])}</a></p></article>'
        for item in events
    ) or "<p>报告所选区间内暂无可追溯的事件条目。</p>"
    span = event_date_range(market)
    event_span = f'<p class="meta event-span">事件筛选区间 {span[0]} 至 {span[1]}</p>' if span else ""
    content = f"""
<header><p class="meta">StockPilot</p><h1>{title}</h1>
<p class="meta">行情区间 {_safe(first)} 至 {_safe(last)} · {source} · 不复权</p>{warning}</header>
<section><h2>行情摘要</h2><div class="cards">{cards}</div></section>
<section><h2>行业背景</h2>{industry_body}</section>
<section class="events"><h2>近期事件</h2>{event_span}{event_body}</section>"""
    return brief_document(title, content)


def brief_document(title, content):
    """为已转义的简报内容套用离线可读的 HTML 布局。"""
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  body {{ margin: 0; background: #f3f7f7; color: #142728; font: 16px/1.65 system-ui, "Microsoft YaHei", sans-serif; }}
  main {{ max-width: 58rem; margin: auto; padding: clamp(1rem, 5vw, 3.25rem); }}
  header {{ border-bottom: 3px solid #176c62; padding-bottom: 1.25rem; }}
  h1 {{ font-size: clamp(1.9rem, 5vw, 3rem); line-height: 1.2; margin: 0 0 .5rem; }}
  h2 {{ margin: 0 0 .75rem; font-size: 1.25rem; }}
  h3 {{ margin: 0 0 .35rem; font-size: 1rem; }}
  p {{ margin: .45rem 0 0; overflow-wrap: anywhere; }}
  section {{ margin-top: 1.8rem; }}
  .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(min(15rem, 100%), 1fr)); gap: .8rem; }}
  article {{ background: white; border: 1px solid #cbdad8; border-radius: .6rem; padding: 1rem; break-inside: avoid; }}
  .events article + article {{ margin-top: .65rem; }}
  .meta {{ color: #455b5a; }}
  article .meta {{ margin: 0 0 .4rem; }}
  .event-span {{ margin: 0 0 .75rem; }}
  .note {{ color: #614700; }}
  .lists {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(min(18rem, 100%), 1fr)); gap: .8rem; margin-top: .85rem; }}
  strong {{ color: #176c62; font-size: 1.55rem; font-variant-numeric: tabular-nums; }}
  ul {{ padding-left: 1.6rem; }}
  a {{ color: #11665d; }}
  @media print {{ body {{ background: white; }} main {{ padding: 0; }} }}
</style></head>
<body><main>{content}</main></body></html>"""
