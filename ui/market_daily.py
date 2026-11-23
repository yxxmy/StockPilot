"""市场日报的网页与 HTML 共用同一套摘要和排行。"""

import streamlit as st

from tools.market_daily import AMOUNT, BENCHMARKS, RETURN
from tools.report_industry import RANKING_SOURCE
from ui.report_share import _safe, brief_document
from ui.service import public_text


TENCENT_SOURCE = "https://gu.qq.com/sh000001/zs"
_BOARDS = {
    "industry": ("行业热点", "个行业", "同花顺行业指数日线"),
    "stocks": ("个股快照", "只股票", "腾讯证券沪深京 A 股快照"),
}


def report_title(report):
    day = report["day"]
    return f"{day:%Y-%m-%d} 市场日报" if day else "市场快报 · 收盘日期未核实"


def _index_cards(report):
    for name in BENCHMARKS.values():
        index = next((entry for entry in report["indices"] if entry["name"] == name), None)
        yield name, f"{index['close']:,.2f} 点" if index else "暂无数据", f"{index['change']:+.2f}%" if index else None


def _overview(report):
    indices = report["indices"]
    if not indices:
        return "暂无可核对的收盘指数。"
    up = sum(entry["change"] > 0 for entry in indices)
    down = sum(entry["change"] < 0 for entry in indices)
    return f"已核对的 {len(indices)} 个指数中，{up} 个上涨，{down} 个下跌，{len(indices) - up - down} 个持平。"


def _board_note(report, key):
    section = report[key]
    noun, source = _BOARDS[key][1:]
    if section is None:
        return f"本次未取得可用的{source}。"
    changes = (f"{section['up']:,} 个上涨，{section['down']:,} 个下跌，{section['flat']:,} 个持平。"
               if section["return_count"] else "")
    data_note = (f"{changes}涨跌幅覆盖 {section['return_count']:,} / {section['total']:,} {noun}，"
                 f"成交额覆盖 {section['amount_count']:,} / {section['total']:,} {noun}。")
    if key == "stocks":
        stamp = report["stock_time"].strftime("%Y-%m-%d %H:%M:%S")
        return f"{source}于 {stamp}（北京时间）抓取。行情所属交易日未核实。{data_note}"
    return (f"{source}。涨跌幅比较 {report['previous']} 与 {report['day']} 收盘，"
            f"成交额取 {report['day']} 当日。{data_note}")


def _leaderboards(section, key):
    name = "行业" if key == "industry" else "名称"
    no_gainers = "暂无可列出的上涨记录。" if section["return_count"] else "暂无可用涨跌幅。"
    for data, header, column, value_format, empty in (
        ("leaders", "涨幅前 5", RETURN, "+.2f", no_gainers),
        ("active", "成交额前 5", AMOUNT, ",.2f", "暂无可用成交额。"),
    ):
        rows = []
        for row in section[data]:
            label = f"{row[name]}（{row['代码']}）" if key == "stocks" else row[name]
            unit = "%" if column == RETURN else " 亿元"
            rows.append(f"{label}  {row[column]:{value_format}}{unit}")
        yield header, rows, empty


def _section_html(report, key):
    section = report[key]
    title = _BOARDS[key][0]
    boards = []
    if section is not None:
        for header, rows, empty in _leaderboards(section, key):
            listing = "<ol>" + "".join(f"<li>{_safe(row)}</li>" for row in rows) + "</ol>" if rows else f"<p>{empty}</p>"
            boards.append(f"<article><h3>{header}</h3>{listing}</article>")
    return (f'<section><h2>{title}</h2><p class="meta">{_safe(_board_note(report, key))}</p>'
            f'<div class="lists">{"".join(boards)}</div></section>')


def daily_html(report):
    title = _safe(report_title(report))
    cards = "".join(f'<article><h3>{name}</h3><strong>{value}</strong>'
                    f'<p>{delta or ""}</p></article>' for name, value, delta in _index_cards(report))
    stamp = report["generated"].strftime("%Y-%m-%d %H:%M:%S")
    content = f"""
<header><p class="meta">StockPilot</p><h1>{title}</h1><p>生成于 {stamp}（北京时间）</p></header>
<section><h2>市场概况</h2><p class="meta">{_safe(_overview(report))}</p><div class="lists">{cards}</div></section>
{_section_html(report, 'industry')}{_section_html(report, 'stocks')}
<section><h2>数据来源</h2><p><a href="{TENCENT_SOURCE}">腾讯证券</a> · <a href="{RANKING_SOURCE}">同花顺行业</a></p></section>"""
    return brief_document(title, content)


def show_daily(report):
    st.subheader(report_title(report))
    st.caption(f"生成于 {report['generated']:%Y-%m-%d %H:%M:%S}（北京时间）")
    st.subheader("市场概况")
    st.caption(_overview(report))
    for column, (name, value, change) in zip(st.columns(3), _index_cards(report)):
        column.metric(name, value, delta=change, help="与上一交易日收盘比较")
    for key, (title, _, _) in _BOARDS.items():
        st.subheader(title)
        st.caption(_board_note(report, key))
        if report[key] is None:
            continue
        for column, (header, rows, empty) in zip(st.columns(2), _leaderboards(report[key], key)):
            with column.container(border=True):
                st.markdown(f"**{header}**")
                st.text("\n".join(f"{index}. {public_text(row)}" for index, row in enumerate(rows, 1)) if rows else empty)

    links = st.columns(2)
    links[0].link_button("腾讯证券指数来源", TENCENT_SOURCE)
    links[1].link_button("同花顺行业来源", RANKING_SOURCE)
    date_suffix = report["day"].isoformat() if report["day"] else report["generated"].strftime("%Y-%m-%d_%H%M%S")
    st.download_button("下载市场日报 HTML", daily_html(report).encode("utf-8"),
                       file_name=f"StockPilot_市场日报_{date_suffix}.html", mime="text/html", on_click="ignore")
