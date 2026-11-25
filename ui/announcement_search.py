"""网页上的财报和公告查找入口。"""

from collections import Counter
from hashlib import sha256
import re

import streamlit as st

from agents.announcement_answer import extract_announcement_answer
from agents.deepseek import AgentError, DeepSeekClient
from agents.retrieval import retrieve_document_facts
from rag.errors import RagError
from tools.announcement_import import import_announcement
from tools.announcement_search import find_announcements, report_year_hint
from tools.stock_data import StockDataError
from ui import service
from ui.report_events import show_official_pdf_link


def _import(result, item):
    try:
        with st.spinner("正在下载并导入官方 PDF…"):
            imported = import_announcement(result["symbol"], item, result["year"])
        item["import"] = imported
        if not imported["duplicate"] or imported.get("source_added"):
            completed = st.session_state.get("completed", {})
            for key in list(completed):
                if key[0] == result["symbol"] and key[-1]:
                    del completed[key]
    except (RagError, ValueError) as exc:
        item["import"] = {"message": service.public_text(exc)}
    except OSError:
        item["import"] = {"message": "本机文件暂时无法保存，请检查项目文件夹的写入权限。"}


def _query_announcement(result, item):
    if "question" in result:
        result.pop("facts", None)
        result.pop("selected", None)
    if "document_id" not in item.get("import", {}):
        _import(result, item)
    if "question" not in result or item["import"].get("message"):
        return
    with st.spinner("正在检索这份公告的原文…"):
        result["facts"] = retrieve_document_facts(
            result["question"], symbol=result["symbol"], document_id=item["import"]["document_id"],
        )
    result["selected"] = item["title"]


def _find_official(result, question):
    result.pop("official_message", None)
    result.pop("records", None)
    result.pop("facts", None)
    result.pop("selected", None)
    try:
        with st.spinner("正在巨潮资讯查找公告…"):
            result.update(find_announcements(result["symbol"], question))
    except ValueError as exc:
        result["official_message"] = service.public_text(exc)
    except StockDataError:
        result["official_message"] = "暂时无法从巨潮资讯读取公告，请稍后重试。"
    if len(result.get("records", [])) == 1:
        _query_announcement(result, result["records"][0])


def search_announcements(stock_query, question, *, query_original=False):
    """独立查询先检索已导入的 PDF，无匹配时再找官方公告。"""
    question = question.strip()
    result = {"question": question} if query_original else {}
    st.session_state.pop("announcement_local_document", None)
    st.session_state["announcement_search"] = result
    try:
        if not question:
            raise ValueError("请先填写财报或公告正文问题。")
        symbol, name = service.resolve_stock(st.session_state, stock_query or "")
        result.update(symbol=symbol, name=name)
        if query_original:
            with st.spinner("正在检索已导入的 PDF…"):
                facts = retrieve_document_facts(
                    question, symbol=symbol, report_year=report_year_hint(question), choose_document=True,
                )
            if facts["status"] == "choose_document":
                result["local_documents"] = facts["documents"]
                return
            if facts["status"] not in {"empty", "no_match"}:
                result["facts"] = facts
                return
        _find_official(result, question)
    except ValueError as exc:
        result["message"] = service.public_text(exc)


def _show_announcement_answer(facts, records):
    st.markdown("**简短答案**")
    answers = extract_announcement_answer(facts["question"], records)
    for label, value, ref_id in answers:
        st.text(f"{label}：{value} [{ref_id}]")
    if not answers:
        st.info("已找到相关段落。当前本地提取无法确认简短答案，可以查看下方原文。")
    cited = {ref_id for _, _, ref_id in answers}
    documents = {hit.get("document_id") for hit in records}
    if len(documents) != 1 or None in documents:
        return cited

    if "deepseek_answer" not in facts:
        identity = next(iter(documents))
        question_key = sha256(facts["question"].encode()).hexdigest()
        if st.button("用 DeepSeek 解读这份原文", key=f"announcement_deepseek_{identity}_{question_key}"):
            facts.pop("deepseek_error", None)
            payload = {"rag": {"question": service.public_text(facts["question"]), "status": "ok", "records": [
                {"ref_id": hit["ref_id"], "page": hit["page"], "text": service.public_text(hit["text"])}
                for hit in records
            ]}}
            try:
                with st.spinner("正在解读所选公告…"):
                    facts["deepseek_answer"] = DeepSeekClient().describe("公告原文助手", payload)
            except AgentError as exc:
                facts["deepseek_error"] = ("请先在本机设置 DEEPSEEK_API_KEY，再重启 StockPilot。"
                                           if exc.category == "configuration" and "DEEPSEEK_API_KEY" in str(exc)
                                           else service.public_text(exc))
        if "deepseek_answer" not in facts:
            st.caption("点击后会把本次问题和引用片段发送给 DeepSeek，可能产生费用。")
    if facts.get("deepseek_error"):
        st.warning(facts["deepseek_error"])
    if answer := facts.get("deepseek_answer"):
        st.markdown("**DeepSeek 解读**")
        st.text(service.public_text(answer))
        cited.update(re.findall(r"\[(R[0-9]+)\]", answer))
    return cited


def show_rag_facts(facts, *, standalone=False):
    st.text(service.public_text("问题：" + facts["question"]))
    if not standalone:
        st.text(service.public_text("检索状态：" + facts["status"] + " · " + facts.get("message", "")))
    if facts["status"] != "ok":
        st.warning(facts["message"] if standalone else "没有取得可用原文，不编造财报或公告内容。")
    records = facts.get("records", [])
    if standalone and facts["status"] == "ok" and records:
        cited = _show_announcement_answer(facts, records)
        records = sorted(records, key=lambda hit: hit["ref_id"] not in cited)
        st.markdown("**引用原文**")
    for hit in records:
        label = f"[{hit['ref_id']}] {service.public_text(hit['document_name'])} · 物理第{hit['page']}页"
        if not standalone:
            label += f" · {hit['score']:.3f}"
        with st.expander(label, expanded=True):
            st.text(service.public_text(hit["text"]))
            show_official_pdf_link(hit)
            if not standalone:
                st.caption("文档标识：" + hit["document_id"] + " · 片段：" + hit["chunk_id"])
    if not standalone:
        st.text(service.public_text(facts.get("limitations", "")))


def _choose_local_document(result):
    documents = {item["document_id"]: item for item in result["local_documents"]}
    labels = {}
    for identity, item in documents.items():
        info = item["metadata"]
        parts = [item["document_name"]]
        if info["report_year"]:
            parts.append(f"报告年份 {info['report_year']}")
        if info["disclosure_date"]:
            parts.append(f"披露于 {info['disclosure_date']}")
        labels[identity] = service.public_text(" · ".join(parts))
    counts = Counter(labels.values())
    for identity, label in labels.items():
        if counts[label] > 1:
            labels[identity] = f"{label} · 文件编号 {identity[:8]}"
    selected = st.selectbox(
        "选择一份已导入的 PDF", list(documents), index=None, format_func=labels.get,
        placeholder="按文件名和披露日期选择", key="announcement_local_document",
    )
    local, official = st.columns(2)
    if local.button("查询所选 PDF", disabled=selected is None):
        item = documents[selected]
        result.pop("records", None)
        result.pop("official_message", None)
        with st.spinner("正在检索所选 PDF…"):
            facts = retrieve_document_facts(result["question"], symbol=result["symbol"],
                report_year=item["metadata"]["report_year"], document_id=selected)
        if facts["status"] in {"empty", "no_match"}:
            facts["message"] = "所选 PDF 没有找到相关片段。可以换一份文件，或去巨潮资讯查找。"
        result.update(facts=facts, selected=labels[selected])
    if official.button("去巨潮资讯查找"):
        _find_official(result, result["question"])


def _show_official_candidates(result, stock, standalone):
    year = f" · 标题年份 {result['year']}" if result["year"] else ""
    st.text(service.public_text(f"{stock} · 巨潮资讯 · {result['keyword']}{year} · 披露日期 {result['start']} 至 {result['end']}"))
    if not result["records"]:
        st.info("没有找到匹配的公告。可以调整问题中的年份或标题关键词。")
        return
    if len(result["records"]) > 1:
        st.caption("找到多份公告。核对标题和披露日期后，选择要查询的文件。" if standalone
                   else "找到多份公告。核对标题和披露日期后，选择要导入的文件。")
    for index, item in enumerate(result["records"], 1):
        st.text(service.public_text(f"{index}. {item['title']} · {item['date']}"))
        detail, action = st.columns(2)
        detail.link_button(f"查看公告 {index}", item["url"])
        status = item.get("import", {})
        if not status or status.get("message") or (standalone and len(result["records"]) > 1):
            label = f"查询第 {index} 条原文" if standalone else f"下载并导入第 {index} 条"
            if action.button(label, key=f"import_announcement_{index}"):
                _query_announcement(result, item)
                status = item["import"]
        if status.get("message"):
            st.warning(status["message"])
        elif status:
            if not standalone:
                message = "已在检索库中，可以直接提问。" if status["duplicate"] else "已下载并导入，可以开始检索原文。"
                st.success(message)
            if status.get("pdf_url"):
                st.link_button(f"打开官方 PDF {index}", status["pdf_url"])


def show_announcement_search():
    result = st.session_state.get("announcement_search")
    if not result:
        return
    standalone = "question" in result
    st.subheader("公告原文查询" if standalone else "巨潮资讯公告")
    if result.get("message"):
        st.warning(result["message"])
        return
    symbol = result["symbol"]
    stock = f"{result['name']}（{symbol}）" if result["name"] else symbol
    if result.get("local_documents"):
        st.caption(service.public_text(f"{stock} · 已导入的 PDF"))
        _choose_local_document(result)
    if result.get("official_message"):
        st.warning(result["official_message"])
    elif "records" in result:
        _show_official_candidates(result, stock, standalone)
    if "facts" in result:
        if "selected" in result:
            st.caption(service.public_text(result["selected"]))
        elif not result.get("local_documents"):
            st.caption(service.public_text(f"{stock} · 已导入的 PDF"))
        show_rag_facts(result["facts"], standalone=True)
