"""从一份公告的引用片段提取格式明确的字段。"""

from datetime import date
from decimal import Decimal
import re
from unicodedata import normalize


_DATE = r"(20[0-9]{2})\s*(?:年|[-/.])\s*([0-9]{1,2})\s*(?:月|[-/.])\s*([0-9]{1,2})(?![0-9])\s*日?"
_REGISTRATION_DATE = re.compile(
    r"股权登记日\s*(?:(?:为|是|定于)\s*)?:?\s*\(?\s*" + _DATE
)
_REGISTRATION_TABLE = re.compile(
    r"股权登记日\s*(?:最后交易日\s*)?(?:除权(?:\s*\(息\)|除息)?|除息)日\s*"
    r"(?:现金)?红利发放日\s*A\s*股\s*" + _DATE
)
_GROSS_DIVIDEND = re.compile(
    r"每\s*(?P<shares>10)?\s*股\s*(?:"
    r"(?:派发|派送|分配|派)?\s*现金(?:红利|股利|股息)?|派发|派送|派)"
    r"\s*(?:人民币\s*)?(?P<amount>[0-9]{1,12}(?:\.[0-9]{1,10})?)\s*元\s*\(?\s*(?:含税|税前)(?=[)\s,。;]|$)"
)


def _registration_dates(text):
    for pattern in (_REGISTRATION_DATE, _REGISTRATION_TABLE):
        for match in pattern.finditer(text):
            try:
                yield date(*map(int, match.groups())).isoformat()
            except ValueError:
                continue


def _gross_dividends(text):
    for match in _GROSS_DIVIDEND.finditer(text):
        yield Decimal(match["amount"]) / int(match["shares"] or 1)


def _unique_value(records, extract):
    sources = {}
    for hit in sorted(records, key=lambda hit: hit["page"]):
        text = normalize("NFKC", hit["text"])
        for value in extract(text):
            sources.setdefault(value, hit["ref_id"])
    if len(sources) == 1:
        return next(iter(sources.items()))
    return None


def extract_announcement_answer(question, records):
    """只返回有唯一原文依据的答案，每个字段保留片段引用。"""
    documents = {hit.get("document_id") for hit in records}
    if len(documents) != 1 or None in documents:
        return []
    question = normalize("NFKC", question)
    if re.search(r"[BH]\s*股", question, re.I):
        return []

    answers = []
    if "股权登记日" in question and not re.search(r"是什么(?:意思|概念)|什么是股权登记日|含义|定义", question):
        if match := _unique_value(records, _registration_dates):
            value, ref_id = match
            answers.append(("股权登记日", value, ref_id))
    if re.search(r"分红|红利|股息|股利|每\s*(?:10\s*)?股.*(?:派|现金)", question) and not re.search(
        r"税后|扣税|实发|实得|净额|到账多少|合计|总额|共计|股息率|同比|历史分红", question
    ):
        # 股权登记日问题常提到“分红”，只有同时问金额时才显示分红金额。
        asks_date = re.search(r"股权登记日|哪[一]?[天日]|什么时候|何时|几号|日期|发放日", question)
        if not asks_date or re.search(r"每\s*(?:10\s*)?股|多少钱|多少元|金额|派多少", question):
            if match := _unique_value(records, _gross_dividends):
                amount, ref_id = match
                per_ten = bool(re.search(r"每\s*10\s*股", question))
                label = "每 10 股现金分红" if per_ten else "每股现金分红"
                value = format((amount * (10 if per_ten else 1)).normalize(), "f")
                answers.append((label, f"{value} 元（含税）", ref_id))
    return answers
