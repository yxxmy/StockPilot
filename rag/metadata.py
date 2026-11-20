"""仅接受导入者显式提供的文档身份；不从文件名或正文猜测股票归属。"""

from datetime import date
import re


FIELDS = ("symbol", "company_name", "report_year", "document_type", "disclosure_date")


def normalize_metadata(value=None):
    if value is None:
        value = {}
    if not isinstance(value, dict) or set(value) - set(FIELDS):
        raise ValueError("文档元数据字段无效")
    result = dict.fromkeys(FIELDS)
    result.update(value)
    symbol = result["symbol"]
    if symbol is not None and (not isinstance(symbol, str) or not re.fullmatch(r"[0-9]{6}", symbol)):
        raise ValueError("元数据股票代码须为六位数字字符串")
    year = result["report_year"]
    if year is not None and (type(year) is not int or not 1900 <= year <= 2100):
        raise ValueError("报告年份须为1900到2100的整数")
    for key in ("company_name", "document_type"):
        text = result[key]
        if text is not None:
            if not isinstance(text, str) or not text.strip() or len(text) > 100:
                raise ValueError("公司名称及文档类型须为1到100字符的非空文本")
            result[key] = text.strip()
    disclosed = result["disclosure_date"]
    if disclosed is not None:
        if not isinstance(disclosed, str) or not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", disclosed):
            raise ValueError("披露日期须为YYYY-MM-DD")
        date.fromisoformat(disclosed)
    return result


def metadata_status(value):
    count = sum(item is not None for item in value.values())
    return "unknown" if count == 0 else "complete" if count == len(FIELDS) else "partial"


def search_filters(symbol=None, report_year=None, document_type=None):
    metadata = normalize_metadata({"symbol": symbol, "report_year": report_year, "document_type": document_type})
    return {key: value for key, value in metadata.items() if value is not None}


def matches(metadata, filters):
    return all(metadata.get(key) == value for key, value in filters.items())
