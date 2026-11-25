"""核对巨潮资讯公告附件并交给现有 PDF 索引导入。"""

from datetime import datetime
from hashlib import file_digest
from pathlib import Path
import re
from tempfile import NamedTemporaryFile
from urllib.parse import parse_qs, urlsplit
from zoneinfo import ZoneInfo

import requests

from rag.errors import RagError
from rag.metadata import verified_pdf_url
from tools.research_data import _text


DOCUMENTS = Path(__file__).resolve().parents[1] / "data" / "documents"
_QUERY_URL = "https://www.cninfo.com.cn/new/hisAnnouncement/query"
_PDF_HOST = "https://static.cninfo.com.cn/"
_MAX_PDF_BYTES = 100 * 1024 * 1024


def _identity(symbol, item):
    url = urlsplit(item["url"])
    params = parse_qs(url.query)
    notice_id = params.get("announcementId", [""])
    org_id = params.get("orgId", [""])
    if (not re.fullmatch(r"[0-9]{6}", symbol) or url.scheme != "https"
            or url.netloc not in {"www.cninfo.com.cn", "cninfo.com.cn"}
            or url.path != "/new/disclosure/detail" or params.get("stockCode") != [symbol]
            or len(notice_id) != 1 or not re.fullmatch(r"[0-9]{1,20}", notice_id[0])
            or len(org_id) != 1 or not re.fullmatch(r"[A-Za-z0-9_-]{2,60}", org_id[0])):
        raise RagError("公告链接缺少可核对的股票代码或公告编号。")
    return notice_id[0], org_id[0]


def _official_attachment(symbol, item, notice_id, org_id):
    day = item["date"].isoformat()
    request = {"pageNum": 1, "pageSize": 30, "column": "szse", "tabName": "fulltext",
               "stock": f"{symbol},{org_id}", "seDate": f"{day}~{day}"}
    try:
        for page in range(1, 11):
            request["pageNum"] = page
            response = requests.post(_QUERY_URL, data=request, timeout=(5, 12), allow_redirects=False)
            response.raise_for_status()
            data = response.json()
            rows = data["announcements"] or []
            for row in rows:
                if (str(row.get("announcementId")), row.get("secCode"), row.get("orgId")) != (notice_id, symbol, org_id):
                    continue
                published = datetime.fromtimestamp(row["announcementTime"] / 1000, ZoneInfo("Asia/Shanghai")).date()
                path = row.get("adjunctUrl") or ""
                if published != item["date"] or _text(row.get("announcementTitle")) != item["title"]:
                    raise RagError("公告披露日期或标题已变化，请重新查找后再导入。")
                url = verified_pdf_url(_PDF_HOST + path.lstrip("/"))
                if not url:
                    raise RagError("该公告未提供可下载的官方 PDF 附件。")
                return url, _text(row.get("secName"), limit=100) or None
            if len(rows) < request["pageSize"]:
                break
    except (requests.RequestException, ValueError, TypeError, KeyError, AttributeError, OverflowError, OSError):
        raise RagError("暂时无法核对巨潮资讯公告附件，请稍后重试。") from None
    raise RagError("未能在巨潮资讯核对这条公告的 PDF 附件。")


def _download(url, path):
    try:
        with requests.get(url, timeout=(5, 20), stream=True, allow_redirects=False) as response:
            response.raise_for_status()
            if response.status_code != 200:
                raise RagError("公告 PDF 下载返回了意外的响应。")
            with path.open("wb") as output:
                length = 0
                for block in response.iter_content(chunk_size=64 * 1024):
                    length += len(block)
                    if length > _MAX_PDF_BYTES:
                        raise RagError("公告 PDF 超过 100 MB，请手动下载后导入。")
                    output.write(block)
            with path.open("rb") as source:
                if b"%PDF-" not in source.read(1024):
                    raise RagError("官方附件返回的内容无法识别为 PDF。")
    except (requests.RequestException, OSError):
        raise RagError("公告 PDF 下载或保存失败，请稍后重试。") from None


def _existing_document(index, path, metadata):
    if not path.is_file():
        return None
    with path.open("rb") as file:
        document_id = file_digest(file, "sha256").hexdigest()
    existing = next((chunk for chunk in index.chunks if chunk["document_id"] == document_id), None)
    if existing:
        saved = existing["metadata"]
        if saved.get("symbol") != metadata["symbol"] or any(
            saved.get(key) is not None and metadata[key] is not None and saved[key] != metadata[key]
            for key in ("report_year", "disclosure_date")
        ):
            raise RagError("这份 PDF 已有不同的股票、报告期或披露日期，请先核对原索引。")
    return existing


def import_announcement(symbol, item, report_year, *, documents_dir=DOCUMENTS, index_dir=None):
    from rag.index import DocumentIndex

    notice_id, org_id = _identity(symbol, item)
    metadata = {"symbol": symbol, "report_year": report_year, "disclosure_date": item["date"].isoformat(),
                "document_type": "annual_report" if re.search(r"(?<!半)年度报告$", item["title"]) else "announcement"}
    index = DocumentIndex(index_dir)
    directory = Path(documents_dir)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{symbol}_cninfo_{notice_id}.pdf"
    existing = _existing_document(index, target, metadata)
    if existing and (url := existing.get("official_pdf_url")):
        return {"document_id": existing["document_id"], "duplicate": True, "chunks_added": 0, "pdf_url": url}

    url, metadata["company_name"] = _official_attachment(symbol, item, notice_id, org_id)
    # 在最终目录中创建临时文件，原子替换后仍继承该目录的访问权限。
    with NamedTemporaryFile(prefix="cninfo-", suffix=".pdf", dir=directory, delete=False) as file:
        staged = Path(file.name)
    try:
        _download(url, staged)
        existing = _existing_document(index, staged, metadata)
        staged.replace(target)
    finally:
        staged.unlink(missing_ok=True)
    stored_metadata = existing["metadata"] if existing else metadata
    result = index.import_pdf(target, metadata=stored_metadata, official_pdf_url=url)
    return {**result, "pdf_url": result["official_pdf_url"]}
