from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest

from rag import index as rag_index
from rag.errors import RagError
from tools import announcement_import as service


NOTICE = {"title": "2023年年度权益分派实施公告", "date": date(2024, 6, 12),
          "url": "https://www.cninfo.com.cn/new/disclosure/detail?stockCode=600519&announcementId=1220324741&orgId=gssh0600519"}
PDF_URL = "https://static.cninfo.com.cn/finalpage/2024-06-12/1220324741.PDF"
SAMPLE = Path(__file__).parent / "fixtures" / "rag_test_document.pdf"


class FixedEmbedder:
    model_id = "TEST-ONLY"

    def encode(self, texts, *, query=False):
        return np.array([[1.0, 0.5] for _ in texts], dtype="float32")


def provider(monkeypatch, payload, *, attachment="finalpage/2024-06-12/1220324741.PDF"):
    metadata = Mock()
    metadata.json.return_value = {"announcements": [{
        "announcementId": "1220324741", "secCode": "600519", "orgId": "gssh0600519", "secName": "贵州茅台",
        "announcementTitle": NOTICE["title"], "adjunctUrl": attachment,
        "announcementTime": int(datetime(2024, 6, 12, tzinfo=timezone.utc).timestamp() * 1000),
    }]}
    request = Mock(return_value=metadata)
    monkeypatch.setattr(service.requests, "post", request)
    content = MagicMock(status_code=200)
    content.__enter__.return_value = content
    content.iter_content.return_value = [payload]
    download = Mock(return_value=content)
    monkeypatch.setattr(service.requests, "get", download)
    return request, download


def test_verified_pdf_import_is_reusable_and_stock_scoped(monkeypatch, tmp_path):
    index_type = rag_index.DocumentIndex
    monkeypatch.setattr(rag_index, "DocumentIndex", lambda path: index_type(path, embedder=FixedEmbedder()))
    request, download = provider(monkeypatch, SAMPLE.read_bytes())
    files, index = tmp_path / "files", tmp_path / "index"
    added = service.import_announcement("600519", NOTICE, 2023, documents_dir=files, index_dir=index)
    assert added["chunks_added"] > 0 and added["pdf_url"] == PDF_URL
    download.assert_called_once_with(PDF_URL, timeout=(5, 20), stream=True, allow_redirects=False)
    assert request.call_args.kwargs["data"]["stock"] == "600519,gssh0600519"
    assert len(list(files.glob("*.pdf"))) == 1
    stored = index_type(index, embedder=FixedEmbedder())
    hit = stored.search_documents("现金流", symbol="600519")[0]
    assert hit["metadata"]["report_year"] == 2023 and hit["official_pdf_url"] == PDF_URL
    assert stored.search_documents("现金流", symbol="000001") == []

    duplicate = service.import_announcement("600519", NOTICE, 2023, documents_dir=files, index_dir=index)
    assert duplicate == {"document_id": hit["document_id"], "duplicate": True, "chunks_added": 0, "pdf_url": PDF_URL}
    request.assert_called_once()
    download.assert_called_once()


def test_previously_imported_pdf_gets_link_after_matching_official_bytes(monkeypatch, tmp_path):
    index_type = rag_index.DocumentIndex
    embedder = FixedEmbedder()
    index = tmp_path / "index"
    manual = index_type(index, embedder=embedder)
    metadata = {"symbol": "600519", "company_name": "贵州茅台酒股份有限公司", "report_year": 2023}
    manual.import_pdf(SAMPLE, metadata=metadata)
    embedder.encode = Mock(side_effect=AssertionError("已有向量应当复用"))
    monkeypatch.setattr(rag_index, "DocumentIndex", lambda path: index_type(path, embedder=embedder))
    provider(monkeypatch, SAMPLE.read_bytes())

    duplicate = service.import_announcement("600519", NOTICE, 2023, documents_dir=tmp_path / "files", index_dir=index)

    assert duplicate["duplicate"] and duplicate["source_added"] and duplicate["pdf_url"] == PDF_URL
    stored = index_type(index, embedder=embedder)
    assert stored.index.ntotal == manual.index.ntotal
    assert all(chunk["official_pdf_url"] == PDF_URL and chunk["document_name"] == SAMPLE.name for chunk in stored.chunks)
    assert stored.chunks[0]["metadata"] == manual.chunks[0]["metadata"]
    embedder.encode.assert_not_called()


@pytest.mark.parametrize("attachment,payload,message", [
    ("https://example.invalid/fake.pdf", b"", "官方 PDF"),
    ("finalpage/2024-06-12/1220324741.PDF", b"<html>not a PDF</html>", "识别为 PDF"),
])
def test_invalid_attachment_does_not_write_pdf_or_index(monkeypatch, tmp_path, attachment, payload, message):
    provider(monkeypatch, payload, attachment=attachment)
    files, index = tmp_path / "files", tmp_path / "index"
    with pytest.raises(RagError, match=message):
        service.import_announcement("600519", NOTICE, 2023, documents_dir=files, index_dir=index)
    assert not list(files.iterdir()) and not (index / "CURRENT.json").exists()
