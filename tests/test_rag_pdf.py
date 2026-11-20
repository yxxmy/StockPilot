from pathlib import Path
from unittest.mock import Mock

import pytest

pytest.importorskip("pypdf")
from pypdf import PdfReader, PdfWriter

from rag.errors import PdfTextError
from rag.pdf import chunk_text, extract_pdf


SAMPLE = Path(__file__).parent / "fixtures" / "rag_test_document.pdf"


@pytest.mark.parametrize("size,overlap,length", [(10, 2, 10), (10, 2, 11), (10, 2, 18), (10, 9, 21), (1, 0, 8), (10, 0, 20)])
def test_chunk_progress_no_duplicate_tail_and_complete_coverage(size, overlap, length):
    text = "甲" * length
    chunks = list(chunk_text(text, size, overlap))
    assert chunks[0]["start"] == 0 and chunks[-1]["end"] == length
    assert len(chunks) <= length
    assert len({(c["start"], c["end"]) for c in chunks}) == len(chunks)
    assert all(c["end"] < length for c in chunks[:-1])
    for previous, current in zip(chunks, chunks[1:]):
        assert current["start"] == previous["end"] - overlap
        assert current["start"] > previous["start"]
    assert all(c["text"] == text[c["start"]:c["end"]] for c in chunks)


def test_empty_chunk_input():
    assert list(chunk_text("")) == []


@pytest.mark.parametrize("size,overlap", [(0, 0), (10, -1), (10, 10), (10, 11), (True, 0), (10, False), (1.5, 0)])
def test_invalid_chunk_options(size, overlap):
    with pytest.raises(ValueError):
        list(chunk_text("测试", size, overlap))


def test_actual_parseable_pdf_provenance_and_no_fictional_results():
    result = extract_pdf(SAMPLE, 100, 20)
    assert len(result["document_id"]) == 64
    assert {c["page"] for c in result["chunks"]} == {1, 2}
    assert all(c["document_name"] == SAMPLE.name for c in result["chunks"])
    assert "测试" in "".join(c["text"] for c in result["chunks"])
    assert any("经营活动现金流量" in c["text"] for c in result["chunks"])
    assert len({c["chunk_id"] for c in result["chunks"]}) == len(result["chunks"])


def test_blank_or_scanned_pdf(tmp_path):
    path = tmp_path / "blank.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=600, height=800)
    writer.write(path)
    with pytest.raises(PdfTextError, match="扫描版"):
        extract_pdf(path)


def test_mixed_blank_page_reports_skipped_physical_page(tmp_path):
    path = tmp_path / "mixed.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=600, height=800)
    writer.add_page(PdfReader(SAMPLE).pages[0])
    writer.write(path)
    result = extract_pdf(path)
    assert {c["page"] for c in result["chunks"]} == {2}
    assert "第1页" in result["warnings"][0]


def test_encrypted_pdf(tmp_path):
    writer = PdfWriter()
    writer.add_page(PdfReader(SAMPLE).pages[0])
    writer.encrypt("test-password")
    path = tmp_path / "encrypted.pdf"
    writer.write(path)
    with pytest.raises(PdfTextError, match="加密"):
        extract_pdf(path)


@pytest.mark.parametrize("kind", ["missing", "corrupt", "not_pdf"])
def test_unreadable_pdf(tmp_path, kind):
    path = tmp_path / ("bad.txt" if kind == "not_pdf" else "bad.pdf")
    if kind != "missing":
        path.write_bytes(b"not a pdf")
    with pytest.raises(PdfTextError):
        extract_pdf(path)


def test_page_extraction_failure_is_not_partial_success(monkeypatch):
    import rag.pdf as module
    page = Mock()
    page.extract_text.side_effect = ValueError("测试失败")
    monkeypatch.setattr(module, "PdfReader", Mock(return_value=Mock(is_encrypted=False, pages=[page])))
    with pytest.raises(PdfTextError, match="第1页文字提取失败"):
        extract_pdf(SAMPLE)
