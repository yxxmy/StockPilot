"""按物理页码提取本地PDF文字，保留内容哈希和字符偏移。"""

from hashlib import sha256
from io import BytesIO
from pathlib import Path
import re

from pypdf import PdfReader
from rag.metadata import normalize_metadata, metadata_status

from rag.errors import PdfTextError


def validate_chunk_options(chunk_size, overlap):
    if (isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0
            or isinstance(overlap, bool) or not isinstance(overlap, int) or not 0 <= overlap < chunk_size):
        raise ValueError("chunk_size必须是正整数，overlap必须满足0 <= overlap < chunk_size")


def chunk_text(text, chunk_size=350, overlap=60):
    """滑动窗口：到达文本末尾立即结束，避免仅含overlap的多余尾片段。"""
    validate_chunk_options(chunk_size, overlap)
    if not isinstance(text, str):
        raise ValueError("切片输入必须是字符串")
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        yield {"text": text[start:end], "start": start, "end": end}
        if end == len(text):
            break
        start = end - overlap  # overlap严格小于chunk_size，保证每次前进。


def extract_pdf(path, chunk_size=350, overlap=60, *, metadata=None):
    validate_chunk_options(chunk_size, overlap)
    metadata = normalize_metadata(metadata)
    path = Path(path)
    if path.suffix.lower() != ".pdf":
        raise PdfTextError("只接受本地.pdf文件")
    try:
        content = path.read_bytes()
        document_id = sha256(content).hexdigest()
        reader = PdfReader(BytesIO(content))
        if reader.is_encrypted:
            raise PdfTextError("加密PDF暂不支持，请先在本机解密")
        chunks, warnings = [], []
        for page_number, page in enumerate(reader.pages, 1):
            try:
                text = page.extract_text() or ""
            except Exception:
                raise PdfTextError(f"PDF第{page_number}页文字提取失败，文档未导入") from None
            # 保留原文用词，仅统一空白；偏移相对该页归一化后的文字。
            text = re.sub(r"\s+", " ", text).strip()
            if not text:
                warnings.append(f"第{page_number}页无可提取文字（可能是扫描页或空白页），已跳过，未执行OCR")
                continue
            if len(text) < 20:
                warnings.append(f"第{page_number}页可提取文字很少，可能只含页眉页脚，请核对PDF")
            for part in chunk_text(text, chunk_size, overlap):
                chunk_id = sha256(f"{document_id}:{page_number}:{part['start']}:{part['end']}".encode()).hexdigest()
                chunks.append({"document_id": document_id, "document_name": path.name,
                               "page": page_number, "chunk_id": chunk_id, **part,
                               "metadata": dict(metadata), "metadata_status": metadata_status(metadata)})
        if not chunks:
            raise PdfTextError("PDF无可提取文字，可能是扫描版或空文档；本阶段不支持OCR")
        return {"document_id": document_id, "document_name": path.name, "chunks": chunks, "warnings": warnings,
                "metadata": metadata, "metadata_status": metadata_status(metadata)}
    except PdfTextError:
        raise
    except Exception:
        raise PdfTextError("PDF无法读取或解析，请检查文件路径、权限和文件完整性") from None
