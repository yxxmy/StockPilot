"""5B：独立PDF检索，不调用大模型生成答案。"""

from rag.errors import RagError


def search_documents(question, top_k=5, *, index_dir=None, symbol=None, report_year=None, document_type=None,
                     document_id=None):
    """返回原文及页码；无索引时返回空列表，不下载模型。"""
    from rag.index import DocumentIndex
    return DocumentIndex(index_dir).search_documents(question, top_k,
        symbol=symbol, report_year=report_year, document_type=document_type, document_id=document_id)


__all__ = ["search_documents", "RagError"]
