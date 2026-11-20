class RagError(RuntimeError):
    """PDF、Embedding或索引错误，绝不以伪造内容回退。"""


class PdfTextError(RagError):
    pass


class EmbeddingError(RagError):
    pass


class IndexCorruptionError(RagError):
    pass
