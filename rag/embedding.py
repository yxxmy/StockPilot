"""中文BGE本地CPU推理。只有显式download-model操作允许下载权重。"""

from pathlib import Path

from rag.errors import EmbeddingError


DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"
DEFAULT_REVISION = "7999e1d3359715c523056ef9478215996d62a620"
QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："
CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "models"


class LocalEmbedder:
    def __init__(self, model_name=DEFAULT_MODEL, *, allow_download=False):
        self.model_name = model_name
        self.revision = DEFAULT_REVISION if model_name == DEFAULT_MODEL else None
        self.model_id = f"{model_name}@{self.revision}" if self.revision else model_name
        self.allow_download = allow_download
        self._model = None

    def load(self):
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(
                self.model_name, revision=self.revision, device="cpu", cache_folder=str(CACHE_DIR),
                local_files_only=not self.allow_download, trust_remote_code=False,
                model_kwargs={"use_safetensors": True},
            )
        except Exception:
            raise EmbeddingError("本地Embedding模型加载失败，请先运行python -m rag download-model，或用--model指定完整本地模型目录") from None

    def encode(self, texts, *, query=False):
        self.load()
        inputs = [QUERY_PREFIX + text for text in texts] if query else texts
        try:
            # 显式检查token长度，避免模型悄悄截断长chunk而丢失检索信息。
            for text in inputs:
                tokens = self._model.tokenizer(text, truncation=False)["input_ids"]
                if len(tokens) > self._model.max_seq_length:
                    raise EmbeddingError("文本超过本地模型长度限制，请减小chunk-size或缩短问题")
            return self._model.encode(inputs, batch_size=16, normalize_embeddings=True,
                                      convert_to_numpy=True, show_progress_bar=False)
        except EmbeddingError:
            raise
        except Exception:
            raise EmbeddingError("本地Embedding计算失败，未更新索引") from None
