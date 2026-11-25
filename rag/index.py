"""FAISS精确余弦检索；JSON保存原文和引用，不使用pickle。"""

from contextlib import contextmanager
from hashlib import sha256
import json
import os
from pathlib import Path
from uuid import uuid4

import faiss
import numpy as np

from rag.embedding import LocalEmbedder, QUERY_PREFIX
from rag.errors import EmbeddingError, IndexCorruptionError, RagError
from rag.pdf import extract_pdf, validate_chunk_options
from rag.metadata import normalize_metadata, metadata_status, search_filters, matches, verified_pdf_url


DEFAULT_INDEX_DIR = Path(__file__).resolve().parents[1] / "data" / "rag"


def _vectors(values, count, dimension=None):
    try:
        array = np.array(values, dtype="float32", copy=True, order="C")
        if array.ndim != 2 or array.shape[0] != count or array.shape[1] == 0:
            raise ValueError()
        if dimension is not None and array.shape[1] != dimension:
            raise ValueError()
        lengths = np.linalg.norm(array, axis=1)
        if not np.isfinite(array).all() or not np.isfinite(lengths).all() or (lengths <= 0).any():
            raise ValueError()
        faiss.normalize_L2(array)
        return array
    except Exception:
        raise EmbeddingError("Embedding向量数量、维度或数值不合法，未更新索引") from None


class DocumentIndex:
    def __init__(self, index_dir=None, *, embedder=None, chunk_size=350, overlap=60):
        validate_chunk_options(chunk_size, overlap)
        self.path = Path(index_dir or DEFAULT_INDEX_DIR).resolve()
        self.embedder = embedder if embedder is not None else LocalEmbedder()
        self.chunk_size, self.overlap = chunk_size, overlap
        self.index, self.chunks = None, []
        self.reload()

    def reload(self):
        """先核对快照哈希和结构，再交给FAISS解析；只加载自己创建的可信索引。"""
        pointer = self.path / "CURRENT.json"
        if not pointer.exists():
            if self.path.exists() and (not self.path.is_dir() or any(self.path.glob("*.faiss"))):
                raise IndexCorruptionError("索引快照存在但缺少有效清单，请用新目录重建")
            self.index, self.chunks = None, []
            return
        try:
            manifest = json.loads(pointer.read_text(encoding="utf-8"))
            generation = manifest["generation"]
            if manifest["version"] != 1 or not isinstance(generation, str) or len(generation) != 32 or any(c not in "0123456789abcdef" for c in generation):
                raise ValueError()
            vector_bytes = (self.path / f"{generation}.faiss").read_bytes()
            metadata_bytes = (self.path / f"{generation}.json").read_bytes()
            if sha256(vector_bytes).hexdigest() != manifest["index_sha256"] or sha256(metadata_bytes).hexdigest() != manifest["metadata_sha256"]:
                raise ValueError()
            metadata = json.loads(metadata_bytes)
            if metadata["model_id"] != self.embedder.model_id:
                raise IndexCorruptionError("索引Embedding模型与当前配置不同，请使用原模型或新索引目录")
            if metadata["query_prefix"] != QUERY_PREFIX:
                raise ValueError()
            chunks = metadata["chunks"]
            if not isinstance(chunks, list) or not chunks:
                raise ValueError()
            ids = set()
            document_metadata = {}
            normalized = []
            for chunk in chunks:
                if (not isinstance(chunk, dict) or not isinstance(chunk["text"], str) or not chunk["text"].strip()
                        or type(chunk["page"]) is not int or chunk["page"] < 1
                        or not isinstance(chunk["document_name"], str) or not chunk["document_name"]
                        or len(chunk["document_id"]) != 64 or len(chunk["chunk_id"]) != 64
                        or type(chunk["start"]) is not int or type(chunk["end"]) is not int
                        or chunk["start"] < 0 or chunk["end"] - chunk["start"] != len(chunk["text"])
                        or chunk["chunk_id"] in ids):
                    raise ValueError()
                ids.add(chunk["chunk_id"])
                info = normalize_metadata(chunk.get("metadata"))
                url = chunk.get("official_pdf_url")
                if url is not None and verified_pdf_url(url) is None:
                    raise ValueError("公告 PDF 链接格式不合法")
                identity = chunk["document_id"]
                if identity in document_metadata and document_metadata[identity] != (info, url):
                    raise ValueError("同一文档的元数据不一致")
                document_metadata[identity] = (info, url)
                # 旧快照仅在内存中补齐未知值，不迁移或改写磁盘文件。
                normalized.append({**chunk, "metadata": info, "metadata_status": metadata_status(info)})
            index = faiss.deserialize_index(np.frombuffer(vector_bytes, dtype="uint8"))
            if not isinstance(index, faiss.IndexFlatIP) or index.ntotal != len(chunks) or index.d != metadata["dimension"]:
                raise ValueError()
            # 归一化和有限值检查，避免损坏向量产生无意义分数。
            vectors = index.reconstruct_n(0, index.ntotal)
            if not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4):
                raise ValueError()
            self.index, self.chunks = index, normalized
        except IndexCorruptionError:
            raise
        except Exception:
            raise IndexCorruptionError("索引损坏或文件不完整，未加载；请保留现场并用新目录重建") from None

    @contextmanager
    def _writer(self):
        try:
            self.path.mkdir(parents=True, exist_ok=True)
            lock = self.path / ".writer.lock"
            handle = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except OSError:
            raise RagError("索引目录不可写或已有导入进程；异常退出后请确认无进程再处理.writer.lock") from None
        try:
            yield
        finally:
            os.close(handle)
            lock.unlink()

    def _persist(self, index, chunks):
        generation = uuid4().hex
        vector_bytes = faiss.serialize_index(index).tobytes()
        metadata = {"model_id": self.embedder.model_id, "dimension": index.d,
                    "query_prefix": QUERY_PREFIX, "chunks": chunks}
        metadata_bytes = json.dumps(metadata, ensure_ascii=False, allow_nan=False).encode("utf-8")
        manifest = {"version": 1, "generation": generation, "index_sha256": sha256(vector_bytes).hexdigest(),
                    "metadata_sha256": sha256(metadata_bytes).hexdigest()}
        temporary = self.path / f"{generation}.pointer.tmp"
        try:
            (self.path / f"{generation}.faiss").write_bytes(vector_bytes)
            (self.path / f"{generation}.json").write_bytes(metadata_bytes)
            temporary.write_text(json.dumps(manifest), encoding="utf-8")
            # 两个快照文件写完后才原子切换指针，失败时旧快照仍可用。
            os.replace(temporary, self.path / "CURRENT.json")
        except OSError:
            raise RagError("索引保存失败，未切换有效快照") from None

    def import_pdf(self, path, *, metadata=None, official_pdf_url=None):
        document = extract_pdf(path, self.chunk_size, self.overlap, metadata=metadata)
        if official_pdf_url is not None and (verified_pdf_url(official_pdf_url) is None or not document["metadata"]["symbol"]):
            raise ValueError("保存官方 PDF 链接时需要已核对的链接和股票代码")
        with self._writer():
            self.reload()  # 避免另一个实例刚导入的内容被旧内存覆盖。
            existing = next((chunk for chunk in self.chunks if chunk["document_id"] == document["document_id"]), None)
            if existing:
                if metadata is not None and existing["metadata"] != document["metadata"]:
                    raise ValueError("文档已存在且元数据不同，请核验后使用set-metadata显式补录或更正")
                result = {"document_id": document["document_id"], "chunks_added": 0, "duplicate": True, "warnings": document["warnings"],
                          "metadata": dict(existing["metadata"]), "metadata_status": existing["metadata_status"]}
                if official_pdf_url and not existing.get("official_pdf_url"):
                    # 同字节文件已在索引中，只补充出处并保留原文向量。
                    chunks = [{**c, "official_pdf_url": official_pdf_url} if c["document_id"] == document["document_id"] else c
                              for c in self.chunks]
                    self._persist(self.index, chunks)
                    self.chunks = chunks
                    result["source_added"] = True
                if url := existing.get("official_pdf_url") or official_pdf_url:
                    result["official_pdf_url"] = url
                return result
            additions = document["chunks"]
            if official_pdf_url:
                for chunk in additions:
                    chunk["official_pdf_url"] = official_pdf_url
            vectors = _vectors(self.embedder.encode([chunk["text"] for chunk in additions]), len(additions),
                               self.index.d if self.index is not None else None)
            index = faiss.clone_index(self.index) if self.index is not None else faiss.IndexFlatIP(vectors.shape[1])
            index.add(vectors)
            chunks = self.chunks + additions
            self._persist(index, chunks)
            self.index, self.chunks = index, chunks
        result = {"document_id": document["document_id"], "chunks_added": len(additions), "duplicate": False, "warnings": document["warnings"],
                  "metadata": document["metadata"], "metadata_status": document["metadata_status"]}
        if official_pdf_url:
            result["official_pdf_url"] = official_pdf_url
        return result

    def update_metadata(self, document_id, metadata):
        """仅显式补录身份，向量/原文/页码不变；保留全部旧快照。"""
        if not isinstance(document_id, str) or len(document_id) != 64 or any(c not in "0123456789abcdef" for c in document_id):
            raise ValueError("文档标识须为64位SHA256")
        if not isinstance(metadata, dict):
            raise ValueError("补录元数据必须为字段字典")
        normalize_metadata(metadata)
        with self._writer():
            self.reload()
            selected = [c for c in self.chunks if c["document_id"] == document_id]
            if not selected:
                raise ValueError("未找到该文档，未修改元数据")
            info = normalize_metadata({**selected[0]["metadata"], **metadata})
            if all(c["metadata"] == info for c in selected):
                return {"document_id": document_id, "updated": False}
            chunks = [{**c, "metadata": dict(info), "metadata_status": metadata_status(info)}
                      if c["document_id"] == document_id else c for c in self.chunks]
            self._persist(self.index, chunks)
            self.chunks = chunks
        return {"document_id": document_id, "updated": True}

    def list_documents(self, *, symbol=None, report_year=None, document_type=None):
        """每份匹配的 PDF 只列一次，不计算向量。"""
        filters = search_filters(symbol, report_year, document_type)
        documents = {}
        for chunk in self.chunks:
            identity = chunk["document_id"]
            if identity not in documents and matches(chunk["metadata"], filters):
                documents[identity] = {
                    "document_id": identity,
                    "document_name": chunk["document_name"],
                    "metadata": dict(chunk["metadata"]),
                }
        return sorted(documents.values(), key=lambda doc: (
            doc["metadata"]["report_year"] or 0, doc["metadata"]["disclosure_date"] or "", doc["document_name"],
        ), reverse=True)

    def search_documents(self, question, top_k=5, *, symbol=None, report_year=None, document_type=None,
                         document_id=None):
        if not isinstance(question, str) or not question.strip():
            raise ValueError("问题不能为空")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 100:
            raise ValueError("top_k必须是1到100的整数")
        filters = search_filters(symbol, report_year, document_type)
        if self.index is None:
            return []
        candidates = [i for i, chunk in enumerate(self.chunks)
                      if (document_id is None or chunk["document_id"] == document_id)
                      and matches(chunk["metadata"], filters)]
        if not candidates:
            return []  # 不加载模型、不放宽股票/年份条件、不把未知归给请求股票。
        index = self.index
        if filters or document_id is not None:
            # 先从元数据确定候选，再对候选构建精确余弦索引，避免全库top_k后过滤漏召回。
            index = faiss.IndexFlatIP(self.index.d)
            index.add(np.stack([self.index.reconstruct(i) for i in candidates]))
        vector = _vectors(self.embedder.encode([question.strip()], query=True), 1, self.index.d)
        scores, positions = index.search(vector, min(top_k, len(candidates)))
        return [{**self.chunks[candidates[int(position)]], "metadata": dict(self.chunks[candidates[int(position)]]["metadata"]), "score": float(score)}
                for score, position in zip(scores[0], positions[0])]
