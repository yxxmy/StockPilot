"""用明确的测试Embedding替身，验证真实FAISS离线导入、保存、加载和检索。"""

from hashlib import sha256
import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

pytest.importorskip("faiss")
pytest.importorskip("pypdf")
from rag.errors import EmbeddingError, IndexCorruptionError, RagError
from rag.index import DocumentIndex


SAMPLE = Path(__file__).parent / "fixtures" / "rag_test_document.pdf"


class TestEmbedder:
    __test__ = False
    model_id = "OFFLINE-TEST-ONLY-keyword-vectors"

    def encode(self, texts, *, query=False):
        return np.array([[1 + t.count("现金流") * 4, 1 + t.count("风险") * 4, 0.1] for t in texts], dtype="float32")


def create(tmp_path):
    index = DocumentIndex(tmp_path / "中文索引", embedder=TestEmbedder())
    index.import_pdf(SAMPLE)
    return index


def test_import_search_and_reload(tmp_path):
    index = create(tmp_path)
    result = index.search_documents("现金流", 5)
    assert result[0]["page"] == 1
    assert result[0]["document_name"] == SAMPLE.name
    assert result[0]["document_id"] and result[0]["text"]
    assert all(-1.0001 <= r["score"] <= 1.0001 for r in result)
    loaded = DocumentIndex(index.path, embedder=TestEmbedder())
    assert loaded.search_documents("现金流", 5) == result
    assert loaded.search_documents("风险", 1)[0]["page"] == 2


def test_empty_index_does_not_call_embedder(tmp_path):
    embedder = TestEmbedder()
    embedder.encode = Mock(side_effect=AssertionError("不能加载或计算模型"))
    assert DocumentIndex(tmp_path, embedder=embedder).search_documents("问题") == []
    embedder.encode.assert_not_called()


def test_duplicate_content_does_not_add_vectors(tmp_path):
    index = create(tmp_path)
    count = len(index.chunks)
    before = (index.path / "CURRENT.json").read_bytes()
    result = index.import_pdf(SAMPLE)
    assert result["duplicate"] is True and result["chunks_added"] == 0
    assert len(index.chunks) == count
    assert (index.path / "CURRENT.json").read_bytes() == before


@pytest.mark.parametrize("question,k", [("", 5), (None, 5), ("问题", 0), ("问题", True), ("问题", 101)])
def test_invalid_query(tmp_path, question, k):
    with pytest.raises(ValueError):
        DocumentIndex(tmp_path, embedder=TestEmbedder()).search_documents(question, k)


@pytest.mark.parametrize("file_kind", ["index", "metadata", "manifest", "missing_index", "missing_manifest"])
def test_corrupted_index_is_not_silently_empty(tmp_path, file_kind):
    index = create(tmp_path)
    manifest = json.loads((index.path / "CURRENT.json").read_text())
    path = index.path / {"index": manifest["generation"] + ".faiss", "metadata": manifest["generation"] + ".json",
                         "manifest": "CURRENT.json", "missing_index": manifest["generation"] + ".faiss", "missing_manifest": "CURRENT.json"}[file_kind]
    if file_kind.startswith("missing"):
        path.unlink()
    else:
        path.write_bytes(b"broken")
    with pytest.raises(IndexCorruptionError):
        DocumentIndex(index.path, embedder=TestEmbedder())


def test_model_mismatch_is_rejected(tmp_path):
    index = create(tmp_path)
    other = TestEmbedder()
    other.model_id = "different-model"
    with pytest.raises(IndexCorruptionError, match="模型"):
        DocumentIndex(index.path, embedder=other)


def test_invalid_metadata_even_with_matching_checksum(tmp_path):
    index = create(tmp_path)
    pointer = index.path / "CURRENT.json"
    manifest = json.loads(pointer.read_text())
    metadata_path = index.path / (manifest["generation"] + ".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["chunks"][0]["page"] = 0
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    manifest["metadata_sha256"] = sha256(metadata_path.read_bytes()).hexdigest()
    pointer.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(IndexCorruptionError):
        DocumentIndex(index.path, embedder=TestEmbedder())


@pytest.mark.parametrize("bad_vector", [[[float("nan"), 1]], [[0, 0]], [[1, 2, 3]], [1, 2]])
def test_invalid_embedding_does_not_create_active_index(tmp_path, bad_vector):
    embedder = TestEmbedder()
    embedder.encode = Mock(return_value=bad_vector)
    index = DocumentIndex(tmp_path, embedder=embedder)
    with pytest.raises(EmbeddingError):
        index.import_pdf(SAMPLE)
    assert not (tmp_path / "CURRENT.json").exists()
    assert not (tmp_path / ".writer.lock").exists()


def test_failed_save_keeps_old_snapshot(tmp_path, monkeypatch):
    import rag.index as module
    index = create(tmp_path)
    before = (index.path / "CURRENT.json").read_bytes()
    changed = tmp_path / "changed.pdf"
    changed.write_bytes(SAMPLE.read_bytes() + b"\n% distinct test content")
    monkeypatch.setattr(module.os, "replace", Mock(side_effect=OSError("测试写失败")))
    with pytest.raises(RagError, match="保存失败"):
        index.import_pdf(changed)
    assert (index.path / "CURRENT.json").read_bytes() == before
    loaded = DocumentIndex(index.path, embedder=TestEmbedder())
    assert len(loaded.chunks) == len(index.chunks)


def test_multiple_instances_reload_before_import(tmp_path):
    first = DocumentIndex(tmp_path / "index", embedder=TestEmbedder())
    second = DocumentIndex(first.path, embedder=TestEmbedder())
    first.import_pdf(SAMPLE)
    changed = tmp_path / "changed.pdf"
    changed.write_bytes(SAMPLE.read_bytes() + b"\n% distinct test document")
    second.import_pdf(changed)
    loaded = DocumentIndex(first.path, embedder=TestEmbedder())
    assert len({c["document_id"] for c in loaded.chunks}) == 2


def test_writer_lock_blocks_import(tmp_path):
    index = DocumentIndex(tmp_path, embedder=TestEmbedder())
    (tmp_path / ".writer.lock").write_text("test lock")
    with pytest.raises(RagError, match="导入进程"):
        index.import_pdf(SAMPLE)


def test_query_dimension_mismatch_is_explicit(tmp_path):
    index = create(tmp_path)
    index.embedder.encode = Mock(return_value=np.ones((1, 4)))
    with pytest.raises(EmbeddingError, match="维度"):
        index.search_documents("现金流")


def test_embedding_failure_keeps_old_index(tmp_path):
    index = create(tmp_path)
    changed = tmp_path / "changed.pdf"
    changed.write_bytes(SAMPLE.read_bytes() + b"\n% another test document")
    before = (index.path / "CURRENT.json").read_bytes()
    index.embedder.encode = Mock(side_effect=EmbeddingError("测试Embedding失败"))
    with pytest.raises(EmbeddingError):
        index.import_pdf(changed)
    assert (index.path / "CURRENT.json").read_bytes() == before
    assert len(DocumentIndex(index.path, embedder=TestEmbedder()).chunks) == 2


def test_unsafe_generation_path_rejected(tmp_path):
    index = create(tmp_path)
    pointer = index.path / "CURRENT.json"
    manifest = json.loads(pointer.read_text())
    manifest["generation"] = "../outside"
    pointer.write_text(json.dumps(manifest))
    with pytest.raises(IndexCorruptionError):
        DocumentIndex(index.path, embedder=TestEmbedder())
