"""身份元数据与候选隔离。所有文档均为明确测试样本，不是公司正式财报。"""

from hashlib import sha256
import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from rag.pdf import extract_pdf
from rag.index import DocumentIndex
from rag.metadata import normalize_metadata, metadata_status
from rag.errors import IndexCorruptionError, RagError
from agents import retrieval, workflow
from memory import ReportStore
from rag.__main__ import main


SAMPLE = Path(__file__).parent / "fixtures" / "rag_test_document.pdf"
INFO = {"symbol": "600519", "company_name": "测试公司甲（仅样本身份）", "report_year": 2023,
        "document_type": "annual_report", "disclosure_date": "2024-04-02"}


class TestVectors:
    __test__ = False
    model_id = "TEST-ONLY-metadata-vectors"

    def encode(self, texts, *, query=False):
        return np.array([[1 + t.count("现金流"), 0.1] for t in texts], dtype="float32")


def document(tmp_path, name):
    path = tmp_path / name
    path.write_bytes(SAMPLE.read_bytes() + f"\n% TEST ONLY {name}".encode())
    return path


def snapshot_files(path):
    return {file.name: file.read_bytes() for file in path.iterdir()}


def rewrite_manifest(index, change):
    pointer = index.path / "CURRENT.json"
    manifest = json.loads(pointer.read_text())
    data_path = index.path / (manifest["generation"] + ".json")
    data = json.loads(data_path.read_text(encoding="utf-8"))
    change(data)
    data_path.write_text(json.dumps(data), encoding="utf-8")
    manifest["metadata_sha256"] = sha256(data_path.read_bytes()).hexdigest()
    pointer.write_text(json.dumps(manifest), encoding="utf-8")


def test_pdf_metadata_provenance_no_guessing_or_source_change(tmp_path):
    labelled = extract_pdf(SAMPLE, metadata=INFO)
    unknown = extract_pdf(SAMPLE)
    assert labelled["document_id"] == unknown["document_id"]
    assert labelled["metadata_status"] == "complete" and unknown["metadata_status"] == "unknown"
    assert all(value is None for value in unknown["metadata"].values())
    for a, b in zip(labelled["chunks"], unknown["chunks"]):
        assert a["metadata"] == INFO
        assert all(a[key] == b[key] for key in ("page", "text", "chunk_id", "start", "end"))
    misleading = tmp_path / "600519_2023_annual_report.pdf"
    misleading.write_bytes(SAMPLE.read_bytes())
    assert extract_pdf(misleading)["metadata"]["symbol"] is None


@pytest.mark.parametrize("metadata", [{"symbol": 600519}, {"symbol": "60051"}, {"symbol": "600519 OR 1=1"},
    {"report_year": True}, {"report_year": "2023"}, {"report_year": 1899}, {"report_year": 2101},
    {"company_name": " "}, {"document_type": ""}, {"disclosure_date": "2024-02-30"},
    {"disclosure_date": "20240402"}, {"unexpected": "value"}])
def test_invalid_metadata_rejected_before_import(tmp_path, metadata):
    index = DocumentIndex(tmp_path / "index", embedder=TestVectors())
    with pytest.raises(ValueError):
        index.import_pdf(SAMPLE, metadata=metadata)
    assert not index.path.exists()


def test_candidate_filter_before_top_k_stock_year_type_and_unknown(tmp_path):
    index = DocumentIndex(tmp_path / "index", embedder=TestVectors())
    docs = []
    for name, info in [("other.pdf", {**INFO, "symbol": "000001", "company_name": "测试公司乙"}),
                       ("older.pdf", {**INFO, "report_year": 2022}),
                       ("notice.pdf", {**INFO, "document_type": "announcement"}),
                       ("unknown.pdf", None), ("target.pdf", INFO)]:
        source = document(tmp_path, name)
        docs.append(index.import_pdf(source, metadata=info)["document_id"])
    loaded = DocumentIndex(index.path, embedder=TestVectors())
    hits = loaded.search_documents("现金流", 1, symbol="600519", report_year=2023, document_type="annual_report")
    assert len(hits) == 1 and hits[0]["document_id"] == docs[-1]
    assert hits[0]["metadata"] == INFO and hits[0]["metadata_status"] == "complete"
    assert all(h["metadata"]["symbol"] == "000001" for h in loaded.search_documents("现金流", 100, symbol="000001"))
    assert all(h["metadata"]["report_year"] == 2022 for h in loaded.search_documents("现金流", 100, report_year=2022))
    assert all(h["metadata"]["document_type"] == "announcement" for h in loaded.search_documents("现金流", 100, document_type="announcement"))
    assert any(h["metadata_status"] == "unknown" for h in loaded.search_documents("现金流", 100))


def test_no_matching_candidates_never_load_model_or_fallback(tmp_path):
    embedder = TestVectors()
    index = DocumentIndex(tmp_path / "index", embedder=embedder)
    index.import_pdf(SAMPLE, metadata=INFO)
    embedder.encode = Mock(side_effect=AssertionError("不该加载模型"))
    assert index.search_documents("现金流", symbol="000001") == []
    assert index.search_documents("现金流", symbol="600519", report_year=2024) == []
    assert index.search_documents("现金流", document_type="interim_report") == []
    embedder.encode.assert_not_called()


def test_selected_document_narrows_candidates_before_ranking(tmp_path, monkeypatch):
    embedder = TestVectors()
    index = DocumentIndex(tmp_path / "index", embedder=embedder)
    index.import_pdf(document(tmp_path, "another-announcement.pdf"), metadata=INFO)
    selected = index.import_pdf(document(tmp_path, "selected-announcement.pdf"), metadata=INFO)["document_id"]
    monkeypatch.setattr("rag.index.LocalEmbedder", lambda: embedder)

    facts = retrieval.retrieve_document_facts("现金流", index_dir=index.path, symbol="600519", document_id=selected)
    assert facts["status"] == "ok"
    assert {item["document_id"] for item in facts["records"]} == {selected}
    assert index.search_documents("现金流", document_id=selected, symbol="000001") == []


def test_returned_metadata_cannot_mutate_stock_identity(tmp_path):
    index = DocumentIndex(tmp_path / "index", embedder=TestVectors())
    index.import_pdf(SAMPLE, metadata=INFO)
    hit = index.search_documents("现金流", symbol="600519")[0]
    hit["metadata"]["symbol"] = "000001"
    duplicate = index.import_pdf(SAMPLE)
    duplicate["metadata"]["symbol"] = "000001"
    assert index.search_documents("现金流", symbol="000001") == []
    assert index.search_documents("现金流", symbol="600519")


@pytest.mark.parametrize("filters", [{"symbol": "bad"}, {"report_year": True}, {"report_year": "2023"}, {"document_type": " "}])
def test_bad_filters_even_for_empty_index(tmp_path, filters):
    with pytest.raises(ValueError):
        DocumentIndex(tmp_path, embedder=TestVectors()).search_documents("现金流", **filters)


def test_legacy_load_unknown_read_only_and_explicit_update(tmp_path):
    index = DocumentIndex(tmp_path / "index", embedder=TestVectors())
    identity = index.import_pdf(SAMPLE)["document_id"]
    def legacy(data):
        for chunk in data["chunks"]:
            chunk.pop("metadata")
            chunk.pop("metadata_status")
    rewrite_manifest(index, legacy)
    before = snapshot_files(index.path)
    loaded = DocumentIndex(index.path, embedder=TestVectors())
    assert all(h["metadata_status"] == "unknown" for h in loaded.search_documents("现金流"))
    assert loaded.search_documents("现金流", symbol="600519") == []
    assert snapshot_files(index.path) == before
    vectors = loaded.index.reconstruct_n(0, loaded.index.ntotal).copy()
    sources = [(c["chunk_id"], c["text"], c["page"]) for c in loaded.chunks]
    loaded.embedder.encode = Mock(side_effect=AssertionError("补录不该重新嵌入"))
    assert loaded.update_metadata(identity, INFO)["updated"]
    assert all((index.path / name).read_bytes() == value for name, value in before.items() if name != "CURRENT.json")
    np.testing.assert_array_equal(vectors, loaded.index.reconstruct_n(0, loaded.index.ntotal))
    assert sources == [(c["chunk_id"], c["text"], c["page"]) for c in loaded.chunks]
    after = snapshot_files(index.path)
    assert not loaded.update_metadata(identity, INFO)["updated"]
    assert snapshot_files(index.path) == after
    loaded.update_metadata(identity, {"report_year": 2024})
    assert loaded.chunks[0]["metadata"]["symbol"] == "600519"  # 部分补录不清除已知字段。


def test_duplicate_conflict_and_update_failure_keep_old_snapshot(tmp_path, monkeypatch):
    import rag.index as module
    index = DocumentIndex(tmp_path / "index", embedder=TestVectors())
    identity = index.import_pdf(SAMPLE, metadata=INFO)["document_id"]
    before = (index.path / "CURRENT.json").read_bytes()
    with pytest.raises(ValueError, match="set-metadata"):
        index.import_pdf(SAMPLE, metadata={**INFO, "symbol": "000001"})
    with pytest.raises(ValueError, match="未找到"):
        index.update_metadata("0" * 64, INFO)
    monkeypatch.setattr(module.os, "replace", Mock(side_effect=OSError("测试失败")))
    with pytest.raises(RagError):
        index.update_metadata(identity, {"report_year": 2024})
    assert (index.path / "CURRENT.json").read_bytes() == before
    assert DocumentIndex(index.path, embedder=TestVectors()).chunks[0]["metadata"] == INFO


def test_corrupt_conflicting_document_metadata_rejected(tmp_path):
    index = DocumentIndex(tmp_path / "index", embedder=TestVectors())
    index.import_pdf(SAMPLE, metadata=INFO)
    rewrite_manifest(index, lambda data: data["chunks"][0]["metadata"].update(symbol="000001"))
    with pytest.raises(IndexCorruptionError):
        DocumentIndex(index.path, embedder=TestVectors())


def test_agent_defense_rejects_cross_stock_year_and_unknown(monkeypatch):
    base = {"document_id": "a" * 64, "document_name": "仅测试.pdf", "chunk_id": "b" * 64,
            "page": 1, "text": "仅为测试样本原文", "score": 0.9}
    hits = [{**base, "metadata": item} for item in [None, {**INFO, "symbol": "000001"}, {**INFO, "report_year": 2022}, INFO]]
    search = Mock(return_value=hits)
    monkeypatch.setattr(retrieval, "search_documents", search)
    facts = retrieval.retrieve_document_facts("测试问题", symbol="600519", report_year=2023, document_type="annual_report")
    assert facts["status"] == "ok" and len(facts["records"]) == 1
    assert facts["records"][0]["metadata"] == facts["sources"][0]["metadata"] == INFO
    search.assert_called_once_with("测试问题", top_k=3, index_dir=None, symbol="600519", report_year=2023, document_type="annual_report")
    search.return_value = hits[:3]
    assert retrieval.retrieve_document_facts("测试问题", symbol="600519", report_year=2023)["status"] == "no_match"


def test_graph_report_model_context_and_sqlite_provenance(monkeypatch):
    import pandas as pd
    hit = {"document_id": "a" * 64, "document_name": "测试样本.pdf", "page": 2, "chunk_id": "b" * 64,
           "text": "仅测试原文", "score": 0.9, "metadata": INFO}
    search = Mock(return_value=[hit])
    monkeypatch.setattr(retrieval, "search_documents", search)
    frame = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=40, freq="B"), "close": [10.] * 40, "volume": [100.] * 40})
    client = Mock()
    client.describe.return_value = "仅测试解读"
    graph = workflow.build_graph(client, history_loader=Mock(return_value=frame))
    state = graph.invoke({"symbol": "600519", "start_date": "20240101", "end_date": "20240630", "rag_question": "测试问题",
                          "rag_report_year": 2023, "rag_document_type": "annual_report", "trace": []})
    assert client.describe.call_args.args[1]["rag"]["sources"][0]["metadata"] == INFO
    assert state["rag_facts"]["filters"] == {"symbol": "600519", "report_year": 2023, "document_type": "annual_report"}
    report_id = ReportStore().save_report(state, mode="local")
    saved = ReportStore().get_report(report_id)
    assert saved["sources"]["rag"]["references"][0]["metadata"] == INFO
    assert '"report_year": 2023' in saved["report"]


def test_cli_import_filter_and_metadata_patch(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("rag.__main__.LocalEmbedder", lambda *a, **kw: TestVectors())
    common = ["--index-dir", str(tmp_path / "index")]
    assert main(common + ["import-pdf", str(SAMPLE), "--symbol", "600519", "--report-year", "2023", "--document-type", "annual_report"]) == 0
    identity = json.loads(capsys.readouterr().out)["document_id"]
    assert main(common + ["search", "现金流", "--symbol", "000001"]) == 0
    assert "没有符合" in capsys.readouterr().out
    assert main(common + ["set-metadata", identity, "--company-name", "仅测试公司"]) == 0
    capsys.readouterr()
    assert main(common + ["search", "现金流", "--symbol", "600519", "--report-year", "2023"]) == 0
    hits = json.loads(capsys.readouterr().out)
    assert hits[0]["metadata"]["company_name"] == "仅测试公司" and hits[0]["metadata_status"] == "partial"
