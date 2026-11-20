import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from rag.embedding import DEFAULT_MODEL, DEFAULT_REVISION, LocalEmbedder, QUERY_PREFIX
from rag.errors import EmbeddingError
from rag.__main__ import main


def test_model_failure_is_explicit(monkeypatch):
    constructor = Mock(side_effect=OSError("模型不存在"))
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=constructor))
    with pytest.raises(EmbeddingError, match="加载失败"):
        LocalEmbedder().encode(["文本"])
    kwargs = constructor.call_args.kwargs
    assert kwargs["local_files_only"] is True and kwargs["trust_remote_code"] is False
    assert kwargs["revision"] == DEFAULT_REVISION


def test_real_wrapper_query_prefix_and_download_flag(monkeypatch):
    model = Mock(max_seq_length=512)
    model.tokenizer.return_value = {"input_ids": [1, 2, 3]}
    model.encode.return_value = np.ones((1, 3))
    constructor = Mock(return_value=model)
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=constructor))
    embedder = LocalEmbedder(allow_download=True)
    embedder.encode(["现金流"], query=True)
    assert model.encode.call_args.args[0] == [QUERY_PREFIX + "现金流"]
    assert model.encode.call_args.kwargs["normalize_embeddings"] is True
    assert constructor.call_args.kwargs["local_files_only"] is False
    embedder.encode(["正文"])
    assert model.encode.call_args.args[0] == ["正文"]
    constructor.assert_called_once()


def test_long_input_not_silently_truncated(monkeypatch):
    model = Mock(max_seq_length=5)
    model.tokenizer.return_value = {"input_ids": list(range(10))}
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=Mock(return_value=model)))
    with pytest.raises(EmbeddingError, match="长度限制"):
        LocalEmbedder().encode(["超长文本"])
    model.encode.assert_not_called()


def test_cli_empty_search_never_loads_model(tmp_path, capsys):
    pytest.importorskip("faiss")
    assert main(["--index-dir", str(tmp_path), "search", "问题"]) == 0
    assert "索引为空" in capsys.readouterr().out


def test_cli_bad_pdf_returns_error(tmp_path, capsys):
    pytest.importorskip("faiss")
    assert main(["--index-dir", str(tmp_path), "import-pdf", str(tmp_path / "missing.pdf")]) == 1
    assert "错误" in capsys.readouterr().err


def test_explicit_download_command(monkeypatch, capsys):
    import rag.__main__ as module
    client = Mock()
    constructor = Mock(return_value=client)
    monkeypatch.setattr(module, "LocalEmbedder", constructor)
    assert main(["download-model"]) == 0
    constructor.assert_called_once_with(DEFAULT_MODEL, allow_download=True)
    client.load.assert_called_once()
    assert "模型已准备" in capsys.readouterr().out
