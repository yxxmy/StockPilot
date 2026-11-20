"""只使用假响应测试 HTTP 协议，绝不调用真实模型。"""

import json
from unittest.mock import Mock

import pytest
import requests

from agents.deepseek import AgentError, DeepSeekClient


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "unit-test-key")
    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)
    return DeepSeekClient()


def response(content, *, finish="stop", status=200):
    return Mock(status_code=status, json=Mock(return_value={"choices": [
        {"finish_reason": finish, "message": {"content": content}},
    ]}))


def test_missing_key(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(AgentError, match="DEEPSEEK_API_KEY"):
        DeepSeekClient()


def test_empty_model(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "unit-test-key")
    monkeypatch.setenv("DEEPSEEK_MODEL", " ")
    with pytest.raises(AgentError, match="DEEPSEEK_MODEL"):
        DeepSeekClient()


def test_official_endpoint_and_request_contract(client, monkeypatch):
    post = Mock(return_value=response(json.dumps({"summary": "测试文字", "evidence": ["latest"]})))
    monkeypatch.setattr(requests, "post", post)
    text = client.describe("Market Agent", {"latest": {"close": 10}})
    assert text == "测试文字\n事实引用：latest"
    assert post.call_args.args == ("https://api.deepseek.com/chat/completions",)
    kwargs = post.call_args.kwargs
    assert kwargs["headers"]["Authorization"] == "Bearer unit-test-key"
    assert kwargs["timeout"] == (10, 60)
    assert kwargs["allow_redirects"] is False
    payload = kwargs["json"]
    assert payload["model"] == "deepseek-flash"
    assert payload["response_format"] == {"type": "json_object"}
    assert json.loads(payload["messages"][1]["content"]) == {"latest": {"close": 10}}
    assert "unit-test-key" not in json.dumps(payload)
    assert '["latest"]' in payload["messages"][0]["content"]
    assert "禁止使用点号路径" in payload["messages"][0]["content"]


@pytest.mark.parametrize("status", [301, 401, 402, 429, 500])
def test_http_failure_no_secret_leak(client, monkeypatch, status):
    post = Mock(return_value=response("unit-test-key", status=status))
    monkeypatch.setattr(requests, "post", post)
    with pytest.raises(AgentError, match=f"HTTP {status}") as error:
        client.describe("Market Agent", {"latest": 10})
    assert "unit-test-key" not in str(error.value)
    post.assert_called_once()


@pytest.mark.parametrize("error", [requests.Timeout("unit-test-key"), requests.ConnectionError("unit-test-key")])
def test_network_failure(client, monkeypatch, error):
    monkeypatch.setattr(requests, "post", Mock(side_effect=error))
    with pytest.raises(AgentError, match="网络请求失败") as caught:
        client.describe("Market Agent", {"latest": 10})
    assert "unit-test-key" not in str(caught.value)


@pytest.mark.parametrize("result", [
    "not json", "{}", "[]", "null",
    json.dumps({"summary": "", "evidence": ["latest"]}),
    json.dumps({"summary": "x" * 1001, "evidence": ["latest"]}),
    json.dumps({"summary": "测试", "evidence": []}),
    json.dumps({"summary": "测试", "evidence": ["invented"]}),
    json.dumps({"summary": "测试", "evidence": ["latest.close"]}),
    json.dumps({"summary": "测试", "evidence": "latest"}),
    json.dumps({"summary": "测试", "evidence": [[]]}),
])
def test_invalid_output(client, monkeypatch, result):
    monkeypatch.setattr(requests, "post", Mock(return_value=response(result)))
    with pytest.raises(AgentError, match="返回格式无效"):
        client.describe("Market Agent", {"latest": 10})


@pytest.mark.parametrize("finish", ["length", "content_filter", None])
def test_truncated_output(client, monkeypatch, finish):
    content = json.dumps({"summary": "测试", "evidence": ["latest"]})
    monkeypatch.setattr(requests, "post", Mock(return_value=response(content, finish=finish)))
    with pytest.raises(AgentError):
        client.describe("Market Agent", {"latest": 10})


@pytest.mark.parametrize("payload", [{}, {"choices": []}, {"choices": [None]}, {"choices": [{}]}])
def test_invalid_envelope(client, monkeypatch, payload):
    monkeypatch.setattr(requests, "post", Mock(return_value=Mock(status_code=200, json=Mock(return_value=payload))))
    with pytest.raises(AgentError):
        client.describe("Market Agent", {"latest": 10})
