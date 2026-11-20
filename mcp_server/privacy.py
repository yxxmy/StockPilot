"""服务端权限和返回内容脱敏；不将配置对象作为工具返回值。"""

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re

from mcp.server.fastmcp.exceptions import ToolError

from memory.store import DEFAULT_DB_PATH


ROOT = Path(__file__).resolve().parents[1]
MAX_PAYLOAD_BYTES = 100000


@dataclass(frozen=True)
class ServerConfig:
    db_path: Path
    index_dir: Path
    allow_reports: bool = False
    allow_documents: bool = False

    @classmethod
    def from_env(cls):
        return cls(
            Path(os.environ.get("STOCKPILOT_MCP_DB_PATH", os.environ.get("STOCKPILOT_DB_PATH", str(DEFAULT_DB_PATH)))).resolve(),
            Path(os.environ.get("STOCKPILOT_MCP_INDEX_DIR", str(ROOT / "data" / "rag"))).resolve(),
            os.environ.get("STOCKPILOT_MCP_ALLOW_REPORTS") == "1",
            os.environ.get("STOCKPILOT_MCP_ALLOW_DOCUMENTS") == "1",
        )


class PublicOutput:
    def __init__(self, config):
        # 仅用于识别并删除敏感值，绝不输出环境配置或认证信息。
        self.hidden = [str(config.db_path), str(config.index_dir), str(ROOT)]
        self.hidden += [value for key, value in os.environ.items()
                        if key.upper().endswith(("_API_KEY", "_TOKEN", "_SECRET")) and len(value) >= 8]
        self.hidden.sort(key=len, reverse=True)

    def text(self, value):
        for secret in self.hidden:
            value = value.replace(secret, "[敏感配置已隐藏]")
        value = re.sub(r"\bsk-[A-Za-z0-9_-]{12,}\b", "[密钥已隐藏]", value)
        value = re.sub(r"(?i)Bearer\s+[^\s\"'<>]+", "[认证信息已隐藏]", value)
        value = re.sub(r"[A-Za-z]:[\\/][^\r\n\"<>]*", "[本地路径已隐藏]", value)
        value = re.sub(r"file://[^\s\"<>]+|(?<![:/\w])/(?:home|Users|tmp|var|etc|mnt)/[^\s\"<>]+", "[本地路径已隐藏]", value)
        return value

    def publish(self, model):
        data = model.model_dump(mode="json")
        changed = False

        def clean(value):
            nonlocal changed
            if isinstance(value, str):
                public = self.text(value)
                changed = changed or public != value
                return public
            if isinstance(value, list):
                return [clean(item) for item in value]
            if isinstance(value, dict):
                return {key: clean(item) for key, item in value.items()}
            return value

        data = clean(data)
        data["redacted"] = changed or data.get("redacted", False)
        if len(json.dumps(data, ensure_ascii=False, allow_nan=False).encode("utf-8")) > MAX_PAYLOAD_BYTES:
            raise ToolError("返回内容超过大小限制，请缩小limit、top_k或max_chars。")
        return type(model).model_validate(data)
