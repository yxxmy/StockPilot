"""使用标准库 sqlite3 保存与查询报告，连接用完即关闭。"""

from contextlib import contextmanager
from datetime import date, datetime
import json
import os
from pathlib import Path
import re
import sqlite3
from uuid import uuid4
from zoneinfo import ZoneInfo


DEFAULT_DB_PATH = Path(__file__).resolve().parents[1] / "data" / "stockpilot.sqlite3"


class MemoryStoreError(RuntimeError):
    """报告校验或数据库操作失败，不包含密钥或 SQL 参数值。"""


def _symbol(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9]{6}", value):
        raise ValueError("股票代码必须是六位数字字符串")
    return value


def _source_metadata(state):
    market = state["market_facts"]
    sources = {"market": {
        "source": market.get("source", "AKShare / 东方财富"), "adjust": "不复权", "volume_unit": "手",
        "source_warning": market.get("source_warning", ""),
        **{key: market[key] for key in ["requested_range", "actual_range", "record_count", "limitations"] if key in market},
    }}
    # 只允许来源与覆盖信息进入元数据，绝不序列化整个 State 或客户端。
    for name, value in [("financial", state.get("financial_facts")),
                        ("news", state.get("news_facts", {}).get("news")),
                        ("notices", state.get("news_facts", {}).get("notices"))]:
        if value is None:
            continue
        sources[name] = {key: value[key] for key in [
            "status", "source", "source_url", "fetched_at", "requested_range", "coverage",
            "upstream_count", "matched_count", "excluded_undated_count", "units", "error",
        ] if key in value}
        sources[name]["references"] = [{key: item[key] for key in [
            "report_date", "published_date", "title", "published_at", "publisher", "url",
        ] if key in item} for item in value.get("records", [])]
    if "rag_facts" in state:
        rag = state["rag_facts"]
        sources["rag"] = {key: rag[key] for key in ["question", "status", "message", "score_threshold", "limitations", "filters"] if key in rag}
        sources["rag"]["references"] = [{key: reference[key] for key in [
            "ref_id", "document_id", "document_name", "page", "chunk_id", "score", "metadata", "metadata_status", "official_pdf_url",
        ] if key in reference} for reference in rag.get("sources", [])]
    return sources


class ReportStore:
    def __init__(self, db_path=None, *, read_only=False):
        configured = db_path if db_path is not None else os.environ.get("STOCKPILOT_DB_PATH", DEFAULT_DB_PATH)
        if not str(configured).strip() or str(configured) == ":memory:":
            raise MemoryStoreError("请指定用于持久化的 SQLite 文件路径")
        self.path = Path(configured).expanduser().resolve()
        self.read_only = read_only
        if read_only:
            # MCP只读模式不创建文件、目录、表或索引，写入由SQLite自身禁止。
            if not self.path.is_file():
                raise MemoryStoreError("SQLite只读数据库不存在")
            return
        with self._connection() as connection:
            # 表结构和索引为固定 SQL；所有运行时数据使用 ? 参数绑定。
            connection.execute("""
                CREATE TABLE IF NOT EXISTS analysis_reports (
                    id TEXT PRIMARY KEY NOT NULL,
                    symbol TEXT NOT NULL,
                    generated_at TEXT NOT NULL,
                    market_end_date TEXT NOT NULL,
                    mode TEXT NOT NULL CHECK(mode IN ('local', 'deepseek')),
                    include_research INTEGER NOT NULL CHECK(include_research IN (0, 1)),
                    report TEXT NOT NULL CHECK(length(trim(report)) > 0),
                    sources_json TEXT NOT NULL
                )
            """)
            connection.execute("CREATE INDEX IF NOT EXISTS reports_symbol_time ON analysis_reports(symbol, generated_at DESC)")
            connection.execute("CREATE INDEX IF NOT EXISTS reports_time ON analysis_reports(generated_at DESC)")

    @contextmanager
    def _connection(self):
        connection = None
        try:
            if self.read_only:
                connection = sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True,
                                             timeout=5, isolation_level="DEFERRED")
            else:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                connection = sqlite3.connect(self.path, timeout=5, isolation_level="DEFERRED")
            connection.row_factory = sqlite3.Row
            # 正常退出自动提交；执行或提交异常时回滚。with 不负责关闭连接。
            with connection:
                yield connection
        except (sqlite3.Error, OSError):
            raise MemoryStoreError("SQLite 操作失败，请检查文件权限、数据库格式或锁占用；未确认保存成功") from None
        finally:
            if connection is not None:
                connection.close()

    def save_report(self, state: dict, *, mode: str, include_research=False) -> str:
        """只接受成功结束的工作流结果，返回每次独立的 UUID。"""
        if state.get("error") or not isinstance(state.get("report"), str) or not state["report"].strip():
            raise MemoryStoreError("失败或空报告不能保存为成功记录")
        trace = state.get("trace", [])
        if "report" not in trace or not trace or trace[-1] != "supervisor -> end":
            raise MemoryStoreError("工作流尚未成功结束，不能保存")
        if mode not in {"local", "deepseek"} or not isinstance(include_research, bool):
            raise ValueError("分析模式必须为 local/deepseek，研究标记必须为布尔值")
        try:
            symbol = _symbol(state["symbol"])
            actual_range = state["market_facts"]["actual_range"]
            if not isinstance(actual_range, list) or len(actual_range) != 2:
                raise ValueError("日期范围不合法")
            first, last = [date.fromisoformat(value) for value in actual_range]
            if first > last:
                raise ValueError("日期范围倒置")
            sources_json = json.dumps(_source_metadata(state), ensure_ascii=False, allow_nan=False)
        except (KeyError, TypeError, ValueError, AttributeError):
            raise MemoryStoreError("成功报告缺少合法的股票代码、实际日期或来源信息") from None
        # 凭证不属于报告数据；若报告意外包含当前密钥，拒绝写入而非改写原报告。
        secret = os.environ.get("DEEPSEEK_API_KEY", "").strip()
        if secret and (secret in state["report"] or secret in sources_json):
            raise MemoryStoreError("报告或来源信息包含认证信息，拒绝保存")
        report_id = str(uuid4())
        generated_at = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="microseconds")
        with self._connection() as connection:
            connection.execute("""
                INSERT INTO analysis_reports
                (id, symbol, generated_at, market_end_date, mode, include_research, report, sources_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (report_id, symbol, generated_at, last.isoformat(), mode,
                  int(include_research), state["report"], sources_json))
        # 只有事务成功提交后，调用方才会得到 ID。
        return report_id

    def list_reports(self, symbol=None, *, limit=10) -> list[dict]:
        """最近记录只返回摘要，不加载所有报告正文。"""
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
            raise ValueError("limit 必须是1到100的整数")
        columns = "id, symbol, generated_at, market_end_date, mode, include_research"
        with self._connection() as connection:
            if symbol is None:
                rows = connection.execute(
                    "SELECT " + columns + " FROM analysis_reports ORDER BY generated_at DESC, rowid DESC LIMIT ?", (limit,),
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT " + columns + " FROM analysis_reports WHERE symbol = ? ORDER BY generated_at DESC, rowid DESC LIMIT ?",
                    (_symbol(symbol), limit),
                ).fetchall()
        return [{**dict(row), "include_research": bool(row["include_research"])} for row in rows]

    def get_report(self, report_id: str) -> dict | None:
        if not isinstance(report_id, str) or not report_id.strip():
            raise ValueError("报告ID不能为空")
        with self._connection() as connection:
            row = connection.execute("SELECT * FROM analysis_reports WHERE id = ?", (report_id,)).fetchone()
        if row is None:
            return None
        result = dict(row)
        try:
            result["sources"] = json.loads(result.pop("sources_json"))
            # JSON语法正确不代表来源结构正确，避免把损坏数据当成正常记录。
            if not isinstance(result["sources"], dict) or not isinstance(result["sources"].get("market"), dict):
                raise ValueError("来源结构不合法")
        except (TypeError, ValueError):
            raise MemoryStoreError("报告来源信息无法解析，数据库可能已损坏") from None
        result["include_research"] = bool(result["include_research"])
        return result
