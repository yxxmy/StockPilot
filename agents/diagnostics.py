"""只暴露固定类别、节点和真实结构轨迹，不携带State、异常正文或配置。"""

import os
import re


def safe_error_text(value):
    """兼容旧异常文字，同时删除认证值与本地路径；不记录环境变量。"""
    text = str(value)
    for key, secret in os.environ.items():
        if key.upper().endswith(("_API_KEY", "_TOKEN", "_SECRET")) and len(secret) >= 8:
            text = text.replace(secret, "[认证信息已隐藏]")
    text = re.sub(r"\bsk-[A-Za-z0-9_-]{12,}\b|(?i:Bearer)\s+[^\s]+", "[认证信息已隐藏]", text)
    return re.sub(r"[A-Za-z]:[\\/][^\r\n\"<>]*|(?<![:/\w])/(?:home|Users|tmp|var|etc|mnt)/[^\s\"<>]+", "[本地路径已隐藏]", text)

MESSAGES = {
    "configuration": "模型配置不可用，请在本机检查环境变量。",
    "market_data": "行情获取或数据校验失败，请检查日期、代码或上游服务。",
    "market_connection": "主行情源连接失败，腾讯备用源也不可用；请稍后再试。",
    "market_timeout": "主行情源请求超时，腾讯备用源也未取得有效数据；本次已停止。",
    "market_rate_limit": "东方财富连接被断开（SDK提示疑似限流），备用源也不可用；请暂停重复提交。",
    "market_empty": "行情源未返回指定区间数据，请核对代码、交易日与上市日期。",
    "market_validation": "行情返回字段或数值校验失败，未使用异常数据生成报告。",
    "model_network": "模型网络请求失败。",
    "model_http": "模型服务返回失败状态，请检查额度、配置或限流。",
    "model_output": "模型输出未通过格式或事实校验。",
    "model_citation": "报告引用未通过校验，未保存为成功报告。",
    "storage_error": "报告持久化失败，请检查本机数据库及历史记录后再手动重试。",
    "analysis_error": "分析未成功完成。",
}
NODES = {"market", "financial", "news", "risk", "report", "rag_retrieval", "configuration", "persistence"}
STEPS = {"supervisor -> " + name for name in NODES | {"end"}}
STEPS |= NODES | {name + ": error" for name in NODES}


def diagnostic(node, category, trace):
    node = node if isinstance(node, str) and node in NODES else "unknown"
    category = category if isinstance(category, str) and category in MESSAGES else "analysis_error"
    trace = trace if isinstance(trace, (list, tuple)) else []
    return {"failed_node": node, "category": category, "message": MESSAGES[category],
            "trace": [step for step in trace if isinstance(step, str) and step in STEPS]}
