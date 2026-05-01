"""
LangSmith 追踪配置（与 LangChain / LangGraph 集成）。

文档: https://docs.smith.langchain.com/
控制台: https://smith.langchain.com/

环境变量（.env）:
  LANGSMITH_TRACING=true
  LANGSMITH_API_KEY=lsv2_pt_...
  LANGSMITH_PROJECT=food-ai-agent   # 可选，默认项目亦可

兼容旧变量: LANGCHAIN_API_KEY、LANGCHAIN_PROJECT、LANGCHAIN_TRACING_V2
"""

from __future__ import annotations

import os


def configure_langsmith_tracing_compat() -> None:
    """
    开启 LANGSMITH_TRACING 时，为 LangChain 兼容层设置 LANGCHAIN_TRACING_V2。
    在首次 import graph_logic / 调用 Runnable 之前调用一次即可（可重复调用）。
    """
    v = (os.getenv("LANGSMITH_TRACING") or "").strip().lower()
    if v in ("true", "1", "yes", "on"):
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")


def log_langsmith_startup_status() -> None:
    """启动时打印一行状态，不输出任何密钥。"""
    configure_langsmith_tracing_compat()
    tracing = (os.getenv("LANGSMITH_TRACING") or "").strip().lower() in ("true", "1", "yes", "on")
    key = (os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY") or "").strip()
    proj = (os.getenv("LANGSMITH_PROJECT") or os.getenv("LANGCHAIN_PROJECT") or "").strip() or "default"

    if tracing and key:
        print(f"📋 LangSmith: tracing enabled | project={proj}", flush=True)
    elif key and not tracing:
        print(
            "ℹ️ LangSmith: API key set; set LANGSMITH_TRACING=true to upload traces",
            flush=True,
        )
    elif tracing and not key:
        print(
            "⚠️ LangSmith: LANGSMITH_TRACING=true but missing LANGSMITH_API_KEY — traces will not upload",
            flush=True,
        )
    else:
        print(
            "ℹ️ LangSmith: tracing off (set LANGSMITH_TRACING=true and LANGSMITH_API_KEY to enable)",
            flush=True,
        )
