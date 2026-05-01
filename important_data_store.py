import json
import os
import time
from typing import Optional

import streamlit as st
from langchain_community.chat_message_histories import RedisChatMessageHistory

from app_config import get_runtime_settings

_settings = get_runtime_settings()
IMPORTANT_DATA_TTL_SECONDS = _settings.important_data_ttl_seconds


def _resolve_session_id() -> str:
    conversation_id = st.session_state.get("current_conversation_id")
    return str(conversation_id) if conversation_id else "default"


def _get_redis_history(session_id: Optional[str] = None) -> Optional[RedisChatMessageHistory]:
    if not _settings.enable_important_qa_redis:
        return None
    redis_url = os.getenv("REDIS_URL", "").strip()
    if not redis_url:
        return None

    sid = session_id or _resolve_session_id()
    try:
        history = RedisChatMessageHistory(
            session_id=sid,
            url=redis_url,
            key_prefix="important_data:",
            ttl=IMPORTANT_DATA_TTL_SECONDS,
        )
        # 连接探测，避免构造后延迟报错
        history.redis_client.ping()
        return history
    except Exception as e:
        print(f"Redis important_data unavailable: {e}")
        return None


def record_important_qa(question: str, answer: str, source: str = "chat") -> None:
    """将关键问答写入 RedisChatMessageHistory（TTL 默认 3 天）。"""
    history = _get_redis_history()
    if history is None:
        return

    meta = {
        "source": source,
        "ts": int(time.time()),
    }

    compact_q = (question or "")[:1200]
    compact_a = (answer or "")[:2400]

    history.add_user_message(
        f"[IMPORTANT_QA]{json.dumps(meta, ensure_ascii=False)}\nQ: {compact_q}")
    history.add_ai_message(f"A: {compact_a}")


def move_to_permanent_storage(question: str, answer: str, source: str = "chat") -> bool:
    """将数据从 Redis 转移到永久存储（data/conversations.json）。"""
    try:
        from convo_store import create_conversation, save_conversation

        # 创建一个标记为“收藏”的对话记录
        title = f"★ Saved: {question[:30]}..."
        messages = [
            {"role": "user", "text": question, "ts": time.time()},
            {"role": "assistant", "text": f"【Saved from Redis】\n\n{answer}",
                "ts": time.time(), "saved": True}
        ]

        conv = create_conversation(title=title)
        conv["messages"] = messages
        save_conversation(conv)
        return True
    except Exception as e:
        print(f"Failed to move to permanent storage: {e}")
        return False
