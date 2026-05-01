import json
import os
import uuid
from datetime import datetime
from typing import List, Optional

from sqlalchemy import text

from app_config import get_runtime_settings
from db import get_engine


CONVO_FILE = os.path.join("data", "conversations.json")
_pg_ready = False
_pg_checked = False
_pg_bootstrap_done = False
_settings = get_runtime_settings()


def _ensure_pg_schema() -> bool:
    global _pg_ready, _pg_checked
    if not _settings.enable_pg_conversation_checkpoint:
        return False
    if _pg_checked:
        return _pg_ready
    _pg_checked = True
    try:
        engine = get_engine()
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS conversations (
                        id TEXT PRIMARY KEY,
                        title TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        pinned BOOLEAN NOT NULL DEFAULT FALSE,
                        messages JSONB NOT NULL DEFAULT '[]'::jsonb
                    )
                    """
                )
            )
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_conversations_updated_at ON conversations(updated_at DESC)"))
        _pg_ready = True
    except Exception:
        _pg_ready = False
    return _pg_ready


def _read_store() -> List[dict]:
    try:
        with open(CONVO_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return []


def _write_store(data: List[dict]):
    os.makedirs(os.path.dirname(CONVO_FILE), exist_ok=True)
    with open(CONVO_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _load_conversations_pg() -> List[dict]:
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT id, title, created_at, updated_at, pinned, messages
                FROM conversations
                ORDER BY updated_at DESC
                """
            )
        ).fetchall()
    return [
        {
            "id": r[0],
            "title": r[1],
            "created_at": r[2],
            "updated_at": r[3],
            "pinned": bool(r[4]),
            "messages": r[5] or [],
        }
        for r in rows
    ]


def _bootstrap_pg_from_file():
    global _pg_bootstrap_done
    if _pg_bootstrap_done:
        return
    _pg_bootstrap_done = True
    file_data = _read_store()
    if not file_data:
        return
    engine = get_engine()
    with engine.begin() as conn:
        row = conn.execute(text("SELECT COUNT(1) FROM conversations")).fetchone()
        count = int(row[0]) if row else 0
        if count > 0:
            return
        for conv in file_data:
            conn.execute(
                text(
                    """
                    INSERT INTO conversations (id, title, created_at, updated_at, pinned, messages)
                    VALUES (:id, :title, :created_at, :updated_at, :pinned, CAST(:messages AS JSONB))
                    ON CONFLICT (id) DO NOTHING
                    """
                ),
                {
                    "id": conv.get("id", str(uuid.uuid4())),
                    "title": conv.get("title", "New conversation"),
                    "created_at": conv.get("created_at", datetime.utcnow().isoformat() + "Z"),
                    "updated_at": conv.get("updated_at", datetime.utcnow().isoformat() + "Z"),
                    "pinned": bool(conv.get("pinned", False)),
                    "messages": json.dumps(conv.get("messages", []), ensure_ascii=False),
                },
            )


def load_conversations() -> List[dict]:
    if _ensure_pg_schema():
        try:
            _bootstrap_pg_from_file()
            return _load_conversations_pg()
        except Exception:
            pass
    return _read_store()


def create_conversation(title: str = "New conversation", messages: List[dict] = None) -> dict:
    conv = {
        "id": str(uuid.uuid4()),
        "title": title,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "updated_at": datetime.utcnow().isoformat() + "Z",
        "pinned": False,
        "messages": messages or [],
    }
    if _ensure_pg_schema():
        try:
            engine = get_engine()
            with engine.begin() as conn:
                conn.execute(
                    text(
                        """
                        INSERT INTO conversations (id, title, created_at, updated_at, pinned, messages)
                        VALUES (:id, :title, :created_at, :updated_at, :pinned, CAST(:messages AS JSONB))
                        """
                    ),
                    {**conv, "messages": json.dumps(conv["messages"], ensure_ascii=False)},
                )
            return conv
        except Exception:
            pass
    convos = _read_store()
    convos.insert(0, conv)
    _write_store(convos)
    return conv


def save_conversation(conv: dict):
    conv["updated_at"] = datetime.utcnow().isoformat() + "Z"
    if _ensure_pg_schema():
        try:
            engine = get_engine()
            with engine.begin() as conn:
                conn.execute(
                    text(
                        """
                        INSERT INTO conversations (id, title, created_at, updated_at, pinned, messages)
                        VALUES (:id, :title, :created_at, :updated_at, :pinned, CAST(:messages AS JSONB))
                        ON CONFLICT (id) DO UPDATE SET
                            title = EXCLUDED.title,
                            updated_at = EXCLUDED.updated_at,
                            pinned = EXCLUDED.pinned,
                            messages = EXCLUDED.messages
                        """
                    ),
                    {
                        "id": conv.get("id"),
                        "title": conv.get("title", "New conversation"),
                        "created_at": conv.get("created_at", datetime.utcnow().isoformat() + "Z"),
                        "updated_at": conv.get("updated_at"),
                        "pinned": bool(conv.get("pinned", False)),
                        "messages": json.dumps(conv.get("messages", []), ensure_ascii=False),
                    },
                )
            return
        except Exception:
            pass
    convos = _read_store()
    for i, c in enumerate(convos):
        if c.get("id") == conv.get("id"):
            convos[i] = conv
            _write_store(convos)
            return
    convos.insert(0, conv)
    _write_store(convos)


def delete_conversation(conv_id: str):
    if _ensure_pg_schema():
        engine = get_engine()
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM conversations WHERE id = :id"), {"id": conv_id})
    convos = _read_store()
    convos = [c for c in convos if c.get("id") != conv_id]
    _write_store(convos)


def find_conversation_by_id(conv_id: str) -> Optional[dict]:
    if _ensure_pg_schema():
        try:
            engine = get_engine()
            with engine.connect() as conn:
                row = conn.execute(
                    text(
                        """
                        SELECT id, title, created_at, updated_at, pinned, messages
                        FROM conversations
                        WHERE id = :id
                        """
                    ),
                    {"id": conv_id},
                ).fetchone()
            if row:
                return {
                    "id": row[0],
                    "title": row[1],
                    "created_at": row[2],
                    "updated_at": row[3],
                    "pinned": bool(row[4]),
                    "messages": row[5] or [],
                }
            return None
        except Exception:
            pass
    convos = _read_store()
    for c in convos:
        if c.get("id") == conv_id:
            return c
    return None
