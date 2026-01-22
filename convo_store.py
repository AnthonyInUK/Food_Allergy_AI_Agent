import os
import json
import uuid
from datetime import datetime
from typing import List, Optional


CONVO_FILE = os.path.join("data", "conversations.json")


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


def load_conversations() -> List[dict]:
    return _read_store()


def create_conversation(title: str = "New conversation", messages: List[dict] = None) -> dict:
    convos = _read_store()
    conv = {
        "id": str(uuid.uuid4()),
        "title": title,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "updated_at": datetime.utcnow().isoformat() + "Z",
        "pinned": False,
        "messages": messages or []
    }
    convos.insert(0, conv)
    _write_store(convos)
    return conv


def save_conversation(conv: dict):
    convos = _read_store()
    for i, c in enumerate(convos):
        if c.get("id") == conv.get("id"):
            conv["updated_at"] = datetime.utcnow().isoformat() + "Z"
            convos[i] = conv
            _write_store(convos)
            return
    # not found -> append
    conv["updated_at"] = datetime.utcnow().isoformat() + "Z"
    convos.insert(0, conv)
    _write_store(convos)


def delete_conversation(conv_id: str):
    convos = _read_store()
    convos = [c for c in convos if c.get("id") != conv_id]
    _write_store(convos)


def find_conversation_by_id(conv_id: str) -> Optional[dict]:
    convos = _read_store()
    for c in convos:
        if c.get("id") == conv_id:
            return c
    return None
