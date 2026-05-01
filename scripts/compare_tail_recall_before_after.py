#!/usr/bin/env python3
import csv
import json
import random
import re
import sys
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from graph_logic import get_vectorstore, _build_retrieval_query


def _old_build_query(question: str) -> str:
    brand_alias = {
        "李锦记": "Lee Kum Kee",
        "海天": "Haday",
        "康师傅": "Master Kong",
    }
    alias_hit = next((en for zh, en in brand_alias.items() if zh in question), "")
    return f"{question} {alias_hit}".strip() if alias_hit else question


def _tokens(text: str, min_len: int) -> List[str]:
    return [
        t
        for t in re.split(r"[^a-zA-Z0-9\u4e00-\u9fff]+", (text or "").lower())
        if len(t) >= min_len
    ]


def _is_hit(name: str, brand: str, docs: List[str]) -> bool:
    blob = "\n".join(docs[:5]).lower()
    name_tokens = _tokens(name, 4)
    brand_tokens = _tokens(brand, 3)
    return any(t in blob for t in name_tokens) or any(t in blob for t in brand_tokens)


def _load_tail_cases(source_csv: Path, n: int = 80) -> List[dict]:
    rows = []
    with source_csv.open("r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            name = (r.get("name") or "").strip()
            brand = (r.get("brand") or "").strip()
            ingredients = (r.get("ingredients") or "").strip()
            if not (name and ingredients):
                continue
            name_low = name.lower()
            brand_low = brand.lower()
            is_multilang = bool(re.search(r"[^\x00-\x7f]", name + brand))
            is_generic = any(k in name_low for k in ["ketchup", "sauce", "noodle", "soup", "beer"])
            has_punct = any(ch in (brand + name) for ch in [",", "&", "/", "(", ")"])
            if is_multilang or is_generic or has_punct:
                rows.append({"name": name, "brand": brand})
    random.seed(42)
    random.shuffle(rows)
    return rows[: min(n, len(rows))]


def _query_docs(vs, query: str, k: int = 5) -> List[str]:
    docs = vs.similarity_search(query, k=k)
    return [d.page_content for d in docs]


def main() -> None:
    source = Path("data/food_products_summary.csv")
    cases = _load_tail_cases(source, n=80)
    vs = get_vectorstore()

    old_hits = 0
    new_hits = 0
    improved_examples: List[dict] = []

    for c in cases:
        q = f"{c['brand']} {c['name']} 的成分是什么？".strip()
        old_q = _old_build_query(q)
        new_q, _ = _build_retrieval_query(q)
        old_docs = _query_docs(vs, old_q, 5)
        new_docs = _query_docs(vs, new_q or q, 5)

        old_ok = _is_hit(c["name"], c["brand"], old_docs)
        new_ok = _is_hit(c["name"], c["brand"], new_docs)
        old_hits += 1 if old_ok else 0
        new_hits += 1 if new_ok else 0

        if (not old_ok) and new_ok and len(improved_examples) < 10:
            improved_examples.append(
                {
                    "question": q[:120],
                    "old_query": old_q[:120],
                    "new_query": (new_q or q)[:120],
                    "new_top1": (new_docs[0][:120] if new_docs else ""),
                }
            )

    total = max(len(cases), 1)
    result = {
        "samples": len(cases),
        "recall_at_5_before": round(old_hits / total, 4),
        "recall_at_5_after": round(new_hits / total, 4),
        "delta": round((new_hits - old_hits) / total, 4),
        "improved_examples": improved_examples,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

