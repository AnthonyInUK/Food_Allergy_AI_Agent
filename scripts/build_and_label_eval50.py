#!/usr/bin/env python3
import argparse
import csv
import json
import random
from pathlib import Path

import requests
from dotenv import load_dotenv

from auto_label_manual_eval import _normalize_prediction
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


def _load_rows(csv_path: Path, limit: int) -> list[dict]:
    rows = []
    with csv_path.open("r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            name = (r.get("name") or "").strip()
            brand = (r.get("brand") or "").strip()
            ing = (r.get("ingredients") or "").strip()
            if name and ing:
                rows.append({"name": name, "brand": brand, "ingredients": ing})
    random.shuffle(rows)
    return rows[:limit]


def _build_question(row: dict) -> str:
    brand = (row.get("brand") or "").strip()
    name = (row.get("name") or "").strip()
    if brand:
        return f"{brand} {name} 的成分是什么？"
    return f"{name} 的成分是什么？"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build 50 eval rows and pre-label with LLM.")
    parser.add_argument("--source", default="data/food_products_summary.csv")
    parser.add_argument("--output", default="eval/manual_eval_50_labeled.csv")
    parser.add_argument("--api", default="http://127.0.0.1:8001")
    parser.add_argument("--size", type=int, default=50)
    parser.add_argument("--model", default="gpt-4o-mini")
    args = parser.parse_args()

    load_dotenv(override=True)

    source = Path(args.source)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    rows = _load_rows(source, args.size)
    if not rows:
        raise RuntimeError("No rows loaded from source CSV.")

    conv = requests.post(f"{args.api}/api/conversations", timeout=30).json()["conversation"]["id"]

    llm = ChatOpenAI(model=args.model, temperature=0)
    prompt = ChatPromptTemplate.from_template(
        """You are a strict evaluator for a food-allergy QA product.
Return JSON only with keys: label, error_type, notes, review_priority.

Allowed label values: correct, partial, wrong, proper_refusal
Allowed error_type values: none, retrieval_miss, hallucination, format_or_language, incomplete, timeout_or_iteration, other
review_priority: high | medium | low

Question:
{question}

Expected key facts:
{expected_key_facts}

Model answer:
{model_answer}
"""
    )
    chain = prompt | llm

    out_rows = []
    for i, r in enumerate(rows, start=1):
        q = _build_question(r)
        expected = (r.get("ingredients") or "")[:180]
        try:
            resp = requests.post(
                f"{args.api}/api/chat",
                json={"text": q, "conversation_id": conv},
                timeout=120,
            )
            if resp.status_code == 200:
                answer = (resp.json().get("response") or "").strip()
            else:
                answer = f"[HTTP {resp.status_code}] {resp.text[:200]}"
        except Exception as e:
            answer = f"[REQUEST_EXCEPTION] {e}"

        label_pred = chain.invoke(
            {
                "question": q,
                "expected_key_facts": expected,
                "model_answer": answer,
            }
        )
        pred_text = label_pred.content if hasattr(label_pred, "content") else str(label_pred)
        pred = _normalize_prediction(pred_text)

        out_rows.append(
            {
                "id": f"case_{i:03d}",
                "question": q,
                "expected_key_facts": expected,
                "model_answer": answer,
                "label": pred["label"],
                "error_type": pred["error_type"],
                "notes": pred["notes"],
                "review_priority": pred["review_priority"],
            }
        )

    fields = [
        "id",
        "question",
        "expected_key_facts",
        "model_answer",
        "label",
        "error_type",
        "notes",
        "review_priority",
    ]
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out_rows)

    print(
        json.dumps(
            {
                "output": str(output),
                "rows_written": len(out_rows),
                "api": args.api,
                "model": args.model,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

