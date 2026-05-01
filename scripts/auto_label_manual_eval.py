#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


ALLOWED_LABELS = {"correct", "partial", "wrong", "proper_refusal"}
ALLOWED_ERROR_TYPES = {
    "none",
    "retrieval_miss",
    "hallucination",
    "format_or_language",
    "incomplete",
    "timeout_or_iteration",
    "other",
}


def _to_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _normalize_prediction(raw: str) -> dict:
    try:
        payload = json.loads(raw)
    except Exception:
        return {
            "label": "partial",
            "error_type": "other",
            "notes": f"json_parse_failed: {raw[:200]}",
            "review_priority": "high",
        }

    label = str(payload.get("label", "partial")).strip()
    if label not in ALLOWED_LABELS:
        label = "partial"

    error_type = str(payload.get("error_type", "other")).strip()
    if error_type not in ALLOWED_ERROR_TYPES:
        error_type = "other"

    notes = str(payload.get("notes", "")).strip()
    review_priority = str(payload.get("review_priority", "medium")).strip().lower()
    if review_priority not in {"high", "medium", "low"}:
        review_priority = "medium"

    return {
        "label": label,
        "error_type": error_type,
        "notes": notes,
        "review_priority": review_priority,
    }


def _should_skip_row(row: dict, overwrite: bool) -> bool:
    if not (row.get("question") or "").strip():
        return True
    if not (row.get("model_answer") or "").strip():
        return True
    if overwrite:
        return False
    return bool((row.get("label") or "").strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM pre-label for manual eval CSV.")
    parser.add_argument(
        "--input",
        default="eval/manual_eval_50_template.csv",
        help="Input CSV path.",
    )
    parser.add_argument(
        "--output",
        default="eval/manual_eval_50_labeled.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--overwrite",
        default="false",
        help="Overwrite rows that already have label: true/false.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model for pre-labeling.",
    )
    args = parser.parse_args()

    load_dotenv(override=True)
    overwrite = _to_bool(args.overwrite)
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    llm = ChatOpenAI(model=args.model, temperature=0)
    prompt = ChatPromptTemplate.from_template(
        """You are a strict evaluator for a food-allergy QA product.
Return JSON only with keys: label, error_type, notes, review_priority.

Allowed label values:
- correct
- partial
- wrong
- proper_refusal

Allowed error_type values:
- none
- retrieval_miss
- hallucination
- format_or_language
- incomplete
- timeout_or_iteration
- other

review_priority must be one of: high, medium, low.

Scoring rules:
- correct: key facts covered, no meaningful contradiction.
- partial: partly correct but misses important facts.
- wrong: factual contradiction or fabricated key facts.
- proper_refusal: clearly states uncertainty/no-data without fabrication.
- timeout text like 'max iterations' should usually be wrong + timeout_or_iteration.

Question:
{question}

Expected key facts:
{expected_key_facts}

Model answer:
{model_answer}
"""
    )
    chain = prompt | llm

    with input_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = list(reader.fieldnames or [])

    if "review_priority" not in fields:
        fields.append("review_priority")

    total = 0
    labeled = 0
    for row in rows:
        total += 1
        if _should_skip_row(row, overwrite):
            continue
        result = chain.invoke(
            {
                "question": row.get("question", ""),
                "expected_key_facts": row.get("expected_key_facts", ""),
                "model_answer": row.get("model_answer", ""),
            }
        )
        text = result.content if hasattr(result, "content") else str(result)
        pred = _normalize_prediction(text)
        row["label"] = pred["label"]
        row["error_type"] = pred["error_type"]
        base_note = (row.get("notes") or "").strip()
        row["notes"] = pred["notes"] if not base_note else f"{base_note} | {pred['notes']}"
        row["review_priority"] = pred["review_priority"]
        labeled += 1

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(
        json.dumps(
            {
                "input": str(input_path),
                "output": str(output_path),
                "rows_total": total,
                "rows_labeled": labeled,
                "overwrite": overwrite,
                "model": args.model,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

