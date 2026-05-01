#!/usr/bin/env python3
"""
Offline index: download product packaging images from PostgreSQL image_url,
encode with CLIP (sentence-transformers), upsert into Chroma collection product_images.

Usage (from repo root, venv activated):
  python scripts/index_product_images.py --limit 500
  python scripts/index_product_images.py   # all rows with image_url
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import chromadb
from PIL import Image
from sqlalchemy import text

from app_config import get_runtime_settings
from db import get_engine

USER_AGENT = "FoodAIAgent-ImageIndexer/1.0 (+https://github.com)"


def fetch_image(url: str, timeout_sec: int) -> Optional[bytes]:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            return resp.read()
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return None


def load_rows(engine, limit: int, offset: int) -> List[Tuple[str, str, str, str, str]]:
    lim_sql = "LIMIT :lim OFFSET :off" if limit > 0 else ""
    sql = f"""
        SELECT id, coalesce(name,''), coalesce(brand,''), coalesce(ingredients,''), coalesce(image_url,'')
        FROM products
        WHERE image_url IS NOT NULL AND btrim(image_url) <> ''
        ORDER BY id
        {lim_sql}
    """
    with engine.connect() as conn:
        if limit > 0:
            rows = conn.execute(text(sql), {"lim": limit, "off": offset}).fetchall()
        else:
            rows = conn.execute(
                text(
                    """
                    SELECT id, coalesce(name,''), coalesce(brand,''), coalesce(ingredients,''), coalesce(image_url,'')
                    FROM products
                    WHERE image_url IS NOT NULL AND btrim(image_url) <> ''
                    ORDER BY id
                    """
                )
            ).fetchall()
    return [(str(r[0]), r[1], r[2], r[3], r[4]) for r in rows]


def build_document(pid: str, brand: str, name: str, ingredients: str) -> str:
    body = (ingredients or "")[:2000]
    return f"[product_id:{pid}] {brand} | {name}\n{body}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Index product images into Chroma (CLIP)")
    parser.add_argument("--limit", type=int, default=0, help="Max rows (0 = all)")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true", help="Only print counts, no Chroma writes")
    args = parser.parse_args()

    settings = get_runtime_settings()
    persist = settings.chroma_persist_directory
    collection_name = settings.chroma_image_collection_name
    model_name = settings.clip_model_name
    timeout = settings.image_download_timeout_sec
    batch_size = max(1, settings.image_index_batch_size)

    print(
        {
            "persist": persist,
            "collection": collection_name,
            "model": model_name,
            "limit": args.limit,
            "offset": args.offset,
            "dry_run": args.dry_run,
        }
    )

    engine = get_engine()
    rows = load_rows(engine, args.limit, args.offset)
    print({"rows_loaded": len(rows)})

    if args.dry_run:
        return

    from sentence_transformers import SentenceTransformer

    os.makedirs(persist, exist_ok=True)
    client = chromadb.PersistentClient(path=persist)
    coll = client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})

    model = SentenceTransformer(model_name)
    ok = 0
    fail_dl = 0
    fail_img = 0
    t0 = time.perf_counter()

    batch_ids: List[str] = []
    batch_docs: List[str] = []
    batch_meta: List[Dict[str, Any]] = []
    batch_images: List[Image.Image] = []

    def flush() -> None:
        nonlocal ok, batch_ids, batch_docs, batch_meta, batch_images
        if not batch_ids:
            return
        embs = model.encode(batch_images, convert_to_numpy=True, show_progress_bar=False)
        embeddings = [embs[i].tolist() for i in range(len(batch_ids))]
        coll.upsert(
            ids=batch_ids,
            embeddings=embeddings,
            documents=batch_docs,
            metadatas=batch_meta,
        )
        ok += len(batch_ids)
        batch_ids = []
        batch_docs = []
        batch_meta = []
        batch_images = []

    for pid, name, brand, ingredients, url in rows:
        raw = fetch_image(url, timeout)
        if not raw:
            fail_dl += 1
            continue
        try:
            img = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception:
            fail_img += 1
            continue

        batch_ids.append(pid)
        batch_docs.append(build_document(pid, brand, name, ingredients))
        batch_meta.append(
            {
                "product_id": str(pid),
                "brand": str((brand or "")[:500]),
                "name": str((name or "")[:500]),
                "image_url": str((url or "")[:2000]),
            }
        )
        batch_images.append(img)

        if len(batch_ids) >= batch_size:
            flush()

    flush()

    elapsed = time.perf_counter() - t0
    print(
        {
            "indexed": ok,
            "download_fail": fail_dl,
            "image_decode_fail": fail_img,
            "elapsed_sec": round(elapsed, 2),
        }
    )


if __name__ == "__main__":
    main()
