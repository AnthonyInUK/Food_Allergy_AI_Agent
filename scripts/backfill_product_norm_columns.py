import time
import sys
from pathlib import Path

from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from db import get_engine


def main():
    engine = get_engine()
    t0 = time.perf_counter()
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE products ADD COLUMN IF NOT EXISTS canonical_brand TEXT"))
        conn.execute(text("ALTER TABLE products ADD COLUMN IF NOT EXISTS canonical_name TEXT"))
        conn.execute(text("ALTER TABLE products ADD COLUMN IF NOT EXISTS lang_tag TEXT"))
        conn.execute(text("ALTER TABLE products ADD COLUMN IF NOT EXISTS ingredients_len INTEGER"))

        updated = conn.execute(
            text(
                """
                UPDATE products
                SET
                    canonical_brand = lower(regexp_replace(coalesce(brand,''), '[^a-z0-9\\u4e00-\\u9fff]+', '', 'g')),
                    canonical_name = lower(regexp_replace(coalesce(name,''), '[^a-z0-9\\u4e00-\\u9fff]+', '', 'g')),
                    lang_tag = CASE
                        WHEN ingredients ~ '[\\u4e00-\\u9fff]' THEN 'zh'
                        WHEN ingredients ~ '[A-Za-z]' THEN 'en'
                        ELSE 'other'
                    END,
                    ingredients_len = length(coalesce(ingredients, ''))
                WHERE canonical_brand IS NULL
                   OR canonical_name IS NULL
                   OR lang_tag IS NULL
                   OR ingredients_len IS NULL
                """
            )
        )

        conn.execute(
            text("CREATE INDEX IF NOT EXISTS idx_products_canonical_brand_name ON products(canonical_brand, canonical_name)")
        )
        conn.execute(
            text("CREATE INDEX IF NOT EXISTS idx_products_canonical_brand_lang ON products(canonical_brand, lang_tag)")
        )

    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(
        {
            "updated_rows": getattr(updated, "rowcount", None),
            "elapsed_ms": round(elapsed_ms, 2),
        }
    )


if __name__ == "__main__":
    main()
