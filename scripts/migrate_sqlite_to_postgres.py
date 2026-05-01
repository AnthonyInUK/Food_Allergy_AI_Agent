import os
import sqlite3

from sqlalchemy import create_engine, text


SQLITE_PATH = os.getenv("SQLITE_DB_PATH", "data/food_data.db")
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://postgres:postgres@localhost:5432/food_ai",
)


def ensure_schema(pg_engine) -> None:
    with pg_engine.begin() as conn:
        existing_type = conn.execute(
            text(
                """
                SELECT data_type
                FROM information_schema.columns
                WHERE table_name = 'products' AND column_name = 'id'
                """
            )
        ).scalar()
        if existing_type and existing_type != "text":
            conn.execute(text("DROP TABLE IF EXISTS products"))

        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS products (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    brand TEXT,
                    ingredients TEXT,
                    allergens TEXT,
                    image_url TEXT,
                    categories TEXT,
                    countries TEXT
                );
                """
            )
        )
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_products_brand ON products(brand);"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_products_name ON products(name);"))


def migrate() -> None:
    if not os.path.exists(SQLITE_PATH):
        raise FileNotFoundError(f"SQLite file not found: {SQLITE_PATH}")

    sqlite_conn = sqlite3.connect(SQLITE_PATH)
    sqlite_cur = sqlite_conn.cursor()
    sqlite_cur.execute(
        """
        SELECT id, name, brand, ingredients, allergens, image_url, categories, countries
        FROM products
        """
    )
    rows = sqlite_cur.fetchall()
    sqlite_conn.close()

    pg_engine = create_engine(DATABASE_URL, future=True)
    ensure_schema(pg_engine)

    insert_sql = text(
        """
        INSERT INTO products (id, name, brand, ingredients, allergens, image_url, categories, countries)
        VALUES (:id, :name, :brand, :ingredients, :allergens, :image_url, :categories, :countries)
        ON CONFLICT (id) DO UPDATE SET
            name = EXCLUDED.name,
            brand = EXCLUDED.brand,
            ingredients = EXCLUDED.ingredients,
            allergens = EXCLUDED.allergens,
            image_url = EXCLUDED.image_url,
            categories = EXCLUDED.categories,
            countries = EXCLUDED.countries
        """
    )

    payload = [
        {
            "id": str(r[0]) if r[0] is not None else "",
            "name": r[1],
            "brand": r[2],
            "ingredients": r[3],
            "allergens": r[4],
            "image_url": r[5],
            "categories": r[6],
            "countries": r[7],
        }
        for r in rows
    ]

    with pg_engine.begin() as conn:
        if payload:
            conn.execute(insert_sql, payload)

    print(f"Migrated {len(rows)} rows from {SQLITE_PATH} to PostgreSQL.")


if __name__ == "__main__":
    migrate()
