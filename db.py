import os
from functools import lru_cache

from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine


def get_database_url() -> str:
    """Get database URL, defaulting to local postgres service."""
    return os.getenv(
        "DATABASE_URL",
        "postgresql+psycopg://postgres:postgres@localhost:5432/food_ai",
    )


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Create a pooled SQLAlchemy engine for PostgreSQL."""
    return create_engine(
        get_database_url(),
        pool_size=int(os.getenv("DB_POOL_SIZE", "10")),
        max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "20")),
        pool_timeout=int(os.getenv("DB_POOL_TIMEOUT", "30")),
        pool_recycle=int(os.getenv("DB_POOL_RECYCLE", "1800")),
        pool_pre_ping=True,
        future=True,
    )


@lru_cache(maxsize=1)
def get_langchain_db() -> SQLDatabase:
    """Get LangChain SQLDatabase backed by PostgreSQL."""
    return SQLDatabase.from_uri(get_database_url())
