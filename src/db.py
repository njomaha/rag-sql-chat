"""
db.py – Database connection and low-level helpers.

The module reads connection parameters from environment variables (loaded via
python-dotenv) so that no credentials are hard-coded in source code.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Generator

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from pgvector.psycopg2 import register_vector

load_dotenv()


def get_connection() -> psycopg2.extensions.connection:
    """Return a new psycopg2 connection with pgvector type registered."""
    conn = psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        user=os.getenv("POSTGRES_USER", "raguser"),
        password=os.getenv("POSTGRES_PASSWORD", "ragpassword"),
        dbname=os.getenv("POSTGRES_DB", "ragdb"),
    )
    register_vector(conn)
    return conn


@contextmanager
def get_cursor(
    conn: psycopg2.extensions.connection,
) -> Generator[psycopg2.extensions.cursor, None, None]:
    """Context manager that yields a dict-row cursor and commits on exit."""
    cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        yield cursor
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cursor.close()
