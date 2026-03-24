"""
retriever.py – Vector similarity search over product embeddings.

Converts a user query into an embedding and performs a cosine-distance nearest-
neighbour search in pgvector to return the most relevant product records.
"""

from __future__ import annotations

import logging
import os
from typing import List

from src.db import get_connection, get_cursor
from src.embeddings import embed_text

logger = logging.getLogger(__name__)


def retrieve(query: str, top_k: int | None = None) -> List[dict]:
    """Return the *top_k* most semantically similar products for *query*.

    Args:
        query:  The natural-language question or search phrase.
        top_k:  Maximum number of results to return.  Defaults to the
                ``TOP_K`` environment variable (5 if unset).

    Returns:
        A list of dicts, each containing:
        - ``product_id``   – FK to the products table.
        - ``name``         – Product name.
        - ``category``     – Product category.
        - ``price``        – Product price.
        - ``in_stock``     – Stock status.
        - ``chunk_text``   – The text chunk that was embedded.
        - ``similarity``   – Cosine similarity score (0–1, higher is better).
    """
    top_k = top_k if top_k is not None else int(os.getenv("TOP_K", "5"))
    query_vector = embed_text(query)

    conn = get_connection()
    try:
        with get_cursor(conn) as cur:
            cur.execute(
                """
                SELECT
                    pe.product_id,
                    p.name,
                    p.category,
                    p.price,
                    p.in_stock,
                    pe.chunk_text,
                    1 - (pe.embedding <=> %s::vector) AS similarity
                FROM   product_embeddings pe
                JOIN   products p ON p.id = pe.product_id
                ORDER  BY pe.embedding <=> %s::vector
                LIMIT  %s
                """,
                (query_vector, query_vector, top_k),
            )
            results = [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()

    logger.debug("Retrieved %d result(s) for query: %r", len(results), query)
    return results
