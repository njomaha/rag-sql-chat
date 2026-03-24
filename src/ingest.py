"""
ingest.py – Embed SQL rows and persist vectors to PostgreSQL.

Each product row is converted to a natural-language text chunk, embedded via
the OpenAI API, and stored in the ``product_embeddings`` table alongside its
source ``product_id`` for full traceability.

Usage (CLI):
    python -m src.ingest
"""

from __future__ import annotations

import logging
from typing import List

from src.db import get_connection, get_cursor
from src.embeddings import embed_texts

logger = logging.getLogger(__name__)


def _row_to_chunk(row: dict) -> str:
    """Convert a products row dict into a natural-language text chunk.

    This textual representation is what gets embedded.  Keeping it structured
    and human-readable produces better embeddings than raw field concatenation.
    """
    stock = "in stock" if row["in_stock"] else "out of stock"
    return (
        f"Product: {row['name']}. "
        f"Category: {row['category']}. "
        f"Description: {row['description']} "
        f"Price: ${row['price']}. "
        f"Availability: {stock}."
    )


def ingest_products(batch_size: int = 50) -> int:
    """Read all products and upsert their embeddings.

    Rows that already have an embedding are skipped so that re-running this
    function is idempotent.

    Args:
        batch_size: Number of rows to embed in a single OpenAI API call.

    Returns:
        Number of new embeddings inserted.
    """
    conn = get_connection()
    inserted = 0

    try:
        with get_cursor(conn) as cur:
            # Fetch products that do not yet have an embedding
            cur.execute(
                """
                SELECT p.id, p.name, p.category, p.description,
                       p.price, p.in_stock
                FROM   products p
                WHERE  NOT EXISTS (
                    SELECT 1 FROM product_embeddings pe
                    WHERE  pe.product_id = p.id
                )
                ORDER  BY p.id
                """
            )
            rows: List[dict] = cur.fetchall()

        if not rows:
            logger.info("All products already have embeddings – nothing to do.")
            return 0

        logger.info("Embedding %d product(s)…", len(rows))

        # Process in batches to stay within OpenAI token limits
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            chunks = [_row_to_chunk(dict(r)) for r in batch]
            vectors = embed_texts(chunks)

            with get_cursor(conn) as cur:
                for row, chunk, vector in zip(batch, chunks, vectors):
                    cur.execute(
                        """
                        INSERT INTO product_embeddings (product_id, chunk_text, embedding)
                        VALUES (%s, %s, %s)
                        """,
                        (row["id"], chunk, vector),
                    )
                    inserted += 1

            logger.info("  Inserted embeddings for batch %d–%d", start, start + len(batch) - 1)

    finally:
        conn.close()

    logger.info("Ingestion complete – inserted %d embedding(s).", inserted)
    return inserted


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ingest_products()
