"""
Tests for src/ingest.py

Database calls and embedding calls are mocked so that no live infrastructure
is required.
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from unittest.mock import MagicMock, call, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ingest import _row_to_chunk, ingest_products  # noqa: E402


# ---------------------------------------------------------------------------
# _row_to_chunk
# ---------------------------------------------------------------------------

class TestRowToChunk:
    def test_in_stock_product(self):
        row = {
            "name": "Widget",
            "category": "Parts",
            "description": "A small widget.",
            "price": "9.99",
            "in_stock": True,
        }
        chunk = _row_to_chunk(row)
        assert "Widget" in chunk
        assert "Parts" in chunk
        assert "A small widget." in chunk
        assert "$9.99" in chunk
        assert "in stock" in chunk

    def test_out_of_stock_product(self):
        row = {
            "name": "Gadget",
            "category": "Electronics",
            "description": "A fancy gadget.",
            "price": "99.00",
            "in_stock": False,
        }
        chunk = _row_to_chunk(row)
        assert "out of stock" in chunk


# ---------------------------------------------------------------------------
# ingest_products
# ---------------------------------------------------------------------------

def _make_fake_rows(n: int) -> list[dict]:
    return [
        {
            "id": i,
            "name": f"Product {i}",
            "category": "Test",
            "description": f"Description {i}.",
            "price": "10.00",
            "in_stock": True,
        }
        for i in range(1, n + 1)
    ]


class TestIngestProducts:
    def test_inserts_new_embeddings(self):
        fake_rows = _make_fake_rows(3)
        fake_vectors = [[float(i)] * 1536 for i in range(3)]

        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.fetchall.return_value = fake_rows

        @contextmanager
        def fake_cursor(conn):
            yield mock_cur

        with patch("src.ingest.get_connection", return_value=mock_conn), \
             patch("src.ingest.get_cursor", side_effect=fake_cursor), \
             patch("src.ingest.embed_texts", return_value=fake_vectors):
            count = ingest_products()

        assert count == 3

    def test_skips_when_no_new_products(self):
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.fetchall.return_value = []

        @contextmanager
        def fake_cursor(conn):
            yield mock_cur

        with patch("src.ingest.get_connection", return_value=mock_conn), \
             patch("src.ingest.get_cursor", side_effect=fake_cursor), \
             patch("src.ingest.embed_texts") as mock_embed:
            count = ingest_products()

        assert count == 0
        mock_embed.assert_not_called()

    def test_respects_batch_size(self):
        fake_rows = _make_fake_rows(5)
        # 5 rows with batch_size=2 → 3 batches → 3 embed calls
        batch_vectors = [[0.0] * 1536, [1.0] * 1536]  # per batch (≤2 items)

        call_count = 0
        embed_calls = []

        def fake_embed(texts, **kwargs):
            nonlocal call_count
            call_count += 1
            vecs = [[float(call_count)] * 1536 for _ in texts]
            embed_calls.append(texts)
            return vecs

        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.fetchall.return_value = fake_rows

        @contextmanager
        def fake_cursor(conn):
            yield mock_cur

        with patch("src.ingest.get_connection", return_value=mock_conn), \
             patch("src.ingest.get_cursor", side_effect=fake_cursor), \
             patch("src.ingest.embed_texts", side_effect=fake_embed):
            count = ingest_products(batch_size=2)

        assert count == 5
        assert call_count == 3  # ceil(5/2) = 3 batches
