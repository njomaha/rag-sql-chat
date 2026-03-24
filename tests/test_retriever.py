"""
Tests for src/retriever.py

All database and embedding calls are mocked.
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.retriever import retrieve  # noqa: E402


def _fake_results() -> list[dict]:
    return [
        {
            "product_id": 1,
            "name": "Widget Pro",
            "category": "Parts",
            "price": "19.99",
            "in_stock": True,
            "chunk_text": "Product: Widget Pro. Category: Parts.",
            "similarity": 0.95,
        },
        {
            "product_id": 2,
            "name": "Gadget X",
            "category": "Electronics",
            "price": "49.99",
            "in_stock": False,
            "chunk_text": "Product: Gadget X. Category: Electronics.",
            "similarity": 0.88,
        },
    ]


class TestRetrieve:
    def test_returns_list_of_dicts(self):
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.fetchall.return_value = _fake_results()

        @contextmanager
        def fake_cursor(conn):
            yield mock_cur

        with patch("src.retriever.get_connection", return_value=mock_conn), \
             patch("src.retriever.get_cursor", side_effect=fake_cursor), \
             patch("src.retriever.embed_text", return_value=[0.1] * 1536):
            results = retrieve("show me some parts")

        assert len(results) == 2
        assert results[0]["name"] == "Widget Pro"

    def test_passes_top_k_to_query(self):
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.fetchall.return_value = []

        @contextmanager
        def fake_cursor(conn):
            yield mock_cur

        with patch("src.retriever.get_connection", return_value=mock_conn), \
             patch("src.retriever.get_cursor", side_effect=fake_cursor), \
             patch("src.retriever.embed_text", return_value=[0.0] * 1536):
            retrieve("anything", top_k=3)

        # The LIMIT parameter is the third positional arg in execute()
        call_args = mock_cur.execute.call_args
        params = call_args[0][1]  # second positional argument to execute()
        assert params[2] == 3

    def test_default_top_k_from_env(self):
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.fetchall.return_value = []

        @contextmanager
        def fake_cursor(conn):
            yield mock_cur

        with patch("src.retriever.get_connection", return_value=mock_conn), \
             patch("src.retriever.get_cursor", side_effect=fake_cursor), \
             patch("src.retriever.embed_text", return_value=[0.0] * 1536), \
             patch.dict(os.environ, {"TOP_K": "7"}):
            retrieve("anything")

        call_args = mock_cur.execute.call_args
        params = call_args[0][1]
        assert params[2] == 7

    def test_returns_empty_list_when_no_results(self):
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.fetchall.return_value = []

        @contextmanager
        def fake_cursor(conn):
            yield mock_cur

        with patch("src.retriever.get_connection", return_value=mock_conn), \
             patch("src.retriever.get_cursor", side_effect=fake_cursor), \
             patch("src.retriever.embed_text", return_value=[0.0] * 1536):
            results = retrieve("obscure query")

        assert results == []
