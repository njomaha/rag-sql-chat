"""
Tests for src/embeddings.py

All OpenAI API calls are mocked so that the tests can run without network
access or a real API key.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# Ensure the repo root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import embeddings  # noqa: E402  (after sys.path manipulation)


@pytest.fixture(autouse=True)
def reset_client():
    """Reset the module-level OpenAI client between tests."""
    embeddings._client = None
    yield
    embeddings._client = None


def _make_mock_client(vectors: list[list[float]]) -> MagicMock:
    """Return a mock OpenAI client whose embeddings.create returns *vectors*."""
    mock_client = MagicMock()
    mock_items = [MagicMock(embedding=v) for v in vectors]
    mock_client.embeddings.create.return_value = MagicMock(data=mock_items)
    return mock_client


# ---------------------------------------------------------------------------
# embed_text
# ---------------------------------------------------------------------------

class TestEmbedText:
    def test_returns_vector_for_single_text(self):
        expected = [0.1] * 1536
        mock_client = _make_mock_client([expected])

        with patch("src.embeddings.OpenAI", return_value=mock_client), \
             patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            result = embeddings.embed_text("Hello world")

        assert result == expected

    def test_strips_whitespace_before_api_call(self):
        mock_client = _make_mock_client([[0.0] * 1536])

        with patch("src.embeddings.OpenAI", return_value=mock_client), \
             patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            embeddings.embed_text("  padded text  ")

        call_kwargs = mock_client.embeddings.create.call_args
        assert call_kwargs.kwargs["input"] == "padded text"

    def test_uses_default_model_from_env(self):
        mock_client = _make_mock_client([[0.0] * 1536])

        with patch("src.embeddings.OpenAI", return_value=mock_client), \
             patch.dict(os.environ, {
                 "OPENAI_API_KEY": "test-key",
                 "EMBEDDING_MODEL": "text-embedding-ada-002",
             }):
            embeddings.embed_text("test")

        call_kwargs = mock_client.embeddings.create.call_args
        assert call_kwargs.kwargs["model"] == "text-embedding-ada-002"

    def test_explicit_model_overrides_env(self):
        mock_client = _make_mock_client([[0.0] * 1536])

        with patch("src.embeddings.OpenAI", return_value=mock_client), \
             patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            embeddings.embed_text("test", model="text-embedding-3-large")

        call_kwargs = mock_client.embeddings.create.call_args
        assert call_kwargs.kwargs["model"] == "text-embedding-3-large"


# ---------------------------------------------------------------------------
# embed_texts
# ---------------------------------------------------------------------------

class TestEmbedTexts:
    def test_returns_vectors_in_order(self):
        vecs = [[float(i)] * 1536 for i in range(3)]
        mock_client = _make_mock_client(vecs)

        with patch("src.embeddings.OpenAI", return_value=mock_client), \
             patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            result = embeddings.embed_texts(["a", "b", "c"])

        assert result == vecs

    def test_sends_stripped_texts_in_batch(self):
        mock_client = _make_mock_client([[0.0] * 1536, [1.0] * 1536])

        with patch("src.embeddings.OpenAI", return_value=mock_client), \
             patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            embeddings.embed_texts([" hello ", " world "])

        call_kwargs = mock_client.embeddings.create.call_args
        assert call_kwargs.kwargs["input"] == ["hello", "world"]
