"""
Tests for src/chat.py

All OpenAI API calls are mocked.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.chat import _build_context, answer  # noqa: E402


def _sample_results() -> list[dict]:
    return [
        {
            "product_id": 1,
            "name": "Widget Pro",
            "category": "Parts",
            "price": "19.99",
            "in_stock": True,
            "chunk_text": "Product: Widget Pro. Category: Parts. Description: A pro widget.",
            "similarity": 0.95,
        },
        {
            "product_id": 2,
            "name": "Gadget X",
            "category": "Electronics",
            "price": "49.99",
            "in_stock": False,
            "chunk_text": "Product: Gadget X. Category: Electronics. Description: A gadget.",
            "similarity": 0.88,
        },
    ]


class TestBuildContext:
    def test_includes_all_product_names(self):
        ctx = _build_context(_sample_results())
        assert "Widget Pro" in ctx
        assert "Gadget X" in ctx

    def test_shows_in_stock_status(self):
        ctx = _build_context(_sample_results())
        assert "In stock" in ctx
        assert "Out of stock" in ctx

    def test_numbered_entries(self):
        ctx = _build_context(_sample_results())
        assert "[1]" in ctx
        assert "[2]" in ctx

    def test_empty_results_returns_empty_string(self):
        assert _build_context([]) == ""


class TestAnswer:
    def test_returns_model_response(self):
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Widget Pro is a great choice."
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[mock_choice]
        )

        with patch("src.chat.OpenAI", return_value=mock_client), \
             patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            result = answer("What is the best part?", _sample_results())

        assert result == "Widget Pro is a great choice."

    def test_uses_chat_model_from_env(self):
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Some answer."
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[mock_choice]
        )

        with patch("src.chat.OpenAI", return_value=mock_client), \
             patch.dict(os.environ, {
                 "OPENAI_API_KEY": "test-key",
                 "CHAT_MODEL": "gpt-4o-mini",
             }):
            answer("What is this?", _sample_results())

        call_kwargs = mock_client.chat.completions.create.call_args
        assert call_kwargs.kwargs["model"] == "gpt-4o-mini"

    def test_explicit_model_overrides_env(self):
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Answer."
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[mock_choice]
        )

        with patch("src.chat.OpenAI", return_value=mock_client), \
             patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            answer("query", _sample_results(), model="gpt-4-turbo")

        call_kwargs = mock_client.chat.completions.create.call_args
        assert call_kwargs.kwargs["model"] == "gpt-4-turbo"

    def test_context_included_in_system_prompt(self):
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Answer."
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[mock_choice]
        )

        with patch("src.chat.OpenAI", return_value=mock_client), \
             patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            answer("tell me about Widget Pro", _sample_results())

        call_kwargs = mock_client.chat.completions.create.call_args
        messages = call_kwargs.kwargs["messages"]
        system_content = messages[0]["content"]
        assert "Widget Pro" in system_content
        assert "Gadget X" in system_content
