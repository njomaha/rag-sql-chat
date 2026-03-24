"""
chat.py – Grounded answer generation with GPT-4o.

Assembles retrieved product context into a system prompt and calls the GPT-4o
Chat Completions API to produce a faithful, context-grounded response.
"""

from __future__ import annotations

import logging
import os
from typing import List

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a helpful product assistant. "
    "Answer the user's question using ONLY the product information provided below. "
    "If the answer cannot be found in the provided context, say so honestly. "
    "Be concise and precise.\n\n"
    "=== PRODUCT CONTEXT ===\n"
    "{context}"
)


def _build_context(results: List[dict]) -> str:
    """Format retrieval results into a numbered context block."""
    lines = []
    for i, r in enumerate(results, start=1):
        stock = "In stock" if r["in_stock"] else "Out of stock"
        lines.append(
            f"[{i}] {r['name']} (Category: {r['category']}, "
            f"Price: ${r['price']}, {stock})\n"
            f"    {r['chunk_text']}"
        )
    return "\n\n".join(lines)


def answer(query: str, results: List[dict], model: str | None = None) -> str:
    """Generate a grounded answer for *query* given retrieval *results*.

    Args:
        query:   The user's question.
        results: Retrieved product rows from :func:`~src.retriever.retrieve`.
        model:   GPT model name.  Defaults to the ``CHAT_MODEL`` environment
                 variable (``gpt-4o`` if unset).

    Returns:
        The model's plain-text answer string.
    """
    model = model or os.getenv("CHAT_MODEL", "gpt-4o")
    context = _build_context(results)
    system_prompt = _SYSTEM_PROMPT.format(context=context)

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
        temperature=0.2,
    )
    answer_text = response.choices[0].message.content or ""
    logger.debug("GPT response for query %r: %s", query, answer_text[:120])
    return answer_text
