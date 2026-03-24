"""
embeddings.py – Wrapper around the OpenAI Embeddings API.

Converts text strings into dense vector representations that can be stored in
pgvector and used for cosine-similarity retrieval.
"""

from __future__ import annotations

import os
from typing import List

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _client


def embed_text(text: str, model: str | None = None) -> List[float]:
    """Return the embedding vector for *text* using the configured model.

    Args:
        text:  The string to embed.  Leading/trailing whitespace is stripped.
        model: OpenAI embedding model name.  Defaults to the ``EMBEDDING_MODEL``
               environment variable (``text-embedding-3-small`` if unset).

    Returns:
        A list of floats representing the dense embedding vector.
    """
    model = model or os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    response = _get_client().embeddings.create(
        input=text.strip(),
        model=model,
    )
    return response.data[0].embedding


def embed_texts(texts: List[str], model: str | None = None) -> List[List[float]]:
    """Return embedding vectors for a batch of texts.

    Batching reduces the number of API round-trips for large ingestion jobs.

    Args:
        texts: List of strings to embed.
        model: OpenAI embedding model name (same default as :func:`embed_text`).

    Returns:
        List of embedding vectors in the same order as *texts*.
    """
    model = model or os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    stripped = [t.strip() for t in texts]
    response = _get_client().embeddings.create(input=stripped, model=model)
    # The API guarantees order is preserved when input is a list.
    return [item.embedding for item in response.data]
