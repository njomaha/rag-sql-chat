"""
main.py – Interactive CLI for RAG SQL Chat.

Run:
    python -m src.main

The CLI first checks that embeddings exist (running ingestion if needed), then
enters an interactive question-answering loop backed by pgvector + GPT-4o.
"""

from __future__ import annotations

import logging
import sys

from src.chat import answer
from src.ingest import ingest_products
from src.retriever import retrieve


def _ensure_embeddings() -> None:
    """Ingest any products that do not yet have embeddings."""
    logging.info("Checking / refreshing embeddings…")
    inserted = ingest_products()
    if inserted:
        logging.info("Inserted %d new embedding(s).", inserted)


def run_interactive() -> None:
    """Start the interactive question-answering loop."""
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s %(message)s",
        stream=sys.stderr,
    )

    print("=== RAG SQL Chat ===")
    print("Type your question and press Enter.  Type 'quit' or 'exit' to stop.\n")

    try:
        _ensure_embeddings()
    except Exception as exc:
        print(f"[Error] Could not ingest embeddings: {exc}", file=sys.stderr)
        sys.exit(1)

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue

        if query.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break

        try:
            results = retrieve(query)
            if not results:
                print("Assistant: I could not find any relevant products in the database.\n")
                continue
            response = answer(query, results)
            print(f"Assistant: {response}\n")
        except Exception as exc:
            print(f"[Error] {exc}", file=sys.stderr)


if __name__ == "__main__":
    run_interactive()
