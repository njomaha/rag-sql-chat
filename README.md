# rag-sql-chat

**RAG Chat over a SQL Database using pgvector + GPT-4o**

This project demonstrates how to build a Retrieval-Augmented Generation (RAG)
chat interface on top of a relational database.  SQL records are converted into
text chunks, embedded with OpenAI's `text-embedding-3-small` model, and stored
in PostgreSQL using the **pgvector** extension.  When a user asks a question
the application:

1. Embeds the question with the same OpenAI model.
2. Performs a **cosine-distance nearest-neighbour search** in pgvector to
   retrieve the most relevant rows.
3. Sends that context to **GPT-4o** to generate a grounded, faithful answer.

## Key design highlights

| Feature | Detail |
|---|---|
| Vector store | PostgreSQL + pgvector (no separate vector DB) |
| Embeddings | OpenAI `text-embedding-3-small` (1536 dims) |
| Answer generation | GPT-4o via Chat Completions API |
| Source of truth | SQL `products` table (seed data included) |
| Traceability | Every embedding row carries a `product_id` FK |
| Idempotent ingestion | Re-running `ingest` only embeds new rows |

## Project structure

```
rag-sql-chat/
├── docker-compose.yml      # PostgreSQL 16 + pgvector
├── .env.example            # Environment-variable template
├── requirements.txt        # Runtime dependencies
├── requirements-dev.txt    # Test dependencies
├── sql/
│   ├── schema.sql          # Products + embeddings tables, IVFFlat index
│   └── seed.sql            # 10 sample product rows
├── src/
│   ├── db.py               # psycopg2 connection helper
│   ├── embeddings.py       # OpenAI Embeddings API wrapper
│   ├── ingest.py           # Embed SQL rows → pgvector
│   ├── retriever.py        # Vector similarity search
│   ├── chat.py             # GPT-4o grounded answer generation
│   └── main.py             # Interactive CLI entry point
└── tests/
    ├── test_embeddings.py
    ├── test_ingest.py
    ├── test_retriever.py
    └── test_chat.py
```

## Quick start

### 1. Prerequisites

- Docker & Docker Compose
- Python 3.11+
- An [OpenAI API key](https://platform.openai.com/account/api-keys)

### 2. Configure environment variables

```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-...
```

### 3. Start PostgreSQL with pgvector

```bash
docker compose up -d
```

The `docker-compose.yml` mounts `sql/schema.sql` and `sql/seed.sql` so the
database is initialised automatically on first start.

### 4. Install Python dependencies

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 5. Ingest product embeddings

```bash
python -m src.ingest
```

This reads every product from the `products` table, builds a natural-language
chunk per row, embeds it via the OpenAI API, and stores the resulting vector in
`product_embeddings`.  Re-running is safe — already-embedded rows are skipped.

### 6. Start the interactive chat

```bash
python -m src.main
```

Example session:

```
=== RAG SQL Chat ===
Type your question and press Enter.  Type 'quit' or 'exit' to stop.

You: Do you have any wireless headphones with long battery life?
Assistant: Yes — the QuietZone Headphones are over-ear wireless headphones with
active noise cancellation and a 40-hour battery life. They are Hi-Res Audio
certified and are currently in stock at $349.00.

You: Which laptops are available under $700?
Assistant: The BudgetBook 14 is available for $599.99. It features an AMD Ryzen 5
processor, 8 GB RAM, a 256 GB SSD, and a Full HD IPS display — great for
students and everyday tasks.
```

## Running the tests

```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -v
```

All tests mock external dependencies (OpenAI API, PostgreSQL) and run entirely
offline.

## Architecture overview

```
User question
     │
     ▼
embed_text(question)  ──── OpenAI Embeddings API ────►  query vector
     │
     ▼
pgvector cosine search  ──── PostgreSQL + pgvector ────►  top-K chunks
     │
     ▼
build_context(chunks)
     │
     ▼
GPT-4o Chat Completions  ────────────────────────────►  grounded answer
     │
     ▼
Print to user
```

## Configuration reference

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | *(required)* | OpenAI secret key |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `CHAT_MODEL` | `gpt-4o` | GPT model for answers |
| `TOP_K` | `5` | Number of similar chunks to retrieve |
| `POSTGRES_HOST` | `localhost` | Database host |
| `POSTGRES_PORT` | `5432` | Database port |
| `POSTGRES_USER` | `raguser` | Database user |
| `POSTGRES_PASSWORD` | `ragpassword` | Database password |
| `POSTGRES_DB` | `ragdb` | Database name |
