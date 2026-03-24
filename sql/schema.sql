-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- ---------------------------------------------------------------
-- Products table: source of truth for the RAG knowledge base
-- ---------------------------------------------------------------
CREATE TABLE IF NOT EXISTS products (
    id          SERIAL PRIMARY KEY,
    name        TEXT        NOT NULL,
    category    TEXT        NOT NULL,
    description TEXT        NOT NULL,
    price       NUMERIC(10, 2) NOT NULL,
    in_stock    BOOLEAN     NOT NULL DEFAULT TRUE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ---------------------------------------------------------------
-- Embeddings table: one row per product, stores the vector
-- alongside the text chunk that was embedded.
-- ---------------------------------------------------------------
CREATE TABLE IF NOT EXISTS product_embeddings (
    id          SERIAL PRIMARY KEY,
    product_id  INT         NOT NULL REFERENCES products(id) ON DELETE CASCADE,
    chunk_text  TEXT        NOT NULL,
    embedding   VECTOR(1536),          -- 1536-dim default output for text-embedding-3-small and text-embedding-ada-002
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Index for fast cosine-distance nearest-neighbour search
CREATE INDEX IF NOT EXISTS idx_product_embeddings_vector
    ON product_embeddings
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 50);
