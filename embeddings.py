import hashlib
import sqlalchemy
from sqlalchemy import text
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL, DB_URL
from db import get_schema_as_text

engine = sqlalchemy.create_engine(DB_URL)

# ── BUILD INDEX ───────────────────────────────────────────────────────────────

def build_schema_index():
    """
    Smart incremental embedding:
    - Only re-embeds tables whose schema actually changed
    - Adds new tables automatically
    - Removes dropped tables automatically
    """
    print("🔍 Checking schema for changes...\n")
    schema_docs = get_schema_as_text()

    model = SentenceTransformer(EMBEDDING_MODEL)

    with engine.connect() as conn:
        # Get already stored checksums
        rows = conn.execute(text("""
            SELECT table_name, schema_checksum 
            FROM schema_embeddings
        """)).fetchall()
        stored = {row[0]: row[1] for row in rows}

        # Current tables in DB
        current_tables = {doc["table"] for doc in schema_docs}
        stored_tables  = set(stored.keys())

        # ── Remove dropped tables ─────────────────────────────────────────
        for table in stored_tables - current_tables:
            conn.execute(text("""
                DELETE FROM schema_embeddings 
                WHERE table_name = :table
            """), {"table": table})
            conn.commit()
            print(f"  🗑️  Removed: {table}")

        # ── Add or update changed tables ──────────────────────────────────
        for doc in schema_docs:
            table    = doc["table"]
            content  = doc["text"]
            checksum = hashlib.md5(content.encode()).hexdigest()

            # Skip if nothing changed
            if table in stored and stored[table] == checksum:
                print(f"  ⏭️  No change:  {table} (skipped)")
                continue

            embedding = model.encode(content).tolist()

            if table not in stored:
                # New table — INSERT
                conn.execute(text("""
                    INSERT INTO schema_embeddings 
                        (table_name, content, embedding, schema_checksum)
                    VALUES 
                        (:table, :content, :embedding, :checksum)
                """), {
                    "table":     table,
                    "content":   content,
                    "embedding": str(embedding),
                    "checksum":  checksum
                })
                print(f"  ✅ New table:   {table}")
            else:
                # Schema changed — UPDATE only this table
                conn.execute(text("""
                    UPDATE schema_embeddings
                    SET content         = :content,
                        embedding       = :embedding,
                        schema_checksum = :checksum,
                        created_at      = NOW()
                    WHERE table_name = :table
                """), {
                    "table":     table,
                    "content":   content,
                    "embedding": str(embedding),
                    "checksum":  checksum
                })
                print(f"  🔄 Updated:    {table}")

            conn.commit()

    print("\n✅ Schema index up to date!")


# ── SEARCH (called on every user request) ────────────────────────────────────

def search_schema(question: str, top_k: int = 4) -> list[str]:
    """
    Find most relevant tables using cosine similarity in pgvector.
    Only embeds the question — schema already stored in pgvector.
    """
    model = SentenceTransformer(EMBEDDING_MODEL)
    question_embedding = model.encode(question).tolist()

    with engine.connect() as conn:
        results = conn.execute(text("""
            SELECT content,
                   1 - (embedding <=> :embedding) AS similarity
            FROM schema_embeddings
            ORDER BY embedding <=> :embedding
            LIMIT :top_k
        """), {
            "embedding": str(question_embedding),
            "top_k":     top_k
        }).fetchall()

    return [row[0] for row in results]


if __name__ == "__main__":
    build_schema_index()