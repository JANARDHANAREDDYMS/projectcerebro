"""pgvector helpers for EEG embedding storage and retrieval."""
from __future__ import annotations

import os
from typing import Any

import psycopg2

PG_CONN_STR = os.getenv(
    "CEREBRO_PG_CONN_STR",
    "host=localhost port=5433 dbname=projectcerebro user=cerebro password=cerebro123 connect_timeout=1",
)


def ensure_eeg_embeddings_table() -> None:
    """Create the serving-time eeg_embeddings table if it is not present."""
    conn = psycopg2.connect(PG_CONN_STR)
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS eeg_embeddings (
          id           SERIAL PRIMARY KEY,
          epoch_id     TEXT NOT NULL,
          subject_id   TEXT NOT NULL,
          session_id   TEXT,
          label_code   INTEGER,
          label_name   TEXT,
          embedding    vector(128),
          confidence   FLOAT,
          timestamp    TIMESTAMPTZ DEFAULT NOW()
        )
        """
    )
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS eeg_embeddings_vector_idx
          ON eeg_embeddings
          USING ivfflat (embedding vector_cosine_ops)
        """
    )
    conn.commit()
    cur.close()
    conn.close()


def _embedding_literal(embedding: list[float]) -> str:
    """Convert an embedding list to pgvector literal syntax."""
    return "[" + ",".join(str(float(value)) for value in embedding) + "]"


def insert_embedding(
    *,
    epoch_id: str,
    subject_id: str,
    session_id: str,
    label_code: int | None,
    label_name: str | None,
    embedding: list[float],
    confidence: float | None,
) -> bool:
    """Insert one EEG embedding into pgvector."""
    try:
        ensure_eeg_embeddings_table()
        conn = psycopg2.connect(PG_CONN_STR)
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO eeg_embeddings
              (epoch_id, subject_id, session_id, label_code, label_name, embedding, confidence)
            VALUES (%s,%s,%s,%s,%s,%s::vector,%s)
            """,
            (epoch_id, subject_id, session_id, label_code, label_name, _embedding_literal(embedding), confidence),
        )
        conn.commit()
        cur.close()
        conn.close()
        return True
    except Exception as exc:
        print(f"[pgvector] insert failed: {exc}")
        return False


def nearest_neighbors(embedding: list[float], *, limit: int = 5) -> list[dict[str, Any]]:
    """Return nearest stored EEG embeddings by cosine similarity."""
    try:
        ensure_eeg_embeddings_table()
        conn = psycopg2.connect(PG_CONN_STR)
        cur = conn.cursor()
        emb = _embedding_literal(embedding)
        cur.execute(
            """
            SELECT epoch_id, subject_id, label_name, confidence,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM eeg_embeddings
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (emb, emb, limit),
        )
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return [
            {
                "epoch_id": row[0],
                "subject_id": row[1],
                "label_name": row[2],
                "confidence": row[3],
                "similarity": float(row[4]),
            }
            for row in rows
        ]
    except Exception as exc:
        print(f"[pgvector] nearest_neighbors failed: {exc}")
        return []
