"""Optional pgvector schema test, gated by RUN_DB_TESTS=1.

Requires `docker compose up -d postgres` and the env var ``CEREBRO_PG_DSN``,
e.g. ``postgresql://cerebro:cerebro@localhost:5433/cerebro``.
"""
from __future__ import annotations

import os

import pytest


@pytest.fixture
def dsn(run_db_tests):
    if not run_db_tests:
        pytest.skip("RUN_DB_TESTS != 1; skipping pgvector test")
    dsn = os.environ.get("CEREBRO_PG_DSN")
    if not dsn:
        pytest.skip("CEREBRO_PG_DSN not set")
    return dsn


def test_pgvector_schema_creates(dsn):
    import psycopg
    from pgvector.psycopg import register_vector

    from ml_core.embeddings.export_pgvector import DDL

    with psycopg.connect(dsn, autocommit=False) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(DDL)
            cur.execute("SELECT to_regclass('public.trial_embeddings');")
            row = cur.fetchone()
            assert row[0] == "trial_embeddings"
        conn.commit()
