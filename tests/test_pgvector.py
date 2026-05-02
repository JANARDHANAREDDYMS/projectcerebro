from __future__ import annotations

import os

import pytest

from ml_core.embeddings.export_pgvector import embedding_to_pgvector


def test_embedding_to_pgvector_format():
    assert embedding_to_pgvector(__import__("numpy").array([1.0, -0.5])) == "[1.00000000,-0.50000000]"


@pytest.mark.skipif(os.environ.get("RUN_DB_TESTS") != "1", reason="requires local docker-compose Postgres")
def test_pgvector_db_tests_are_explicitly_gated():
    assert os.environ["RUN_DB_TESTS"] == "1"
