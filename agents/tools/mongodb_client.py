"""MongoDB helper functions for agent persistence."""
from __future__ import annotations

import os
from typing import Any

from pymongo import MongoClient

MONGO_URI = os.getenv("CEREBRO_MONGO_URI", "mongodb://cerebro:cerebro123@localhost:27017/?authSource=admin")
MONGO_DB = os.getenv("CEREBRO_MONGO_DB", "projectcerebro")


def insert_one(collection: str, document: dict[str, Any]) -> bool:
    """Insert one MongoDB document, returning whether it succeeded."""
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=1000)
        client.admin.command("ping")
        client[MONGO_DB][collection].insert_one(document)
        client.close()
        return True
    except Exception as exc:
        print(f"[MongoDB] insert_one({collection}) failed: {exc}")
        return False


def find_many(collection: str, query: dict[str, Any], projection: dict[str, int] | None = None) -> list[dict]:
    """Fetch documents from MongoDB, returning an empty list on failure."""
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=1000)
        client.admin.command("ping")
        rows = list(client[MONGO_DB][collection].find(query, projection))
        client.close()
        return rows
    except Exception as exc:
        print(f"[MongoDB] find_many({collection}) failed: {exc}")
        return []
