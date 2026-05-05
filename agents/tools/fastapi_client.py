"""HTTP client helpers for the local FastAPI serving layer."""
from __future__ import annotations

import os
from typing import Any

import httpx

FASTAPI_URL = os.getenv("CEREBRO_FASTAPI_URL", "http://127.0.0.1:8001")


def post_json(path: str, payload: dict[str, Any], *, timeout: float = 10.0) -> tuple[int, dict[str, Any]]:
    """POST JSON to the FastAPI server and return status plus JSON body."""
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(f"{FASTAPI_URL}{path}", json=payload)
    except httpx.HTTPError as exc:
        print(f"[FastAPI] POST {path} failed: {exc}")
        return 0, {"detail": str(exc)}
    try:
        body = response.json()
    except ValueError:
        body = {"detail": response.text}
    return response.status_code, body


def get_json(path: str, *, timeout: float = 5.0) -> tuple[int, dict[str, Any]]:
    """GET JSON from the FastAPI server and return status plus JSON body."""
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.get(f"{FASTAPI_URL}{path}")
    except httpx.HTTPError as exc:
        print(f"[FastAPI] GET {path} failed: {exc}")
        return 0, {"detail": str(exc)}
    try:
        body = response.json()
    except ValueError:
        body = {"detail": response.text}
    return response.status_code, body
