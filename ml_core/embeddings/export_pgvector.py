"""Export 128-dim trial embeddings from a trained EEGNet checkpoint to pgvector.

Schema (created if absent):

    CREATE TABLE IF NOT EXISTS trial_embeddings (
        trial_id          text PRIMARY KEY,
        dataset           text NOT NULL,
        subject_id        text NOT NULL,
        label_code        int  NOT NULL,
        filter_version    text NOT NULL,
        model_version     text NOT NULL,
        embedding         vector(128) NOT NULL,
        created_at        timestamptz NOT NULL DEFAULT now()
    );

Inserts use ``ON CONFLICT (trial_id) DO NOTHING`` so re-runs are idempotent.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data import EpochDataset, NormStats, read_epochs, validate_schema
from ..models import EEGNet
from ..training.checkpoint import load_checkpoint
from ..training.trainer import pick_device

log = logging.getLogger(__name__)

DDL = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS trial_embeddings (
    trial_id        text PRIMARY KEY,
    dataset         text NOT NULL,
    subject_id      text NOT NULL,
    label_code      int  NOT NULL,
    filter_version  text NOT NULL,
    model_version   text NOT NULL,
    embedding       vector(128) NOT NULL,
    created_at      timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS trial_embeddings_subject_idx
    ON trial_embeddings (subject_id);
"""


def _connect(dsn: str):
    """Connect with psycopg (v3) and register the pgvector adapter."""
    import psycopg
    from pgvector.psycopg import register_vector

    conn = psycopg.connect(dsn, autocommit=False)
    register_vector(conn)
    return conn


def _embed_batches(
    model: EEGNet,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, list[dict]]:
    model.to(device)
    model.train(False)
    embeds: list[np.ndarray] = []
    metas: list[dict] = []
    with torch.no_grad():
        for batch in loader:
            x, _y, meta = batch  # dataset must yield meta
            x = x.to(device)
            _logits, emb = model(x, return_embedding=True)
            embeds.append(emb.detach().cpu().numpy().astype(np.float32))
            # meta is a dict-of-lists when DataLoader collates; normalize to list-of-dicts.
            keys = list(meta.keys())
            n = len(meta[keys[0]])
            for i in range(n):
                metas.append({k: meta[k][i] for k in keys})
    return np.concatenate(embeds, axis=0), metas


def export_embeddings(
    *,
    delta_path: str | Path,
    checkpoint_path: str | Path,
    norm_stats_path: str | Path,
    pg_dsn: str,
    model_version: str,
    filter_version: str,
    datasets: Iterable[str] | None = None,
    batch_size: int = 128,
    device: str | None = None,
) -> int:
    """Compute embeddings for every epoch in `delta_path` and upsert into pgvector.

    Returns the number of newly inserted rows.
    """
    df = read_epochs(delta_path, datasets=datasets, filter_version=filter_version)
    validate_schema(df)
    df = df.assign(_label=df["label_code"])

    norm_stats = NormStats.from_json(norm_stats_path)
    ds = EpochDataset(df, norm_stats=norm_stats, shape_mode="bcnt", return_meta=True)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = EEGNet()
    load_checkpoint(checkpoint_path, model, map_location="cpu", strict=False)
    dev = pick_device(device)
    embeds, metas = _embed_batches(model, loader, dev)
    assert len(embeds) == len(metas) == len(df), (len(embeds), len(metas), len(df))

    # Sanity: pull labels from df in order to keep alignment exact.
    label_codes = df["label_code"].astype(int).tolist()
    inserted = 0
    with _connect(pg_dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(DDL)
            for emb, meta, lab in zip(embeds, metas, label_codes):
                cur.execute(
                    """
                    INSERT INTO trial_embeddings
                        (trial_id, dataset, subject_id, label_code,
                         filter_version, model_version, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (trial_id) DO NOTHING
                    """,
                    (
                        meta["epoch_id"],
                        meta["dataset"],
                        meta["subject_id"],
                        int(lab),
                        filter_version,
                        model_version,
                        emb.tolist(),
                    ),
                )
                inserted += cur.rowcount or 0
        conn.commit()
    log.info("Inserted %d new rows into trial_embeddings.", inserted)
    return inserted
