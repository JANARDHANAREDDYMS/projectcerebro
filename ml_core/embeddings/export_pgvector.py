from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import psycopg2
import torch
from torch.utils.data import DataLoader

from ml_core.data.dataset import EpochDataset, collate_epoch_batch
from ml_core.data.delta_loader import read_delta
from ml_core.data.normalize import load_stats
from ml_core.data.schema import LABEL_MAP
from ml_core.experiments.common import filter_to_delta_path
from ml_core.models.eegnet import EEGNet
from ml_core.training.trainer import get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export EEGNet trial embeddings to pgvector")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--filter", default="bp8_30", choices=["bp8_30", "bp4_38"])
    parser.add_argument("--delta-path", default=None)
    parser.add_argument("--stats", default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=5433)
    parser.add_argument("--dbname", default="projectcerebro")
    parser.add_argument("--user", default="cerebro")
    parser.add_argument("--password", default="cerebro123")
    return parser.parse_args()


def embedding_to_pgvector(value: np.ndarray) -> str:
    return "[" + ",".join(f"{float(x):.8f}" for x in value.tolist()) + "]"


def main() -> None:
    args = parse_args()
    ckpt_path = Path(args.ckpt)
    stats_path = Path(args.stats) if args.stats else ckpt_path.parent / "normalization_stats.json"
    stats = load_stats(stats_path)
    df = read_delta(args.delta_path or filter_to_delta_path(args.filter), filter_version=args.filter)
    dataset = EpochDataset(df, stats=stats)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_epoch_batch)

    model = EEGNet()
    payload = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(payload["model_state"])
    device = get_device()
    model.to(device)
    model.eval()

    conn = psycopg2.connect(
        host=args.host,
        port=args.port,
        dbname=args.dbname,
        user=args.user,
        password=args.password,
    )
    inserted = 0
    sql = """
        INSERT INTO trial_embeddings (
            trial_id, subject_id, dataset, run_id, label, label_code, embedding, model_version
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s::vector, %s)
        ON CONFLICT (trial_id) DO UPDATE SET
            embedding = EXCLUDED.embedding,
            model_version = EXCLUDED.model_version
    """
    with conn, conn.cursor() as cur, torch.no_grad():
        for x, y, metas in loader:
            embeddings = model.encode(x.to(device)).detach().cpu().numpy()
            for emb, label, meta in zip(embeddings, y.tolist(), metas):
                cur.execute(
                    sql,
                    (
                        meta.epoch_id,
                        meta.subject_id,
                        meta.dataset,
                        meta.run_id,
                        LABEL_MAP[int(label)],
                        int(label),
                        embedding_to_pgvector(emb),
                        ckpt_path.stem,
                    ),
                )
                inserted += 1
    conn.close()
    print(f"Upserted {inserted} embeddings into trial_embeddings")


if __name__ == "__main__":
    main()
