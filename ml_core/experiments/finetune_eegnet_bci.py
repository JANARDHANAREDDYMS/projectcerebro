from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ml_core.experiments.common import load_config, make_arg_parser, prepare_dataloaders, write_run_manifests
from ml_core.models.eegnet import EEGNet, load_encoder_weights
from ml_core.training.callbacks import MLflowLogger
from ml_core.training.trainer import train_model


def main() -> None:
    parser = make_arg_parser("ProjectCerebro EEGNet BCI IV-2a fine-tuning")
    parser.add_argument("--pretrained", type=str, required=True)
    args = parser.parse_args()
    config_path = args.config or Path(__file__).resolve().parents[1] / "configs" / "eegnet_finetune.yaml"
    config = load_config(config_path)
    output_dir = Path(f"artifacts/checkpoints/eegnet_finetune_bci_{args.filter}")
    loaders, split, stats, path = prepare_dataloaders(
        config,
        filter_key=args.filter,
        delta_path=args.delta_path,
        dataset="bci_iv_2a",
        limit=args.limit,
        include_synthetic_rest=args.include_synthetic_rest,
    )
    model = EEGNet()
    checkpoint = torch.load(args.pretrained, map_location="cpu")
    load_encoder_weights(model, checkpoint["model_state"])
    write_run_manifests(output_dir, split, stats, {**config, "delta_path": str(path), "pretrained": args.pretrained})
    with MLflowLogger(experiment_name="projectcerebro-eegnet-finetune") as logger:
        result = train_model(model, loaders["train"], loaders["val"], config, output_dir, logger)
    print(f"Best val macro F1: {result.best_metric:.4f} | {result.best_checkpoint}")


if __name__ == "__main__":
    main()
