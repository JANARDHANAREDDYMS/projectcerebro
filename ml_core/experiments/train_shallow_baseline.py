from __future__ import annotations

from pathlib import Path

from ml_core.experiments.common import load_config, make_arg_parser, prepare_dataloaders, write_run_manifests
from ml_core.models.shallowconv import ShallowConvNet
from ml_core.training.callbacks import MLflowLogger
from ml_core.training.trainer import train_model


def main() -> None:
    parser = make_arg_parser("ProjectCerebro ShallowConvNet baseline")
    args = parser.parse_args()
    config_path = args.config or Path(__file__).resolve().parents[1] / "configs" / "shallow_baseline.yaml"
    config = load_config(config_path)
    output_dir = Path(f"artifacts/checkpoints/shallow_{args.filter}")
    loaders, split, stats, path = prepare_dataloaders(
        config,
        filter_key=args.filter,
        delta_path=args.delta_path,
        dataset=None,
        limit=args.limit,
        include_synthetic_rest=args.include_synthetic_rest,
    )
    write_run_manifests(output_dir, split, stats, {**config, "delta_path": str(path)})
    with MLflowLogger(experiment_name="projectcerebro-baseline") as logger:
        result = train_model(ShallowConvNet(), loaders["train"], loaders["val"], config, output_dir, logger)
    print(f"Best val macro F1: {result.best_metric:.4f} | {result.best_checkpoint}")


if __name__ == "__main__":
    main()
