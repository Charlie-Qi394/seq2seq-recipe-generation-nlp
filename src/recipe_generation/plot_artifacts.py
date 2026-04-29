"""Plot stored training curves from exported CSV artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_epoch_metrics(metrics_dir: str | Path, output_path: str | Path) -> None:
    metrics_dir = Path(metrics_dir)
    output_path = Path(output_path)
    frames = []
    for csv_path in sorted(metrics_dir.glob("*_epoch_metrics.csv")):
        frames.append(pd.read_csv(csv_path))
    if not frames:
        raise FileNotFoundError(f"No *_epoch_metrics.csv files found in {metrics_dir}")

    plt.figure(figsize=(9, 4))
    for frame in frames:
        label = frame["model"].iloc[0]
        plt.plot(frame["epoch"], frame["train_loss"], label=f"{label} train")
        plt.plot(frame["epoch"], frame["dev_loss"], linestyle="--", label=f"{label} dev")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Development Loss")
    plt.grid(alpha=0.2)
    plt.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training curves from artifact CSVs.")
    parser.add_argument("--metrics-dir", default="artifacts/metrics")
    parser.add_argument("--output", default="artifacts/plots/loss_curves_from_csv.png")
    args = parser.parse_args()
    plot_epoch_metrics(args.metrics_dir, args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
