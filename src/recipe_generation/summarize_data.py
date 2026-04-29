"""Summarize local recipe data without redistributing it."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd

from .data import RecipeSample, load_samples
from .tokenization import tokenize


def split_stats(samples: list[RecipeSample]) -> dict[str, float]:
    ingredient_lengths = [len(tokenize(sample.ingredients)) for sample in samples]
    recipe_lengths = [len(tokenize(sample.recipe)) for sample in samples]
    ingredient_vocab = Counter(token for sample in samples for token in tokenize(sample.ingredients))
    recipe_vocab = Counter(token for sample in samples for token in tokenize(sample.recipe))
    return {
        "samples": len(samples),
        "ingredient_vocab": len(ingredient_vocab),
        "recipe_vocab": len(recipe_vocab),
        "avg_ingredient_len": sum(ingredient_lengths) / len(ingredient_lengths) if ingredient_lengths else 0.0,
        "avg_recipe_len": sum(recipe_lengths) / len(recipe_lengths) if recipe_lengths else 0.0,
        "max_recipe_len": max(recipe_lengths) if recipe_lengths else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize train/dev/test recipe folders.")
    parser.add_argument("data_root", type=Path, help="Directory containing train, dev, and test folders.")
    args = parser.parse_args()

    rows = []
    for split in ("train", "dev", "test"):
        split_dir = args.data_root / split
        rows.append({"split": split, **split_stats(load_samples(split_dir))})
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
