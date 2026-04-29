"""Recipe file parsing utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


RECIPE_SUFFIXES = {".mmf", ".txt"}


@dataclass(frozen=True)
class RecipeSample:
    ingredients: str
    recipe: str
    path: Path


def iter_recipe_files(root: str | Path, suffixes: set[str] | None = None) -> list[Path]:
    suffixes = suffixes or RECIPE_SUFFIXES
    root = Path(root)
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in suffixes)


def parse_recipe_blocks(path: str | Path) -> list[list[str]]:
    """Split one source file into records beginning at `Title:` lines."""
    path = Path(path)
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    starts = [idx for idx, line in enumerate(lines) if line.lower().startswith("title:")]
    starts.append(len(lines))
    return [lines[start:end] for start, end in zip(starts, starts[1:]) if end > start]


def parse_record(block: Iterable[str], path: str | Path) -> RecipeSample | None:
    ingredients: list[str] = []
    instructions: list[str] = []
    in_ingredients = False
    in_instructions = False

    for raw_line in block:
        line = raw_line.strip()
        lower = line.lower()
        if not line:
            continue
        if lower.startswith("ingredients:"):
            in_ingredients = True
            in_instructions = False
            value = line.split(":", 1)[1].strip()
            if value:
                ingredients.append(value)
            continue
        if lower.startswith("directions:") or lower.startswith("instructions:") or lower.startswith("recipe:"):
            in_ingredients = False
            in_instructions = True
            value = line.split(":", 1)[1].strip()
            if value:
                instructions.append(value)
            continue
        if lower.startswith("end recipe"):
            break
        if lower.startswith(("title:", "categories:", "yield:", "servings:")):
            continue

        if in_ingredients:
            ingredients.append(line)
        elif in_instructions:
            instructions.append(line)
        elif ingredients:
            instructions.append(line)

    ingredient_text = " ".join(ingredients).strip()
    recipe_text = " ".join(instructions).strip()
    if not ingredient_text or not recipe_text:
        return None
    return RecipeSample(ingredients=ingredient_text, recipe=recipe_text, path=Path(path))


def load_samples(root: str | Path) -> list[RecipeSample]:
    samples: list[RecipeSample] = []
    for recipe_file in iter_recipe_files(root):
        for block in parse_recipe_blocks(recipe_file):
            sample = parse_record(block, recipe_file)
            if sample is not None:
                samples.append(sample)
    return samples
