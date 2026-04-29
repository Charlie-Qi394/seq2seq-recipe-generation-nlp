"""Evaluation metrics for generated recipes."""

from __future__ import annotations

import re
from dataclasses import dataclass

from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import meteor_score

from .tokenization import tokenize


INGREDIENT_SPLIT = re.compile(r"\t+|,|;")
PAREN_PATTERN = re.compile(r"\([^)]*\)")
QUANTITY_PATTERN = re.compile(r"^(?:\d+(?:[./-]\d+)?|\d*\.\d+)$")
UNIT_WORDS = {
    "c", "cup", "cups", "tb", "tbsp", "tablespoon", "tablespoons", "ts", "tsp", "teaspoon", "teaspoons",
    "oz", "ounce", "ounces", "lb", "lbs", "pound", "pounds", "g", "gram", "grams", "kg", "ml", "l",
    "can", "cans", "jar", "jars", "package", "packages", "pkg", "clove", "cloves", "slice", "slices",
}
SIZE_WORDS = {"small", "medium", "large", "lg", "md", "sm", "extra", "jumbo", "whole", "half", "halves"}


def normalize_ingredient_item(item: str) -> str:
    item = PAREN_PATTERN.sub(" ", item.lower())
    kept = []
    for token in tokenize(item):
        if token in UNIT_WORDS or token in SIZE_WORDS or QUANTITY_PATTERN.match(token):
            continue
        if token in {",", ".", "/", "-", "and"}:
            continue
        kept.append(token)
    return " ".join(kept).strip()


def extract_ingredients(ingredient_text: str) -> set[str]:
    ingredients = set()
    for raw_item in INGREDIENT_SPLIT.split(ingredient_text):
        normalized = normalize_ingredient_item(raw_item)
        if normalized:
            ingredients.add(normalized)
    return ingredients


def ingredient_coverage(source_ingredients: str, generated_recipe: str) -> float:
    source_items = extract_ingredients(source_ingredients)
    if not source_items:
        return 0.0
    generated = generated_recipe.lower()
    matched = sum(1 for ingredient in source_items if ingredient and ingredient in generated)
    return matched / len(source_items)


def extra_ingredient_count(source_ingredients: str, generated_recipe: str) -> int:
    source_items = extract_ingredients(source_ingredients)
    generated_tokens = set(tokenize(generated_recipe))
    extras = 0
    for token in generated_tokens:
        if len(token) > 2 and all(token not in source for source in source_items):
            extras += 1
    return extras


@dataclass(frozen=True)
class RecipeMetrics:
    bleu4: float
    meteor: float
    ingredient_coverage: float
    extra_ingredients: int


def score_recipe(source_ingredients: str, reference_recipe: str, generated_recipe: str) -> RecipeMetrics:
    reference_tokens = tokenize(reference_recipe)
    generated_tokens = tokenize(generated_recipe)
    smoothie = SmoothingFunction().method1
    bleu = sentence_bleu([reference_tokens], generated_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)
    try:
        meteor = meteor_score([reference_tokens], generated_tokens)
    except LookupError:
        overlap = len(set(reference_tokens) & set(generated_tokens))
        meteor = overlap / max(1, len(set(reference_tokens)))
    return RecipeMetrics(
        bleu4=bleu,
        meteor=meteor,
        ingredient_coverage=ingredient_coverage(source_ingredients, generated_recipe),
        extra_ingredients=extra_ingredient_count(source_ingredients, generated_recipe),
    )
