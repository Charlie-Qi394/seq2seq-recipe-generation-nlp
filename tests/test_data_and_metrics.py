from pathlib import Path

from recipe_generation.data import load_samples
from recipe_generation.metrics import ingredient_coverage, score_recipe
from recipe_generation.tokenization import Vocabulary


def test_load_samples_from_recipe_file(tmp_path: Path) -> None:
    recipe_file = tmp_path / "sample.mmf"
    recipe_file.write_text(
        """Title: Test Recipe
ingredients:
1 cup rice
2 cups water
recipe:
Boil water. Add rice and simmer.
END RECIPE
""",
        encoding="utf-8",
    )

    samples = load_samples(tmp_path)

    assert len(samples) == 1
    assert "rice" in samples[0].ingredients.lower()
    assert "simmer" in samples[0].recipe.lower()


def test_vocabulary_encode_decode_round_trip() -> None:
    vocab = Vocabulary.build(["rice water rice", "add water"], min_freq=1)

    ids = vocab.encode("rice water", max_len=10)

    assert vocab.decode(ids) == "rice water"


def test_recipe_metrics_include_grounding() -> None:
    source = "1 cup rice, 2 cups water"
    reference = "Boil water and add rice."
    generated = "Add rice to boiling water."

    metrics = score_recipe(source, reference, generated)

    assert metrics.bleu4 >= 0.0
    assert metrics.meteor >= 0.0
    assert ingredient_coverage(source, generated) == 1.0
