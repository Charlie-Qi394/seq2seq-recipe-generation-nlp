# Seq2Seq Recipe Generation with Attention

## Overview

This project explores neural recipe generation using ingredient lists as input and recipe instructions as output. It compares a baseline LSTM encoder-decoder model with an attention-based Seq2Seq model.

This is a sanitized portfolio version of a university learning project. It removes assignment-specific material and focuses on the technical approach, experiment design, evaluation, and limitations.

## Problem Statement

Given a list of ingredients, generate a plausible recipe instruction sequence. The core challenge is maintaining ingredient grounding while producing fluent procedural text.

## Dataset

The original experiment used a recipe corpus split into training, development, and test folders. This repository does not redistribute the raw dataset. See `data/README.md` for the expected local input structure.

Supported recipe files use records with:

- `Title:`
- `ingredients:`
- recipe instruction text
- `END RECIPE`

## Methodology

- Parsed recipe files into ingredient and instruction records.
- Lowercased and tokenized text.
- Built separate train-only source and target vocabularies.
- Truncated long sequences to a maximum length of 150 tokens.
- Implemented two PyTorch models:
  - LSTM encoder-decoder without attention
  - LSTM encoder-decoder with masked additive attention
- Used packed encoder sequences for variable-length batches.
- Used beam-search decoding for reported outputs.
- Evaluated with BLEU-4, METEOR, ingredient coverage, and extra-ingredient count.

## Key Results

The attention model improved the recorded test metrics compared with the no-attention baseline:

| model | BLEU-4 | METEOR | Ingredient Coverage | Extra Ingredients |
| --- | ---: | ---: | ---: | ---: |
| no_attention | 0.0429 | 0.1957 | 0.0827 | 20.35 |
| attention | 0.0653 | 0.2308 | 0.2075 | 19.81 |

The model still showed limitations including repetition, missing ingredients, and occasional unsupported ingredients.

## Training Curves

![Loss curves](artifacts/plots/loss_curves.png)

## Repository Structure

```text
seq2seq-recipe-generation-nlp/
  README.md
  academic-integrity.md
  requirements.txt
  configs/
    baseline.yaml
  data/
    README.md
  src/
    recipe_generation/
      data.py
      dataset.py
      tokenization.py
      model.py
      decoding.py
      metrics.py
      summarize_data.py
      plot_artifacts.py
  artifacts/
    metrics/
    plots/
    sample_outputs/
  tests/
```

## How To Run

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
pytest
```

Summarize a local dataset:

```bash
python -m recipe_generation.summarize_data data/raw
```

Regenerate the loss plot from stored epoch metrics:

```bash
python -m recipe_generation.plot_artifacts
```

## Artifacts

- `artifacts/metrics/test_metrics.csv`: final test metrics for no-attention and attention models.
- `artifacts/metrics/*_epoch_metrics.csv`: train/dev loss by epoch.
- `artifacts/metrics/model_config_table.csv`: shared model and training configuration.
- `artifacts/sample_outputs/qualitative_outputs.csv`: two qualitative generation examples.
- `artifacts/plots/loss_curves.png`: stored training/development loss plot.

## Lessons Learned

- Attention improved ingredient grounding, but did not fully solve hallucination.
- Evaluation needs both text-overlap metrics and task-specific grounding metrics.
- Beam search improved inference control, but does not replace explicit coverage modelling.
- Train-only vocabulary construction avoids data leakage across splits.

## Future Improvements

- Add coverage penalty or constrained decoding.
- Compare LSTM Seq2Seq against Transformer baselines.
- Add experiment tracking.
- Package full training and checkpoint evaluation as reproducible CLI commands.

## Academic Integrity

See `academic-integrity.md`. This repository excludes raw assignment briefs, submitted PDFs, student identifiers, restricted datasets, and large model checkpoints.
