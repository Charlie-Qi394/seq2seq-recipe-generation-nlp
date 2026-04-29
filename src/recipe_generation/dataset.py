"""PyTorch dataset and batching helpers."""

from __future__ import annotations

from typing import Sequence

import torch
from torch.utils.data import Dataset

from .data import RecipeSample
from .tokenization import PAD_IDX, Vocabulary


class Seq2SeqDataset(Dataset):
    def __init__(self, samples: Sequence[RecipeSample], source_vocab: Vocabulary, target_vocab: Vocabulary, max_len: int = 150):
        self.samples = list(samples)
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[list[int], list[int]]:
        sample = self.samples[idx]
        return (
            self.source_vocab.encode(sample.ingredients, max_len=self.max_len),
            self.target_vocab.encode(sample.recipe, max_len=self.max_len),
        )


def pad_sequences(sequences: Sequence[Sequence[int]], pad_value: int = PAD_IDX) -> torch.Tensor:
    max_len = max(len(seq) for seq in sequences)
    batch = torch.full((len(sequences), max_len), pad_value, dtype=torch.long)
    for row, seq in enumerate(sequences):
        batch[row, : len(seq)] = torch.tensor(seq, dtype=torch.long)
    return batch


def collate_batch(batch: Sequence[tuple[list[int], list[int]]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    src_sequences, tgt_sequences = zip(*batch)
    src_lengths = torch.tensor([len(seq) for seq in src_sequences], dtype=torch.long)
    return pad_sequences(src_sequences), src_lengths, pad_sequences(tgt_sequences)
