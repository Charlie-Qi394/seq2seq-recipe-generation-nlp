"""Text tokenisation and vocabulary helpers."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable


SPECIAL_TOKENS = ["<pad>", "<sos>", "<eos>", "<unk>"]
PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX = range(4)
TOKEN_PATTERN = re.compile(r"[a-z0-9']+|[.,;:!?()/-]")


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


@dataclass
class Vocabulary:
    token_to_idx: dict[str, int]
    idx_to_token: list[str]

    @classmethod
    def build(cls, texts: Iterable[str], min_freq: int = 2, max_vocab: int | None = None) -> "Vocabulary":
        counts: Counter[str] = Counter()
        for text in texts:
            counts.update(tokenize(text))

        tokens = [token for token, freq in counts.most_common() if freq >= min_freq]
        if max_vocab is not None:
            tokens = tokens[: max(0, max_vocab - len(SPECIAL_TOKENS))]

        idx_to_token = SPECIAL_TOKENS + [token for token in tokens if token not in SPECIAL_TOKENS]
        token_to_idx = {token: idx for idx, token in enumerate(idx_to_token)}
        return cls(token_to_idx=token_to_idx, idx_to_token=idx_to_token)

    def encode(self, text: str, max_len: int = 150, add_sos: bool = True, add_eos: bool = True) -> list[int]:
        ids = [self.token_to_idx.get(token, UNK_IDX) for token in tokenize(text)]
        if add_sos:
            ids = [SOS_IDX] + ids
        if add_eos:
            ids = ids + [EOS_IDX]
        return ids[:max_len]

    def decode(self, ids: Iterable[int], skip_special: bool = True) -> str:
        tokens: list[str] = []
        for idx in ids:
            if idx < 0 or idx >= len(self.idx_to_token):
                token = "<unk>"
            else:
                token = self.idx_to_token[idx]
            if skip_special and token in SPECIAL_TOKENS:
                continue
            tokens.append(token)
        return " ".join(tokens)
