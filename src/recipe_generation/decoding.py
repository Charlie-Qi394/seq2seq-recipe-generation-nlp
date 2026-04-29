"""Greedy and beam-search decoding helpers."""

from __future__ import annotations

import torch

from .model import Seq2Seq
from .tokenization import EOS_IDX, PAD_IDX, SOS_IDX, Vocabulary


def greedy_decode(model: Seq2Seq, ingredient_text: str, source_vocab: Vocabulary, target_vocab: Vocabulary, max_len: int = 150) -> str:
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        src_ids = source_vocab.encode(ingredient_text, max_len=max_len)
        src = torch.tensor([src_ids], dtype=torch.long, device=device)
        src_lengths = torch.tensor([len(src_ids)], dtype=torch.long, device=device)
        src_mask = src.ne(PAD_IDX)
        encoder_outputs, (hidden, cell) = model.encoder(src, src_lengths)
        input_token = torch.tensor([[SOS_IDX]], dtype=torch.long, device=device)
        generated: list[int] = []
        for _ in range(max_len):
            logits, hidden, cell, _ = model.decoder(input_token, hidden, cell, encoder_outputs, src_mask=src_mask)
            next_token = int(logits.argmax(dim=1).item())
            if next_token == EOS_IDX:
                break
            generated.append(next_token)
            input_token = torch.tensor([[next_token]], dtype=torch.long, device=device)
    return target_vocab.decode(generated)


def beam_search_decode(
    model: Seq2Seq,
    ingredient_text: str,
    source_vocab: Vocabulary,
    target_vocab: Vocabulary,
    max_len: int = 150,
    beam_width: int = 3,
) -> str:
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        src_ids = source_vocab.encode(ingredient_text, max_len=max_len)
        src = torch.tensor([src_ids], dtype=torch.long, device=device)
        src_lengths = torch.tensor([len(src_ids)], dtype=torch.long, device=device)
        src_mask = src.ne(PAD_IDX)
        encoder_outputs, state = model.encoder(src, src_lengths)
        beams = [([SOS_IDX], 0.0, state)]

        for _ in range(max_len):
            candidates = []
            for tokens, score, (hidden, cell) in beams:
                if tokens[-1] == EOS_IDX:
                    candidates.append((tokens, score, (hidden, cell)))
                    continue
                input_token = torch.tensor([[tokens[-1]]], dtype=torch.long, device=device)
                logits, next_hidden, next_cell, _ = model.decoder(input_token, hidden, cell, encoder_outputs, src_mask=src_mask)
                log_probs = torch.log_softmax(logits, dim=1)
                values, indices = torch.topk(log_probs, beam_width, dim=1)
                for value, index in zip(values[0], indices[0]):
                    candidates.append((tokens + [int(index.item())], score + float(value.item()), (next_hidden, next_cell)))
            beams = sorted(candidates, key=lambda item: item[1] / max(1, len(item[0])), reverse=True)[:beam_width]
            if all(tokens[-1] == EOS_IDX for tokens, _, _ in beams):
                break

    best_tokens = beams[0][0]
    return target_vocab.decode(best_tokens)
