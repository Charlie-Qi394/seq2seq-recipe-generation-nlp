"""LSTM encoder-decoder model with optional additive attention."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .tokenization import PAD_IDX


@dataclass(frozen=True)
class ModelConfig:
    source_vocab_size: int
    target_vocab_size: int
    embedding_dim: int = 256
    hidden_size: int = 256
    dropout: float = 0.1
    use_attention: bool = True
    use_packed_sequences: bool = True


class Encoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.source_vocab_size, config.embedding_dim, padding_idx=PAD_IDX)
        self.rnn = nn.LSTM(config.embedding_dim, config.hidden_size, batch_first=True)

    def forward(self, src: torch.Tensor, src_lengths: torch.Tensor | None = None):
        embedded = self.embedding(src)
        if self.config.use_packed_sequences and src_lengths is not None:
            packed = pack_padded_sequence(embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=False)
            outputs, state = self.rnn(packed)
            outputs, _ = pad_packed_sequence(outputs, batch_first=True)
            return outputs, state
        return self.rnn(embedded)


class AdditiveAttention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.energy = nn.Linear(hidden_size * 2, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, decoder_hidden: torch.Tensor, encoder_outputs: torch.Tensor, mask: torch.Tensor | None = None):
        seq_len = encoder_outputs.size(1)
        repeated_hidden = decoder_hidden[-1].unsqueeze(1).repeat(1, seq_len, 1)
        scores = self.score(torch.tanh(self.energy(torch.cat((repeated_hidden, encoder_outputs), dim=2)))).squeeze(2)
        if mask is not None:
            scores = scores.masked_fill(~mask[:, :seq_len], float("-inf"))
        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs)
        return context, weights


class Decoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.target_vocab_size, config.embedding_dim, padding_idx=PAD_IDX)
        self.attention = AdditiveAttention(config.hidden_size) if config.use_attention else None
        rnn_input_size = config.embedding_dim + (config.hidden_size if config.use_attention else 0)
        self.rnn = nn.LSTM(rnn_input_size, config.hidden_size, batch_first=True)
        self.dropout = nn.Dropout(config.dropout)
        self.output = nn.Linear(config.hidden_size, config.target_vocab_size)

    def forward(
        self,
        input_token: torch.Tensor,
        hidden: torch.Tensor,
        cell: torch.Tensor,
        encoder_outputs: torch.Tensor,
        src_mask: torch.Tensor | None = None,
    ):
        embedded = self.dropout(self.embedding(input_token))
        attention_weights = None
        if self.attention is not None:
            context, attention_weights = self.attention(hidden, encoder_outputs, mask=src_mask)
            embedded = torch.cat((embedded, context), dim=2)
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        logits = self.output(output.squeeze(1))
        return logits, hidden, cell, attention_weights


class Seq2Seq(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def forward(self, src: torch.Tensor, src_lengths: torch.Tensor, tgt: torch.Tensor, teacher_forcing_ratio: float = 1.0):
        batch_size, tgt_len = tgt.shape
        outputs = torch.zeros(batch_size, tgt_len, self.config.target_vocab_size, device=tgt.device)
        src_mask = src.ne(PAD_IDX)
        encoder_outputs, (hidden, cell) = self.encoder(src, src_lengths)
        input_token = tgt[:, 0].unsqueeze(1)
        for step in range(1, tgt_len):
            logits, hidden, cell, _ = self.decoder(input_token, hidden, cell, encoder_outputs, src_mask=src_mask)
            outputs[:, step] = logits
            use_teacher = torch.rand(1).item() < teacher_forcing_ratio
            next_token = tgt[:, step] if use_teacher else logits.argmax(dim=1)
            input_token = next_token.unsqueeze(1)
        return outputs
