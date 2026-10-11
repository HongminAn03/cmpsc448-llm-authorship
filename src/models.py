"""The two required architectures, written from scratch in PyTorch.

Both learn their embeddings from random init: no pretrained vectors and no
transformer, which keeps the comparison between the convolutional and the
recurrent inductive bias honest and keeps training feasible on CPU.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tokenizer import PAD_ID


class TextCNN(nn.Module):
    """Kim (2014) sentence CNN.

    Parallel convolutions of width 3/4/5 act as learned n-gram detectors;
    global max-pooling keeps the strongest activation of each filter, so the
    model answers "does this stylistic n-gram appear anywhere" regardless of
    where in the response it occurs.
    """

    def __init__(self, vocab_size: int, n_classes: int, emb_dim: int = 128,
                 n_filters: int = 128, kernel_sizes: tuple[int, ...] = (3, 4, 5),
                 dropout: float = 0.5):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_ID)
        self.convs = nn.ModuleList([
            nn.Conv1d(emb_dim, n_filters, k) for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * len(kernel_sizes), n_classes)
        self.min_len = max(kernel_sizes)

    def forward(self, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if ids.size(1) < self.min_len:  # pad short batches up to the widest kernel
            ids = F.pad(ids, (0, self.min_len - ids.size(1)), value=PAD_ID)
        x = self.emb(ids).transpose(1, 2)           # B, E, T
        feats = [F.relu(conv(x)).max(dim=2).values for conv in self.convs]
        return self.fc(self.dropout(torch.cat(feats, dim=1)))


class BiLSTM(nn.Module):
    """Single-layer bidirectional LSTM with max-pooling over time.

    Sequences are packed so padding never enters the recurrence. Max-pooling
    the hidden states (rather than taking only the last one) gave a markedly
    better validation score in the smoke tests and keeps long responses from
    washing out early stylistic cues.
    """

    def __init__(self, vocab_size: int, n_classes: int, emb_dim: int = 128,
                 hidden: int = 96, dropout: float = 0.5):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_ID)
        self.lstm = nn.LSTM(emb_dim, hidden, num_layers=1, batch_first=True,
                            bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden * 2, n_classes)

    def forward(self, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        x = self.emb(ids)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        mask = (ids[:, : out.size(1)] != PAD_ID).unsqueeze(-1)
        out = out.masked_fill(~mask, float("-inf"))
        pooled = out.max(dim=1).values
        pooled = torch.nan_to_num(pooled, neginf=0.0)
        return self.fc(self.dropout(pooled))


def build_model(arch: str, vocab_size: int, n_classes: int) -> nn.Module:
    if arch == "cnn":
        return TextCNN(vocab_size, n_classes)
    if arch == "lstm":
        return BiLSTM(vocab_size, n_classes)
    raise ValueError(f"unknown arch {arch!r}")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
