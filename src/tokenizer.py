"""Whitespace/punctuation tokenizer plus a train-split-only vocabulary.

Design choices that matter for this project:

* Case is PRESERVED. Capitalisation habits ("Certainly!", "Sure,") are part of
  the stylistic fingerprint we are trying to detect; lower-casing would throw
  signal away before the model ever sees it.
* Punctuation and markdown glyphs are kept as their own tokens, so "**", "###"
  and "```" survive tokenisation. RQ4 removes them deliberately in an ablation
  rather than silently in preprocessing.
* The vocabulary is fitted on the TRAIN SPLIT ONLY. Fitting it on everything
  would leak test-set lexical statistics into the model.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, Sequence

PAD, UNK, SEP = "<pad>", "<unk>", "<sep>"
PAD_ID, UNK_ID, SEP_ID = 0, 1, 2

# letters | digits | any single non-space, non-alphanumeric glyph
_TOKEN_RE = re.compile(r"[A-Za-z]+|\d+|[^\sA-Za-z\d]")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(str(text))


class Vocab:
    def __init__(self, itos: Sequence[str]):
        self.itos = list(itos)
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    @classmethod
    def build(cls, texts: Iterable[str], max_size: int = 30_000,
              min_freq: int = 2) -> "Vocab":
        counter: Counter[str] = Counter()
        for t in texts:
            counter.update(tokenize(t))
        keep = [tok for tok, c in counter.most_common() if c >= min_freq]
        itos = [PAD, UNK, SEP] + keep[: max_size - 3]
        return cls(itos)

    def encode(self, text: str, max_len: int) -> list[int]:
        ids = [self.stoi.get(t, UNK_ID) for t in tokenize(text)[:max_len]]
        return ids if ids else [UNK_ID]

    def encode_pair(self, a: str, b: str, max_a: int, max_b: int) -> list[int]:
        return (self.encode(a, max_a) + [SEP_ID] + self.encode(b, max_b))
