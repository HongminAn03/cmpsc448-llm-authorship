"""One configurable training/evaluation routine shared by every experiment.

Every number in the report comes out of `run()`, so the training protocol is
identical across research questions and only the stated factor changes.
"""
from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass, field, asdict

import env_boot  # noqa: F401  (must precede pandas; see env_boot docstring)
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from torch.utils.data import DataLoader, Dataset

from models import build_model, count_params
from text_views import MAX_INPUT_IN_BOTH, MAX_LEN, MAX_OUTPUT_IN_BOTH, render
from tokenizer import PAD_ID, Vocab

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRIMARY = os.path.join(ROOT, "data", "curated", "dataset.parquet")
RQ3 = os.path.join(ROOT, "data", "curated", "dataset_rq3.parquet")
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "figures")
CLASSES = ["GPT", "Claude", "Llama", "Mistral", "Vicuna"]


@dataclass
class Config:
    name: str
    arch: str = "cnn"                 # cnn | lstm
    view: str = "output"              # input | output | both
    transform: str = "none"
    dataset: str = PRIMARY
    split_col: str = "split"
    seed: int = 448
    epochs: int = 8
    batch_size: int = 64
    lr: float = 1e-3
    vocab_size: int = 30_000
    patience: int = 2
    max_train: int | None = None      # subsample for smoke tests
    max_len_override: int | None = None  # window-size study only
    notes: str = ""
    tags: dict = field(default_factory=dict)


class EncodedSet(Dataset):
    def __init__(self, ids: list[list[int]], labels: list[int]):
        self.ids, self.labels = ids, labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i):
        return self.ids[i], self.labels[i]


def collate(batch):
    """Pad to the longest sequence in the batch, not to the global maximum."""
    seqs, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    width = int(lengths.max())
    out = torch.full((len(seqs), width), PAD_ID, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    return out, lengths, torch.tensor(labels, dtype=torch.long)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


def _texts(df: pd.DataFrame, cfg: Config) -> tuple[list[str], list[str] | None]:
    rendered = [render(i, o, cfg.view, cfg.transform)
                for i, o in zip(df["llm_input"], df["llm_output"])]
    first = [r[0] for r in rendered]
    second = [r[1] for r in rendered] if cfg.view == "both" else None
    return first, second


def _encode(vocab: Vocab, first: list[str], second: list[str] | None,
            view: str, max_len_override: int | None = None) -> list[list[int]]:
    if view == "both":
        assert second is not None
        out_cap = max_len_override or MAX_OUTPUT_IN_BOTH
        return [vocab.encode_pair(a, b, MAX_INPUT_IN_BOTH, out_cap)
                for a, b in zip(first, second)]
    return [vocab.encode(t, max_len_override or MAX_LEN[view]) for t in first]


class BucketedBatches(torch.utils.data.Sampler):
    """Group similar-length sequences into the same batch.

    collate() pads to the longest member of a batch, so mixing a 12-token and
    a 160-token response wastes most of the batch on padding. Shuffling, then
    sorting inside large chunks, then shuffling the batch order keeps epochs
    stochastic while cutting padded positions substantially.
    """

    def __init__(self, lengths: list[int], batch_size: int, seed: int,
                 chunk_multiple: int = 50):
        self.lengths = lengths
        self.batch_size = batch_size
        self.chunk = batch_size * chunk_multiple
        self.epoch = 0
        self.seed = seed

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        order = list(range(len(self.lengths)))
        rng.shuffle(order)
        batches = []
        for start in range(0, len(order), self.chunk):
            chunk = sorted(order[start: start + self.chunk],
                           key=lambda i: self.lengths[i])
            for b in range(0, len(chunk), self.batch_size):
                batches.append(chunk[b: b + self.batch_size])
        rng.shuffle(batches)
        return iter(batches)

    def __len__(self) -> int:
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size


def evaluate(model: nn.Module, loader: DataLoader) -> tuple[list[int], list[int]]:
    model.eval()
    preds, gold = [], []
    with torch.no_grad():
        for ids, lengths, y in loader:
            logits = model(ids, lengths)
            preds.extend(logits.argmax(1).tolist())
            gold.extend(y.tolist())
    return gold, preds


def run(cfg: Config, save_dir: str = RESULTS, verbose: bool = True,
        resume: bool = True) -> dict:
    """Train and evaluate one configuration.

    If an identical configuration has already been saved, its result is
    returned unchanged. Re-running a chain after an interruption therefore
    costs nothing for the runs that already finished, and a changed config
    never silently reuses a stale result because the saved config must match
    field for field.
    """
    cached = os.path.join(save_dir, f"{cfg.name}.json") if save_dir else None
    if resume and cached and os.path.exists(cached):
        with open(cached, encoding="utf-8") as fh:
            prev = json.load(fh)
        if prev.get("config") == asdict(cfg):
            if verbose:
                print(f"  [{cfg.name}] cached: acc "
                      f"{prev['test_accuracy']:.4f}", flush=True)
            return prev
        if verbose:
            print(f"  [{cfg.name}] config changed; retraining", flush=True)
    t0 = time.time()
    set_seed(cfg.seed)
    df = pd.read_parquet(cfg.dataset)
    label_of = {c: i for i, c in enumerate(CLASSES)}

    parts = {}
    for split in ("train", "val", "test"):
        sub = df[df[cfg.split_col] == split]
        if cfg.max_train and split == "train":
            sub = sub.sample(n=min(cfg.max_train, len(sub)),
                             random_state=cfg.seed)
        parts[split] = sub.reset_index(drop=True)

    rendered = {s: _texts(parts[s], cfg) for s in parts}
    vocab = Vocab.build(rendered["train"][0] +
                        (rendered["train"][1] or []),
                        max_size=cfg.vocab_size)

    loaders = {}
    for split in parts:
        ids = _encode(vocab, *rendered[split], cfg.view, cfg.max_len_override)
        labels = [label_of[n] for n in parts[split]["llm_name"]]
        ds = EncodedSet(ids, labels)
        sampler = BucketedBatches([len(x) for x in ids], cfg.batch_size,
                                  cfg.seed)
        loaders[split] = DataLoader(ds, batch_sampler=sampler,
                                    collate_fn=collate)

    model = build_model(cfg.arch, len(vocab), len(CLASSES))
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.CrossEntropyLoss()

    best_f1, best_state, best_epoch, stale = -1.0, None, -1, 0
    history = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total, nb = 0.0, 0
        for ids, lengths, y in loaders["train"]:
            opt.zero_grad()
            loss = loss_fn(model(ids, lengths), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach())
            nb += 1
        gold, preds = evaluate(model, loaders["val"])
        vf1 = f1_score(gold, preds, average="macro")
        vacc = accuracy_score(gold, preds)
        history.append({"epoch": epoch, "train_loss": total / max(nb, 1),
                        "val_acc": vacc, "val_macro_f1": vf1})
        if verbose:
            print(f"  [{cfg.name}] epoch {epoch}  loss {total/max(nb,1):.4f}"
                  f"  val_acc {vacc:.4f}  val_f1 {vf1:.4f}")
        if vf1 > best_f1:
            best_f1, best_epoch, stale = vf1, epoch, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= cfg.patience:
                if verbose:
                    print(f"  [{cfg.name}] early stop at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    gold, preds = evaluate(model, loaders["test"])

    result = {
        "config": asdict(cfg),
        "vocab_size": len(vocab),
        "params": count_params(model),
        "n_train": len(parts["train"]), "n_val": len(parts["val"]),
        "n_test": len(parts["test"]),
        "best_epoch": best_epoch,
        "val_macro_f1": best_f1,
        "test_accuracy": accuracy_score(gold, preds),
        "test_macro_f1": f1_score(gold, preds, average="macro"),
        "per_class": classification_report(
            gold, preds, labels=list(range(len(CLASSES))),
            target_names=CLASSES, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(
            gold, preds, labels=list(range(len(CLASSES)))).tolist(),
        "history": history,
        "seconds": round(time.time() - t0, 1),
    }
    if verbose:
        print(f"  [{cfg.name}] TEST acc {result['test_accuracy']:.4f}"
              f"  macro-F1 {result['test_macro_f1']:.4f}"
              f"  ({result['seconds']}s)")
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        with open(f"{save_dir}/{cfg.name}.json", "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
    return result


if __name__ == "__main__":
    # smoke test: tiny subsample, one epoch, both architectures
    for arch in ("cnn", "lstm"):
        run(Config(name=f"smoke_{arch}", arch=arch, view="output",
                   epochs=1, max_train=2000, patience=1), save_dir="")
