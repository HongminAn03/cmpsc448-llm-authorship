"""RQ3 -- do LLM fingerprints generalise across task domains?

Design. Training data is held fixed and only the test set moves:

  train / val : source domains (code + math + general), prompt-hash buckets
                0-7 and 8 respectively
  in-domain   : source domains, prompt-hash bucket 9
  cross-domain: target domains (creative writing + roleplay), never seen
                during training

Because both evaluations share one training run's data distribution, the gap
between them isolates domain shift rather than training-set size, vocabulary
coverage, or class balance. Both test sets are downsampled to the same number
of examples per family so the two accuracies sit on the same 20% floor.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import env_boot  # noqa: F401
import hashlib

import pandas as pd

from train import CLASSES, RQ3, RESULTS, Config, run

CROSS = os.path.join(os.path.dirname(RQ3), "dataset_rq3_cross.parquet")
INDOMAIN = os.path.join(os.path.dirname(RQ3), "dataset_rq3_indomain.parquet")
SEED = 448


def bucket(prompt: str) -> int:
    return int(hashlib.md5(prompt.encode("utf-8", "ignore")).hexdigest(),
               16) % 10


def prepare() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_parquet(RQ3)
    df["bucket"] = df["llm_input"].map(bucket)
    src = df[df["rq3_role"] == "source"].copy()
    tgt = df[df["rq3_role"] == "target"].copy()

    train = src[src["bucket"] < 8].copy()
    val = src[src["bucket"] == 8].copy()
    held = src[src["bucket"] == 9].copy()

    # equalise the two test sets so their accuracies are directly comparable
    cap = int(min(held.groupby("llm_name").size().min(),
                  tgt.groupby("llm_name").size().min()))
    print(f"test cap per family: {cap}")
    held = pd.concat([held[held.llm_name == f].sample(cap, random_state=SEED)
                      for f in CLASSES], ignore_index=True)
    tgt = pd.concat([tgt[tgt.llm_name == f].sample(cap, random_state=SEED)
                     for f in CLASSES], ignore_index=True)

    train["split"], val["split"] = "train", "val"
    cross = pd.concat([train, val, tgt.assign(split="test")],
                      ignore_index=True)
    indomain = pd.concat([train, val, held.assign(split="test")],
                         ignore_index=True)

    for name, frame in (("cross", cross), ("in-domain", indomain)):
        overlap = int((frame.groupby("llm_input")["split"].nunique() > 1).sum())
        assert overlap == 0, f"{name}: {overlap} prompts cross a split"
        print(f"{name:10s} train={int((frame.split=='train').sum()):,} "
              f"val={int((frame.split=='val').sum()):,} "
              f"test={int((frame.split=='test').sum()):,}")
    cross.to_parquet(CROSS, index=False)
    indomain.to_parquet(INDOMAIN, index=False)
    return cross, indomain


def audit_domain_labels(n: int = 60) -> None:
    """Dump a sample of heuristic domain labels for manual inspection.

    The domain labels come from keyword rules, so the report must state how
    wrong they are rather than assume they are clean.
    """
    df = pd.read_parquet(RQ3).drop_duplicates("llm_input")
    per_domain = max(n // df["domain"].nunique(), 1)
    picks = [g.sample(min(len(g), per_domain), random_state=SEED)
             for _, g in df.groupby("domain")]
    sample = pd.concat(picks, ignore_index=True)
    out = os.path.join(RESULTS, "rq3_label_audit.csv")
    sample.loc[:, ["domain", "llm_input"]].assign(
        llm_input=sample["llm_input"].str.slice(0, 220).str.replace(
            r"\s+", " ", regex=True)).to_csv(out, index=False)
    print(f"wrote {out} ({len(sample)} prompts for manual audit)")


def main() -> None:
    os.makedirs(RESULTS, exist_ok=True)
    prepare()
    audit_domain_labels()

    summary = {}
    for arch in ("cnn", "lstm"):
        for setting, path in (("indomain", INDOMAIN), ("cross", CROSS)):
            name = f"rq3_{arch}_{setting}"
            print(f"\n=== {name} ===", flush=True)
            res = run(Config(name=name, arch=arch, view="output",
                             dataset=path, split_col="split", epochs=6,
                             batch_size=64, seed=SEED,
                             notes=f"RQ3 {setting} evaluation"))
            summary[name] = {
                "accuracy": res["test_accuracy"],
                "macro_f1": res["test_macro_f1"],
                "per_class": {c: res["per_class"][c]["f1-score"]
                              for c in CLASSES},
            }
    with open(os.path.join(RESULTS, "rq3_summary.json"), "w",
              encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    for k, v in summary.items():
        print(f"{k:22s} acc {v['accuracy']:.4f}  macro-F1 {v['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
