"""Curate the LLM-authorship dataset from Chatbot Arena battles.

Source: lmarena-ai/arena-human-preference-55k (HuggingFace, Apache-2.0).
Each raw row is a *battle*: one prompt answered by two named models.
We explode it into (llm_name, llm_input, llm_output) records.

Outputs
-------
data/curated/dataset.parquet       primary 5-family balanced set (RQ1/RQ2/RQ4)
data/curated/dataset_rq3.parquet   task-domain split set (RQ3)
data/curated/sample_100.csv        human-inspectable sample
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys

import pandas as pd

RAW = "data/raw/arena55k.parquet"
OUT_DIR = "data/curated"
SEED = 448

# ---------------------------------------------------------------- families
# Order matters: first match wins. Claude is tested before GPT so that a
# hypothetical "claude-instant-gpt4-judge" style name cannot be miscredited,
# and mixtral is folded into the Mistral family.
FAMILY_RULES = [
    ("claude", "Claude"),
    ("gpt", "GPT"),
    ("llama", "Llama"),
    ("mixtral", "Mistral"),
    ("mistral", "Mistral"),
    ("vicuna", "Vicuna"),
    ("gemini", "Gemini"),
    ("bard", "Gemini"),
]
KEEP_FAMILIES = ["GPT", "Claude", "Llama", "Mistral", "Vicuna"]
PER_CLASS = 6000
SPLIT_FRACTIONS = {"train": 0.8, "val": 0.1, "test": 0.1}
MIN_RESPONSE_CHARS = 20


def family_of(checkpoint: str) -> str | None:
    s = str(checkpoint).lower()
    for key, fam in FAMILY_RULES:
        if key in s:
            return fam
    return None


# ------------------------------------------------------------------ domains
# Heuristic task domains, matched against the *prompt* (a property of the task,
# shared by both sides of a battle). Noisy by construction; the label error
# rate is audited by src/experiments/rq3.py and reported in the paper.
DOMAIN_RULES = [
    ("code", r"```|\bpython\b|\bjavascript\b|\bc\+\+\b|\bjava\b|\bsql\b|"
             r"\bfunction\b|\bcode\b|\bscript\b|\bdebug\b|\bcompile\b|\bapi\b"),
    ("math", r"\bcalculat|\bsolve\b|\bequation\b|\bderivat|\bintegral\b|"
             r"\bprobabilit|\bmatrix\b|\bsum of\b|\bhow many\b|"
             r"\d+\s*[\+\-\*/x]\s*\d+"),
    ("write", r"\bwrite (?:a|an|me)\b.*\b(?:story|poem|essay|email|letter|song|"
              r"article|blog|script)\b|\bpoem\b|\bshort story\b|\bessay\b"),
    ("roleplay", r"\byou are\b|\bpretend\b|\bact as\b|\broleplay\b|"
                 r"\bimagine you\b"),
]


def domain_of(prompt: str) -> str:
    s = str(prompt).lower()
    for name, rx in DOMAIN_RULES:
        if re.search(rx, s):
            return name
    return "general"


def first_turn(blob: str) -> str:
    """Raw prompt/response cells are JSON lists (multi-turn). Take turn 1."""
    try:
        v = json.loads(blob)
    except Exception:
        return ""
    if isinstance(v, list) and v:
        return v[0] if isinstance(v[0], str) else ""
    return v if isinstance(v, str) else ""


def split_of(prompt: str) -> str:
    """Deterministic split assignment keyed on the PROMPT, not the row.

    A prompt recurs across battles and, inside a single battle, carries two
    different author labels. Hashing the prompt guarantees every record that
    shares a prompt lands in the same split, so no classifier can memorise a
    prompt-to-author association across the train/test boundary.
    """
    h = int(hashlib.md5(prompt.encode("utf-8", "ignore")).hexdigest(), 16)
    bucket = h % 10
    if bucket < 8:
        return "train"
    if bucket == 8:
        return "val"
    return "test"


def build_pool() -> pd.DataFrame:
    df = pd.read_parquet(RAW)
    n_models = pd.concat([df.model_a, df.model_b]).nunique()
    print(f"raw battles: {len(df):,}  distinct checkpoints: {n_models}")

    df = df.assign(prompt_t1=df["prompt"].map(first_turn))
    frames = []
    for side in ("a", "b"):
        frames.append(pd.DataFrame({
            "battle_id": df["id"].values,
            "side": side,
            "checkpoint": df[f"model_{side}"].values,
            "llm_input": df["prompt_t1"].values,
            "llm_output": df[f"response_{side}"].map(first_turn).values,
        }))
    pool = pd.concat(frames, ignore_index=True)
    print(f"exploded records: {len(pool):,}")

    pool["llm_name"] = pool["checkpoint"].map(family_of)
    pool = pool[pool["llm_name"].notna()]
    print(f"records in a known family: {len(pool):,}")

    n0 = len(pool)
    pool = pool[pool["llm_output"].str.len() >= MIN_RESPONSE_CHARS]
    pool = pool[pool["llm_input"].str.len() >= 1]
    print(f"dropped {n0 - len(pool):,} short/empty records")

    n0 = len(pool)
    pool = pool.drop_duplicates(subset=["llm_input", "llm_output"])
    print(f"dropped {n0 - len(pool):,} duplicate (input,output) pairs")

    pool["domain"] = pool["llm_input"].map(domain_of)
    pool["split"] = pool["llm_input"].map(split_of)
    return pool.reset_index(drop=True)


def balance(frame: pd.DataFrame, per_class: int, seed: int) -> pd.DataFrame:
    """Downsample every family in every split to exactly per_class * fraction."""
    out = []
    for split, frac in SPLIT_FRACTIONS.items():
        target = int(round(per_class * frac))
        sub = frame[frame["split"] == split]
        avail = sub.groupby("llm_name").size()
        short = avail[avail < target]
        if len(short):
            sys.exit(f"cannot balance split={split} to {target}/class; "
                     f"short: {short.to_dict()}")
        for fam in KEEP_FAMILIES:
            out.append(sub[sub["llm_name"] == fam]
                       .sample(n=target, random_state=seed))
    return pd.concat(out, ignore_index=True)


def build_primary(pool: pd.DataFrame) -> pd.DataFrame:
    frame = pool[pool["llm_name"].isin(KEEP_FAMILIES)]
    print("\navailable records per family x split:")
    print(frame.pivot_table(index="llm_name", columns="split",
                            values="battle_id", aggfunc="count").to_string())
    ds = balance(frame, PER_CLASS, SEED)
    return ds.sample(frac=1.0, random_state=SEED).reset_index(drop=True)


SOURCE_DOMAINS = ["code", "math", "general"]
TARGET_DOMAINS = ["write", "roleplay"]


def build_rq3(pool: pd.DataFrame) -> pd.DataFrame:
    """Cross-domain set: train on code+math+general, test on write+roleplay.

    Both halves are balanced across families so accuracy stays comparable to
    the primary experiment and to the 1/5 chance floor.
    """
    frame = pool[pool["llm_name"].isin(KEEP_FAMILIES)].copy()
    src = frame[frame["domain"].isin(SOURCE_DOMAINS)]
    tgt = frame[frame["domain"].isin(TARGET_DOMAINS)]

    tgt_cap = int(tgt.groupby("llm_name").size().min())
    print(f"\nrq3 target-domain cap per family: {tgt_cap}")
    tgt_bal = pd.concat(
        [tgt[tgt.llm_name == f].sample(tgt_cap, random_state=SEED)
         for f in KEEP_FAMILIES], ignore_index=True)

    src_cap = int(min(src.groupby("llm_name").size().min(), 5000))
    print(f"rq3 source-domain cap per family: {src_cap}")
    src_bal = pd.concat(
        [src[src.llm_name == f].sample(src_cap, random_state=SEED)
         for f in KEEP_FAMILIES], ignore_index=True)

    src_bal["rq3_role"] = "source"
    tgt_bal["rq3_role"] = "target"
    # A validation slice is carved out of the source half for early stopping;
    # it is keyed on the prompt hash so it inherits the no-leakage guarantee.
    src_bal["rq3_split"] = [
        "val" if int(hashlib.md5(p.encode("utf-8", "ignore")).hexdigest(), 16)
        % 10 == 8 else "train" for p in src_bal["llm_input"]
    ]
    tgt_bal["rq3_split"] = "test"
    ds = pd.concat([src_bal, tgt_bal], ignore_index=True)
    return ds.sample(frac=1.0, random_state=SEED).reset_index(drop=True)


def verify(ds: pd.DataFrame, name: str, split_col: str,
           require_balance: bool = True) -> None:
    print(f"\n--- verifying {name} ---")
    print(ds.pivot_table(index="llm_name", columns=split_col,
                         values="battle_id", aggfunc="count").to_string())
    overlap = int((ds.groupby("llm_input")[split_col].nunique() > 1).sum())
    assert overlap == 0, f"{overlap} prompts appear in more than one split"
    dups = int(ds.duplicated(subset=["llm_input", "llm_output"]).sum())
    assert dups == 0, f"{dups} duplicate (input,output) rows"
    if require_balance:
        for s, grp in ds.groupby(split_col):
            counts = grp.groupby("llm_name").size()
            assert counts.nunique() == 1, \
                f"split {s} unbalanced: {counts.to_dict()}"
    print(f"OK  rows={len(ds):,}  prompt-leakage=0  duplicates=0")


def main() -> None:
    if not os.path.exists(RAW):
        sys.exit(f"missing {RAW} -- see README for the download command")
    os.makedirs(OUT_DIR, exist_ok=True)
    pool = build_pool()

    primary = build_primary(pool)
    verify(primary, "primary (RQ1/RQ2/RQ4)", "split")
    primary.to_parquet(f"{OUT_DIR}/dataset.parquet", index=False)

    rq3 = build_rq3(pool)
    # the val slice is a hash carve-out, so it is near-balanced but not exact
    verify(rq3, "cross-domain (RQ3)", "rq3_split", require_balance=False)
    print(rq3.groupby(["rq3_role", "domain"]).size().to_string())
    rq3.to_parquet(f"{OUT_DIR}/dataset_rq3.parquet", index=False)

    primary.head(100).to_csv(f"{OUT_DIR}/sample_100.csv", index=False)
    print("\ndomain mix in the primary set:")
    print(primary["domain"].value_counts().to_string())
    print("\nwrote", OUT_DIR)


if __name__ == "__main__":
    main()
