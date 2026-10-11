"""RQ4, part 1 -- descriptive characteristics of each LLM family.

Classification accuracy says the families are separable. These statistics say
along which axes. Everything here is measured on the TRAINING split only, so
nothing reported as a "characteristic" has been read off the test set.

A one-way ANOVA F statistic per feature ranks how strongly each measurement
separates the five families, which gives the report an ordering rather than a
wall of undifferentiated numbers.
"""
from __future__ import annotations

import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import env_boot  # noqa: F401
import numpy as np
import pandas as pd
from scipy import stats as sps

from tokenizer import tokenize
from train import CLASSES, PRIMARY, RESULTS

CANNED = {
    "as_an_ai": r"\bas an ai\b|\bas a language model\b|\bas an? (?:large )?language model\b",
    "i_cannot": r"\bi cannot\b|\bi can't\b|\bi am unable\b|\bi'm unable\b",
    "certainly": r"\bcertainly\b|\bof course\b|\bsure[,!]",
    "hope_helps": r"\bhope (?:this|that) helps\b|\blet me know if\b",
    "important_note": r"\bit(?:'s| is) important to\b|\bkeep in mind\b|\bnote that\b",
}


def features(text: str) -> dict[str, float]:
    s = str(text)
    toks = tokenize(s)
    words = [t for t in toks if t.isalpha()]
    n_words = max(len(words), 1)
    sentences = [x for x in re.split(r"[.!?]+", s) if x.strip()]
    lines = s.split("\n")
    return {
        "chars": len(s),
        "tokens": len(toks),
        "words": len(words),
        "type_token_ratio": len(set(w.lower() for w in words)) / n_words,
        "mean_word_len": float(np.mean([len(w) for w in words])) if words else 0.0,
        "mean_sentence_words": n_words / max(len(sentences), 1),
        "n_lines": len(lines),
        "blank_line_frac": sum(1 for x in lines if not x.strip()) / max(len(lines), 1),
        "md_bold": s.count("**") / 2,
        "md_header": len(re.findall(r"^\s{0,3}#{1,6}\s", s, re.M)),
        "md_bullet": len(re.findall(r"^\s*[-*+]\s", s, re.M)),
        "md_numbered": len(re.findall(r"^\s*\d+[.)]\s", s, re.M)),
        "code_fence": s.count("```") / 2,
        "any_markdown": float(bool(re.search(r"\*\*|^\s{0,3}#{1,6}\s|^\s*[-*+]\s|```",
                                             s, re.M))),
        "exclaim": s.count("!"),
        "question": s.count("?"),
        "comma_per_word": s.count(",") / n_words,
        "colon_per_word": s.count(":") / n_words,
        "emdash": s.count("—"),
        "uppercase_frac": sum(1 for c in s if c.isupper()) / max(len(s), 1),
        **{f"phrase_{k}": float(bool(re.search(rx, s, re.I)))
           for k, rx in CANNED.items()},
    }


def main() -> None:
    df = pd.read_parquet(PRIMARY)
    train = df[df["split"] == "train"]
    print(f"profiling {len(train):,} training responses")

    rows = [features(t) for t in train["llm_output"]]
    feat = pd.DataFrame(rows)
    feat["llm_name"] = train["llm_name"].values

    by_family = feat.groupby("llm_name").mean().reindex(CLASSES)

    # rank features by how strongly they separate the families
    ranking = []
    for col in feat.columns.drop("llm_name"):
        groups = [feat.loc[feat.llm_name == c, col].values for c in CLASSES]
        if all(np.ptp(g) == 0 for g in groups):
            continue
        f_stat, p = sps.f_oneway(*groups)
        spread = by_family[col].max() - by_family[col].min()
        rel = spread / abs(by_family[col].mean()) if by_family[col].mean() else 0
        ranking.append({"feature": col, "f_stat": float(f_stat),
                        "p_value": float(p), "range_across_families": float(spread),
                        "relative_range": float(rel)})
    ranking.sort(key=lambda r: r["f_stat"], reverse=True)

    out = {
        "n_profiled": int(len(train)),
        "per_family_means": by_family.to_dict(orient="index"),
        "discriminative_ranking": ranking,
    }
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "rq4_features.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    by_family.to_csv(os.path.join(RESULTS, "rq4_family_means.csv"))

    print("\ntop 12 most family-discriminative features (one-way ANOVA F):")
    for r in ranking[:12]:
        print(f"  {r['feature']:22s} F={r['f_stat']:9.1f}  p={r['p_value']:.2e}")
    print("\nheadline per-family means:")
    cols = ["tokens", "type_token_ratio", "any_markdown", "md_bullet",
            "mean_sentence_words", "phrase_as_an_ai"]
    print(by_family[cols].round(3).to_string())


if __name__ == "__main__":
    main()
