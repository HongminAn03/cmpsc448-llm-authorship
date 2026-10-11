"""Non-neural reference points for the results table.

A from-scratch CNN or LSTM on 24k training documents is not automatically
better than a well-tuned linear model on sparse n-grams, and a report that
omits that comparison cannot claim its networks learned anything useful.
Two baselines are reported:

* majority class -- the 20% floor implied by five balanced classes;
* TF-IDF (word 1-2 grams + character 3-5 grams) with multinomial logistic
  regression, which is a strong stylometry baseline because character n-grams
  capture punctuation and formatting habits directly.

Both see exactly the same 160-token window as the neural models.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import env_boot  # noqa: F401
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.pipeline import FeatureUnion, Pipeline

from text_views import MAX_LEN, render
from tokenizer import tokenize
from train import CLASSES, PRIMARY, RESULTS


def windowed(text: str, cap: int) -> str:
    return " ".join(tokenize(text)[:cap])


def view_text(row, view: str) -> str:
    a, b = render(row["llm_input"], row["llm_output"], view)
    if view == "both":
        return windowed(a, 96) + " <sep> " + windowed(b, 160)
    return windowed(a, MAX_LEN[view])


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("feats", FeatureUnion([
            ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2),
                                     min_df=3, max_features=60_000,
                                     sublinear_tf=True, lowercase=False)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                     min_df=3, max_features=60_000,
                                     sublinear_tf=True, lowercase=False)),
        ])),
        ("clf", LogisticRegression(max_iter=2000, C=4.0, n_jobs=-1)),
    ])


def main() -> None:
    df = pd.read_parquet(PRIMARY)
    label_of = {c: i for i, c in enumerate(CLASSES)}
    out = {}

    for view in ("input", "output", "both"):
        parts = {s: df[df["split"] == s] for s in ("train", "val", "test")}
        xs = {s: [view_text(r, view) for _, r in parts[s].iterrows()]
              for s in parts}
        ys = {s: [label_of[n] for n in parts[s]["llm_name"]] for s in parts}

        pipe = build_pipeline()
        pipe.fit(xs["train"], ys["train"])
        pred = pipe.predict(xs["test"])
        res = {
            "view": view,
            "n_features": len(pipe.named_steps["feats"].get_feature_names_out()),
            "test_accuracy": accuracy_score(ys["test"], pred),
            "test_macro_f1": f1_score(ys["test"], pred, average="macro"),
            "per_class": classification_report(
                ys["test"], pred, labels=list(range(len(CLASSES))),
                target_names=CLASSES, output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(
                ys["test"], pred, labels=list(range(len(CLASSES)))).tolist(),
        }
        out[f"tfidf_{view}"] = res
        print(f"tfidf {view:7s} acc {res['test_accuracy']:.4f} "
              f"macro-F1 {res['test_macro_f1']:.4f} "
              f"({res['n_features']:,} features)", flush=True)

    n_test = int((df["split"] == "test").sum())
    out["majority"] = {"test_accuracy": 1.0 / len(CLASSES),
                       "test_macro_f1": 0.0,
                       "note": f"five balanced classes, n_test={n_test}"}

    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "baselines.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print("wrote results/baselines.json")


if __name__ == "__main__":
    main()
