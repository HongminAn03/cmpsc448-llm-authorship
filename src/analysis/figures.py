"""Render every figure used in the report.

Figures are built for a printed PDF, so they commit to the light surface only.
Colour roles come from a validated categorical palette (lightness band, chroma
floor, CVD separation and normal-vision separation all checked); because three
of the five hues sit below 3:1 contrast against the surface, every chart that
uses them also carries direct value labels, which is the required relief.

Family colours are assigned in fixed order and never cycled, so a family keeps
its hue in every figure.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import env_boot  # noqa: F401
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "figures")

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#dcdbd6"

CLASSES = ["GPT", "Claude", "Llama", "Mistral", "Vicuna"]
FAMILY_COLOR = {
    "GPT": "#2a78d6", "Claude": "#eb6834", "Llama": "#1baf7a",
    "Mistral": "#eda100", "Vicuna": "#e87ba4",
}
MODEL_COLOR = {"CNN": "#2a78d6", "BiLSTM": "#eb6834", "TF-IDF": "#1baf7a"}
BLUES = LinearSegmentedColormap.from_list(
    "seq_blue", ["#eef5fd", "#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6",
                 "#256abf", "#184f95", "#0d366b"])
CHANCE = 0.20

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "text.color": INK,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.edgecolor": GRID, "axes.linewidth": 0.8,
    "font.family": "DejaVu Sans",
})


def load(name: str) -> dict | None:
    path = os.path.join(RESULTS, name)
    if not os.path.exists(path):
        print(f"  skip (missing) {name}")
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _tidy(ax, ylabel: str = "", ymax: float | None = None) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel(ylabel)
    if ymax:
        ax.set_ylim(0, ymax)


def _save(fig, name: str) -> None:
    os.makedirs(FIGURES, exist_ok=True)
    path = os.path.join(FIGURES, name)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote figures/{name}")


# --------------------------------------------------------------- RQ1 figures
def fig_confusion() -> None:
    panels = [("CNN", "cnn_output_s448.json"), ("BiLSTM", "lstm_output_s448.json")]
    data = [(t, load(f)) for t, f in panels]
    data = [(t, d) for t, d in data if d]
    if not data:
        return
    fig, axes = plt.subplots(1, len(data), figsize=(5.2 * len(data), 4.4))
    axes = np.atleast_1d(axes)
    for ax, (title, res) in zip(axes, data):
        cm = np.array(res["confusion_matrix"], dtype=float)
        pct = cm / cm.sum(axis=1, keepdims=True) * 100
        im = ax.imshow(pct, cmap=BLUES, vmin=0, vmax=70)
        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
                ax.text(j, i, f"{pct[i, j]:.0f}", ha="center", va="center",
                        fontsize=9,
                        color="#ffffff" if pct[i, j] > 38 else INK)
        ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=30, ha="right")
        ax.set_yticks(range(len(CLASSES)), CLASSES)
        ax.set_xlabel("predicted family")
        if ax is axes[0]:            # only the leftmost panel carries it
            ax.set_ylabel("true family")
        ax.set_title(f"{title}  ({res['test_accuracy']*100:.1f}% accuracy)")
        for s in ax.spines.values():
            s.set_visible(False)
    fig.colorbar(im, ax=axes.tolist(), shrink=0.8,
                 label="row-normalised %")
    _save(fig, "fig_rq1_confusion.png")


def fig_per_class() -> None:
    res = load("cnn_output_s448.json")
    res2 = load("lstm_output_s448.json")
    if not res:
        return
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    x = np.arange(len(CLASSES))
    w = 0.38
    for off, (label, r) in zip((-w/2, w/2),
                               [("CNN", res), ("BiLSTM", res2)]):
        if not r:
            continue
        vals = [r["per_class"][c]["f1-score"] for c in CLASSES]
        bars = ax.bar(x + off, vals, w * 0.92, label=label,
                      color=MODEL_COLOR[label], zorder=3)
        ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2, color=INK2)
    ax.axhline(CHANCE, color=INK2, linestyle=(0, (4, 3)), linewidth=1.2,
               zorder=2)
    ax.text(len(CLASSES) - 0.45, CHANCE + 0.012, "chance (0.20)",
            fontsize=8, color=INK2, ha="right")
    ax.set_xticks(x, CLASSES)
    _tidy(ax, "test F1 score", 1.0)
    ax.set_title("Per-family F1, response-only classification")
    ax.legend(frameon=False, loc="upper right")
    _save(fig, "fig_rq1_per_class.png")


def fig_training_curves() -> None:
    pairs = [("CNN", "cnn_output_s448.json"), ("BiLSTM", "lstm_output_s448.json")]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    drew = False
    for label, f in pairs:
        r = load(f)
        if not r:
            continue
        hist = r["history"]
        ep = [h["epoch"] for h in hist]
        ax.plot(ep, [h["val_macro_f1"] for h in hist], marker="o",
                markersize=5, linewidth=2, color=MODEL_COLOR[label],
                label=label, zorder=3)
        ax.annotate(f"{hist[-1]['val_macro_f1']:.3f}",
                    (ep[-1], hist[-1]["val_macro_f1"]),
                    textcoords="offset points", xytext=(6, 0),
                    fontsize=8, color=INK2, va="center")
        drew = True
    if not drew:
        plt.close(fig)
        return
    ax.axhline(CHANCE, color=INK2, linestyle=(0, (4, 3)), linewidth=1.1)
    ax.set_xlabel("epoch")
    _tidy(ax, "validation macro-F1", 0.62)
    ax.set_title("Validation macro-F1 by epoch (response-only)")
    ax.legend(frameon=False)
    _save(fig, "fig_training_curves.png")


# ---------------------------------------------------------------- RQ2 figure
def fig_views() -> None:
    base = load("baselines.json")
    views = ["input", "output", "both"]
    labels = {"input": "prompt only", "output": "response only",
              "both": "prompt + response"}
    series = {"CNN": [], "BiLSTM": [], "TF-IDF": []}
    for v in views:
        c = load(f"cnn_{v}_s448.json")
        l = load(f"lstm_{v}_s448.json")
        series["CNN"].append(c["test_accuracy"] if c else np.nan)
        series["BiLSTM"].append(l["test_accuracy"] if l else np.nan)
        series["TF-IDF"].append(
            base[f"tfidf_{v}"]["test_accuracy"] if base else np.nan)
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    x = np.arange(len(views))
    w = 0.26
    for i, (name, vals) in enumerate(series.items()):
        bars = ax.bar(x + (i - 1) * w, vals, w * 0.9, label=name,
                      color=MODEL_COLOR[name], zorder=3)
        ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2, color=INK2)
    ax.axhline(CHANCE, color=INK2, linestyle=(0, (4, 3)), linewidth=1.2)
    # sit the annotation under the line at the right edge, clear of the
    # prompt-only bars whose values land right at chance
    ax.text(len(views) - 0.55, CHANCE - 0.035, "chance (0.20)", fontsize=8,
            color=INK2, ha="right")
    ax.set_xticks(x, [labels[v] for v in views])
    _tidy(ax, "test accuracy", 0.72)
    ax.set_title("What the classifier is allowed to see (RQ2)")
    ax.legend(frameon=False, ncols=3, loc="upper left")
    _save(fig, "fig_rq2_views.png")


# ---------------------------------------------------------------- RQ3 figure
def fig_rq3() -> None:
    s = load("rq3_summary.json")
    if not s:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.0),
                                   gridspec_kw={"width_ratios": [1, 1.5]})
    archs = [("CNN", "cnn"), ("BiLSTM", "lstm")]
    x = np.arange(len(archs))
    w = 0.34
    for off, setting, shade in ((-w/2, "indomain", "#2a78d6"),
                                (w/2, "cross", "#eb6834")):
        vals = [s.get(f"rq3_{k}_{setting}", {}).get("accuracy", np.nan)
                for _, k in archs]
        bars = ax1.bar(x + off, vals, w * 0.9, color=shade, zorder=3,
                       label="same domains" if setting == "indomain"
                       else "unseen domains")
        ax1.bar_label(bars, fmt="%.3f", fontsize=8, padding=2, color=INK2)
    ax1.axhline(CHANCE, color=INK2, linestyle=(0, (4, 3)), linewidth=1.2)
    ax1.set_xticks(x, [a for a, _ in archs])
    _tidy(ax1, "test accuracy", 0.62)
    ax1.set_title("Domain transfer (RQ3)")
    ax1.legend(frameon=False, fontsize=9)

    # per-family F1, in-domain vs cross-domain, CNN
    a = s.get("rq3_cnn_indomain", {}).get("per_class", {})
    b = s.get("rq3_cnn_cross", {}).get("per_class", {})
    xf = np.arange(len(CLASSES))
    for off, d, shade, lab in ((-w/2, a, "#2a78d6", "same domains"),
                               (w/2, b, "#eb6834", "unseen domains")):
        vals = [d.get(c, np.nan) for c in CLASSES]
        bars = ax2.bar(xf + off, vals, w * 0.9, color=shade, zorder=3,
                       label=lab)
        ax2.bar_label(bars, fmt="%.2f", fontsize=8, padding=2, color=INK2)
    ax2.set_xticks(xf, CLASSES, rotation=20, ha="right")
    _tidy(ax2, "F1 score", 0.85)
    ax2.set_title("Which families survive the domain shift (CNN)")
    ax2.legend(frameon=False, fontsize=9)
    _save(fig, "fig_rq3_transfer.png")


# --------------------------------------------------------------- RQ4 figures
PRETTY_ABL = {
    "none": "unmodified",
    "no_markdown": "markdown removed",
    "fixed_length": "length equalised\n(first 60 tokens)",
    "shuffled": "word order shuffled",
    "function_words": "content words masked",
}


def fig_ablations() -> None:
    s = load("rq4_ablations.json")
    if not s:
        return
    rows = [(k, v) for k, v in s.items()
            if not k.startswith("_") and v["arch"] == "cnn"]
    order = ["none", "no_markdown", "fixed_length", "shuffled",
             "function_words"]
    rows.sort(key=lambda kv: order.index(kv[1]["transform"]))
    labels = [PRETTY_ABL[v["transform"]] for _, v in rows]
    vals = [v["accuracy"] for _, v in rows]
    deltas = [v.get("delta_accuracy", 0.0) for _, v in rows]

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    y = np.arange(len(rows))[::-1]
    colors = ["#2a78d6"] + ["#eb6834"] * (len(rows) - 1)
    bars = ax.barh(y, vals, 0.6, color=colors, zorder=3)
    for yy, v, d in zip(y, vals, deltas):
        txt = f"{v:.3f}" + (f"   ({d:+.3f})" if abs(d) > 1e-9 else "   baseline")
        ax.text(v + 0.006, yy, txt, va="center", fontsize=9, color=INK2)
    ax.axvline(CHANCE, color=INK2, linestyle=(0, (4, 3)), linewidth=1.2)
    ax.text(CHANCE + 0.004, y.min() - 0.52, "chance", fontsize=8, color=INK2)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 0.62)
    ax.set_xlabel("test accuracy")
    ax.xaxis.grid(True, color=GRID, linewidth=0.7)
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_title("What the CNN depends on (RQ4 ablations)")
    _save(fig, "fig_rq4_ablations.png")


def fig_family_profile() -> None:
    s = load("rq4_features.json")
    if not s:
        return
    means = s["per_family_means"]
    panels = [
        ("tokens", "response length\n(tokens)", "%.0f"),
        ("type_token_ratio", "lexical diversity\n(type-token ratio)", "%.3f"),
        ("any_markdown", "uses markdown\n(fraction of replies)", "%.2f"),
        ("md_bullet", "bullet lines\nper reply", "%.2f"),
        ("phrase_hope_helps", "closing offer of help\n(fraction)", "%.2f"),
        ("mean_sentence_words", "mean sentence\nlength (words)", "%.1f"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(11.6, 6.0))
    for ax, (key, title, fmt) in zip(axes.ravel(), panels):
        vals = [means[c][key] for c in CLASSES]
        bars = ax.bar(range(len(CLASSES)), vals, 0.66,
                      color=[FAMILY_COLOR[c] for c in CLASSES], zorder=3)
        ax.bar_label(bars, fmt=fmt, fontsize=8, padding=2, color=INK2)
        ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=30, ha="right",
                      fontsize=9)
        ax.set_title(title, fontsize=10)
        _tidy(ax)
        ax.set_ylim(0, max(vals) * 1.22)
    fig.suptitle("Measured stylistic profile of each family (training split)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, "fig_rq4_profile.png")


def fig_window() -> None:
    s = load("window_study.json")
    if not s:
        return
    ws = sorted(int(k) for k in s)
    vals = [s[str(w)]["accuracy"] for w in ws]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot(ws, vals, marker="o", markersize=7, linewidth=2,
            color="#2a78d6", zorder=3)
    for w, v in zip(ws, vals):
        ax.annotate(f"{v:.3f}", (w, v), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8, color=INK2)
    ax.axvline(160, color=INK2, linestyle=(0, (4, 3)), linewidth=1.1)
    ax.text(168, min(vals), "window used\nin the main grid", fontsize=8,
            color=INK2)
    ax.set_xscale("log")
    ax.set_xticks(ws, [str(w) for w in ws])
    ax.set_xlabel("response tokens visible to the model")
    _tidy(ax, "test accuracy")
    ax.set_ylim(min(vals) * 0.95, max(vals) * 1.06)
    ax.set_title("Cost of the truncation window (CNN, response only)")
    _save(fig, "fig_window_study.png")


def main() -> None:
    print("rendering figures")
    for fn in (fig_confusion, fig_per_class, fig_training_curves, fig_views,
               fig_rq3, fig_ablations, fig_family_profile, fig_window):
        try:
            fn()
        except Exception as exc:  # a missing run must not kill the rest
            print(f"  FAILED {fn.__name__}: {exc}")


if __name__ == "__main__":
    main()
