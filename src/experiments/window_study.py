"""How much does the 160-token window cost?

The main grid caps responses at 160 tokens so the bidirectional LSTM stays
trainable on CPU. That cap is a real limitation and the report should price it
rather than hide it. The CNN is cheap enough to retrain at several window
sizes, which shows whether the cap sits on a plateau or on a slope.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import env_boot  # noqa: F401
from train import RESULTS, Config, run

WINDOWS = [40, 80, 160, 320, 640]


def main() -> None:
    summary = {}
    for w in WINDOWS:
        name = f"win_cnn_{w}"
        print(f"\n=== {name} (first {w} tokens) ===", flush=True)
        res = run(Config(name=name, arch="cnn", view="output",
                         max_len_override=w, epochs=6, batch_size=64,
                         seed=448, notes=f"window study: {w} tokens"))
        summary[str(w)] = {"accuracy": res["test_accuracy"],
                           "macro_f1": res["test_macro_f1"],
                           "seconds": res["seconds"]}
    with open(os.path.join(RESULTS, "window_study.json"), "w",
              encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print("\nwindow study:")
    for w, v in summary.items():
        print(f"  {w:>4s} tokens: acc {v['accuracy']:.4f} "
              f"macro-F1 {v['macro_f1']:.4f} ({v['seconds']}s)")


if __name__ == "__main__":
    main()
