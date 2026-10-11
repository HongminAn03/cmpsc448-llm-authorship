"""RQ4, part 2 -- which signals is the classifier actually using?

Descriptive statistics show that the families differ. They do not show that
the classifier relies on those differences. Each ablation deletes one
candidate signal from the response, retrains from scratch, and reports the
accuracy that survives. A signal the model depends on produces a large drop;
one it ignores produces none.

  no_markdown     removes headers, bold, bullets, code fences, line structure
  fixed_length    truncates every response to 60 tokens, so response length
                  carries no information
  shuffled        randomises word order, leaving the bag of words intact;
                  what survives is the lexical ceiling, with syntax destroyed
  function_words  masks every content word, leaving only grammatical habit

Run on the CNN, whose training cost allows the full set. One LSTM control is
included to check the attribution is not an artefact of the architecture.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import env_boot  # noqa: F401
from train import RESULTS, Config, run

ABLATIONS = ["none", "no_markdown", "fixed_length", "shuffled",
             "function_words"]


def main() -> None:
    summary = {}
    jobs = [("cnn", t) for t in ABLATIONS]
    jobs.append(("lstm", "no_markdown"))   # architecture control
    jobs.append(("lstm", "fixed_length"))  # architecture control

    for arch, transform in jobs:
        name = f"abl_{arch}_{transform}"
        print(f"\n=== {name} ===", flush=True)
        res = run(Config(name=name, arch=arch, view="output",
                         transform=transform, epochs=6, batch_size=64,
                         seed=448, notes=f"RQ4 ablation: {transform}"))
        summary[name] = {"arch": arch, "transform": transform,
                         "accuracy": res["test_accuracy"],
                         "macro_f1": res["test_macro_f1"]}

    # Express each ablation as a drop from the matching unablated run. The CNN
    # control is retrained here; the LSTM control is the main-grid output-only
    # run, which used the identical configuration.
    bases = {"cnn": summary["abl_cnn_none"]["accuracy"]}
    grid_lstm = os.path.join(RESULTS, "lstm_output_s448.json")
    if os.path.exists(grid_lstm):
        with open(grid_lstm, encoding="utf-8") as fh:
            bases["lstm"] = json.load(fh)["test_accuracy"]
    for v in summary.values():
        if v["arch"] in bases:
            v["delta_accuracy"] = v["accuracy"] - bases[v["arch"]]
    summary["_baselines"] = bases

    with open(os.path.join(RESULTS, "rq4_ablations.json"), "w",
              encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print("\nablation summary:")
    for k, v in summary.items():
        if k.startswith("_"):
            continue
        d = v.get("delta_accuracy")
        d_s = f"{d:+.4f}" if d is not None else "   n/a"
        print(f"  {k:28s} acc {v['accuracy']:.4f}  delta {d_s}")


if __name__ == "__main__":
    main()
