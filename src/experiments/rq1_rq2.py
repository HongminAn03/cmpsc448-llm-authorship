"""RQ1 and RQ2 in one grid.

RQ1 ("can we identify which LLM generated a response?") is the output-only
column of the RQ2 grid, so training it once serves both questions instead of
paying for the same run twice.

Grid: {CNN, BiLSTM} x {input-only, output-only, input+output}, plus extra
seeds on the output-only cell so the headline RQ1 numbers carry a spread
rather than a single lucky initialisation.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import env_boot  # noqa: F401
from train import Config, run

SEEDS_EXTRA = {"cnn": [449, 450], "lstm": [449]}


def main() -> None:
    jobs: list[Config] = []
    for arch in ("cnn", "lstm"):
        for view in ("input", "output", "both"):
            jobs.append(Config(name=f"{arch}_{view}_s448", arch=arch,
                               view=view, seed=448, epochs=6,
                               batch_size=64,
                               notes="RQ1/RQ2 main grid"))
    # replicate the output-only cell (the RQ1 headline) across seeds
    for arch, seeds in SEEDS_EXTRA.items():
        for s in seeds:
            jobs.append(Config(name=f"{arch}_output_s{s}", arch=arch,
                               view="output", seed=s, epochs=6,
                               batch_size=64, notes="RQ1 seed replicate"))

    print(f"{len(jobs)} runs queued")
    for i, cfg in enumerate(jobs, 1):
        print(f"\n=== [{i}/{len(jobs)}] {cfg.name} "
              f"(arch={cfg.arch} view={cfg.view} seed={cfg.seed}) ===",
              flush=True)
        run(cfg)


if __name__ == "__main__":
    main()
