# Lessons from this build

Operational notes worth keeping, recorded as they happened.

## Torch and pandas fight over OpenMP on Windows

`pandas` 3.0 imports `pyarrow` eagerly, and `pyarrow` loads its own Intel
OpenMP runtime. If that happens before `torch` is imported, `torch`'s
`c10.dll` fails with `WinError 1114` — and the error names a DLL, not the real
cause, so it looks like a broken torch install. The fix is import order:
`src/env_boot.py` imports torch and is the first import of every entry point.
Separately, `pyarrow` must be pinned to 17.0.0 because 18+ requires numpy 2.0
while this environment is on numpy 1.26.

## Buffered stdout makes a healthy job look hung

Background runs wrote epoch lines through plain `print()`, which block-buffers
when stdout is a file. Section headers used `flush=True` and appeared
immediately, so the log showed a run starting and then nothing for ten
minutes. The job was fine; only the log was stale. Launch background training
with `PYTHONUNBUFFERED=1`, or flush every progress line.

## Running three training chains at once was slower than running them in order

Three concurrent processes on 20 cores thrashed: an LSTM epoch that benchmarks
at 2.5 minutes alone had not finished one epoch in eight minutes. Total CPU
work is fixed, so oversubscription only adds cache pressure and context
switching. Serialising the chains was strictly faster.

## Benchmark before committing to a grid

The first BiLSTM configuration implied 85 minutes per run, which would have
made the experiment grid impossible before the deadline. A five-minute
benchmark across hidden size, sequence length, batch size and thread count
found a configuration 5x faster, and length-bucketed batching (grouping
similar-length sequences so padding does not dominate a batch) supplied most
of that. Measuring first turned an infeasible plan into a feasible one.

## Make the generated artifact read from the data

The report builder reads every number out of `results/*.json` at build time.
Nothing is transcribed by hand, so re-running an experiment cannot leave a
stale figure in the prose. For a deadline project this also means the report
can be regenerated the moment the last run lands.

## `pgrep` does not exist in Git Bash

Two background watchers exited immediately and silently because `pgrep`
returned nothing, which the loop read as "no processes left". Watch a known
PID with `kill -0` instead, and always test the exit condition before trusting
a watcher.

## pandas 3.0 `groupby().apply()` drops the grouping column

The RQ3 label-audit helper crashed with `KeyError: ['domain'] not in index`
because the grouping key moved into the index. Build per-group samples with an
explicit loop and `pd.concat` rather than relying on `apply` to preserve
columns.
