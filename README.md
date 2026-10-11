# Who Wrote It? Identifying LLMs from Their Responses

CMPSC 448 midterm project. Individual submission.

Given a piece of text produced by a large language model, can a classifier
name the model family that wrote it? This repository curates 30,000 real
`(LLM_name, LLM_input, LLM_output)` triples across five model families and
trains a CNN and a bidirectional LSTM to answer that question, plus three
follow-up experiments on whether the prompt helps, whether the fingerprint
survives a change of task, and which textual signals the classifier actually
relies on.

The written report is **`report/report.pdf`**.

## Headline results

All numbers are test-set accuracy over five balanced classes, so chance is
0.200. See the report for macro-F1, per-family breakdowns and confusion
matrices.

| What the model sees | CNN | BiLSTM | TF-IDF + logistic regression |
|---|---|---|---|
| prompt only | 0.204 | 0.218 | 0.217 |
| response only | 0.495 | see report | 0.514 |
| prompt + response | 0.497 | see report | 0.500 |

Three findings drive the report:

1. **Model families are identifiable from their responses** at roughly 2.5x
   chance, but not trivially so.
2. **The prompt carries essentially no authorship signal.** Prompt-only
   classification sits at chance, and adding the prompt to the response does
   not help. This is a consequence of how the data is built: in a Chatbot
   Arena battle the same prompt is answered by two different models, so the
   prompt cannot predict the author.
3. **A classical stylometry baseline is competitive with, and here slightly
   better than, the from-scratch neural models** at this data scale.

## Data

Source: [`lmarena-ai/arena-human-preference-55k`](https://huggingface.co/datasets/lmarena-ai/arena-human-preference-55k)
(Apache-2.0), 57,477 real Chatbot Arena battles over 64 model checkpoints.
Each battle supplies one prompt and two named model responses, which we
explode into one record per response. Families kept: GPT, Claude, Llama,
Mistral, Vicuna, balanced at 6,000 examples each.

Full provenance, cleaning rules and known biases: **`data/curated/DATA_CARD.md`**.

The raw 97 MB download is gitignored. Reproduce it with:

```bash
mkdir -p data/raw
curl -L -o data/raw/arena55k.parquet \
  "https://huggingface.co/api/datasets/lmarena-ai/arena-human-preference-55k/parquet/default/train/0.parquet"
```

## Reproducing

```bash
pip install -r requirements.txt
python src/build_dataset.py              # curate + verify splits
python src/baselines.py                  # TF-IDF and majority baselines
python src/experiments/rq1_rq2.py        # RQ1 + RQ2 grid (9 runs)
python src/experiments/rq3.py            # RQ3 domain transfer
python src/experiments/rq4_features.py   # RQ4 descriptive statistics
python src/experiments/rq4_ablations.py  # RQ4 ablations
python src/experiments/window_study.py   # truncation-window study
python src/analysis/figures.py           # all figures
python src/report/build_report.py        # render report/report.pdf
```

Everything is seeded (`SEED = 448`) and runs on CPU.

## Layout

```
src/build_dataset.py      curation, family mapping, leakage-safe splits
src/tokenizer.py          tokenizer + train-split-only vocabulary
src/text_views.py         input views (RQ2) and signal-removing transforms (RQ4)
src/models.py             TextCNN and BiLSTM
src/train.py              the single training/evaluation routine every run uses
src/baselines.py          TF-IDF + logistic regression, majority class
src/experiments/          one script per research question
src/analysis/figures.py   every figure in the report
results/                  raw metrics as JSON, plus the manual label audit
figures/                  rendered PNGs
report/                   report source and PDF
```

## A note on the environment

On Windows, `pandas` 3.0 imports `pyarrow` eagerly, and `pyarrow`'s Intel
OpenMP runtime prevents `torch`'s `c10.dll` from loading if it is imported
first. Every entry point therefore imports `src/env_boot.py` before `pandas`.
`pyarrow` is pinned to 17.0.0 for `numpy` 1.26 compatibility.
