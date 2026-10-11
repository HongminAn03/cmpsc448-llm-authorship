# Data card: LLM authorship corpus

## Provenance

| | |
|---|---|
| Source dataset | `lmarena-ai/arena-human-preference-55k` on Hugging Face |
| URL | https://huggingface.co/datasets/lmarena-ai/arena-human-preference-55k |
| License | Apache-2.0 |
| Publisher | LMArena / LMSYS, released for a Kaggle human-preference competition |
| Underlying system | Chatbot Arena, a public site where users send a prompt and two anonymous models answer |
| Downloaded | 11 October 2026 |
| Raw size | 97 MB parquet, 57,477 rows, 64 distinct model checkpoints |

Authorship labels are **not** inferred by us. The arena logs which checkpoint
produced each response, and those logged names are the labels.

Nothing private, confidential, or separately copyrighted was added. The
prompts are text that members of the public typed into a public website, and
they are redistributed here only as the already-public dataset distributes
them. No attempt is made to identify any user, and no user identifiers exist
in the source.

## From battles to triples

A raw row is a *battle*: one prompt plus two responses from two named models.
The assignment asks for `(LLM_name, LLM_input, LLM_output)`, so each battle is
exploded into two records, one per side.

| step | rows |
|---|---|
| raw battles | 57,477 |
| exploded, one record per response | 114,954 |
| restricted to the five kept families | 80,419 |
| after dropping responses under 20 characters | 78,899 |
| after dropping duplicate (input, output) pairs | 78,187 |
| **final balanced corpus** | **30,000** |

Other decisions:

* **Turn 1 only.** The prompt and response cells are JSON lists, because arena
  conversations can be multi-turn. We keep only the first exchange so every
  example is context-free and comparable; later turns would mix a model's
  style with its own earlier output.
* **Family, not checkpoint, is the label.** `gpt-4-0613`, `gpt-4-1106-preview`
  and `chatgpt-3.5-turbo` all map to GPT. The question is "which lab's model
  wrote this", and checkpoint-level labels would create dozens of tiny,
  hopelessly confusable classes.
* **Five families kept**: GPT, Claude, Llama, Mistral, Vicuna. These are the
  five most abundant families, they satisfy the three-family minimum, and they
  mix proprietary (GPT, Claude) with open-weight (Llama, Mistral, Vicuna)
  models.
* **Gemini was excluded.** Only 2,924 Gemini records exist, so including it
  would have forced every class down to roughly 2,900 and cut the corpus by
  half. The balance was judged more valuable than the sixth class.

## Composition

6,000 examples per family, split 80/10/10:

| family | train | val | test |
|---|---|---|---|
| GPT | 4,800 | 600 | 600 |
| Claude | 4,800 | 600 | 600 |
| Llama | 4,800 | 600 | 600 |
| Mistral | 4,800 | 600 | 600 |
| Vicuna | 4,800 | 600 | 600 |

Chance accuracy is therefore exactly 0.200 in every split.

Heuristic task domains (keyword rules over the prompt): general 22,688,
code 2,702, math 2,340, roleplay 1,212, write 1,058. These labels are 83%
accurate against a 60-prompt manual audit; see
`results/rq3_label_audit_verdicts.md`.

## Splits are keyed on the prompt, not the row

This is the most important curation decision in the project.

A prompt recurs across battles, and **inside a single battle one prompt is
paired with two different author labels**. Splitting records at random would
place the same prompt in both train and test, letting a model memorise
prompt-to-author associations and inflating every number, most severely in the
prompt-only condition of RQ2.

Each record's split is therefore `md5(prompt) % 10` (buckets 0-7 train,
8 validation, 9 test). Every record sharing a prompt lands in the same split.
`src/build_dataset.py` asserts zero prompt overlap across splits, zero
duplicate `(input, output)` pairs, and exact class balance, and fails loudly
rather than writing a corpus that violates them.

## Known biases and limitations

* **Era.** The battles date from 2023 to early 2024. The checkpoints are of
  that period, so results describe those model versions, not current ones.
* **Prompt distribution.** Arena prompts are what volunteers type into a
  model-comparison site: heavy on tests, jailbreak probes, coding questions
  and short curiosities. They are not a sample of ordinary LLM use.
* **Within-family heterogeneity.** The GPT class mixes GPT-3.5 and several
  GPT-4 checkpoints, which differ from each other. Family labels therefore
  describe a lineage, not one fixed system, and set a ceiling on achievable
  accuracy.
* **Response-length confound.** Families differ systematically in how long
  their replies are, which is a legitimate stylistic signal but also an easy
  shortcut. The RQ4 `fixed_length` ablation measures how much of the
  classifier's performance rests on it.
* **Truncation.** Models see the first 160 tokens of a response. The
  window-size study quantifies the cost.
* **Unbalanced source.** GPT is over-represented in the raw data (31,840
  records) and Vicuna under-represented (8,759). Downsampling to 6,000 each
  discards most GPT data; the retained sample is random, not curated.
