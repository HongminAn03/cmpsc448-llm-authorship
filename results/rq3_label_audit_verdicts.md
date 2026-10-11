# Manual audit of the heuristic task-domain labels

The RQ3 domain labels come from keyword rules over the prompt, so they are
noisy. All 60 prompts sampled by `src/experiments/rq3.py` (12 per domain,
`results/rq3_label_audit.csv`) were read and judged by hand. A label counts as
wrong when a human reader would not put that prompt in that domain.

| domain | sampled | correct | accuracy |
|---|---|---|---|
| code | 12 | 10 | 83% |
| general | 12 | 12 | 100% |
| math | 12 | 6 | 50% |
| roleplay | 12 | 10 | 83% |
| write | 12 | 12 | 100% |
| **overall** | **60** | **50** | **83%** |

## The errors, and why they happen

**math is the weak rule (50%).** Six of twelve are misfiled, and the cause is
one pattern: `\bhow many\b` captures ordinary factual questions. "How many
layoffs have there been in the tech industry and why?" is general knowledge,
not arithmetic. The other math misses are an extraction task ("extract the
numbers from the USER input"), a prompt-engineering request, an equity-analyst
persona, and a dated general question.

**code misses twice**, both from substring greed: "a compiled list of swiss
activities" matches `compile`, and one adversarial prompt matched a keyword
outside the visible excerpt. Neither is a programming task.

**roleplay misses twice**: a praise poem addressed to a public figure (really
creative writing) and a prompt-injection probe that merely quotes the string
"You are a ".

## Why this is still sound for RQ3

RQ3 trains on code + math + general and tests on write + roleplay. The claim
depends mainly on the **target** domains being genuinely creative, and those
are the two cleanest rules in the table: writing audited 12/12 and roleplay
10/12, so the held-out set is roughly 92% pure. Source-side noise is less
damaging because all three source domains are pooled into one training set;
a prompt misfiled from math into general stays inside the training pool
either way and never leaks into the test set.

The residual risk runs one direction: the two roleplay errors are both closer
to creative writing than to the source domains, so they do not make the target
set look more like the training distribution than it is. Any measured transfer
drop is therefore, if anything, slightly conservative.

This is a heuristic, not a task taxonomy, and the report states the 83% figure
rather than presenting the domains as clean.
