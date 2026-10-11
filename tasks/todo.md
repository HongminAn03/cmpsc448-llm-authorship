# CMPSC 448 midterm — task log

Deadline: 23:59, 11 Oct 2026. Work started 18:43 the same day.

## Plan

- [x] Read the assignment PDF, extract requirements and the grading rubric
- [x] Decide data strategy (public dataset), scope (all four RQs), submission
      type (individual)
- [x] Source a dataset with real authorship labels across 3+ families
- [x] Curate 30,000 balanced triples with leakage-safe, prompt-keyed splits
- [x] Write the data card (provenance, composition, biases)
- [x] Implement the tokenizer and train-split-only vocabulary
- [x] Implement TextCNN and BiLSTM from scratch
- [x] Implement one shared training/eval routine for every run
- [x] Add TF-IDF and majority baselines
- [x] RQ1 + RQ2 grid (2 architectures x 3 views, plus seed replicates)
- [x] RQ3 cross-domain transfer, with a manual audit of the domain labels
- [x] RQ4 descriptive statistics, ranked by ANOVA F
- [x] RQ4 ablations (markdown, length, word order, content words)
- [x] Window-size study to price the truncation cap
- [x] Figures from a CVD-validated palette
- [x] Report generator that reads every number from results/*.json
- [ ] Final run of figures + report once all chains finish
- [ ] Create the GitHub repository and push
- [ ] Verify the pushed repo renders and contains report.pdf

## Decisions worth remembering

**Prompt-keyed splitting.** Each arena battle pairs one prompt with two
different models, so a random row split leaks identical prompt text across the
train/test boundary with conflicting labels. Splits are keyed on
`md5(prompt) % 10` and the builder asserts zero overlap. This is what makes the
RQ2 prompt-only result interpretable.

**Same truncation window for both architectures.** The BiLSTM forced a
160-token cap for CPU tractability. Applying it to the CNN as well costs the
CNN some accuracy but keeps the architecture comparison unconfounded; the
window study prices the cap separately.

**Combining RQ1 and RQ2 into one grid.** RQ1 is the output-only column of the
RQ2 grid, so training it once serves both.

**Baselines are not optional.** TF-IDF with character n-grams beat both neural
models. Omitting it would have implied a conclusion the data does not support.

## Review

Results are summarised in `report/report.pdf`. Headline: response-only
classification reaches roughly 2.5x chance; the prompt contributes nothing;
fingerprints partly survive a change of task domain; and the signal is
redundant across formatting, length, lexis and function-word habits rather
than concentrated in any one of them.
