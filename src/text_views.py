"""Input views (RQ2) and signal-removing transforms (RQ4).

A "view" decides which part of the record the classifier is allowed to see:
the user prompt, the model response, or both. The transforms are used by the
RQ4 ablations to delete one candidate signal at a time and measure how much
accuracy depends on it.
"""
from __future__ import annotations

import random
import re

from tokenizer import tokenize

VIEWS = ("input", "output", "both")

# Token budgets. Measured on the curated set: responses have a median of 226
# tokens (mean 245, p90 462) and prompts a median of 17 (p90 157).
#
# The 160-token response window is a compute-driven cap: a bidirectional LSTM
# unrolls one step per token on CPU, and 256 steps made the full experiment
# grid infeasible in the available time. The SAME cap is imposed on the CNN,
# which would happily take more, so that the architecture comparison measures
# inductive bias rather than context length. What the cap costs is quantified
# separately by the window-size study in src/experiments/window_study.py.
MAX_LEN = {"input": 96, "output": 160, "both": 257}
MAX_INPUT_IN_BOTH = 96
MAX_OUTPUT_IN_BOTH = 160  # +1 for <sep> keeps the pair at 257

# ------------------------------------------------------------- RQ4 ablations
_MD_PATTERNS = [
    (re.compile(r"```[\s\S]*?```"), " "),      # fenced code blocks
    (re.compile(r"`[^`]*`"), " "),             # inline code
    (re.compile(r"^\s{0,3}#{1,6}\s*", re.M), ""),   # ATX headers
    (re.compile(r"\*\*|__|\*|_|~~"), ""),      # bold / italic / strike
    (re.compile(r"^\s*([-*+]|\d+[.)])\s+", re.M), ""),  # list bullets
    (re.compile(r"^\s*>\s?", re.M), ""),       # block quotes
    (re.compile(r"\|"), " "),                  # table pipes
    (re.compile(r"\r?\n"), " "),               # all line structure
    (re.compile(r"\s{2,}"), " "),
]

# ~150 high-frequency English function words. Content words are masked out so
# the classifier can only use grammatical habit, not topic.
FUNCTION_WORDS = set("""
a an the and or but if then than that this these those there here it its it's
is are was were be been being am do does did doing done have has had having
i you he she we they me him her us them my your his our their mine yours
to of in on at by for with from into onto about over under above below
between through during before after since until while as so because though
although however therefore thus hence moreover furthermore also too very
much many more most less least some any no not none nor neither either both
each every all few several such only just even still yet again once
can could will would shall should may might must ought need
what which who whom whose when where why how whether
s t re ve ll d m n o
""".split())


def strip_markdown(text: str) -> str:
    out = str(text)
    for rx, rep in _MD_PATTERNS:
        out = rx.sub(rep, out)
    return out.strip()


def truncate_tokens(text: str, n: int = 60) -> str:
    """Length-blind view: every response contributes exactly n tokens."""
    return " ".join(tokenize(text)[:n])


def shuffle_words(text: str, seed: int = 0) -> str:
    """Destroys word order, preserves the bag of words."""
    toks = tokenize(text)
    rng = random.Random(seed ^ (len(toks) * 2654435761))
    rng.shuffle(toks)
    return " ".join(toks)


def function_words_only(text: str) -> str:
    """Masks content words, keeping only grammatical scaffolding."""
    out = []
    for t in tokenize(text):
        out.append(t if (t.lower() in FUNCTION_WORDS or not t.isalnum())
                   else "<c>")
    return " ".join(out)


TRANSFORMS = {
    "none": lambda s: s,
    "no_markdown": strip_markdown,
    "fixed_length": truncate_tokens,
    "shuffled": shuffle_words,
    "function_words": function_words_only,
}


def render(row_input: str, row_output: str, view: str, transform: str = "none"
           ) -> tuple[str, str | None]:
    """Return the text(s) the model may see for this view."""
    fn = TRANSFORMS[transform]
    if view == "input":
        return fn(row_input), None
    if view == "output":
        return fn(row_output), None
    if view == "both":
        return fn(row_input), fn(row_output)
    raise ValueError(f"unknown view {view!r}")
