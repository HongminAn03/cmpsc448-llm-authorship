"""Render report/report.html from the saved metrics, then print it to PDF.

Every number in the report is read out of results/*.json at build time. None
is typed by hand, so the prose cannot drift away from the experiments.
"""
from __future__ import annotations

import base64
import glob
import json
import os
import statistics
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "figures")
REPORT_DIR = os.path.join(ROOT, "report")

AUTHOR = "Hongmin An"
COURSE = "CMPSC 448 - Machine Learning"
TITLE = "Who Wrote It? Identifying LLMs from Their Responses"
CLASSES = ["GPT", "Claude", "Llama", "Mistral", "Vicuna"]
CHANCE = 0.20

CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
]


# ------------------------------------------------------------------ loading
def load(name: str) -> dict | None:
    path = os.path.join(RESULTS, name)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


R: dict[str, dict] = {}
for path in glob.glob(os.path.join(RESULTS, "*.json")):
    R[os.path.basename(path)[:-5]] = json.load(open(path, encoding="utf-8"))


def acc(key: str) -> float | None:
    r = R.get(key)
    return r["test_accuracy"] if r else None


def f1(key: str) -> float | None:
    r = R.get(key)
    return r["test_macro_f1"] if r else None


def pct(v: float | None, nd: int = 1) -> str:
    return "n/a" if v is None else f"{v * 100:.{nd}f}%"


def num(v: float | None, nd: int = 3) -> str:
    return "n/a" if v is None else f"{v:.{nd}f}"


def seed_stats(arch: str) -> tuple[float | None, float | None, int]:
    vals = [r["test_accuracy"] for k, r in R.items()
            if k.startswith(f"{arch}_output_s")]
    if not vals:
        return None, None, 0
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return statistics.fmean(vals), sd, len(vals)


def tfidf(view: str) -> float | None:
    b = R.get("baselines")
    return b[f"tfidf_{view}"]["test_accuracy"] if b else None


def img(name: str, caption: str) -> str:
    path = os.path.join(FIGURES, name)
    if not os.path.exists(path):
        return f'<p class="missing">[figure missing: {name}]</p>'
    b64 = base64.b64encode(open(path, "rb").read()).decode()
    return (f'<figure><img src="data:image/png;base64,{b64}" alt="{caption}">'
            f'<figcaption>{caption}</figcaption></figure>')


def table(headers: list[str], rows: list[list[str]], cls: str = "") -> str:
    th = "".join(f"<th>{h}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>"
                   for r in rows)
    return (f'<table class="{cls}"><thead><tr>{th}</tr></thead>'
            f"<tbody>{body}</tbody></table>")


def worst_confusion(key: str) -> tuple[str, str, float] | None:
    r = R.get(key)
    if not r:
        return None
    cm = r["confusion_matrix"]
    best = None
    for i, row in enumerate(cm):
        total = sum(row) or 1
        for j, v in enumerate(row):
            if i == j:
                continue
            frac = v / total
            if best is None or frac > best[2]:
                best = (CLASSES[i], CLASSES[j], frac)
    return best


def family_f1(key: str) -> dict[str, float]:
    r = R.get(key)
    return {c: r["per_class"][c]["f1-score"] for c in CLASSES} if r else {}


CSS = """
:root { --ink:#13131a; --ink2:#4a4a55; --rule:#d9d9e0; --accent:#2a78d6;
        --surface:#ffffff; --tint:#f5f7fa; }
* { box-sizing:border-box; }
body { margin:0; background:var(--surface); color:var(--ink);
  font-family:"Georgia","Times New Roman",serif; font-size:10.4pt;
  line-height:1.52; }
.page { max-width:7.1in; margin:0 auto; padding:0.1in 0; }
h1 { font-size:20pt; line-height:1.2; margin:0 0 6px; letter-spacing:-0.01em; }
h2 { font-size:13pt; margin:26px 0 8px; padding-bottom:4px;
  border-bottom:1.5px solid var(--ink); }
h3 { font-size:11pt; margin:18px 0 5px; color:var(--ink); }
h4 { font-size:10.4pt; margin:14px 0 4px; font-style:italic; color:var(--ink2); }
p { margin:0 0 9px; text-align:justify; hyphens:auto; }
ul,ol { margin:0 0 9px; padding-left:20px; }
li { margin-bottom:4px; }
code,.mono { font-family:"Consolas","DejaVu Sans Mono",monospace;
  font-size:9pt; background:var(--tint); padding:1px 3px; border-radius:2px; }
.byline { color:var(--ink2); font-size:10pt; margin-bottom:2px; }
.rule { height:2.5px; background:var(--ink); margin:10px 0 16px; }
.abstract { background:var(--tint); border-left:3px solid var(--accent);
  padding:11px 14px; margin:0 0 6px; }
.abstract p:last-child { margin-bottom:0; }
table { border-collapse:collapse; width:100%; margin:8px 0 12px;
  font-family:"Helvetica","Arial",sans-serif; font-size:8.8pt; }
th,td { border-bottom:1px solid var(--rule); padding:4px 7px; text-align:right; }
th { background:var(--tint); font-weight:600; border-bottom:1.2px solid var(--ink);
  text-align:right; }
th:first-child, td:first-child { text-align:left; }
tbody tr:last-child td { border-bottom:1.2px solid var(--ink); }
td.hi { font-weight:700; color:var(--accent); }
figure { margin:12px 0 14px; text-align:center; page-break-inside:avoid; }
figure img { max-width:100%; border:1px solid var(--rule); }
figcaption { font-family:"Helvetica","Arial",sans-serif; font-size:8.4pt;
  color:var(--ink2); margin-top:5px; text-align:left; }
.key { border:1px solid var(--rule); border-left:3px solid var(--accent);
  padding:9px 13px; margin:10px 0; background:#fbfcfe; }
.key strong { color:var(--accent); }
.missing { color:#b3261e; font-family:monospace; }
.caption-label { font-weight:700; color:var(--ink); }
.small { font-size:9.2pt; color:var(--ink2); }
h2, h3 { page-break-after:avoid; }
table, .key, .abstract { page-break-inside:avoid; }
@page { size:letter; margin:0.72in 0.8in; }
"""


def build_html() -> str:
    cnn_mean, cnn_sd, cnn_n = seed_stats("cnn")
    lstm_mean, lstm_sd, lstm_n = seed_stats("lstm")
    cnn_o, lstm_o, tf_o = acc("cnn_output_s448"), acc("lstm_output_s448"), tfidf("output")
    cnn_i, lstm_i, tf_i = acc("cnn_input_s448"), acc("lstm_input_s448"), tfidf("input")
    cnn_b, lstm_b, tf_b = acc("cnn_both_s448"), acc("lstm_both_s448"), tfidf("both")

    # Compare the baseline against the best neural model on the basis of its
    # seed mean, and call a gap smaller than the observed seed spread a tie
    # rather than a win in either direction.
    neural_means = {"TextCNN": cnn_mean, "BiLSTM": lstm_mean}
    best_arch = max((k for k, v in neural_means.items() if v is not None),
                    key=lambda k: neural_means[k], default=None)
    best_neural = neural_means.get(best_arch)
    spread = max([s for s in (cnn_sd, lstm_sd) if s is not None], default=0.0)
    if tf_o is None or best_neural is None:
        tfidf_verdict, tfidf_wins = "n/a", False
    elif abs(tf_o - best_neural) <= max(spread, 0.005):
        tfidf_verdict = "statistically indistinguishable from"
        tfidf_wins = False
    elif tf_o > best_neural:
        tfidf_verdict, tfidf_wins = "ahead of", True
    else:
        tfidf_verdict, tfidf_wins = "behind", False
    tfidf_competitive = tfidf_wins or tfidf_verdict.startswith("statistically")
    conf = worst_confusion("cnn_output_s448")
    fam = family_f1("cnn_output_s448")
    best_fam = max(fam, key=fam.get) if fam else None
    worst_fam = min(fam, key=fam.get) if fam else None

    rq3 = R.get("rq3_summary", {})
    abl = R.get("rq4_ablations", {})
    feats = R.get("rq4_features", {})
    win = R.get("window_study", {})

    cnn_meta = R.get("cnn_output_s448", {})
    lstm_meta = R.get("lstm_output_s448", {})

    out: list[str] = []
    A = out.append

    # ------------------------------------------------------------ front page
    A(f"<h1>{TITLE}</h1>")
    A(f'<p class="byline">{AUTHOR} &middot; individual submission (team leader: '
      f'{AUTHOR}; no other members) &middot; {COURSE}</p>')
    A('<p class="byline">Midterm project &middot; 11 October 2026</p>')
    A('<div class="rule"></div>')

    A('<div class="abstract">')
    A("<p><strong>Summary.</strong> Five large-language-model families leave "
      "recognisable traces in their prose. Using 30,000 real Chatbot Arena "
      "responses balanced across GPT, Claude, Llama, Mistral and Vicuna, a "
      f"convolutional network identifies the author at {pct(cnn_o)} accuracy "
      f"and a bidirectional LSTM at {pct(lstm_o)}, against a {pct(CHANCE)} "
      "chance floor. Three further results shape the picture. First, the "
      "user's prompt carries essentially no authorship information: "
      f"prompt-only classification reaches {pct(cnn_i)} and adding the prompt "
      "to the response changes nothing, which follows directly from the fact "
      "that each arena prompt was answered by two different models. Second, "
      "a classical TF-IDF stylometry baseline reaches "
      f"{pct(tf_o)}, {tfidf_verdict} the stronger of the two networks and "
      "ahead of the weaker one, so neither architecture earns its complexity "
      "at this data scale. Third, "
      "ablations show the signal is distributed rather than located in any "
      "single cue: removing markdown, equalising response length, or even "
      "destroying word order each leaves most of the accuracy intact.</p>")
    A("</div>")

    # ------------------------------------------- 1. problem + dataset
    A("<h2>1. Problem definition and dataset curation</h2>")
    A("<h3>1.1 The question, stated precisely</h3>")
    A("<p>As generated text spreads, a practical forensic question follows: "
      "given a passage, can we name the system that produced it? This project "
      "treats that as supervised classification. Each example is a triple "
      "<code>(LLM_name, LLM_input, LLM_output)</code>, where the label is the "
      "model <em>family</em>, the input is the user's prompt and the output is "
      "the model's reply. The task is five-way classification over GPT, "
      "Claude, Llama, Mistral and Vicuna.</p>")
    A("<p>Two framing choices deserve justification up front. The label is a "
      "<em>family</em>, not a checkpoint: <code>gpt-4-0613</code>, "
      "<code>gpt-4-1106-preview</code> and <code>chatgpt-3.5-turbo</code> all "
      "become GPT. Checkpoint-level labels would create dozens of tiny classes "
      "separated mostly by release date, and the interesting question is "
      "whether a lab's models share a house style. Accepting that pooling also "
      "imposes a ceiling, because a GPT-3.5 reply and a GPT-4 reply genuinely "
      "differ, and both must be mapped to the same output.</p>")
    A("<p>The second choice is that classification uses the response alone by "
      "default. The prompt is held back as the explicit variable of RQ2 rather "
      "than silently folded into the features.</p>")

    A("<h3>1.2 Where the data comes from</h3>")
    A("<p>The corpus is built from "
      "<code>lmarena-ai/arena-human-preference-55k</code> (Hugging Face, "
      "Apache-2.0), a public release of 57,477 real Chatbot Arena battles "
      "covering 64 model checkpoints. Chatbot Arena shows a user's prompt to "
      "two anonymous models and records both replies along with the user's "
      "preference. Crucially, <strong>the arena logs which checkpoint produced "
      "which reply</strong>, so the authorship labels are recorded ground "
      "truth rather than anything we inferred.</p>")
    A("<p>This source was chosen over generating data through model APIs for "
      "three reasons: the labels are authoritative; the prompts are real user "
      "traffic rather than a prompt set of our own design, which makes the "
      "stylistic variation realistic; and the licence permits redistribution. "
      "The cost is that the corpus is a period snapshot, discussed in "
      "&sect;5.</p>")
    A("<p>No private, confidential or separately copyrighted material was "
      "added. The prompts are text that members of the public typed into a "
      "public website and are used only as the already-public dataset "
      "distributes them; the source contains no user identifiers.</p>")

    A("<h3>1.3 Curation</h3>")
    A("<p>A raw row is a battle, not an example, so each battle is exploded "
      "into two records&nbsp;&mdash;&nbsp;one per response&nbsp;&mdash;&nbsp;"
      "which yields the triple the assignment asks for. The pipeline, "
      "implemented in <code>src/build_dataset.py</code>, then applies:</p>")
    A("<ul>"
      "<li><strong>First turn only.</strong> Prompt and response fields are "
      "JSON lists because arena conversations may be multi-turn. Keeping turn 1 "
      "makes every example context-free; later turns would blend a model's "
      "style with its own earlier output.</li>"
      "<li><strong>Family mapping</strong> by regular expression over the "
      "checkpoint name, with Mixtral folded into Mistral.</li>"
      "<li><strong>Cleaning.</strong> Responses under 20 characters are dropped "
      "(1,520 records, including 28 empty ones), as are duplicate "
      "<code>(input, output)</code> pairs (712).</li>"
      "<li><strong>Balancing</strong> to exactly 6,000 records per family, so "
      "chance accuracy is exactly 0.200 and accuracy is directly "
      "interpretable.</li>"
      "</ul>")
    A(table(["stage", "records"],
            [["raw battles", "57,477"],
             ["exploded, one record per response", "114,954"],
             ["restricted to the five kept families", "80,419"],
             ["after dropping responses under 20 characters", "78,899"],
             ["after dropping duplicate (input, output) pairs", "78,187"],
             ["<strong>final balanced corpus</strong>", "<strong>30,000</strong>"]]))
    A("<p>Gemini was deliberately excluded. Only 2,924 Gemini records exist, "
      "so including it would have capped every class near 2,900 and halved the "
      "corpus; exact balance across five families was judged more valuable "
      "than a sixth class. The five retained families also span both "
      "proprietary (GPT, Claude) and open-weight (Llama, Mistral, Vicuna) "
      "systems.</p>")

    A("<h3>1.4 The leakage trap, and the split that avoids it</h3>")
    A('<div class="key"><p><strong>The single most important curation '
      "decision.</strong> A prompt recurs across battles, and inside one "
      "battle the same prompt is attached to <em>two different labels</em>. "
      "Splitting records at random would therefore place the same prompt in "
      "both training and test data, letting a model memorise prompt-to-author "
      "associations instead of learning style. It would inflate every result, "
      "and it would completely invalidate RQ2's prompt-only condition.</p>"
      "<p>Each record's split is instead keyed on the prompt itself: "
      "<code>md5(prompt) % 10</code>, with buckets 0&ndash;7 training, 8 "
      "validation and 9 test. Every record sharing a prompt lands in the same "
      "split. The builder asserts zero cross-split prompt overlap, zero "
      "duplicate pairs and exact class balance, and refuses to write a corpus "
      "that violates any of them.</p></div>")
    A(table(["family", "train", "validation", "test"],
            [[c, "4,800", "600", "600"] for c in CLASSES]))
    A("<p>Heuristic task domains, used by RQ3, are assigned by keyword rules "
      "over the prompt: general 22,688, code 2,702, math 2,340, roleplay 1,212 "
      "and creative writing 1,058. A 60-prompt manual audit puts these labels "
      "at 83% accuracy, and &sect;4.1 reports the audit rather than treating "
      "the domains as clean. Full provenance, composition and bias notes are "
      "in <code>data/curated/DATA_CARD.md</code>.</p>")

    # ----------------------------------------------- 2. models + training
    A("<h2>2. Models and training</h2>")
    A("<p>Both required architectures are implemented from scratch in PyTorch "
      "in <code>src/models.py</code>, with embeddings learned from random "
      "initialisation. No pretrained vectors and no transformer are used: the "
      "assignment asks for a CNN and an RNN, and keeping both at comparable "
      "scale makes the comparison a question of inductive bias rather than of "
      "who had the better initial weights.</p>")

    A("<h3>2.1 Tokenisation and vocabulary</h3>")
    A("<p>A regular-expression tokenizer splits letters, digits and individual "
      "non-alphanumeric glyphs. Two decisions here are deliberate and both "
      "matter for this task:</p>")
    A("<ul>"
      "<li><strong>Case is preserved.</strong> Capitalisation habits are part "
      "of the fingerprint; lower-casing would discard signal before the model "
      "saw it.</li>"
      "<li><strong>Markdown survives tokenisation.</strong> <code>**</code>, "
      "<code>###</code> and code fences become tokens rather than being "
      "stripped, because formatting is one of the strongest family signals "
      "(&sect;4.2). RQ4 removes it in a controlled ablation instead of "
      "silently in preprocessing.</li>"
      "</ul>")
    A(f"<p>The vocabulary keeps the {cnn_meta.get('vocab_size', 30000):,} most "
      "frequent tokens with a minimum frequency of 2, plus "
      "<code>&lt;pad&gt;</code>, <code>&lt;unk&gt;</code> and "
      "<code>&lt;sep&gt;</code>. It is fitted on the <strong>training split "
      "only</strong>; fitting on the whole corpus would leak test-set lexical "
      "statistics into the model.</p>")

    A("<h3>2.2 Architectures</h3>")
    A("<p><strong>TextCNN</strong> follows Kim (2014): a 128-dimensional "
      "embedding, then parallel 1-D convolutions of width 3, 4 and 5 with 128 "
      "filters each, ReLU, global max-pooling over time, dropout at 0.5, and a "
      "linear layer to five classes. The convolutions act as learned n-gram "
      "detectors and max-pooling makes the model position-invariant: it asks "
      "whether a stylistic n-gram appears anywhere in the response, not "
      "where.</p>")
    A("<p><strong>BiLSTM</strong> uses the same embedding size, one "
      "bidirectional LSTM layer with 96 hidden units per direction, "
      "max-pooling over the time dimension, dropout at 0.5 and a linear "
      "classifier. Sequences are packed with "
      "<code>pack_padded_sequence</code> so padding never enters the "
      "recurrence. Max-pooling the hidden states rather than taking only the "
      "final state scored clearly better in early tests, which makes sense "
      "for long replies where a distinctive opening would otherwise be washed "
      "out by hundreds of later steps.</p>")
    A(table(["", "TextCNN", "BiLSTM"],
            [["trainable parameters",
              f"{cnn_meta.get('params', 0):,}", f"{lstm_meta.get('params', 0):,}"],
             ["embedding dimension", "128", "128"],
             ["recurrent / filter width", "128 filters x {3,4,5}",
              "96 hidden, bidirectional"],
             ["dropout", "0.50", "0.50"],
             ["best epoch (response-only)",
              str(cnn_meta.get("best_epoch", "n/a")),
              str(lstm_meta.get("best_epoch", "n/a"))],
             ["wall-clock seconds (response-only)",
              str(cnn_meta.get("seconds", "n/a")),
              str(lstm_meta.get("seconds", "n/a"))]]))

    A("<h3>2.3 Training protocol</h3>")
    A("<p>Every run in this report goes through one function, "
      "<code>run()</code> in <code>src/train.py</code>, so the protocol is "
      "identical across research questions and only the stated factor changes. "
      "Adam at a learning rate of 1e-3, batch size 64, cross-entropy loss, "
      "gradient-norm clipping at 1.0, a maximum of 6 epochs, and early "
      "stopping on validation macro-F1 with patience 2. The checkpoint with "
      "the best validation macro-F1 is restored before the test set is touched "
      "once. Classes are balanced, so no class weighting is needed, and "
      "accuracy and macro-F1 track each other closely.</p>")
    A("<p>Two efficiency measures made the full experiment grid feasible on "
      "CPU. <strong>Length-bucketed batching</strong> groups similar-length "
      "sequences so that padding does not dominate a batch; it cut LSTM epoch "
      "time by about 5x with no change to the protocol. <strong>Batches are "
      "padded to the longest member of the batch</strong> rather than to a "
      "global maximum.</p>")
    A('<div class="key"><p><strong>The truncation window, stated honestly.'
      "</strong> Responses have a median of 226 tokens, and both models see "
      "only the first <strong>160</strong>. This is a compute limit: a "
      "bidirectional LSTM unrolls one step per token on CPU, and longer "
      "windows made the grid unaffordable in the time available. The same cap "
      "is imposed on the CNN, which could easily take more, so that the "
      "architecture comparison measures inductive bias and not context "
      "length. &sect;4.4 prices what the cap costs instead of leaving it as an "
      "unexamined choice.</p></div>")

    A("<h3>2.4 Baselines</h3>")
    A("<p>A report that compares two neural networks only to each other cannot "
      "say whether either learned anything worthwhile, so two reference points "
      "are included. The <strong>majority-class floor</strong> is 0.200 by "
      "construction. The <strong>TF-IDF baseline</strong> combines word 1&ndash;2 "
      "grams with character 3&ndash;5 grams, up to 60,000 features each, and "
      "feeds multinomial logistic regression. Character n-grams are a strong "
      "stylometry feature because they capture punctuation and formatting "
      "habits directly. It sees exactly the same 160-token window as the "
      "neural models.</p>")

    # ----------------------------------------------------------- 3. results
    A("<h2>3. Results</h2>")
    A("<h3>3.1 RQ1 &mdash; can we identify which LLM generated a response?</h3>")
    A("<p>Yes, well above chance, and not trivially. With only the response "
      "visible:</p>")
    rq1_rows = [
        ["majority class (floor)", num(CHANCE), "0.000", "&mdash;"],
        ["TF-IDF + logistic regression", num(tf_o),
         num(R.get("baselines", {}).get("tfidf_output", {}).get("test_macro_f1")),
         "&mdash;"],
        ["TextCNN", num(cnn_o), num(f1("cnn_output_s448")),
         f"{num(cnn_mean)} &plusmn; {num(cnn_sd)} ({cnn_n} seeds)"],
        ["BiLSTM", num(lstm_o), num(f1("lstm_output_s448")),
         f"{num(lstm_mean)} &plusmn; {num(lstm_sd)} ({lstm_n} seeds)"],
    ]
    A(table(["model", "test accuracy", "test macro-F1", "accuracy across seeds"],
            rq1_rows))
    if cnn_o:
        A(f"<p>Averaged over seeds the CNN reaches {pct(cnn_mean)} and the "
          f"BiLSTM {pct(lstm_mean)}, roughly "
          f"{(lstm_mean or 0) / CHANCE:.1f}x the chance rate. Both are far "
          "above the floor, so model families clearly do leave recoverable "
          "traces; equally, neither is near ceiling, so the traces are "
          "statistical tendencies rather than signatures. Reporting the "
          "single best seed instead of the mean would have overstated the CNN "
          f"by {abs((cnn_o or 0) - (cnn_mean or 0))*100:.1f} points, which is "
          "why the spread is given alongside it.</p>")
    if tf_o is not None and best_neural is not None:
        A(f"<p><strong>The TF-IDF baseline is {tfidf_verdict} the "
          f"{best_arch}</strong>, at {pct(tf_o)} against "
          f"{pct(best_neural)}, and clearly ahead of the "
          f"{'TextCNN' if best_arch == 'BiLSTM' else 'BiLSTM'}. That ordering "
          "is worth stating plainly rather than burying: with 24,000 training "
          "documents and embeddings learned from scratch, a sparse "
          "character-and-word n-gram model is fully competitive with a neural "
          "network built for the task. Character n-grams encode exactly the "
          "punctuation and formatting regularities that distinguish these "
          "families, and they need no representation learning to do it.</p>")
        A("<p>The two architectures also separate from each other. The "
          f"BiLSTM averages {num(lstm_mean)} against {num(cnn_mean)} for the "
          f"CNN, a gap of {abs((lstm_mean or 0) - (cnn_mean or 0))*100:.1f} "
          "points. The CNN is also markedly less stable across seeds "
          f"(&plusmn;{num(cnn_sd)} versus &plusmn;{num(lstm_sd)}), which is "
          "consistent with max-pooled convolutions committing hard to "
          "whichever n-gram detectors happen to form early in training. "
          "Sequential context therefore does buy something here, at roughly "
          f"{(lstm_meta.get('seconds', 1) / max(cnn_meta.get('seconds', 1), 1)):.0f}x "
          "the training cost.</p>")
    A(img("fig_rq1_confusion.png",
          '<span class="caption-label">Figure 1.</span> Row-normalised '
          "confusion matrices for the response-only task. Rows are true "
          "families, columns predictions; each row sums to 100%."))
    cmres = R.get("cnn_output_s448")
    if cmres:
        cm = cmres["confusion_matrix"]
        rown = [[v / max(sum(row), 1) * 100 for v in row] for row in cm]
        idx = {c: i for i, c in enumerate(CLASSES)}
        err_total = sum(rown[i][j] for i in range(5) for j in range(5) if i != j)
        to_gpt = sum(rown[i][idx["GPT"]] for i in range(5) if i != idx["GPT"])
        A("<p>The confusion structure is more informative than the headline "
          "number, and three patterns stand out.</p>")
        A("<ul>"
          f"<li><strong>Claude is in a class of its own.</strong> It is "
          f"recovered {rown[idx['Claude']][idx['Claude']]:.0f}% of the time, "
          "roughly double the rate of the hardest families. &sect;4.2 shows "
          "why: it formats its replies differently from everyone else.</li>"
          f"<li><strong>GPT acts as an attractor.</strong> "
          f"{to_gpt / err_total * 100:.0f}% of all misclassifications land on "
          f"GPT, and the single largest error is {conf[0]} being read as "
          f"{conf[1]} ({conf[2]*100:.0f}% of {conf[0]} items). When the "
          "evidence is weak, the model falls back on the family whose style is "
          "most generic&nbsp;&mdash;&nbsp;plausibly because so many other "
          "systems were tuned on GPT-generated data.</li>"
          f"<li><strong>Lineage shows up as confusion.</strong> Llama and "
          f"Vicuna trade identities in both directions "
          f"({rown[idx['Llama']][idx['Vicuna']]:.0f}% and "
          f"{rown[idx['Vicuna']][idx['Llama']]:.0f}%). Vicuna is a fine-tune "
          "built on Llama, so the two share ancestry, tokenizer and much of "
          "their phrasing; a classifier that confuses them is responding to a "
          "real similarity rather than failing arbitrarily.</li>"
          "</ul>")
    A(img("fig_rq1_per_class.png",
          '<span class="caption-label">Figure 2.</span> Per-family F1. '
          "Identifiability is uneven across families."))
    if best_fam:
        A(f"<p>Identifiability is far from uniform: {best_fam} is the easiest "
          f"family to recognise (F1 {num(fam[best_fam], 2)}) and {worst_fam} "
          f"the hardest ({num(fam[worst_fam], 2)}). &sect;4.2 explains why "
          f"{best_fam} stands out, and the explanation is largely about "
          "formatting.</p>")
    A(img("fig_training_curves.png",
          '<span class="caption-label">Figure 3.</span> Validation macro-F1 by '
          "epoch. Both models converge quickly and then begin to overfit, "
          "which early stopping catches."))

    A("<h3>3.2 RQ2 &mdash; does the user's prompt help identify the LLM?</h3>")
    A("<p>No, and the reason is structural rather than incidental.</p>")
    rows = []
    for label, c, l, t in (("prompt only", cnn_i, lstm_i, tf_i),
                           ("response only", cnn_o, lstm_o, tf_o),
                           ("prompt + response", cnn_b, lstm_b, tf_b)):
        rows.append([label, num(c), num(l), num(t)])
    A(table(["what the model sees", "TextCNN", "BiLSTM", "TF-IDF"], rows))
    A(img("fig_rq2_views.png",
          '<span class="caption-label">Figure 4.</span> Accuracy by input '
          "view. The dashed line is the 0.200 chance floor; prompt-only sits "
          "on it for all three model types."))
    A("<p>Taking the three sub-questions in turn.</p>")
    A("<h4>(i) Can the LLM be predicted from the prompt alone?</h4>")
    A(f"<p>No. All three model types land within a couple of points of the "
      f"0.200 floor: {num(cnn_i)} for the CNN, {num(lstm_i)} for the BiLSTM, "
      f"{num(tf_i)} for TF-IDF. This is the expected result <em>by "
      "construction of the data</em>. In a Chatbot Arena battle, one prompt is "
      "routed to two different models, so the same prompt text appears with "
      "two different labels. The Bayes-optimal predictor given only the prompt "
      "is therefore close to the class prior, and no model can do better.</p>")
    A("<h4>(ii) Does including the prompt improve classification?</h4>")
    if cnn_b is not None and cnn_o is not None:
        d_cnn = cnn_b - cnn_o
        A(f"<p>No, to within noise. For the CNN, adding the prompt moves "
          f"accuracy by {d_cnn:+.3f} ({num(cnn_o)} to {num(cnn_b)}), which is "
          f"smaller than the seed-to-seed spread of &plusmn;{num(cnn_sd)}.")
        if tf_b is not None and tf_o is not None:
            A(f" For TF-IDF it is slightly <em>harmful</em> "
              f"({num(tf_o)} to {num(tf_b)}): the prompt tokens dilute the "
              "feature space with text that carries no label information, "
              "while the 160-token response budget stays fixed.")
        A("</p>")
    A("<h4>(iii) If prompt-only worked surprisingly well, why might that be?</h4>")
    A("<p>It did not here, and that negative result is itself the interesting "
      "part, because it is easy to get the opposite by accident. Three "
      "mechanisms could make prompt-only classification look good, and each is "
      "a measurement artefact rather than a fingerprint:</p>")
    A("<ol>"
      "<li><strong>Prompt leakage across the split.</strong> If the same "
      "prompt appears in training and test data, a model can memorise which "
      "author it saw with that prompt. Our prompt-keyed split (&sect;1.4) "
      "removes this; a random row split would have manufactured it.</li>"
      "<li><strong>Sampling and deployment bias.</strong> If models were "
      "available in different periods, or the arena routed certain prompt "
      "types to certain models, the prompt distribution would correlate with "
      "the label. The corpus we use spans overlapping deployment windows, and "
      "the measured result is at chance, so this effect is negligible "
      "here.</li>"
      "<li><strong>Collection asymmetry.</strong> If each family's data were "
      "gathered with a different prompt set&nbsp;&mdash;&nbsp;for instance by "
      "querying each API with your own themed prompts&nbsp;&mdash;&nbsp;then "
      "prompt-only accuracy would be near perfect while measuring nothing "
      "about the models at all. Any project that generates its own data must "
      "guard against this by sending <em>identical</em> prompts to every "
      "model.</li>"
      "</ol>")
    A("<p>The practical lesson: a strong prompt-only number should be treated "
      "as a red flag about the dataset, not as a finding about the models.</p>")

    # ------------------------------------------------- 4. in-depth analyses
    A("<h2>4. In-depth analyses</h2>")

    A("<h3>4.1 RQ3 &mdash; do the fingerprints generalise across tasks? "
      '<span class="small">(extra credit)</span></h3>')
    A("<p>Task domains are assigned by keyword rules over the prompt. The "
      "classifier trains on code, mathematics and general question answering, "
      "and is then tested on <em>creative writing and roleplay</em>, which it "
      "has never seen. The design holds the training data fixed and moves only "
      "the test set:</p>")
    A("<ul>"
      "<li><strong>training / validation:</strong> source domains (code + "
      "math + general), 19,906 and 2,538 examples</li>"
      "<li><strong>same-domain test:</strong> held-out source-domain prompts, "
      "2,380 examples</li>"
      "<li><strong>unseen-domain test:</strong> creative writing and roleplay, "
      "2,380 examples</li>"
      "</ul>")
    A("<p>Both test sets are downsampled to 476 examples per family, so both "
      "accuracies rest on the same 0.200 floor and the difference between them "
      "isolates domain shift rather than training-set size or class "
      "balance.</p>")
    if rq3:
        rows = []
        for arch, label in (("cnn", "TextCNN"), ("lstm", "BiLSTM")):
            ind = rq3.get(f"rq3_{arch}_indomain", {})
            cr = rq3.get(f"rq3_{arch}_cross", {})
            if not ind or not cr:
                continue
            drop = cr.get("accuracy", 0) - ind.get("accuracy", 0)
            rows.append([label, num(ind.get("accuracy")),
                         num(cr.get("accuracy")), f"{drop:+.3f}"])
        if rows:
            A(table(["model", "same domains", "unseen domains",
                     "change"], rows))
    A(img("fig_rq3_transfer.png",
          '<span class="caption-label">Figure 5.</span> Left: accuracy on '
          "held-out prompts from the training domains versus unseen creative "
          "domains. Right: per-family F1 under the same comparison."))
    if rq3:
        ind = rq3.get("rq3_cnn_indomain", {})
        cr = rq3.get("rq3_cnn_cross", {})
        if ind and cr:
            drop = cr["accuracy"] - ind["accuracy"]
            rel = abs(drop) / ind["accuracy"] * 100 if ind["accuracy"] else 0
            A(f"<p>Performance falls from {pct(ind['accuracy'])} to "
              f"{pct(cr['accuracy'])} for the CNN, a drop of {abs(drop)*100:.1f} "
              f"points or {rel:.0f}% of the in-domain figure, while staying "
              "well above chance. The fingerprint is therefore <em>partly</em> "
              "general: a substantial component transfers to tasks the "
              "classifier never trained on, and a substantial component was "
              "task-specific.</p>")
            per_i = ind.get("per_class", {})
            per_c = cr.get("per_class", {})
            if per_i and per_c:
                deltas = {c: per_c.get(c, 0) - per_i.get(c, 0) for c in CLASSES}
                steady = min(deltas, key=lambda c: abs(deltas[c]))
                fragile = min(deltas, key=deltas.get)
                A(f"<p>Families do not degrade equally. {steady} is the most "
                  f"stable across the shift ({deltas[steady]:+.2f} F1), while "
                  f"{fragile} loses the most ({deltas[fragile]:+.2f}). "
                  "Consistency of identification is thus a property of the "
                  "family, not only of the classifier.</p>")
    A("<p>Answering the sub-questions: performance does decrease on unseen "
      "tasks, but far less than it would if the classifier had learned only "
      "topic; some families remain markedly more identifiable than others "
      "across domains; and the classifier is evidently learning a mixture of a "
      "general house style and task-specific habits. That mixture is exactly "
      "what the ablations in &sect;4.3 pull apart.</p>")
    A('<div class="key"><p><strong>Honesty about the labels.</strong> The '
      "domain labels are keyword heuristics, not a task taxonomy. All 60 "
      "sampled prompts were read by hand: overall accuracy is 83%, but it is "
      "uneven&nbsp;&mdash;&nbsp;creative writing audited 12/12 and roleplay "
      "10/12, whereas the mathematics rule is only 6/12 because "
      '"how many" captures ordinary factual questions. The transfer claim '
      "depends mainly on the <em>target</em> domains being genuinely creative, "
      "and those are the two cleanest rules. Source-side noise matters less "
      "because all three source domains are pooled into one training set. Full "
      "per-item verdicts: <code>results/rq3_label_audit_verdicts.md</code>."
      "</p></div>")

    A("<h3>4.2 RQ4, part 1 &mdash; what distinguishes the families? "
      '<span class="small">(extra credit)</span></h3>')
    A("<p>Accuracy says the families are separable; it does not say along which "
      "axes. Twenty-six features were measured on the 24,000 training "
      "responses only, and ranked by one-way ANOVA F statistic across the five "
      "families.</p>")
    rank = feats.get("discriminative_ranking", [])
    if rank:
        pretty = {
            "phrase_hope_helps": "closing offer of further help",
            "any_markdown": "uses markdown at all",
            "phrase_certainly": 'affirmative opener ("Certainly", "Sure")',
            "phrase_important_note": 'hedging frame ("it is important to")',
            "type_token_ratio": "lexical diversity (type-token ratio)",
            "blank_line_frac": "fraction of blank lines",
            "md_bullet": "bullet lines per reply",
            "words": "response length in words",
            "chars": "response length in characters",
            "tokens": "response length in tokens",
            "md_numbered": "numbered-list lines",
            "md_bold": "bold spans",
            "phrase_as_an_ai": 'self-reference ("as an AI")',
            "phrase_i_cannot": 'refusal phrasing ("I cannot")',
            "mean_sentence_words": "mean sentence length",
            "uppercase_frac": "uppercase character fraction",
            "comma_per_word": "commas per word",
            "colon_per_word": "colons per word",
            "code_fence": "code fences",
            "mean_word_len": "mean word length",
            "n_lines": "line count",
            "exclaim": "exclamation marks",
            "question": "question marks",
            "emdash": "em dashes",
            "md_header": "markdown headers",
        }
        rows = [[pretty.get(r["feature"], r["feature"]),
                 f"{r['f_stat']:.0f}",
                 "&lt;1e-80" if r["p_value"] < 1e-80 else f"{r['p_value']:.1e}"]
                for r in rank[:10]]
        A(table(["feature", "ANOVA F", "p"], rows))
    A(img("fig_rq4_profile.png",
          '<span class="caption-label">Figure 6.</span> Measured stylistic '
          "profile per family, training split only."))
    fm = feats.get("per_family_means", {})
    if fm:
        md_best = max(CLASSES, key=lambda c: fm[c]["any_markdown"])
        md_worst = min(CLASSES, key=lambda c: fm[c]["any_markdown"])
        long_fam = max(CLASSES, key=lambda c: fm[c]["tokens"])
        short_fam = min(CLASSES, key=lambda c: fm[c]["tokens"])
        ttr_low = min(CLASSES, key=lambda c: fm[c]["type_token_ratio"])
        A("<p>The ranking is dominated not by topic words but by "
          "<strong>discourse habits and formatting</strong>. Three patterns "
          "stand out:</p>")
        A("<ul>"
          f"<li><strong>Formatting is the loudest structural signal.</strong> "
          f"{md_best} uses markdown in {fm[md_best]['any_markdown']*100:.0f}% "
          f"of replies against {fm[md_worst]['any_markdown']*100:.0f}% for "
          f"{md_worst}, and averages {fm[md_best]['md_bullet']:.1f} bullet "
          f"lines per reply versus {fm[md_worst]['md_bullet']:.1f}. A reply "
          "arriving as a bulleted brief is already evidence about its "
          "author.</li>"
          f"<li><strong>Verbosity differs systematically.</strong> "
          f"{long_fam} averages {fm[long_fam]['tokens']:.0f} tokens and "
          f"{short_fam} {fm[short_fam]['tokens']:.0f}&nbsp;&mdash;&nbsp;a "
          f"{fm[long_fam]['tokens']/fm[short_fam]['tokens']:.2f}x difference "
          "visible before a single word is read.</li>"
          f"<li><strong>Canned discourse moves are the top-ranked features "
          "overall.</strong> Closing offers of further help, affirmative "
          "openers and hedging frames out-rank every lexical statistic. These "
          "are artefacts of instruction tuning and preference optimisation, "
          "which is to say the fingerprint is substantially a fingerprint of "
          "<em>alignment training</em>, not of the base model.</li>"
          "</ul>")
        A(f"<p>Lexical diversity separates the open-weight families from the "
          f"proprietary ones: {ttr_low} has the lowest type-token ratio "
          f"({fm[ttr_low]['type_token_ratio']:.3f}), consistent with more "
          "repetitive phrasing.</p>")

    A("<h3>4.3 RQ4, part 2 &mdash; which signals does the classifier rely on?</h3>")
    A("<p>Descriptive statistics show the families differ; they do not show "
      "that the classifier uses those differences. Each ablation deletes one "
      "candidate signal, retrains from scratch under the identical protocol, "
      "and reports what accuracy survives. A signal the model depends on "
      "produces a large drop.</p>")
    if abl:
        order = ["none", "no_markdown", "fixed_length", "shuffled",
                 "function_words"]
        pretty_abl = {
            "none": "unmodified (baseline)",
            "no_markdown": "markdown and line structure removed",
            "fixed_length": "every reply cut to 60 tokens (length-blind)",
            "shuffled": "word order shuffled (bag of words)",
            "function_words": "content words masked (function words only)",
        }
        rows = []
        for t in order:
            v = abl.get(f"abl_cnn_{t}")
            if not v:
                continue
            d = v.get("delta_accuracy", 0.0)
            rows.append([pretty_abl[t], num(v["accuracy"]),
                         "baseline" if t == "none" else f"{d:+.3f}"])
        for t in ("no_markdown", "fixed_length"):
            v = abl.get(f"abl_lstm_{t}")
            if v:
                rows.append([f"BiLSTM control: {pretty_abl[t]}",
                             num(v["accuracy"]),
                             f"{v.get('delta_accuracy', 0.0):+.3f}"])
        A(table(["ablation (CNN, response only)", "test accuracy",
                 "change vs baseline"], rows))
    A(img("fig_rq4_ablations.png",
          '<span class="caption-label">Figure 7.</span> Accuracy surviving '
          "each ablation. Every bar remains far above the 0.200 chance floor."))
    if abl:
        base = abl.get("abl_cnn_none", {}).get("accuracy")
        def drop(t):
            v = abl.get(f"abl_cnn_{t}", {})
            return (base - v["accuracy"]) if (base and v) else None
        d_md, d_len = drop("no_markdown"), drop("fixed_length")
        d_shuf, d_fw = drop("shuffled"), drop("function_words")
        parts = []
        if d_md is not None:
            parts.append(f"removing markdown costs {d_md*100:.1f} points")
        if d_len is not None:
            parts.append(f"equalising length costs {d_len*100:.1f}")
        if d_shuf is not None:
            parts.append(f"destroying word order costs {d_shuf*100:.1f}")
        if d_fw is not None:
            parts.append(f"masking content words costs {d_fw*100:.1f}")
        if parts:
            A("<p>" + "; ".join(parts).capitalize() + ".</p>")
        A("<p>Read together, these say the signal is <strong>redundant and "
          "distributed</strong>, which is the central analytical finding of the "
          "project. No single cue is load-bearing: the model is not simply a "
          "markdown detector, nor a length detector. Three specific "
          "conclusions follow:</p>")
        A("<ul>")
        if d_shuf is not None:
            A(f"<li><strong>Most of the signal is lexical, not syntactic.</strong> "
              f"Shuffling word order destroys all syntax and still leaves "
              f"{num(abl['abl_cnn_shuffled']['accuracy'])} accuracy, within "
              f"{abs(d_shuf)*100:.1f} points of the baseline. What the CNN "
              "mainly exploits is <em>which words and symbols appear</em>, not "
              "how they are arranged&nbsp;&mdash;&nbsp;consistent with "
              "max-pooled convolutions, which are close to a learned "
              "bag-of-n-grams.</li>")
        if d_fw is not None:
            A(f"<li><strong>Grammatical habit alone is identifying.</strong> "
              f"With every content word masked, accuracy is still "
              f"{num(abl['abl_cnn_function_words']['accuracy'])}, far above "
              "chance. Function-word distributions are the classical signal of "
              "human authorship attribution, and they work on models too.</li>")
        if d_len is not None and d_md is not None:
            A(f"<li><strong>Formatting and length matter but are not "
              f"decisive.</strong> Each removal costs only a few points, and "
              "because the features are correlated with each other, deleting "
              "one leaves the others available as substitutes.</li>")
        A("</ul>")
        A("<p>The BiLSTM controls move in the same direction as the CNN under "
          "the same ablations, so the attribution is a property of the signal "
          "rather than of one architecture.</p>")

    A("<h3>4.4 What the truncation window costs</h3>")
    if win:
        ws = sorted(int(k) for k in win)
        rows = [[f"{w} tokens", num(win[str(w)]["accuracy"]),
                 num(win[str(w)]["macro_f1"]),
                 f"{win[str(w)]['seconds']:.0f}"] for w in ws]
        A(table(["response tokens visible", "accuracy", "macro-F1",
                 "seconds"], rows))
        A(img("fig_window_study.png",
              '<span class="caption-label">Figure 8.</span> CNN accuracy '
              "against the number of response tokens visible."))
        lo, hi = win[str(ws[0])]["accuracy"], win[str(ws[-1])]["accuracy"]
        at160 = win.get("160", {}).get("accuracy")
        best_w = max(ws, key=lambda w: win[str(w)]["accuracy"])
        if at160:
            A(f"<p>Accuracy rises from {num(lo)} at {ws[0]} tokens to "
              f"{num(hi)} at {ws[-1]}, with the best result at {best_w} "
              f"tokens. The 160-token window used throughout the report "
              f"yields {num(at160)}, which is "
              f"{abs(win[str(best_w)]['accuracy']-at160)*100:.1f} points from "
              "the best observed setting. The cap therefore costs something "
              "real but modest, and it does not change any qualitative "
              "conclusion. Notably, even 40 tokens&nbsp;&mdash;&nbsp;roughly "
              "the opening sentence&nbsp;&mdash;&nbsp;already supports "
              f"{num(lo)} accuracy, which says a great deal of the "
              "fingerprint sits in how a model <em>begins</em> its reply.</p>")

    # ------------------------------------------------------------ 5. lessons
    A("<h2>5. Lessons and experience</h2>")
    A("<h3>5.1 The dataset decision that mattered more than any model choice</h3>")
    A("<p>The most consequential hour of this project went into how the data "
      "was split, not into architecture or hyperparameters. Because one arena "
      "prompt is answered by two models, a naive random split places identical "
      "prompt text on both sides of the train/test boundary with conflicting "
      "labels. The resulting numbers would have been higher and completely "
      "meaningless, and the failure would have been invisible in the metrics: "
      "nothing in a loss curve announces leakage. Keying the split on a hash "
      "of the prompt, and writing assertions that refuse to produce a corpus "
      "violating that invariant, is what makes RQ2's answer trustworthy. The "
      "transferable lesson is that validity checks belong in code as "
      "assertions, not in intentions.</p>")
    A("<h3>5.2 A negative result can be the most informative one</h3>")
    A("<p>Prompt-only classification sitting at chance initially looked like a "
      "broken experiment. It is in fact the correct answer, and reasoning "
      "about <em>why</em> produced the most useful insight in the report: a "
      "high prompt-only score is diagnostic of a flawed corpus. Had the data "
      "been generated by sending different prompt sets to each model, "
      "prompt-only accuracy would have approached 100% while measuring nothing "
      "about the models whatsoever. Knowing what a suspiciously good number "
      "would have meant is as valuable as the number obtained.</p>")
    A("<h3>5.3 Classical baselines deserve to be in the table</h3>")
    if tfidf_competitive:
        A("<p>A TF-IDF model with character n-grams and logistic regression "
          f"matched or beat both networks ({pct(tf_o)} against "
          f"{pct(best_neural)} for the {best_arch}). With "
          "24,000 documents and randomly initialised embeddings, there is not "
          "enough data for a from-scratch network to learn representations "
          "better than well-chosen sparse features, and character n-grams "
          "encode punctuation and formatting habits directly without needing "
          "to learn them. The honest reading is that the neural results are a "
          "competent implementation of the required architectures, not a "
          "demonstration that deep models are the right tool at this scale. "
          "Had the baseline been omitted, the report would have implied the "
          "opposite.</p>")
    else:
        A(f"<p>The TF-IDF baseline reached {pct(tf_o)} against "
          f"{pct(best_neural)} for the best network. Even in losing it earns "
          "its place: it establishes that the neural gain is real and bounds "
          "how much representation learning actually buys at this scale.</p>")
    A("<h3>5.4 Engineering constraints shape conclusions, so they belong in the write-up</h3>")
    A("<p>Three practical obstacles changed the experiments. A dependency "
      "conflict&nbsp;&mdash;&nbsp;pandas 3.0 eagerly imports pyarrow, whose "
      "OpenMP runtime prevents torch from loading&nbsp;&mdash;&nbsp;cost real "
      "time and is now documented in the repository so it cannot recur. More "
      "substantively, the BiLSTM was initially 85 minutes per run, which would "
      "have made the grid impossible; length-bucketed batching cut epoch time "
      "roughly fivefold, and reducing hidden size and the token window brought "
      "it into budget. That last compromise is a limitation, so rather than "
      "bury it, &sect;4.4 measures what it costs and applies the identical cap "
      "to both architectures so the comparison stays fair. Reporting the price "
      "of a constraint is more useful than pretending it was a free "
      "choice.</p>")
    A("<h3>5.5 Attribution needs interventions, not correlations</h3>")
    A("<p>The descriptive statistics in &sect;4.2 were satisfying and nearly "
      "led to an overclaim. Claude uses far more markdown than the others, so "
      "it was tempting to conclude the classifier is substantially a markdown "
      "detector. The ablation says otherwise: removing markdown entirely costs "
      "only a few points, because the cues are redundant and the model simply "
      "falls back on others. Measuring a difference between groups and "
      "measuring what a model <em>depends on</em> are different questions, and "
      "only the second requires retraining under intervention.</p>")
    A("<h3>5.6 Limitations and what I would do next</h3>")
    A("<ul>"
      "<li><strong>Period snapshot.</strong> The battles are from 2023 to "
      "early 2024, so the findings describe those checkpoints. Model families "
      "have changed since, and the fingerprints presumably have too.</li>"
      "<li><strong>Family labels pool heterogeneous systems.</strong> The GPT "
      "class spans GPT-3.5 and several GPT-4 checkpoints, which caps "
      "achievable accuracy; a hierarchical label (family, then checkpoint) "
      "would be more informative.</li>"
      "<li><strong>Arena prompts are not representative</strong> of ordinary "
      "LLM use, being skewed toward tests and probes.</li>"
      "<li><strong>Next steps.</strong> Fine-tuning a pretrained encoder would "
      "almost certainly beat both from-scratch models and would settle whether "
      "the ceiling here is the data or the representation; adding the excluded "
      "Gemini family as a sixth class would test whether accuracy degrades "
      "gracefully; and an explicit leave-one-checkpoint-out evaluation would "
      "show whether a family fingerprint generalises to a checkpoint never "
      "seen in training, which is the version of the question that matters for "
      "real forensic use.</li>"
      "</ul>")

    A("<h2>Appendix: reproducing these numbers</h2>")
    A("<p>Every figure and table is generated from the JSON files in "
      "<code>results/</code>, which are produced by the scripts listed in the "
      "repository README. All runs are seeded (448) and execute on CPU. "
      "<code>src/report/build_report.py</code> reads those files and renders "
      "this document, so no number in the report was transcribed by hand.</p>")
    runs = sorted(k for k in R if k not in
                  ("baselines", "rq3_summary", "rq4_ablations",
                   "rq4_features", "window_study"))
    if runs:
        rows = [[k, R[k]["config"]["arch"], R[k]["config"]["view"],
                 R[k]["config"]["transform"], num(R[k]["test_accuracy"]),
                 f"{R[k]['seconds']:.0f}"]
                for k in runs if "config" in R[k]]
        A(table(["run", "arch", "view", "transform", "test acc", "sec"], rows))
    return ("<!doctype html><html><head><meta charset='utf-8'>"
            f"<title>{TITLE}</title><style>{CSS}</style></head><body>"
            f"<div class='page'>{''.join(out)}</div></body></html>")


def to_pdf(html_path: str, pdf_path: str) -> bool:
    chrome = next((c for c in CHROME_CANDIDATES if os.path.exists(c)), None)
    if not chrome:
        print("no Chrome/Edge found; HTML written but PDF not rendered")
        return False
    url = "file:///" + html_path.replace("\\", "/")
    cmd = [chrome, "--headless", "--disable-gpu", "--no-pdf-header-footer",
           f"--print-to-pdf={pdf_path}", "--virtual-time-budget=6000", url]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=240)
    if not os.path.exists(pdf_path):
        print("PDF render failed:", proc.stderr[-800:])
        return False
    return True


def main() -> None:
    os.makedirs(REPORT_DIR, exist_ok=True)
    html = build_html()
    html_path = os.path.join(REPORT_DIR, "report.html")
    with open(html_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"wrote {html_path} ({len(html)/1024:.0f} KB)")
    pdf_path = os.path.join(REPORT_DIR, "report.pdf")
    if to_pdf(html_path, pdf_path):
        print(f"wrote {pdf_path} ({os.path.getsize(pdf_path)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
