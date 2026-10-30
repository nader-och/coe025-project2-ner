"""Make the result figures (results/figures/*.png) from the saved result files.

Every number in a figure is read from results/test_results.json,
results/error_analysis_test.json, results/timing/timing_colab.json or
results/validation/crf_ablation.json - nothing is typed in by hand.

Run from the project root:  python -m src.figures
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils import RESULTS_DIR, load_json

FIG_DIR = os.path.join(RESULTS_DIR, "figures")
METHODS = ["baseline", "hmm", "crf", "bert_softmax", "bert_crf"]
LABELS = {"all_O": "All-O", "baseline": "Baseline", "hmm": "HMM", "crf": "Feature CRF",
          "bert_softmax": "BERT + softmax", "bert_crf": "BERT + CRF"}
# Not in slides: a validated colorblind-safe palette (blue, orange, aqua, yellow).
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
BLUE_DARK, GRAY = "#104281", "#9a9994"
TEXT, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"

plt.rcParams.update({"font.size": 12, "axes.edgecolor": GRAY, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": TEXT,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                     "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, name), dpi=200)
    plt.close(fig)
    print("saved results/figures/" + name)


def f1_by_method(res):
    """Week 6, Slides 38-39: test micro F1 per method (best method darker)."""
    vals = [res[m]["micro"]["f1"] for m in METHODS]
    best = max(range(len(vals)), key=lambda i: vals[i])
    fig, ax = plt.subplots(figsize=(8, 4))
    y = range(len(METHODS))[::-1]
    ax.barh(list(y), vals, height=0.6, color=[BLUE_DARK if i == best else BLUE for i in range(len(vals))])
    for yi, v in zip(y, vals):
        ax.text(v + 0.01, yi, f"{v:.3f}", va="center", color=TEXT)
    ax.set_yticks(list(y), [LABELS[m] for m in METHODS])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Entity micro F1 (exact match, test set)")
    ax.xaxis.grid(True, color="#e6e5e0"); ax.set_axisbelow(True)
    ax.set_title("Entity F1 on the BC5CDR test set", loc="left")
    save(fig, "f1_by_method.png")


def grouped(res_pairs, series, colors, title, xlabel, name, keys=None):
    """Grouped horizontal bars with a legend and a number on every bar."""
    keys = keys or METHODS
    fig, ax = plt.subplots(figsize=(8, 5.0))
    n = len(series)
    h = 0.8 / n
    for k, (label, getter) in enumerate(series):
        for i, m in enumerate(keys):
            yi = len(keys) - 1 - i + (n - 1 - k - (n - 1) / 2) * h
            v = getter(res_pairs[m])
            ax.barh(yi, v, height=h * 0.9, color=colors[k], label=label if i == 0 else None)
            ax.text(v + 0.01, yi, f"{v:.3f}", va="center", fontsize=10, color=TEXT)
    ax.set_yticks(range(len(keys))[::-1], [LABELS[m] for m in keys])
    ax.set_xlim(0, 1.1)
    ax.set_xlabel(xlabel)
    ax.xaxis.grid(True, color="#e6e5e0"); ax.set_axisbelow(True)
    ax.legend(ncol=n, loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=False)
    ax.set_title(title, loc="left")
    save(fig, name)


def error_categories(err):
    """Week 6, Slide 41: boundary / type / missed / spurious errors per method."""
    cats = [("boundary", BLUE), ("type", ORANGE), ("missed", AQUA), ("spurious", YELLOW)]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for i, m in enumerate(METHODS):
        left, yi = 0, len(METHODS) - 1 - i
        for c, color in cats:
            v = err[m]["error_categories"][c]
            ax.barh(yi, v, left=left, height=0.6, color=color, edgecolor=SURFACE, linewidth=2,
                    label=c if i == 0 else None)
            if v >= 300:
                ax.text(left + v / 2, yi, str(v), ha="center", va="center", fontsize=9, color=TEXT)
            left += v
        ax.text(left + 40, yi, f"total {left}", va="center", fontsize=10, color=MUTED)
    ax.set_yticks(range(len(METHODS))[::-1], [LABELS[m] for m in METHODS])
    ax.set_xlabel("Number of errors (test set)")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.18), frameon=False)
    ax.set_title("Error types per method (Week 6, Slide 41)", loc="left")
    save(fig, "error_categories.png")


def speed(timing):
    """Week 1, Slide 27: prediction time per sentence, same machine, log scale."""
    rows = [(f"{LABELS[m]} (CPU)", timing[f"{m}_cpu"]["ms_per_sentence"]) for m in METHODS]
    rows += [(f"{LABELS[m]} (GPU)", timing[f"{m}_gpu"]["ms_per_sentence"])
             for m in ("bert_softmax", "bert_crf")]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    y = list(range(len(rows)))[::-1]
    ax.barh(y, [v for _, v in rows], height=0.6, color=BLUE)
    for yi, (_, v) in zip(y, rows):
        text = f"{v:.3f}" if v < 0.1 else f"{v:.2f}" if v < 10 else f"{v:.0f}"
        ax.text(v * 1.15, yi, f"{text} ms", va="center", fontsize=10, color=TEXT)
    ax.set_xscale("log")
    ax.set_xlim(0.003, 2000)
    ticks = [0.01, 0.1, 1, 10, 100, 1000]
    ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
    ax.minorticks_off()
    ax.set_yticks(y, [r for r, _ in rows])
    ax.set_xlabel("Milliseconds per sentence (log scale, same Colab machine)")
    ax.set_title("Prediction speed", loc="left")
    save(fig, "speed.png")


def ablation(abl):
    """Feature ablation of the CRF on validation (rubric: feature engineering)."""
    keys = ["full", "without_word", "without_suffix", "without_neighbors", "without_case_shape"]
    names = {"full": "All features", "without_word": "− word identity", "without_suffix": "− suffix",
             "without_neighbors": "− neighboring words", "without_case_shape": "− capitalization/shape"}
    vals = [abl[k]["val_micro_f1"] for k in keys]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    y = list(range(len(keys)))[::-1]
    ax.barh(y, vals, height=0.6, color=[BLUE_DARK] + [BLUE] * (len(keys) - 1))
    for yi, v, k in zip(y, vals, keys):
        delta = "" if k == "full" else f"  ({v - vals[0]:+.3f})"
        ax.text(v + 0.005, yi, f"{v:.3f}{delta}", va="center", fontsize=10, color=TEXT)
    ax.set_yticks(y, [names[k] for k in keys])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Entity micro F1 (validation set)")
    ax.set_title("Feature CRF: remove one feature group at a time", loc="left")
    save(fig, "crf_ablation.png")


def main():
    res = load_json(RESULTS_DIR, "test_results.json")["results"]
    err = load_json(RESULTS_DIR, "error_analysis_test.json")["methods"]
    timing = load_json(RESULTS_DIR, "timing", "timing_colab.json")["methods"]
    abl = load_json(RESULTS_DIR, "validation", "crf_ablation.json")

    f1_by_method(res)
    grouped(res, [("Chemical", lambda r: r["per_type"]["Chemical"]["f1"]),
                  ("Disease", lambda r: r["per_type"]["Disease"]["f1"])],
            [BLUE, ORANGE], "F1 per entity type (test set)", "F1 (exact match)", "f1_by_type.png")
    grouped(res, [("Token accuracy", lambda r: r["token_accuracy"]),
                  ("Entity micro F1", lambda r: r["micro"]["f1"])],
            [BLUE, ORANGE], "Token accuracy hides missed entities (Slide 36)", "Score (test set)",
            "accuracy_vs_f1.png", keys=["all_O"] + METHODS)
    seen = {m: {"seen": err[m]["slices"]["all/seen"]["recall"],
                "unseen": err[m]["slices"]["all/unseen"]["recall"]} for m in METHODS}
    grouped(seen, [("Seen in training", lambda r: r["seen"]),
                   ("Unseen name", lambda r: r["unseen"])],
            [BLUE, ORANGE], "Recall on seen vs unseen entity names (Slide 41)", "Recall (test set)",
            "seen_vs_unseen.png")
    error_categories(err)
    speed(timing)
    ablation(abl)


if __name__ == "__main__":
    main()
