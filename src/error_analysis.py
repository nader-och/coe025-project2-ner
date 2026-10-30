"""Error analysis (rubric: "error analysis and insights").

Week 6, Slide 41: inspect error categories (boundary, type, missed, spurious)
and inspect slices (rare type, unseen name), because averages hide weak groups.
Week 6, Slide 87: "Use the simplest suitable baseline, then test whether
structured decoding fixes the observed errors."
Week 6, Slides 23, 80: count invalid IOB2 sequences per method.

Run from the project root:  python -m src.error_analysis            (test, at the end)
                            python -m src.error_analysis validation (during development)
"""
import sys
from collections import Counter

from data.bc5cdr import load_split
from src.bio import START, get_spans, is_allowed
from src.final_eval import METHODS, NAMES, classical_predictions, load_all_predictions
from src.metrics import safe_divide
from src.utils import RESULTS_DIR, save_json

CATEGORIES = ["boundary", "type", "missed", "spurious"]


def overlaps(a, b):
    return a[0] < b[1] and b[0] < a[1]


def categorize_errors(gold_tags, pred_tags):
    """Week 6, Slide 41 error categories, for one sentence.

    Every predicted span that is not an exact match is:
      type     - same boundaries as a gold span, different type
      boundary - overlaps a gold span but the boundaries differ
      spurious - overlaps no gold span (an extra entity)
    Every gold span that no prediction overlaps is: missed.
    (A gold span overlapped by a wrong prediction is counted once, through
    that prediction's boundary/type error.)
    """
    gold, pred = set(get_spans(gold_tags)), set(get_spans(pred_tags))
    counts = Counter()
    for p in pred - gold:
        same_bounds = [g for g in gold if g[0] == p[0] and g[1] == p[1]]
        if same_bounds:
            counts["type"] += 1
        elif any(overlaps(p, g) for g in gold):
            counts["boundary"] += 1
        else:
            counts["spurious"] += 1
    for g in gold - pred:
        if not any(overlaps(g, p) for p in pred):
            counts["missed"] += 1
    return counts


def training_entity_names():
    names = set()
    for s in load_split("train"):
        for start, end, t in get_spans(s["tags"]):
            names.add((" ".join(s["tokens"][start:end]).lower(), t))
    return names


def slice_recall(sentences, preds, train_names):
    """Week 6, Slide 41 slices: recall on 'seen' vs 'unseen name' gold entities.

    seen   = the same text (lower case) with the same type is an entity in training
    unseen = it never appears as an entity of that type in training
    Recall is used because a slice is a set of GOLD entities.
    """
    found, total = Counter(), Counter()
    for s, p in zip(sentences, preds):
        pred_spans = set(get_spans(p))
        for g in get_spans(s["tags"]):
            seen = (" ".join(s["tokens"][g[0]:g[1]]).lower(), g[2]) in train_names
            key = f"{g[2]}/{'seen' if seen else 'unseen'}"
            for k in (key, "all/seen" if seen else "all/unseen"):
                total[k] += 1
                found[k] += g in pred_spans
    return {k: {"recall": safe_divide(found[k], total[k]), "found": found[k], "total": total[k]}
            for k in sorted(total)}


def invalid_patterns(preds):
    """Which illegal transitions appear (e.g. O -> I-Disease)? Slides 23, 80."""
    patterns = Counter()
    for tags in preds:
        prev = START
        for t in tags:
            if not is_allowed(prev, t):
                patterns[f"{prev} -> {t}"] += 1
            prev = t
    return dict(patterns.most_common())


def spans_text(tokens, tags):
    return ", ".join(f"[{' '.join(tokens[a:b])}]{t}" for a, b, t in get_spans(tags)) or "(none)"


def tags_text(tokens, tags):
    """Raw word/tag pairs (only the non-O tags), to make invalid transitions visible."""
    return " ".join(f"{w}/{t}" for w, t in zip(tokens, tags) if t != "O") or "(all O)"


def structured_decoding_check(sentences, preds):
    """Week 6, Slide 87: "test whether structured decoding fixes the observed errors".

    On the sentences where BERT + softmax predicted an invalid IOB2 sequence,
    compare span counts of BERT + softmax and BERT + CRF (same encoder, other decoder).
    """
    from src.metrics import evaluate
    if "bert_softmax" not in preds or "bert_crf" not in preds:
        return None
    from src.bio import count_invalid
    idx = [i for i, p in enumerate(preds["bert_softmax"]) if count_invalid(p) > 0]
    gold = [sentences[i]["tags"] for i in idx]
    out = {"sentences_with_invalid_softmax_output": len(idx)}
    for m in ("bert_softmax", "bert_crf"):
        r = evaluate(gold, [preds[m][i] for i in idx], entity_types=["Chemical", "Disease"])
        out[m] = {"micro_f1": r["micro"]["f1"], "tp": r["micro"]["tp"], "fp": r["micro"]["fp"],
                  "fn": r["micro"]["fn"],
                  "fully_correct_sentences": sum(preds[m][i] == sentences[i]["tags"] for i in idx)}
    return out


def pick_examples(sentences, preds, train_names, max_examples=10):
    """Choose real sentences where the methods disagree (Slide 87 comparison).

    One rule per question we want to answer; we keep the shortest sentence for
    each rule so that it fits on a slide.
    """
    methods = [m for m in METHODS if m in preds]
    rules = {
        "baseline wrong, HMM right (label transitions help)":
            lambda ok, s, i: not ok["baseline"] and ok["hmm"],
        "HMM wrong, CRF right (rich features help)":
            lambda ok, s, i: not ok["hmm"] and ok["crf"],
        "CRF wrong, BERT + softmax right (context from the encoder)":
            lambda ok, s, i: not ok["crf"] and ok.get("bert_softmax", False),
        "BERT + softmax invalid IOB2, BERT + CRF right":
            lambda ok, s, i: "bert_softmax" in preds and
            any(not is_allowed(a, b) for a, b in zip([START] + preds["bert_softmax"][i],
                                                    preds["bert_softmax"][i])) and ok.get("bert_crf"),
        "BERT + CRF wrong, BERT + softmax right (CRF is not always better)":
            lambda ok, s, i: "bert_crf" in preds and not ok["bert_crf"] and ok["bert_softmax"],
        "unseen name: only BERT methods right":
            lambda ok, s, i: not ok["crf"] and not ok["hmm"] and ok.get("bert_crf", False)
            and ok.get("bert_softmax", False) and any(
                (" ".join(s["tokens"][a:b]).lower(), t) not in train_names
                for a, b, t in get_spans(s["tags"])),
        "every method wrong":
            lambda ok, s, i: not any(ok.values()),
        "boundary error by the best method":
            lambda ok, s, i: "bert_crf" in preds and
            categorize_errors(s["tags"], preds["bert_crf"][i])["boundary"] > 0,
    }
    examples, used = [], set()
    for rule, test in rules.items():
        candidates = []
        for i, s in enumerate(sentences):
            if not (6 <= len(s["tokens"]) <= 25) or not get_spans(s["tags"]) or i in used:
                continue
            ok = {m: preds[m][i] == s["tags"] for m in methods}
            try:
                if test(ok, s, i):
                    candidates.append(i)
            except KeyError:
                break
        if candidates:
            i = min(candidates, key=lambda j: len(sentences[j]["tokens"]))
            used.add(i)
            examples.append({"rule": rule, "sentence_index": i,
                             "tokens": sentences[i]["tokens"], "gold": sentences[i]["tags"],
                             "predictions": {m: preds[m][i] for m in methods}})
        if len(examples) >= max_examples:
            break
    return examples


def main():
    split = sys.argv[1] if len(sys.argv) > 1 else "test"
    sentences = load_split(split)
    preds = load_all_predictions(split)
    if "baseline" not in preds:          # validation: predict methods 1-3 here
        preds.update(classical_predictions(split)[0])
    train_names = training_entity_names()

    report = {"split": split, "methods": {}}
    for m in [m for m in METHODS if m in preds]:
        cats, cats_by_type = Counter(), Counter()
        for s, p in zip(sentences, preds[m]):
            cats += categorize_errors(s["tags"], p)
        for t in ("Chemical", "Disease"):
            only_t = lambda tags: [x if x.endswith(t) else "O" for x in tags]
            for s, p in zip(sentences, preds[m]):
                for c, n in categorize_errors(only_t(s["tags"]), only_t(p)).items():
                    cats_by_type[f"{t}/{c}"] += n
        report["methods"][m] = {"error_categories": {c: cats[c] for c in CATEGORIES},
                                "error_categories_by_type": dict(cats_by_type),
                                "slices": slice_recall(sentences, preds[m], train_names),
                                "invalid_patterns": invalid_patterns(preds[m])}
        print(f"{NAMES[m]:<26}", {c: cats[c] for c in CATEGORIES},
              "| unseen recall:", round(report["methods"][m]["slices"]["all/unseen"]["recall"], 3),
              "| seen recall:", round(report["methods"][m]["slices"]["all/seen"]["recall"], 3))

    report["structured_decoding_check"] = structured_decoding_check(sentences, preds)
    print("Slide 87 check:", report["structured_decoding_check"])
    examples = pick_examples(sentences, preds, train_names)
    report["examples"] = examples
    save_json(report, RESULTS_DIR, f"error_analysis_{split}.json")

    lines = [f"# Disagreement examples ({split} split)\n",
             "Chosen automatically by src/error_analysis.py (one rule per row; shortest sentence).\n"]
    for e in examples:
        lines.append(f"## {e['rule']}  (sentence {e['sentence_index']})\n")
        lines.append("Sentence: " + " ".join(e["tokens"]) + "\n")
        lines.append(f"- **Gold**: {spans_text(e['tokens'], e['gold'])}"
                     f"  — tags: `{tags_text(e['tokens'], e['gold'])}`")
        for m, p in e["predictions"].items():
            mark = "correct" if p == e["gold"] else "wrong"
            lines.append(f"- {NAMES[m]} ({mark}): {spans_text(e['tokens'], p)}"
                         + ("" if p == e["gold"] else f"  — tags: `{tags_text(e['tokens'], p)}`"))
        lines.append("")
    with open(f"{RESULTS_DIR}/error_examples_{split}.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"saved results/error_analysis_{split}.json and results/error_examples_{split}.md")


if __name__ == "__main__":
    main()
