"""Method 1: frequent-tag baseline.

Week 6, Slide 11: each word gets its most frequent label in the TRAINING data.
An unseen word uses a declared fallback label (here: O). This baseline
ignores sentence context. "Never use the test set to determine frequent tags."
Week 6, Slide 12: the frequent tag is one cue; context is another.

Run from the project root:  python -m src.baseline
"""
from collections import Counter, defaultdict

from data.bc5cdr import load_id2label, load_split
from src.bio import count_invalid
from src.metrics import evaluate, format_result
from src.utils import MODELS_DIR, RESULTS_DIR, save_json, load_json, time_prediction, timing_record

FALLBACK = "O"  # Week 6, Slide 11: declared unknown-word policy


def normalize(word, lowercase):
    # Week 6, Slide 11 notes: whether "Book" and "book" are the same word
    # depends on preprocessing (case normalization). We test both on validation.
    return word.lower() if lowercase else word


def train(train_sentences, id2label, lowercase=False):
    """Count word-tag pairs in training and keep the most frequent tag per word."""
    counts = defaultdict(Counter)
    for s in train_sentences:
        for word, tag in zip(s["tokens"], s["tags"]):
            counts[normalize(word, lowercase)][tag] += 1
    label_order = [id2label[i] for i in sorted(id2label)]
    most_frequent = {}
    for word, c in counts.items():
        # Tie policy (Week 6, Slide 56 notes ask for one): the larger count wins;
        # on a tie, the label with the smaller id in id2label wins.
        most_frequent[word] = max(label_order, key=lambda t: (c[t], -label_order.index(t)))
    return {"lowercase": lowercase, "most_frequent": most_frequent, "id2label": id2label}


def predict(model, sentences):
    table, lowercase = model["most_frequent"], model["lowercase"]
    return [[table.get(normalize(w, lowercase), FALLBACK) for w in s["tokens"]]
            for s in sentences]


def predict_all_o(sentences):
    """Week 6, Slide 36: predict O everywhere, to show why accuracy misleads."""
    return [["O"] * len(s["tokens"]) for s in sentences]


def save_model(model, path):
    save_json(model, path)


def load_model(path):
    model = load_json(path)
    model["id2label"] = {int(k): v for k, v in model["id2label"].items()}
    return model


def main():
    id2label = load_id2label()
    train_sents = load_split("train")
    val_sents = load_split("validation")   # tune on validation only (Slide 41)
    gold = [s["tags"] for s in val_sents]

    results = {}
    # Slide 36 demonstration on real data: an all-O "model".
    r = evaluate(gold, predict_all_o(val_sents))
    results["all_O"] = r
    print(format_result("all-O (Slide 36)", r))

    for lowercase in (False, True):
        model = train(train_sents, id2label, lowercase=lowercase)
        preds, seconds = time_prediction(lambda s: predict(model, s), val_sents)
        r = evaluate(gold, preds)
        r["invalid_transitions"] = sum(count_invalid(p) for p in preds)
        r["timing_validation"] = timing_record(seconds, val_sents)
        name = f"baseline_lowercase={lowercase}"
        results[name] = r
        print(format_result(name, r), "| invalid IOB2:", r["invalid_transitions"])

    # Choose the setting with the best validation micro F1 (Slide 41: tune on validation).
    best = max([False, True], key=lambda lc: results[f"baseline_lowercase={lc}"]["micro"]["f1"])
    results["chosen_setting"] = {"lowercase": best}
    save_model(train(train_sents, id2label, lowercase=best), f"{MODELS_DIR}/baseline.json")
    save_json(results, RESULTS_DIR, "validation", "baseline.json")
    print("chosen setting: lowercase =", best)


if __name__ == "__main__":
    main()
