"""Method 2: Hidden Markov Model (HMM) + Viterbi.

Week 6, Slides 49-51: hidden states y (here: BIO labels, Slide 66) produce the
observed words x. Assumptions: y_t depends only on y_{t-1}; x_t depends only on y_t.
Week 6, Slide 50: P(X, Y) = pi(y1) b(y1, x1) a(y1, y2) b(y2, x2) ...
Week 6, Slide 62: learn pi, a, b by counting in the training data and normalizing.
Week 6, Slide 63: add-alpha smoothing, P(j) = (count(j) + alpha) / (total + alpha K).
Week 6, Slide 64: reserve an UNK symbol for unseen words before estimating b.
Week 6, Slide 65: work with natural logs.
Week 6, Slides 56-57: decode with Viterbi (src/viterbi.py).

Run from the project root:  python -m src.hmm
"""
from collections import Counter

import numpy as np

from data.bc5cdr import load_id2label, load_split
from src.bio import count_invalid
from src.metrics import evaluate, format_result
from src.utils import MODELS_DIR, RESULTS_DIR, load_json, save_json, time_prediction, timing_record
from src.viterbi import viterbi

UNK = "<UNK>"


def smoothed_distribution(counts, alpha):
    """Week 6, Slide 63: (count + alpha) / (total + alpha * K) for each outcome.

    counts: 1-D array of counts over all K possible outcomes.
    """
    counts = np.asarray(counts, dtype=float)
    return (counts + alpha) / (counts.sum() + alpha * len(counts))


def normalize(word, lowercase):
    # Week 6, Slide 11 notes: case normalization is a preprocessing choice.
    return word.lower() if lowercase else word


def train(train_sentences, id2label, lowercase=True, emission_alpha=1.0, min_count=1,
          transition_alpha=1.0):
    """Estimate start (pi), transition (a) and emission (b) tables from counts."""
    labels = [id2label[i] for i in sorted(id2label)]
    label_index = {l: i for i, l in enumerate(labels)}
    K = len(labels)

    # Week 6, Slide 64: vocabulary design. Words seen fewer than min_count times
    # in training are mapped to UNK, so UNK is reserved before estimating b.
    word_counts = Counter(normalize(w, lowercase) for s in train_sentences for w in s["tokens"])
    vocab = [w for w, c in word_counts.items() if c >= min_count] + [UNK]
    word_index = {w: i for i, w in enumerate(vocab)}

    start_counts = np.zeros(K)
    trans_counts = np.zeros((K, K))
    emit_counts = np.zeros((K, len(vocab)))
    for s in train_sentences:
        tag_ids = [label_index[t] for t in s["tags"]]
        start_counts[tag_ids[0]] += 1                          # sentence-start counts
        for prev, cur in zip(tag_ids, tag_ids[1:]):
            trans_counts[prev, cur] += 1                       # Slide 62: count transitions
        for w, t in zip(s["tokens"], tag_ids):
            w = normalize(w, lowercase)
            emit_counts[t, word_index.get(w, word_index[UNK])] += 1

    # Slide 62: normalize each row; Slide 63: with add-alpha smoothing.
    start = smoothed_distribution(start_counts, transition_alpha)
    trans = np.vstack([smoothed_distribution(row, transition_alpha) for row in trans_counts])
    emit = np.vstack([smoothed_distribution(row, emission_alpha) for row in emit_counts])

    return {"labels": labels, "vocab": vocab, "lowercase": lowercase,
            "settings": {"lowercase": lowercase, "emission_alpha": emission_alpha,
                         "min_count": min_count, "transition_alpha": transition_alpha},
            # Slide 65: store natural logs
            "log_start": np.log(start), "log_trans": np.log(trans), "log_emit": np.log(emit),
            "id2label": id2label}


def predict(model, sentences):
    word_index = {w: i for i, w in enumerate(model["vocab"])}
    unk = word_index[UNK]
    labels = model["labels"]
    predictions = []
    for s in sentences:
        ids = [word_index.get(normalize(w, model["lowercase"]), unk) for w in s["tokens"]]
        unary = model["log_emit"][:, ids].T          # (T, K): log b(j, x_t)
        path, _ = viterbi(model["log_start"], model["log_trans"], unary)
        predictions.append([labels[j] for j in path])
    return predictions


def save_model(model, path):
    out = dict(model)
    for key in ("log_start", "log_trans", "log_emit"):
        out[key] = model[key].tolist()
    save_json(out, path)


def load_model(path):
    model = load_json(path)
    for key in ("log_start", "log_trans", "log_emit"):
        model[key] = np.array(model[key])
    model["id2label"] = {int(k): v for k, v in model["id2label"].items()}
    return model


def main():
    id2label = load_id2label()
    train_sents = load_split("train")
    val_sents = load_split("validation")
    gold = [s["tags"] for s in val_sents]

    # Slide 63 notes: "Add-one is not automatically the best ... Choose settings
    # with validation data." A small grid, scored on validation only.
    results = {}
    for lowercase in (False, True):
        for emission_alpha in (1.0, 0.1, 0.01):
            for min_count in (1, 2):
                model = train(train_sents, id2label, lowercase, emission_alpha, min_count)
                preds, seconds = time_prediction(lambda s: predict(model, s), val_sents)
                r = evaluate(gold, preds)
                r["invalid_transitions"] = sum(count_invalid(p) for p in preds)
                r["settings"] = model["settings"]
                r["timing_validation"] = timing_record(seconds, val_sents)
                name = f"lower={lowercase},alpha={emission_alpha},min_count={min_count}"
                results[name] = r
                print(format_result("HMM", r), "|", name, "| invalid:", r["invalid_transitions"])

    best_name = max(results, key=lambda n: results[n]["micro"]["f1"])
    best = results[best_name]["settings"]
    results["chosen_setting"] = best
    print("chosen:", best_name)
    save_model(train(train_sents, id2label, best["lowercase"], best["emission_alpha"],
                     best["min_count"]), f"{MODELS_DIR}/hmm.json")
    save_json(results, RESULTS_DIR, "validation", "hmm.json")


if __name__ == "__main__":
    main()
