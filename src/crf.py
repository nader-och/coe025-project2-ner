"""Method 3: feature-based linear-chain CRF, written in PyTorch.

Week 6, Slide 67: a CRF combines rich input clues and label compatibility.
Week 6, Slide 68: P(Y|X) = exp(S(X,Y)) / Z(X).
Week 6, Slide 69: S(X,Y) = sum_i u_i(y_i) + sum_i a(y_{i-1}, y_i)  (+ start/end scores).
Week 6, Slide 79: feature CRF -> u_i(y) is a weighted sum of feature values.
Week 6, Slides 75-76: forward algorithm with log-sum-exp gives ln Z.
Week 6, Slide 77: loss = ln Z - gold score.
Week 6, Slide 74: decode with Viterbi (src/viterbi.py).
Week 6, Slides 80-81: a hard IOB2 mask sets illegal transitions to -inf.
Slide 80 notes: use the same mask for training (Z) and for decoding.

The LinearChainCRF layer below is reused by BERT + CRF (src/bert_crf.py).

Run from the project root:  python -m src.crf
"""
import time

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from data.bc5cdr import load_id2label, load_split
from src.bio import START, count_invalid, is_allowed
from src.metrics import evaluate, format_result
from src.utils import (MODELS_DIR, RESULTS_DIR, save_json, set_seed, time_prediction,
                       timing_record)
from src.viterbi import viterbi

# Slide 80 says -inf. Not in slides: we use a large negative number instead,
# because exp(-10000) is 0 in practice and -inf can create NaN gradients.
FORBIDDEN = -10000.0


def iob2_constraints(labels):
    """Week 6, Slide 81: build the IOB2 mask from the annotation rules.

    allowed[i, j] is True if labels[i] -> labels[j] is legal.
    allowed_start[j] is True if a sentence may start with labels[j] (no I-X).
    """
    allowed = torch.tensor([[is_allowed(p, c) for c in labels] for p in labels])
    allowed_start = torch.tensor([is_allowed(START, c) for c in labels])
    return allowed, allowed_start


class LinearChainCRF(nn.Module):
    """Learned start, transition and end scores + the hard mask (Slides 69, 80)."""

    def __init__(self, num_labels, allowed=None, allowed_start=None):
        super().__init__()
        self.start = nn.Parameter(torch.zeros(num_labels))
        self.trans = nn.Parameter(torch.zeros(num_labels, num_labels))
        self.end = nn.Parameter(torch.zeros(num_labels))
        if allowed is None:          # no constraints (used by the slide unit tests)
            allowed = torch.ones(num_labels, num_labels, dtype=torch.bool)
            allowed_start = torch.ones(num_labels, dtype=torch.bool)
        # The mask is fixed (not learned): 0 if legal, FORBIDDEN if illegal.
        self.register_buffer("trans_mask", (~allowed).float() * FORBIDDEN)
        self.register_buffer("start_mask", (~allowed_start).float() * FORBIDDEN)

    def masked_scores(self):
        # The same masked scores are used in training and in decoding (Slide 80 notes).
        return self.start + self.start_mask, self.trans + self.trans_mask, self.end

    def log_partition(self, unary, mask):
        """Week 6, Slides 75-76: forward algorithm in log space -> ln Z(X).

        unary: (B, T, K) scores u_t(j); mask: (B, T) True for real words.
        alpha_t(j) = u_t(j) + LSE_i [alpha_{t-1}(i) + a(i, j)]
        """
        start, trans, end = self.masked_scores()
        alpha = start + unary[:, 0]                                    # (B, K)
        for t in range(1, unary.shape[1]):
            # LSE across alternatives (Slide 76): sum path weights, in log space.
            new_alpha = torch.logsumexp(alpha.unsqueeze(2) + trans, dim=1) + unary[:, t]
            alpha = torch.where(mask[:, t].unsqueeze(1), new_alpha, alpha)  # skip padding
        return torch.logsumexp(alpha + end, dim=1)                     # (B,)

    def gold_score(self, unary, tags, mask):
        """Week 6, Slide 69: S(X, Y_gold) = start + sum of unary + sum of transitions + end."""
        start, trans, end = self.masked_scores()
        B, T, _ = unary.shape
        mask_f = mask.float()
        score = start[tags[:, 0]]
        score = score + (unary.gather(2, tags.unsqueeze(2)).squeeze(2) * mask_f).sum(1)
        score = score + (trans[tags[:, :-1], tags[:, 1:]] * mask_f[:, 1:]).sum(1)
        last = tags.gather(1, (mask.sum(1) - 1).unsqueeze(1)).squeeze(1)
        return score + end[last]

    def neg_log_likelihood(self, unary, tags, mask):
        """Week 6, Slide 77: loss = ln Z - gold score (mean over sentences)."""
        return (self.log_partition(unary, mask) - self.gold_score(unary, tags, mask)).mean()

    @torch.no_grad()
    def decode(self, unary):
        """Week 6, Slide 74: Viterbi on one sentence. unary: (T, K) tensor."""
        start, trans, end = (x.detach().cpu().numpy() for x in self.masked_scores())
        path, _ = viterbi(start, trans, unary.detach().cpu().numpy(), end)
        return path


# ---------------------------------------------------------------------------
# Features (Week 6, Slides 12 and 79): word identity, capitalization / word
# shape, suffix, neighboring words.
# ---------------------------------------------------------------------------
FEATURE_GROUPS = ("word", "case_shape", "suffix", "neighbors")


def word_shape(word):
    """Week 6, Slide 64 notes: "unknown-word models can use suffixes or word shape".
    Upper -> X, lower -> x, digit -> d, other characters stay; repeats collapse.
    Example: 'Naloxone' -> 'Xx', '5-HT' -> 'd-X'."""
    out = []
    for ch in word:
        c = "X" if ch.isupper() else "x" if ch.islower() else "d" if ch.isdigit() else ch
        if not out or out[-1] != c:
            out.append(c)
    return "".join(out)


def token_features(tokens, i, groups):
    """List of feature strings for token i. Each feature has value 1."""
    word = tokens[i]
    low = word.lower()
    feats = ["bias"]
    if "word" in groups:                       # Slide 79: word identity
        feats.append("w=" + low)
    if "case_shape" in groups:                 # Slide 12: capitalization (Janet)
        if word[0].isupper():
            feats.append("init_cap")
        if word.isupper():
            feats.append("all_caps")
        if any(ch.isdigit() for ch in word):
            feats.append("has_digit")
        feats.append("shape=" + word_shape(word))
    if "suffix" in groups:                     # Slide 12: word form, e.g. ends in -ly
        for k in (2, 3, 4):
            if len(low) > k:
                feats.append(f"suf{k}=" + low[-k:])
    if "neighbors" in groups:                  # Slide 12: neighboring words
        feats.append("w-1=" + (tokens[i - 1].lower() if i > 0 else "<s>"))
        feats.append("w+1=" + (tokens[i + 1].lower() if i + 1 < len(tokens) else "</s>"))
    return feats


def build_feature_vocab(train_sentences, groups):
    """Feature index built from TRAINING data only (Week 6, Slide 11; Week 3
    lecture notes: fit the representation on training data only)."""
    vocab = {}
    for s in train_sentences:
        for i in range(len(s["tokens"])):
            for f in token_features(s["tokens"], i, groups):
                if f not in vocab:
                    vocab[f] = len(vocab)
    return vocab


def encode_sentence(tokens, vocab, groups):
    """Feature ids for each token; features never seen in training are skipped."""
    return [[vocab[f] for f in token_features(tokens, i, groups) if f in vocab]
            for i in range(len(tokens))]


class FeatureCRF(nn.Module):
    """u_i(y) = sum of weights of the active features (Slide 79: weighted sum)."""

    def __init__(self, num_features, labels):
        super().__init__()
        K = len(labels)
        # EmbeddingBag(mode="sum") adds the weight rows of the active features:
        # this is exactly w . f(X, i, label) with binary feature values.
        self.weights = nn.EmbeddingBag(num_features, K, mode="sum")
        nn.init.zeros_(self.weights.weight)
        self.crf = LinearChainCRF(K, *iob2_constraints(labels))

    def unary(self, encoded_sentences):
        """Return padded unary scores (B, T, K) and the mask (B, T)."""
        flat, offsets, lengths = [], [], []
        for sent in encoded_sentences:
            lengths.append(len(sent))
            for feats in sent:
                offsets.append(len(flat))
                flat.extend(feats)
        scores = self.weights(torch.tensor(flat, dtype=torch.long),
                              torch.tensor(offsets, dtype=torch.long))
        per_sentence = torch.split(scores, lengths)
        unary = pad_sequence(per_sentence, batch_first=True)
        mask = pad_sequence([torch.ones(n, dtype=torch.bool) for n in lengths], batch_first=True)
        return unary, mask


def predict(model, vocab, groups, labels, sentences):
    model.eval()
    predictions = []
    with torch.no_grad():
        for s in sentences:
            unary, _ = model.unary([encode_sentence(s["tokens"], vocab, groups)])
            predictions.append([labels[j] for j in model.crf.decode(unary[0])])
    return predictions


def train(train_sents, val_sents, labels, groups=FEATURE_GROUPS, epochs=8, lr=0.02,
          batch_size=32, l2=0.0, verbose=True):
    """Train with Adam; keep the epoch with the best VALIDATION micro F1 (Slide 77:
    "check results on separate validation data")."""
    set_seed()
    label_index = {l: i for i, l in enumerate(labels)}
    vocab = build_feature_vocab(train_sents, groups)
    encoded = [encode_sentence(s["tokens"], vocab, groups) for s in train_sents]
    tags = [torch.tensor([label_index[t] for t in s["tags"]]) for s in train_sents]
    model = FeatureCRF(len(vocab), labels)
    # Not in slides: Adam is the optimizer; weight_decay is the L2 regularization
    # that Slide 77 notes mention ("Regularization, optimizer settings").
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
    gold_val = [s["tags"] for s in val_sents]

    history, best_f1, best_state = [], -1.0, None
    rng = np.random.default_rng(42)
    for epoch in range(1, epochs + 1):
        model.train()
        start = time.perf_counter()
        order = rng.permutation(len(encoded))
        total_loss = 0.0
        for b in range(0, len(order), batch_size):
            idx = order[b:b + batch_size]
            unary, mask = model.unary([encoded[i] for i in idx])
            gold = pad_sequence([tags[i] for i in idx], batch_first=True)
            loss = model.crf.neg_log_likelihood(unary, gold, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)
        preds = predict(model, vocab, groups, labels, val_sents)
        r = evaluate(gold_val, preds)
        history.append({"epoch": epoch, "train_loss": total_loss / len(encoded),
                        "val_micro_f1": r["micro"]["f1"], "val_macro_f1": r["macro_f1"],
                        "seconds": time.perf_counter() - start})
        if verbose:
            print(f"  epoch {epoch}: loss={history[-1]['train_loss']:.4f} "
                  f"val micro F1={r['micro']['f1']:.4f} ({history[-1]['seconds']:.0f}s)")
        if r["micro"]["f1"] > best_f1:
            best_f1 = r["micro"]["f1"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, vocab, history


def save_model(model, vocab, groups, labels, id2label, path):
    torch.save({"state_dict": model.state_dict(), "vocab": vocab, "groups": list(groups),
                "labels": labels, "id2label": id2label}, path)


def load_model(path):
    ckpt = torch.load(path, weights_only=False)
    model = FeatureCRF(len(ckpt["vocab"]), ckpt["labels"])
    model.load_state_dict(ckpt["state_dict"])
    return model, ckpt


def main():
    import os
    id2label = load_id2label()
    labels = [id2label[i] for i in sorted(id2label)]
    train_sents = load_split("train")
    val_sents = load_split("validation")
    gold = [s["tags"] for s in val_sents]
    results = {}

    # 1) Small regularization choice on validation (Slide 77 notes).
    for l2 in (0.0, 1e-6):
        print(f"Full feature CRF, l2={l2}")
        model, vocab, history = train(train_sents, val_sents, labels, l2=l2)
        preds, seconds = time_prediction(
            lambda s: predict(model, vocab, FEATURE_GROUPS, labels, s), val_sents)
        r = evaluate(gold, preds)
        r["invalid_transitions"] = sum(count_invalid(p) for p in preds)
        r["history"] = history
        r["timing_validation"] = timing_record(seconds, val_sents)
        r["n_features"] = len(vocab)
        results[f"full_l2={l2}"] = r
        print(format_result("CRF", r), "| invalid:", r["invalid_transitions"])
        if l2 == 0.0 or r["micro"]["f1"] > results["full_l2=0.0"]["micro"]["f1"]:
            best_l2, best_model, best_vocab = l2, model, vocab

    results["chosen_setting"] = {"l2": best_l2, "groups": list(FEATURE_GROUPS)}
    os.makedirs(MODELS_DIR, exist_ok=True)
    save_model(best_model, best_vocab, FEATURE_GROUPS, labels, id2label, f"{MODELS_DIR}/crf.pt")
    save_json(results, RESULTS_DIR, "validation", "crf.json")

    # 2) Feature ablation on validation: remove one feature group at a time.
    ablation = {"full": {"val_micro_f1": results[f"full_l2={best_l2}"]["micro"]["f1"],
                         "val_macro_f1": results[f"full_l2={best_l2}"]["macro_f1"],
                         "per_type_f1": {t: v["f1"] for t, v in
                                         results[f"full_l2={best_l2}"]["per_type"].items()}}}
    for removed in FEATURE_GROUPS:
        groups = tuple(g for g in FEATURE_GROUPS if g != removed)
        print(f"Ablation: without {removed}")
        model, vocab, history = train(train_sents, val_sents, labels, groups=groups, l2=best_l2)
        r = evaluate(gold, predict(model, vocab, groups, labels, val_sents))
        ablation[f"without_{removed}"] = {
            "val_micro_f1": r["micro"]["f1"], "val_macro_f1": r["macro_f1"],
            "per_type_f1": {t: v["f1"] for t, v in r["per_type"].items()},
            "change_vs_full": r["micro"]["f1"] - ablation["full"]["val_micro_f1"],
            "best_epoch": max(history, key=lambda h: h["val_micro_f1"])["epoch"]}
        print(format_result(f"CRF -{removed}", r))
    save_json(ablation, RESULTS_DIR, "validation", "crf_ablation.json")


if __name__ == "__main__":
    main()
