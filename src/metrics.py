"""Entity-level exact-match precision, recall and F1, written from the slides.

Week 6, Slide 37: list the spans before counting errors. A span is identified
by its boundaries and its type, not by its text alone.
Week 6, Slide 38: TP = exact matches, FP = predictions - matches,
FN = gold - matches; P = TP/(TP+FP), R = TP/(TP+FN), F1 = 2TP/(2TP+FP+FN).
Week 6, Slide 39: micro F1 pools the counts; macro F1 averages the type F1s.
O is not an entity type.
Week 6, Slide 36: token accuracy can hide missed entities.
"""
from src.bio import get_spans


def safe_divide(numerator, denominator):
    """Zero-division rule (Week 6, Slides 36 and 39 ask us to state one):
    if the denominator is 0, the score is 0."""
    return numerator / denominator if denominator > 0 else 0.0


def prf_from_counts(tp, fp, fn):
    """Week 6, Slide 38: precision, recall and F1 from TP, FP and FN."""
    return {"precision": safe_divide(tp, tp + fp),
            "recall": safe_divide(tp, tp + fn),
            "f1": safe_divide(2 * tp, 2 * tp + fp + fn),
            "tp": tp, "fp": fp, "fn": fn}


def micro_macro(counts_by_type):
    """Week 6, Slide 39.

    counts_by_type: {"Chemical": (tp, fp, fn), "Disease": (tp, fp, fn)}
    micro = pool the counts of all types, then compute F1 once.
    macro = compute F1 for each type, then take the plain average.
    """
    per_type = {t: prf_from_counts(*c) for t, c in counts_by_type.items()}
    tp = sum(c[0] for c in counts_by_type.values())
    fp = sum(c[1] for c in counts_by_type.values())
    fn = sum(c[2] for c in counts_by_type.values())
    micro = prf_from_counts(tp, fp, fn)
    macro_f1 = sum(r["f1"] for r in per_type.values()) / len(per_type)
    return {"per_type": per_type, "micro": micro, "macro_f1": macro_f1}


def span_set(tag_sequences):
    """All spans of a dataset as (sentence index, start, end, type).

    Week 6, Slide 37 notes: use span positions, not surface strings, because
    the same name can appear many times.
    """
    spans = set()
    for sent_id, tags in enumerate(tag_sequences):
        for start, end, entity_type in get_spans(tags):
            spans.add((sent_id, start, end, entity_type))
    return spans


def evaluate(gold_sequences, pred_sequences, entity_types=None):
    """Full evaluation: span P/R/F1 per type, micro, macro, and token accuracy.

    gold_sequences and pred_sequences are lists of word-level tag lists
    (Week 6, Slide 86: return to word units before scoring).
    """
    assert len(gold_sequences) == len(pred_sequences)
    for g, p in zip(gold_sequences, pred_sequences):
        assert len(g) == len(p), "predictions must be word-level (Slide 35)"

    gold = span_set(gold_sequences)
    pred = span_set(pred_sequences)
    if entity_types is None:
        entity_types = sorted({s[3] for s in gold})

    counts = {}
    for t in entity_types:
        gold_t = {s for s in gold if s[3] == t}
        pred_t = {s for s in pred if s[3] == t}
        tp = len(gold_t & pred_t)        # exact match: same boundaries and type
        fp = len(pred_t) - tp            # unmatched predictions
        fn = len(gold_t) - tp            # unmatched gold spans
        counts[t] = (tp, fp, fn)
    result = micro_macro(counts)

    # Week 6, Slide 36: token accuracy = correct token labels / evaluated tokens
    correct = sum(g == p for gs, ps in zip(gold_sequences, pred_sequences)
                  for g, p in zip(gs, ps))
    total = sum(len(gs) for gs in gold_sequences)
    result["token_accuracy"] = safe_divide(correct, total)
    result["n_tokens"] = total
    result["n_gold_spans"] = len(gold)
    result["n_pred_spans"] = len(pred)
    return result


def format_result(name, r):
    """One readable line for the terminal."""
    pt = r["per_type"]
    parts = [f"{name:<22}", f"micro P={r['micro']['precision']:.4f}",
             f"R={r['micro']['recall']:.4f}", f"F1={r['micro']['f1']:.4f}",
             f"macro F1={r['macro_f1']:.4f}"]
    parts += [f"{t} F1={pt[t]['f1']:.4f}" for t in pt]
    parts.append(f"token acc={r['token_accuracy']:.4f}")
    return " | ".join(parts)
