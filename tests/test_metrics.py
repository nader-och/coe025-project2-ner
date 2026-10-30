"""Unit tests: the metric must reproduce the worked examples on the slides."""
import math

from src.bio import count_invalid, get_spans, is_allowed, START
from src.metrics import evaluate, micro_macro, prf_from_counts


def test_slide_38_precision_recall_f1():
    # Week 6, Slides 37-38. Gold: Ada Lovelace/PER, New York/LOC, Acme Labs/ORG.
    # Predicted: Ada Lovelace/PER, New York/LOC, Acme/ORG (boundary error),
    # Tuesday/DATE (extra). Expected: TP=2, FP=2, FN=1.
    tokens = ["Ada", "Lovelace", "met", "New", "York", "at", "Acme", "Labs", "on", "Tuesday"]
    gold = ["B-PER", "I-PER", "O", "B-LOC", "I-LOC", "O", "B-ORG", "I-ORG", "O", "O"]
    pred = ["B-PER", "I-PER", "O", "B-LOC", "I-LOC", "O", "B-ORG", "O", "O", "B-DATE"]
    assert len(tokens) == len(gold) == len(pred)
    r = evaluate([gold], [pred], entity_types=["PER", "LOC", "ORG", "DATE"])
    m = r["micro"]
    assert (m["tp"], m["fp"], m["fn"]) == (2, 2, 1)
    assert m["precision"] == 0.5
    assert round(m["recall"], 3) == 0.667
    assert round(m["f1"], 3) == 0.571
    assert math.isclose(m["f1"], 4 / 7)


def test_slide_39_micro_and_macro():
    # Week 6, Slide 39: PER = 8/2/0, LOC = 1/0/3 (TP/FP/FN).
    r = micro_macro({"PER": (8, 2, 0), "LOC": (1, 0, 3)})
    assert round(r["per_type"]["PER"]["f1"], 3) == 0.889
    assert round(r["per_type"]["LOC"]["f1"], 3) == 0.400
    assert round(r["micro"]["f1"], 3) == 0.783
    assert round(r["macro_f1"], 3) == 0.644
    assert math.isclose(r["macro_f1"], 29 / 45)


def test_slide_39_from_real_sequences():
    # Same counts as Slide 39, but produced by real tag sequences.
    gold, pred = [], []
    for _ in range(8):                      # 8 PER true positives
        gold.append(["B-PER"]); pred.append(["B-PER"])
    for _ in range(2):                      # 2 PER false positives
        gold.append(["O"]); pred.append(["B-PER"])
    gold.append(["B-LOC"]); pred.append(["B-LOC"])   # 1 LOC true positive
    for _ in range(3):                      # 3 LOC false negatives
        gold.append(["B-LOC"]); pred.append(["O"])
    r = evaluate(gold, pred, entity_types=["PER", "LOC"])
    assert round(r["micro"]["f1"], 3) == 0.783
    assert round(r["macro_f1"], 3) == 0.644


def test_slide_36_accuracy_hides_missed_entities():
    # Week 6, Slide 36: 100 tokens, 90 O + 10 entity tokens, predict O everywhere.
    gold = [["B-PER", "I-PER"] + ["O"] * 18 for _ in range(5)]   # 10 entity tokens
    pred = [["O"] * 20 for _ in range(5)]
    r = evaluate(gold, pred, entity_types=["PER"])
    assert r["n_tokens"] == 100
    assert r["token_accuracy"] == 0.90
    assert r["micro"]["recall"] == 0.0
    # Zero-division rule: no predicted spans -> precision 0 (stated convention).
    assert r["micro"]["precision"] == 0.0
    assert r["micro"]["f1"] == 0.0


def test_slides_42_43_practice():
    # Week 6, Slides 42-43: Ada visited New York . ; prediction finds only York.
    gold = ["B-PER", "O", "B-LOC", "I-LOC", "O"]
    pred = ["B-PER", "O", "O", "B-LOC", "O"]
    r = evaluate([gold], [pred], entity_types=["PER", "LOC"])
    m = r["micro"]
    assert (m["tp"], m["fp"], m["fn"]) == (1, 1, 1)
    assert m["f1"] == 0.5


def test_prf_zero_division():
    assert prf_from_counts(0, 0, 0) == {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                                        "tp": 0, "fp": 0, "fn": 0}


def test_slide_23_bio_rules():
    # Week 6, Slide 23: which sequences are valid BIO?
    assert count_invalid(["B-PER", "I-PER", "O"]) == 0
    assert count_invalid(["O", "I-PER"]) == 1
    assert count_invalid(["B-ORG", "I-PER"]) == 1
    assert count_invalid(["B-PER", "B-PER"]) == 0


def test_slide_80_81_transitions():
    # Week 6, Slides 80-81.
    assert not is_allowed("O", "I-PER")
    assert is_allowed("B-PER", "I-PER")
    assert is_allowed("I-PER", "B-ORG")
    assert not is_allowed(START, "I-PER")
    assert not is_allowed("B-PER", "I-ORG")


def test_slide_21_22_spans():
    # Week 6, Slides 21-22: Jane Villanueva of United , a unit of United Airlines Holding ...
    tags = ["B-PER", "I-PER", "O", "B-ORG", "O", "O", "O", "O", "B-ORG", "I-ORG", "I-ORG"]
    assert get_spans(tags) == [(0, 2, "PER"), (3, 4, "ORG"), (8, 11, "ORG")]
    # Slide 23: B-PER B-PER is two adjacent person spans.
    assert get_spans(["B-PER", "B-PER"]) == [(0, 1, "PER"), (1, 2, "PER")]
