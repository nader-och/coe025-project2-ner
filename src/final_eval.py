"""Final evaluation on the TEST set - run once, at the end.

Week 6, Slide 41: "Keep a final test set. Tune on validation, not test."
Week 3 lecture notes (Slide 36): "Train to fit; validate to choose; test once after choosing."
All settings were chosen on validation before this script was run.

- Methods 1-3: load the saved models and predict the test set here (laptop CPU).
- Methods 4-5: read the test predictions made in Colab (results/predictions/).
Metrics (Week 6, Slides 36-39): entity-level exact-match P/R/F1, micro and macro
F1, F1 per type, token accuracy, plus invalid IOB2 transitions (Slides 23, 80).

Run from the project root:  python -m src.final_eval
"""
import os

from data.bc5cdr import load_id2label, load_split
from src import baseline, crf, hmm
from src.bio import count_invalid
from src.metrics import evaluate, format_result
from src.utils import (MODELS_DIR, RESULTS_DIR, load_json, machine_info, save_json,
                       time_prediction, timing_record)

METHODS = ["baseline", "hmm", "crf", "bert_softmax", "bert_crf"]
NAMES = {"all_O": "All-O (Slide 36 demo)", "baseline": "1. Frequent-tag baseline",
         "hmm": "2. HMM + Viterbi", "crf": "3. Feature CRF", "bert_softmax": "4. BERT + softmax",
         "bert_crf": "5. BERT + CRF"}


def classical_predictions(split):
    """Predict a split with the saved models of methods 1-3. Returns preds and timing."""
    sentences = load_split(split)
    b = baseline.load_model(f"{MODELS_DIR}/baseline.json")
    h = hmm.load_model(f"{MODELS_DIR}/hmm.json")
    c, ck = crf.load_model(f"{MODELS_DIR}/crf.pt")
    predictors = {
        "baseline": lambda s: baseline.predict(b, s),
        "hmm": lambda s: hmm.predict(h, s),
        "crf": lambda s: crf.predict(c, ck["vocab"], ck["groups"], ck["labels"], s)}
    preds, timing = {}, {}
    for name, fn in predictors.items():
        preds[name], seconds = time_prediction(fn, sentences)
        timing[name] = timing_record(seconds, sentences)
    return preds, timing


def load_all_predictions(split):
    """Predictions of all five methods for one split (saved as JSON files)."""
    out = {}
    for m in METHODS:
        path = os.path.join(RESULTS_DIR, "predictions", f"{split}_{m}.json")
        if os.path.exists(path):
            out[m] = load_json(path)
    return out


def invalid_report(preds):
    """Week 6, Slides 23, 80: how many illegal IOB2 transitions were predicted?"""
    per_sentence = [count_invalid(p) for p in preds]
    return {"invalid_transitions": sum(per_sentence),
            "sentences_with_invalid": sum(n > 0 for n in per_sentence)}


def results_table(results, timing):
    """A markdown table; every number comes from results/test_results.json."""
    header = ("| Method | Micro P | Micro R | Micro F1 | Macro F1 | Chemical F1 | Disease F1 "
              "| Token acc. | Invalid IOB2 | ms / sentence (CPU) | ms / sentence (GPU) |\n"
              "|---|---|---|---|---|---|---|---|---|---|---|\n")
    rows = []
    for key, r in results.items():
        cpu = timing.get(f"{key}_cpu", {}).get("ms_per_sentence")
        gpu = timing.get(f"{key}_gpu", {}).get("ms_per_sentence")
        rows.append(f"| {NAMES[key]} | {r['micro']['precision']:.3f} | {r['micro']['recall']:.3f} "
                    f"| **{r['micro']['f1']:.3f}** | {r['macro_f1']:.3f} "
                    f"| {r['per_type']['Chemical']['f1']:.3f} | {r['per_type']['Disease']['f1']:.3f} "
                    f"| {r['token_accuracy']:.3f} | {r['invalid_transitions']} "
                    f"| {f'{cpu:.2f}' if cpu is not None else '-'} "
                    f"| {f'{gpu:.2f}' if gpu is not None else '-'} |")
    return header + "\n".join(rows) + "\n"


def main():
    test = load_split("test")
    gold = [s["tags"] for s in test]

    preds, local_timing = classical_predictions("test")
    for name, p in preds.items():
        save_json(p, RESULTS_DIR, "predictions", f"test_{name}.json")
    preds.update({k: v for k, v in load_all_predictions("test").items() if k not in preds})
    missing = [m for m in METHODS if m not in preds]
    if missing:
        print("WARNING: no test predictions for", missing, "- run the Colab notebook first.")

    results = {"all_O": evaluate(gold, [["O"] * len(s["tokens"]) for s in test])}
    results["all_O"].update(invalid_report([["O"] * len(s["tokens"]) for s in test]))
    for m in METHODS:
        if m in preds:
            results[m] = evaluate(gold, preds[m])
            results[m].update(invalid_report(preds[m]))
    for key, r in results.items():
        print(format_result(key, r), "| invalid:", r["invalid_transitions"])

    colab_path = os.path.join(RESULTS_DIR, "timing", "timing_colab.json")
    colab = load_json(colab_path) if os.path.exists(colab_path) else {"methods": {}, "machine": None}
    save_json({"split": "test", "results": results,
               "local_test_prediction_timing": local_timing,
               "local_machine": machine_info(),
               "speed_table_source": "results/timing/timing_colab.json (same machine for all methods)",
               "colab_machine": colab["machine"]},
              RESULTS_DIR, "test_results.json")
    table = results_table(results, colab["methods"])
    with open(os.path.join(RESULTS_DIR, "results_table.md"), "w", encoding="utf-8") as f:
        f.write("# Test results (BC5CDR test split, used once)\n\n" + table +
                "\nMicro/macro F1 = entity-level exact match (Week 6, Slides 37-39). "
                "Zero-division rule: a score with denominator 0 is 0.\n"
                "Speed = prediction time per sentence on the validation split, all methods on the "
                f"same Colab machine (see results/timing/timing_colab.json): {colab['machine']}\n")
    print(table)


if __name__ == "__main__":
    main()
