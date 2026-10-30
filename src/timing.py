"""Measure the prediction time of every method on ONE machine.

Week 1, Slide 27: "Response time matters" (LLM 100-1000 ms vs simple classifier < 10 ms).
Week 3 lecture notes (Slide 36): "measure runtime under a stated environment";
record "data/splits, settings, metric, runtime protocol".

Protocol: predict the VALIDATION sentences (no labels are used, and the test set
is not touched). The time includes the full pipeline from words to word-level
labels (for BERT: tokenization, model, decoding). Classical methods run on the
CPU. BERT runs on the GPU if there is one, and also on the CPU for a subset of
sentences, so all methods can be compared on the same CPU.

Run (in Colab, after training):  python -m src.timing --tag colab
"""
import argparse
import os

import torch

from data.bc5cdr import load_id2label, load_split
from src.utils import MODELS_DIR, RESULTS_DIR, machine_info, save_json, time_prediction


def classical_predictors(id2label):
    from src import baseline, crf, hmm
    out = {}
    if os.path.exists(f"{MODELS_DIR}/baseline.json"):
        m = baseline.load_model(f"{MODELS_DIR}/baseline.json")
        out["baseline"] = lambda s, m=m: baseline.predict(m, s)
    if os.path.exists(f"{MODELS_DIR}/hmm.json"):
        m = hmm.load_model(f"{MODELS_DIR}/hmm.json")
        out["hmm"] = lambda s, m=m: hmm.predict(m, s)
    if os.path.exists(f"{MODELS_DIR}/crf.pt"):
        m, ck = crf.load_model(f"{MODELS_DIR}/crf.pt")
        out["crf"] = lambda s, m=m, ck=ck: crf.predict(m, ck["vocab"], ck["groups"], ck["labels"], s)
    return out


def bert_predictor(name, id2label, device):
    """Load a fine-tuned BERT model saved by src/bert_common.py."""
    from transformers import AutoTokenizer
    from src.bert_common import encode, predict
    path = os.path.join(MODELS_DIR, name)
    if not os.path.isdir(path):
        return None
    tokenizer = AutoTokenizer.from_pretrained(path)
    if name == "bert_softmax":
        from src.bert_softmax import BertSoftmax
        model = BertSoftmax(path, id2label)
    else:
        from src.bert_crf import BertCRF
        ckpt = torch.load(os.path.join(path, "bert_crf.pt"), map_location="cpu", weights_only=False)
        model = BertCRF("bert-base-cased", ckpt["id2label"])
        model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    label2id = {l: i for i, l in id2label.items()}

    def run(sentences):
        sentences = [{"tokens": s["tokens"]} for s in sentences]     # no labels needed
        feats, _ = encode(tokenizer, sentences, label2id, 512)
        return predict(model, feats, tokenizer.pad_token_id, id2label, device)
    return run


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="local")
    p.add_argument("--cpu_subset", type=int, default=500, help="sentences for BERT on CPU")
    args = p.parse_args()

    id2label = load_id2label()
    sentences = load_split("validation")
    report = {"machine": machine_info(), "split": "validation",
              "protocol": "wall-clock time of predict() on the listed sentences, "
                          "words in -> word-level labels out", "methods": {}}

    def record(name, fn, sents, device):
        fn(sents[:32])                       # warm-up (first call loads/compiles things)
        _, seconds = time_prediction(fn, sents)
        report["methods"][f"{name}_{device}"] = {"method": name, "device": device,
                                                 "n_sentences": len(sents),
                                   "seconds": seconds,
                                   "ms_per_sentence": 1000 * seconds / len(sents)}
        print(f"{name:<22} {device:<4} {len(sents):>5} sentences "
              f"{seconds:8.2f} s  {1000 * seconds / len(sents):8.3f} ms/sentence")

    for name, fn in classical_predictors(id2label).items():
        record(name, fn, sentences, "cpu")
    for name in ("bert_softmax", "bert_crf"):
        if torch.cuda.is_available():
            fn = bert_predictor(name, id2label, torch.device("cuda"))
            if fn:
                record(name, fn, sentences, "gpu")
        fn = bert_predictor(name, id2label, torch.device("cpu"))
        if fn:
            record(name, fn, sentences[:args.cpu_subset], "cpu")
    save_json(report, RESULTS_DIR, "timing", f"timing_{args.tag}.json")


if __name__ == "__main__":
    main()
