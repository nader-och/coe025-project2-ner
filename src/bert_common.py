"""Shared code for the two BERT methods (BERT + softmax, BERT + CRF).

Week 1, Slide 34: HuggingFace (load BERT) + PyTorch.
Week 6, Slide 82: words -> subword tokens; encoder gives h_i; head gives z_i = W h_i + b.
Week 6, Slide 33: first-subword alignment: the first piece of a word gets the
word label; later pieces and special tokens ([CLS], [SEP]) get -100.
Week 6, Slide 34 / 86: -100 means "ignore in the loss". O is a REAL label and
is never replaced by -100.
Week 6, Slides 35 / 86: before scoring, go back to one label per word.
Week 6, Slide 40: check truncation (a cut entity cannot be rebuilt) and save id2label.
"""
import argparse
import os
import time

import numpy as np
import torch

from data.bc5cdr import load_id2label, load_split
from src.bio import count_invalid
from src.metrics import evaluate, format_result
from src.utils import MODELS_DIR, RESULTS_DIR, machine_info, save_json, set_seed

IGNORE = -100   # Week 6, Slide 34: ignored target position (not a label)


def get_args(default_name):
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="bert-base-cased")  # BERT, as named in the slides
    # Not in slides: these training settings follow the Hugging Face
    # token-classification guide (lr 2e-5, batch 16, weight decay 0.01).
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--limit", type=int, default=None, help="use only N sentences (quick test)")
    p.add_argument("--results_dir", default=RESULTS_DIR)
    p.add_argument("--models_dir", default=MODELS_DIR)
    p.add_argument("--name", default=default_name)
    return p.parse_args()


def align_sentence(word_ids, tags, label2id):
    """Week 6, Slide 33: build subword labels from word labels (first-piece policy).

    word_ids: for each subword position, the index of its word (None = special token).
    Returns (labels, first_positions). first_positions[k] = position of the first
    piece of word k; we use it later to go back to word-level labels (Slide 35).
    """
    labels, first_positions = [], []
    previous = None
    for position, word in enumerate(word_ids):
        if word is None:
            labels.append(IGNORE)                     # [CLS], [SEP], padding
        elif word != previous:
            labels.append(label2id[tags[word]] if tags else IGNORE)  # first piece
            first_positions.append(position)
        else:
            labels.append(IGNORE)                     # later pieces: -100, NOT "O"
        previous = word
    return labels, first_positions


def encode(tokenizer, sentences, label2id, max_length):
    """Tokenize sentences into subwords and align labels (Slides 33-34, 40)."""
    enc = tokenizer([s["tokens"] for s in sentences], is_split_into_words=True,
                    truncation=True, max_length=max_length)
    features, truncated = [], 0
    for i, s in enumerate(sentences):
        word_ids = enc.word_ids(i)
        labels, first_positions = align_sentence(word_ids, s.get("tags"), label2id)
        # Slide 40: a word with no first piece was cut off (truncation) or has no
        # subwords at all. Not in slides: such a word is predicted as "O".
        seen_words = {w for w in word_ids if w is not None}
        missing = [w for w in range(len(s["tokens"])) if w not in seen_words]
        if max(seen_words, default=-1) < len(s["tokens"]) - 1 and len(word_ids) >= max_length:
            truncated += 1
        features.append({"input_ids": enc["input_ids"][i],
                         "attention_mask": enc["attention_mask"][i],
                         "labels": labels,
                         "first_positions": first_positions,
                         "word_index": sorted(seen_words),
                         "n_words": len(s["tokens"]),
                         "missing_words": missing})
    return features, truncated


def collate(features, pad_id):
    """Pad a batch. Padding has attention_mask 0 and label -100 (Slide 34)."""
    longest = max(len(f["input_ids"]) for f in features)
    def pad(key, value):
        return torch.tensor([f[key] + [value] * (longest - len(f[key])) for f in features])
    return {"input_ids": pad("input_ids", pad_id),
            "attention_mask": pad("attention_mask", 0),
            "labels": pad("labels", IGNORE)}


def to_word_tags(word_label_ids, feature, id2label):
    """Week 6, Slides 35, 86: one label per original word, ready for scoring."""
    tags = ["O"] * feature["n_words"]          # words without subwords stay "O"
    for word, label_id in zip(feature["word_index"], word_label_ids):
        tags[word] = id2label[label_id]
    return tags


def batches(features, batch_size, shuffle, rng=None):
    order = np.arange(len(features))
    if shuffle:
        order = rng.permutation(len(features))
    for b in range(0, len(order), batch_size):
        yield [features[i] for i in order[b:b + batch_size]]


def predict(model, features, pad_id, id2label, device, batch_size=32):
    """Word-level predictions for all sentences (model.predict_word_ids does the decoding)."""
    model.eval()
    out = []
    with torch.no_grad():
        for feats in batches(features, batch_size, shuffle=False):
            batch = {k: v.to(device) for k, v in collate(feats, pad_id).items()}
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=device.type == "cuda"):
                word_ids = model.predict_word_ids(batch, [f["first_positions"] for f in feats])
            out += [to_word_tags(w, f, id2label) for w, f in zip(word_ids, feats)]
    return out


def run(model_builder, default_name):
    """Train on train, choose the epoch on validation, then predict the test set once.

    The test predictions are saved but NOT scored here; scoring happens once in
    src/final_eval.py (Week 6, Slide 41: tune on validation, keep a final test set).
    """
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    args = get_args(default_name)
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    id2label = load_id2label()
    label2id = {l: i for i, l in id2label.items()}
    splits = {s: load_split(s) for s in ("train", "validation", "test")}
    if args.limit:
        splits = {k: v[:args.limit] for k, v in splits.items()}

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    feats, truncation_report = {}, {}
    for name, sents in splits.items():
        feats[name], truncation_report[name] = encode(tokenizer, sents, label2id, args.max_length)
    print("truncated sentences per split (Slide 40):", truncation_report)
    # Sanity check of the alignment (Slide 40: inspect paired tokens and labels).
    f0 = feats["train"][0]
    print("example alignment:", list(zip(tokenizer.convert_ids_to_tokens(f0["input_ids"]),
                                         f0["labels"]))[:12])

    model = model_builder(args.model_name, id2label).to(device)
    optimizer = torch.optim.AdamW(model.parameter_groups(args.lr), lr=args.lr,
                                  weight_decay=args.weight_decay)
    steps = args.epochs * ((len(feats["train"]) + args.batch_size - 1) // args.batch_size)
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, steps)
    # Not in slides: mixed precision makes GPU training faster; it does not change the method.
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    gold_val = [s["tags"] for s in splits["validation"]]
    rng = np.random.default_rng(42)
    history, best = [], {"f1": -1.0}
    for epoch in range(1, args.epochs + 1):
        model.train()
        start, total = time.perf_counter(), 0.0
        for feats_batch in batches(feats["train"], args.batch_size, shuffle=True, rng=rng):
            batch = {k: v.to(device) for k, v in collate(feats_batch, tokenizer.pad_token_id).items()}
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=device.type == "cuda"):
                loss = model.loss(batch, [f["first_positions"] for f in feats_batch])
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total += loss.item()
        train_seconds = time.perf_counter() - start
        val_pred = predict(model, feats["validation"], tokenizer.pad_token_id, id2label, device)
        r = evaluate(gold_val, val_pred)
        history.append({"epoch": epoch, "train_loss": total, "val_micro_f1": r["micro"]["f1"],
                        "val_macro_f1": r["macro_f1"], "train_seconds": train_seconds})
        print(f"epoch {epoch}: " + format_result(args.name, r))
        if r["micro"]["f1"] > best["f1"]:      # select the epoch on validation only
            best = {"f1": r["micro"]["f1"], "epoch": epoch, "result": r, "val_pred": val_pred,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}

    model.load_state_dict(best["state"])
    result = best["result"]
    result["invalid_transitions"] = sum(count_invalid(p) for p in best["val_pred"])
    result["history"] = history
    result["chosen_epoch"] = best["epoch"]
    result["settings"] = {k: v for k, v in vars(args).items() if k not in ("results_dir", "models_dir")}
    result["truncated_sentences"] = truncation_report
    result["machine"] = machine_info()

    # Predict the test set ONCE with the chosen model; save without scoring.
    start = time.perf_counter()
    test_pred = predict(model, feats["test"], tokenizer.pad_token_id, id2label, device)
    result["test_prediction_seconds"] = time.perf_counter() - start

    save_json(result, args.results_dir, "validation", f"{args.name}.json")
    save_json(best["val_pred"], args.results_dir, "predictions", f"validation_{args.name}.json")
    save_json(test_pred, args.results_dir, "predictions", f"test_{args.name}.json")
    os.makedirs(args.models_dir, exist_ok=True)
    model.save(os.path.join(args.models_dir, args.name), tokenizer, id2label)  # Slide 40: keep id2label
    print(f"chosen epoch {best['epoch']}, validation micro F1 {best['f1']:.4f}, "
          f"invalid IOB2 transitions on validation: {result['invalid_transitions']}")
