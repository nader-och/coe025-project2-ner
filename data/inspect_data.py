"""Inspect BC5CDR before writing any model (Week 6, Slide 32: check the task
contract - labels, splits, entity definitions).

Reports: id2label, split sizes, label counts, entity counts, whether the gold
labels follow IOB2 (Slides 23, 80-81), and an overlap audit between splits
(Slide 41: "Split rule and overlap audit"). If `transformers` is installed it
also checks BERT subword lengths for truncation (Slide 40).

Run from the project root:  python -m data.inspect_data
"""
from collections import Counter

from data.bc5cdr import load_all, load_id2label
from src.bio import count_invalid, get_spans
from src.utils import RESULTS_DIR, save_json


def split_stats(sentences):
    lengths = [len(s["tokens"]) for s in sentences]
    label_counts = Counter(t for s in sentences for t in s["tags"])
    entity_counts = Counter(sp[2] for s in sentences for sp in get_spans(s["tags"]))
    invalid = sum(count_invalid(s["tags"]) for s in sentences)
    return {"sentences": len(sentences),
            "tokens": sum(lengths),
            "max_sentence_length": max(lengths),
            "mean_sentence_length": round(sum(lengths) / len(lengths), 2),
            "label_counts": dict(label_counts),
            "entity_counts": dict(entity_counts),
            "O_token_share": round(label_counts["O"] / sum(lengths), 4),
            "invalid_iob2_transitions_in_gold": invalid}


def overlap_audit(splits):
    """Week 6, Slides 40-41: count identical sentences shared by two splits."""
    as_text = {name: Counter(" ".join(s["tokens"]) for s in sents)
               for name, sents in splits.items()}
    audit = {}
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        shared = set(as_text[a]) & set(as_text[b])
        long_shared = [t for t in shared if len(t.split()) >= 10]
        audit[f"{a}&{b}"] = {"identical_sentences": len(shared),
                             "identical_sentences_with_10+_tokens": len(long_shared),
                             "examples": sorted(shared, key=len)[-3:]}
    return audit


def unseen_entity_share(splits):
    """Week 6, Slide 41 ("unseen name" slice): how many validation/test entities
    never appear (same text, same type) as an entity in training?"""
    def entity_strings(sentences):
        out = []
        for s in sentences:
            for start, end, t in get_spans(s["tags"]):
                out.append((" ".join(s["tokens"][start:end]), t))
        return out
    train_names = set(entity_strings(splits["train"]))
    report = {}
    for split in ("validation", "test"):
        ents = entity_strings(splits[split])
        unseen = sum(e not in train_names for e in ents)
        report[split] = {"entities": len(ents), "unseen_in_train": unseen,
                         "unseen_share": round(unseen / len(ents), 4)}
    return report


def bert_length_check(splits, model_name="bert-base-cased", max_length=512):
    """Week 6, Slide 40: a truncated entity cannot be reconstructed.
    Count sentences whose subword length (with [CLS] and [SEP]) is too long."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    report = {"model": model_name, "max_length_limit": max_length}
    for name, sents in splits.items():
        lengths = [len(tok(s["tokens"], is_split_into_words=True)["input_ids"]) for s in sents]
        report[name] = {"max_subword_length": max(lengths),
                        "sentences_over_limit": sum(n > max_length for n in lengths),
                        "sentences_over_256": sum(n > 256 for n in lengths)}
    return report


def main():
    id2label = load_id2label()
    splits = load_all()
    report = {"dataset": "tner/bc5cdr",
              "id2label": id2label,
              "scheme_check": "IOB2 if no gold sequence has an illegal transition",
              "splits": {name: split_stats(s) for name, s in splits.items()},
              "overlap_audit": overlap_audit(splits),
              "unseen_entities": unseen_entity_share(splits),
              "example_sentence": splits["train"][0]}
    try:
        report["bert_length_check"] = bert_length_check(splits)
    except ImportError:
        report["bert_length_check"] = "transformers not installed; run in Colab"
    path = save_json(report, RESULTS_DIR, "data_inspection.json")

    print("id2label:", id2label)
    for name, st in report["splits"].items():
        print(f"{name:<11} sentences={st['sentences']:>5} tokens={st['tokens']:>6} "
              f"entities={st['entity_counts']} invalid IOB2 in gold={st['invalid_iob2_transitions_in_gold']}")
    print("overlap audit:", {k: v["identical_sentences"] for k, v in report["overlap_audit"].items()})
    print("unseen entities:", report["unseen_entities"])
    print("BERT length check:", report["bert_length_check"])
    print("saved", path)


if __name__ == "__main__":
    main()
