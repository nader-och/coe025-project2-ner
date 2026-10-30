"""Load the BC5CDR dataset (tner/bc5cdr) from Hugging Face.

Week 6, Slides 14, 31-32: BC5CDR is a biomedical NER dataset with two entity
types, Chemical and Disease. Slide 32 says: check the task contract (labels,
splits, entity definitions) before using a dataset.
Week 1, Slide 34: HuggingFace gives standardized datasets.
"""
import json
import os

# Not in slides: the new `datasets` library no longer runs the loading script
# `bc5cdr.py` of this repo, so we download the raw JSON files of the same repo.
from huggingface_hub import hf_hub_download

REPO_ID = "tner/bc5cdr"
RAW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")
SPLIT_FILES = {"train": "dataset/train.json",
               "validation": "dataset/valid.json",
               "test": "dataset/test.json"}


def _download(filename):
    return hf_hub_download(REPO_ID, filename, repo_type="dataset", local_dir=RAW_DIR)


def load_id2label():
    """Read the label map from the dataset itself (Week 6, Slide 40: save id2label)."""
    with open(_download("dataset/label.json"), encoding="utf-8") as f:
        label2id = json.load(f)
    return {int(i): label for label, i in label2id.items()}


def load_split(split):
    """Return a list of sentences. Each sentence is {"tokens": [...], "tags": [...]}.

    Tags are converted from numbers to label strings with id2label, so every
    method works with readable labels such as "B-Chemical".
    """
    id2label = load_id2label()
    sentences = []
    # Not in slides: file format detail. Each line of the file is one JSON sentence.
    with open(_download(SPLIT_FILES[split]), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            sentences.append({"tokens": item["tokens"],
                              "tags": [id2label[t] for t in item["tags"]]})
    return sentences


def load_all():
    """Return train, validation and test splits (the same split for every method)."""
    return {split: load_split(split) for split in SPLIT_FILES}
