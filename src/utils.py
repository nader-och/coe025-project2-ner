"""Small shared helpers: seeds, JSON files, timing, machine info."""
import json
import os
import platform
import random
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results")
MODELS_DIR = os.path.join(ROOT, "models")
SEED = 42


def set_seed(seed=SEED):
    """Fix random seeds so runs can be repeated."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def save_json(obj, *path_parts):
    path = os.path.join(*path_parts)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    return path


def load_json(*path_parts):
    with open(os.path.join(*path_parts), encoding="utf-8") as f:
        return json.load(f)


def machine_info():
    """Record the machine used for timing (Week 1 latency slide; Week 3 lecture
    notes: "measure runtime under a stated environment")."""
    info = {"platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count()}
    try:
        import torch
        info["torch"] = torch.__version__
        info["torch_threads"] = torch.get_num_threads()
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        pass
    return info


def time_prediction(predict_fn, sentences):
    """Run predict_fn on all sentences and measure the wall-clock time.

    Returns (predictions, seconds). Week 1, Slide 27: response time matters.
    """
    start = time.perf_counter()
    predictions = predict_fn(sentences)
    seconds = time.perf_counter() - start
    return predictions, seconds


def timing_record(seconds, sentences):
    n_sent = len(sentences)
    n_tok = sum(len(s["tokens"]) for s in sentences)
    return {"seconds_total": seconds, "n_sentences": n_sent, "n_tokens": n_tok,
            "ms_per_sentence": 1000 * seconds / n_sent,
            "sentences_per_second": n_sent / seconds if seconds > 0 else None,
            "machine": machine_info()}
