"""Method 5: BERT + CRF.

Week 6, Slide 79: neural CRF -> unary score u_i = W h_i + b from the encoder;
the CRF adds transition scores and normalizes complete label sequences.
Week 6, Slide 82: "A transformer plus CRF is an option, not a universal upgrade.
Compare against a simpler contextual softmax baseline on the actual task."
We reuse the LinearChainCRF layer of method 3 (src/crf.py), with the same hard
IOB2 mask in training and decoding (Slides 80-81).

The CRF works on WORDS: we take the unary scores of the first piece of each
word (Slide 33 first-piece policy), so the label sequence is the word-level
BIO sequence that we evaluate (Slides 35, 86).

Run in Google Colab (GPU):  python -m src.bert_crf
"""
import os

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from src.bert_common import IGNORE, run
from src.crf import LinearChainCRF, iob2_constraints


class BertCRF(nn.Module):
    def __init__(self, model_name, id2label):
        super().__init__()
        from transformers import AutoModel
        self.id2label = id2label
        labels = [id2label[i] for i in sorted(id2label)]
        self.bert = AutoModel.from_pretrained(model_name)
        # Same dropout as the softmax token-classification head, for a fair comparison.
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.head = nn.Linear(self.bert.config.hidden_size, len(labels))   # u_i = W h_i + b
        self.crf = LinearChainCRF(len(labels), *iob2_constraints(labels))

    def parameter_groups(self, lr):
        # Not in slides: the CRF scores start at zero and are few, so they get a
        # larger learning rate than the pre-trained BERT weights.
        return [{"params": list(self.bert.parameters()) + list(self.head.parameters()), "lr": lr},
                {"params": self.crf.parameters(), "lr": 1e-3}]

    def word_unary(self, batch, first_positions):
        """Unary scores of the first piece of each word -> (B, W, K) plus mask (B, W)."""
        h = self.bert(input_ids=batch["input_ids"],
                      attention_mask=batch["attention_mask"]).last_hidden_state
        scores = self.head(self.dropout(h)).float()      # the CRF runs in float32
        per_sentence = [scores[i, pos] for i, pos in enumerate(first_positions)]
        mask = pad_sequence([torch.ones(len(p), dtype=torch.bool, device=scores.device)
                             for p in first_positions], batch_first=True)
        return pad_sequence(per_sentence, batch_first=True), mask

    def loss(self, batch, first_positions):
        unary, mask = self.word_unary(batch, first_positions)
        gold = pad_sequence([batch["labels"][i, pos] for i, pos in enumerate(first_positions)],
                            batch_first=True, padding_value=0)
        assert (gold != IGNORE).all()                   # first pieces always carry a label
        return self.crf.neg_log_likelihood(unary, gold, mask)   # Slide 77: ln Z - gold score

    def predict_word_ids(self, batch, first_positions):
        unary, mask = self.word_unary(batch, first_positions)
        return [self.crf.decode(unary[i, :mask[i].sum()]) for i in range(len(first_positions))]

    def save(self, path, tokenizer, id2label):
        os.makedirs(path, exist_ok=True)
        torch.save({"state_dict": self.state_dict(), "id2label": id2label},
                   os.path.join(path, "bert_crf.pt"))
        tokenizer.save_pretrained(path)


if __name__ == "__main__":
    run(BertCRF, "bert_crf")
