"""Method 4: BERT + softmax token classifier.

Week 6, Slide 82: contextual vector h_i -> scores z_i = W h_i + b -> per-token softmax.
Week 6, Slide 83: softmax turns the scores of one position into probabilities;
the predicted label is the argmax at each position.
Week 6, Slide 85: cross-entropy loss averaged over the supervised positions only;
-100 positions are ignored. O stays a real target.
Week 6, Slide 87: "Contextual softmax: each label given the full observed input;
argmax at each position" (no label-transition scores).

Run in Google Colab (GPU):  python -m src.bert_softmax
"""
import torch
import torch.nn as nn

from src.bert_common import run


class BertSoftmax(nn.Module):
    def __init__(self, model_name, id2label):
        super().__init__()
        from transformers import AutoModelForTokenClassification
        # Week 6, Slide 40: store id2label inside the model config.
        self.bert = AutoModelForTokenClassification.from_pretrained(
            model_name, num_labels=len(id2label), id2label=id2label,
            label2id={l: i for i, l in id2label.items()},
            ignore_mismatched_sizes=True)  # Not in slides: lets small test models load too

    def parameter_groups(self, lr):
        return [{"params": self.parameters(), "lr": lr}]

    def loss(self, batch, first_positions):
        # Hugging Face uses CrossEntropyLoss(ignore_index=-100): Slides 34 and 85.
        return self.bert(**batch).loss

    def predict_word_ids(self, batch, first_positions):
        logits = self.bert(input_ids=batch["input_ids"],
                           attention_mask=batch["attention_mask"]).logits   # (B, L, K), Slide 83
        best = logits.argmax(-1)                     # argmax at each position (Slide 83)
        # Keep only the first piece of each word (Slide 35: back to word units).
        return [best[i, pos].tolist() for i, pos in enumerate(first_positions)]

    def save(self, path, tokenizer, id2label):
        self.bert.save_pretrained(path)               # config.json contains id2label
        tokenizer.save_pretrained(path)


if __name__ == "__main__":
    run(BertSoftmax, "bert_softmax")
