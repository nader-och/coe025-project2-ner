# Slide map: every file and function → the course slide it comes from

Slide decks (in `slides/`): **W1** = Week 1 presentation, **W2** = Week 2, **W3L** = Week 3 lecture deck (speaker notes), **W6** = Week 6 study deck (`week6-pro-standalone.pptx`).
Lines marked `# Not in slides:` in the code are small technical details the slides do not cover; each one says why it is needed.

## Task, data and labels

| File / function | What it does | Slide |
|---|---|---|
| (whole project) | NER on medical text; "medical records" use case | W1 Slide 31 |
| (whole project) | NER = token classification, BIO labels | W6 Slides 2, 13, 23 |
| `data/bc5cdr.py` | Loads `tner/bc5cdr` (Chemical, Disease) from Hugging Face | W6 Slides 14, 31–32; W1 Slide 34 |
| `data/bc5cdr.py: load_id2label` | Reads the label map from the dataset | W6 Slide 40 (save id2label) |
| `data/inspect_data.py` | Split sizes, label counts, IOB2 check | W6 Slide 32 (check the task contract) |
| `data/inspect_data.py: overlap_audit` | Identical sentences shared by splits | W6 Slides 40–41 (split rule and overlap audit) |
| `data/inspect_data.py: bert_length_check` | No sentence longer than BERT's limit | W6 Slide 40 (truncated entity) |
| `src/bio.py: is_allowed` | IOB2 transition rules | W6 Slides 23, 80–81 |
| `src/bio.py: count_invalid` | Counts illegal transitions | W6 Slides 23, 80 |
| `src/bio.py: get_spans` | BIO tags → entity spans | W6 Slides 13, 21–22 |

## Evaluation

| File / function | What it does | Slide |
|---|---|---|
| `src/metrics.py: prf_from_counts` | P = TP/(TP+FP), R = TP/(TP+FN), F1 = 2TP/(2TP+FP+FN) | W6 Slide 38 |
| `src/metrics.py: safe_divide` | Zero-division rule (score = 0) | W6 Slides 36, 39 |
| `src/metrics.py: span_set` | Spans identified by position + type, not text | W6 Slide 37 |
| `src/metrics.py: micro_macro` | Micro (pool counts) and macro (average type F1) | W6 Slide 39 |
| `src/metrics.py: evaluate` | Token accuracy next to span F1 | W6 Slide 36 |
| `tests/test_metrics.py` | Reproduces Slides 36, 38 (0.5 / 0.667 / 0.571), 39 (0.783 / 0.644), 42–43, 23, 80–81 | W6 Slides 23, 36–39, 42–43, 80–81 |
| `src/final_eval.py` | Test set used once, at the end | W6 Slide 41; W3L Slide 36 notes |
| `src/timing.py` | Prediction time on one stated machine | W1 Slide 27; W3L Slide 36 notes |
| `src/error_analysis.py: categorize_errors` | Boundary / type / missed / spurious | W6 Slide 41 |
| `src/error_analysis.py: slice_recall` | Seen vs unseen names; Chemical vs Disease | W6 Slide 41 |
| `src/error_analysis.py: invalid_patterns` | Which illegal transitions each method makes | W6 Slides 23, 80, 87 |
| `src/error_analysis.py: pick_examples` | Real sentences where methods disagree | W6 Slide 87 |
| `src/error_analysis.py: structured_decoding_check` | Does the CRF decoder fix the sentences where softmax was invalid? | W6 Slide 87 ("test whether structured decoding fixes the observed errors") |
| `src/figures.py` | Charts made from the saved result files | W6 Slides 36, 38–39, 41; W1 Slide 27 |
| `results/ERROR_ANALYSIS.md` | Written insights with real examples | W6 Slides 41, 87, 14, 18, 45 |

## Method 1: frequent-tag baseline

| File / function | What it does | Slide |
|---|---|---|
| `src/baseline.py: train` | Most frequent training tag per word | W6 Slide 11 |
| `src/baseline.py: FALLBACK` | Unknown word → `O` (declared policy) | W6 Slide 11 |
| `src/baseline.py: normalize` | Case normalization, chosen on validation | W6 Slide 11 notes |
| `src/baseline.py: predict_all_o` | All-O demo: high accuracy, zero recall | W6 Slide 36 |

## Method 2: HMM + Viterbi

| File / function | What it does | Slide |
|---|---|---|
| `src/hmm.py` (hidden states = BIO labels) | HMM structure and assumptions | W6 Slides 49–51, 66 |
| `src/hmm.py: train` | Start, transition, emission tables from counts | W6 Slides 50, 62 |
| `src/hmm.py: smoothed_distribution` | Add-α smoothing | W6 Slide 63 |
| `src/hmm.py: UNK`, `min_count` | Reserved unknown-word symbol, vocabulary design | W6 Slide 64 |
| `src/hmm.py` (`log_*` tables) | Natural logs, no underflow | W6 Slide 65 |
| `src/hmm.py: main` (settings grid) | Settings chosen with validation data | W6 Slide 63 notes |
| `src/viterbi.py: viterbi` | Viterbi from scratch, backpointers, tie policy | W6 Slides 56–58, 65, 74 |
| `tests/test_hmm.py` | "can fish" example: VN = 0.1344, ln = −2.0069; Slides 63, 64, 89 | W6 Slides 54–58, 63–65, 88–89 |

## Method 3: feature-based linear-chain CRF

| File / function | What it does | Slide |
|---|---|---|
| `src/crf.py: token_features` | Word identity, capitalization, suffix, neighboring words | W6 Slides 12, 79 |
| `src/crf.py: word_shape` | Word shape feature | W6 Slide 64 notes |
| `src/crf.py: build_feature_vocab` | Features from training data only | W6 Slide 11; W3L leakage notes |
| `src/crf.py: FeatureCRF` | Unary score = weighted sum of features | W6 Slides 69, 79 |
| `src/crf.py: LinearChainCRF.log_partition` | Forward algorithm with log-sum-exp → ln Z | W6 Slides 59, 75–76 |
| `src/crf.py: LinearChainCRF.gold_score` | S(X,Y) = unary + transitions | W6 Slide 69 |
| `src/crf.py: LinearChainCRF.neg_log_likelihood` | Loss = ln Z − gold score | W6 Slides 72, 77 |
| `src/crf.py: LinearChainCRF.decode` | Viterbi on scores | W6 Slide 74 |
| `src/crf.py: iob2_constraints`, `FORBIDDEN` | Hard IOB2 mask, same in training and decoding | W6 Slides 80–81 (+ notes) |
| `src/crf.py: main` (ablation) | Remove one feature group at a time (validation) | W2 Slide 27 (feature engineering); W6 Slide 12 |
| `tests/test_crf.py` | Z = 16, loss 0.6931, Viterbi VN, gradient −0.5, Slide 91, mask | W6 Slides 70–78, 80–81, 90–91 |

## Method 4: BERT + softmax

| File / function | What it does | Slide |
|---|---|---|
| `src/bert_common.py: align_sentence` | First-subword labels, −100 for later pieces and special tokens | W6 Slides 33–34 |
| `src/bert_common.py: collate` | Padding: attention mask 0, label −100 | W6 Slides 34, 82, 85 |
| `src/bert_common.py: encode` | Truncation check | W6 Slide 40 |
| `src/bert_common.py: to_word_tags` | Back to one label per word before scoring | W6 Slides 35, 86 |
| `src/bert_common.py: run` | Choose the epoch on validation; predict test once | W6 Slide 41 |
| `src/bert_softmax.py: BertSoftmax` | z = W h + b, softmax, argmax per position | W6 Slides 82–85, 87 |
| `src/bert_softmax.py` (model config) | id2label saved with the model | W6 Slide 40 |
| `notebooks/colab_bert.ipynb` | Runs methods 4–5 on a free Colab GPU | W1 (Colab), W1 Slide 34 |

## Method 5: BERT + CRF

| File / function | What it does | Slide |
|---|---|---|
| `src/bert_crf.py: BertCRF.word_unary` | Neural unary scores u = W h + b at the first piece of each word | W6 Slides 33, 79 |
| `src/bert_crf.py: BertCRF.loss` | CRF loss (reuses method 3's layer) | W6 Slides 77, 79 |
| `src/bert_crf.py: BertCRF.predict_word_ids` | Viterbi with the IOB2 mask | W6 Slides 74, 80–81 |
| (comparison 4 vs 5) | "An option, not a universal upgrade" | W6 Slide 82 notes; Slide 87 |
