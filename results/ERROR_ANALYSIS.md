# Error analysis (test set)

All numbers come from `results/error_analysis_test.json` and `results/test_results.json`,
made by `src/error_analysis.py` and `src/final_eval.py`. Examples come from
`results/error_examples_test.md` (real BC5CDR test sentences).

## 1. Four error types (Week 6, Slide 41)

| Method | Boundary | Type | Missed | Spurious | Total errors |
|---|---|---|---|---|---|
| 1. Frequent-tag baseline | 1690 | 9 | 2742 | 849 | 5290 |
| 2. HMM + Viterbi | 694 | 11 | 3217 | 549 | 4471 |
| 3. Feature CRF | 689 | 84 | 1367 | 660 | 2800 |
| 4. BERT + softmax | 721 | 64 | 546 | 1096 | 2427 |
| 5. BERT + CRF | 584 | 52 | 706 | 806 | 2148 |

How we count (one rule for every method):
- **boundary**: the prediction overlaps a gold entity but the start or the end is different.
- **type**: same start and end as a gold entity, but the wrong type (Chemical ↔ Disease).
- **missed**: a gold entity that no prediction touches.
- **spurious**: a predicted entity that touches no gold entity (an extra entity).

What we learn:
- **Type errors are rare** for every method. The hard part is **finding** the entity and its **boundaries**, not choosing Chemical or Disease.
- The **HMM misses the most entities** (3217). It is careful (precision 0.822) but finds little (recall 0.592).
- The **feature CRF halves the missed entities** (3217 → 1367) compared with the HMM.
- **BERT + softmax misses the fewest** (546), but it makes the **most extra entities** (1096 spurious).
- **BERT + CRF** has the fewest boundary errors (584) and the fewest errors in total (2148).

## 2. Slices: seen vs unseen names, Chemical vs Disease (Week 6, Slide 41)

An entity is **unseen** if its text (lower case) never appears as an entity of the same type in the training data. In the test set, 3719 of 9809 entities (38%) are unseen.

| Method | Recall, seen names | Recall, unseen names |
|---|---|---|
| 1. Frequent-tag baseline | 0.907 | 0.038 |
| 2. HMM + Viterbi | 0.872 | 0.134 |
| 3. Feature CRF | 0.944 | 0.504 |
| 4. BERT + softmax | 0.937 | 0.760 |
| 5. BERT + CRF | 0.927 | 0.748 |

What we learn:
- On **seen** names, all methods are similar (0.87–0.94). The averages hide the real difference.
- On **unseen** names the methods are very different. The baseline and the HMM only "remember" words. The HMM gives every unknown word the same `UNK` emission (Slide 64), so it cannot tell an unknown drug name from an unknown normal word.
- The feature CRF uses **suffix and word shape** (Slides 12, 79). In the feature ablation, removing the suffix features costs 0.055 F1 on validation (`results/validation/crf_ablation.json`).
- BERT has the best recall on unseen names. It splits unknown words into known subword pieces (Slide 33) and uses the whole sentence as context (Slide 82).

**Chemical vs Disease.** Disease is harder for every method (test F1 for BERT + CRF: Chemical 0.903, Disease 0.801). Disease names are often several words long ("optic neuropathy", "acute renal failure"), so boundary errors are common. For BERT + CRF, Disease has 477 boundary errors and Chemical has only 88.

## 3. Invalid IOB2 sequences (Week 6, Slides 23, 80)

| Method | Invalid transitions | Sentences with an invalid transition |
|---|---|---|
| 1. Frequent-tag baseline | 1265 | 1027 |
| 2. HMM + Viterbi | 0 | 0 |
| 3. Feature CRF | 0 | 0 |
| 4. BERT + softmax | 342 | 294 |
| 5. BERT + CRF | 0 | 0 |

- The baseline and BERT + softmax choose every label **alone** (Slide 87: "argmax at each position"), so nothing stops `O → I-Disease`. The most common invalid pattern is `O → I-Disease` (baseline 1124 times, BERT + softmax 240 times).
- The HMM has learned transition probabilities. `O → I-X` never occurs in training, so its probability is very small, and Viterbi never chose it.
- The two CRF models use a **hard IOB2 mask** (Slides 80–81), so an invalid output is impossible.

**Does structured decoding fix the errors? (Slide 87)** We took the 294 test sentences where BERT + softmax produced an invalid sequence. On these sentences:

| | Entity micro F1 | Fully correct sentences |
|---|---|---|
| BERT + softmax | 0.680 | 0 of 294 |
| BERT + CRF (same encoder, CRF decoder) | 0.769 | 88 of 294 |

So yes: on exactly these sentences, the CRF decoder fixes many errors. But it is not a "universal upgrade" (Slide 82). In the whole test set, BERT + CRF misses more entities than BERT + softmax (706 vs 546) and has slightly lower recall (0.859 vs 0.870). Its gain comes from higher precision (0.854 vs 0.819). Overall, micro F1 goes up by 0.013 (0.844 → 0.857).

## 4. Real examples where the methods disagree (Week 6, Slide 87)

**Example 1: "Baboon syndrome induced by ketoconazole ."** Gold: [Baboon syndrome] Disease, [ketoconazole] Chemical.
- Baseline: `syndrome/I-Disease` with `Baboon/O`, which gives the invalid `O → I-Disease`. The span becomes only [syndrome]. "Baboon" never appears in training, so the fallback is `O`. "syndrome" is most often `I-Disease` in training (66 of 88 times). The baseline ignores context (Slide 11).
- HMM: correct. Viterbi scores the **whole path** (Slide 56). `I-Disease` for "syndrome" needs a `B-Disease` before it, so the best path labels "Baboon" `B-Disease`, even though "Baboon" is `UNK`.
- Feature CRF: wrong, it misses the disease. This shows that a method that is better on average can still fail on one sentence.

**Example 2: "Fluconazole associated agranulocytosis and thrombocytopenia ."**
- "Fluconazole" never appears in training. HMM: the `UNK` word gets `O`, so the chemical is missed.
- Feature CRF: correct. The suffix features (last 2–4 letters: "-le", "-ole", "-zole") look like other drug names (Slides 12, 79).

**Example 3: "Linezolid - induced optic neuropathy ."**
- Baseline, HMM and feature CRF all miss [Linezolid] Chemical (it is not in training). The CRF only sees one word to the left and right.
- Both BERT models are correct. BERT reads the whole sentence (Slide 82), including the pattern "X - induced".

**Example 4 (invalid sequence): "Oral procaterol 50 micrograms b ."** Gold: `procaterol/B-Chemical`.
- BERT + softmax: `Oral/O procaterol/I-Chemical 50/I-Chemical`. This is invalid (`O → I-Chemical`), and the span becomes [procaterol 50], a boundary error.
- BERT + CRF: `procaterol/B-Chemical 50/O`, which is correct. The mask forbids `O → I-Chemical`, so the CRF must start the entity with B, and the path with "50" = O scores best.

**Example 5 (CRF is not always better): "The Olanzipine HGGW Study Group ."** Gold: [Olanzipine] Chemical.
- BERT + softmax: correct.
- BERT + CRF and feature CRF: [Olanzipine HGGW], a boundary error. The learned transition `B-Chemical → I-Chemical` pulled the unknown code "HGGW" into the entity. This is Slide 82: "an option, not a universal upgrade".

**Example 6 (unseen names): "Hypertension in diabetics prone to hypoglycaemia"**
- "diabetics" and the British spelling "hypoglycaemia" never appear in training.
- HMM finds only [Hypertension]. The feature CRF also finds [hypoglycaemia] (suffix "-emia", as in "anemia"), but misses [diabetics].
- Both BERT models find all three.

**Example 7 (every method wrong, annotation ambiguity): "Patterns of sulfadiazine acute nephrotoxicity ."** Gold: [acute nephrotoxicity] Disease.
- Every method predicts only [nephrotoxicity]. In the training data, "acute" is labeled `O` 99 times and `B-Disease` 52 times. The annotators sometimes include "acute" and sometimes do not. This is an annotation-policy question (Week 6, Slides 14, 18), not only a model error.

**Example 8 (boundary error by the best method): "Viracept and irregular heartbeat warning ."** Gold: [Viracept] Chemical, [irregular heartbeat] Disease.
- BERT + CRF: [Viracept and irregular heartbeat] Disease, one long wrong span. The output is a **valid** IOB2 sequence, but it is semantically wrong. As Slide 45 says, a valid output can still be wrong.
- BERT + softmax: a spurious [and] (`and/I-Disease`, which is invalid), plus the correct [irregular heartbeat].

## 5. Summary of insights
1. Token accuracy hides the real quality. "All O" gets 89.1% token accuracy but 0 entity F1 (Slide 36). BERT + softmax even has slightly **higher** token accuracy than BERT + CRF (0.9704 vs 0.9700) but **lower** entity F1 (0.844 vs 0.857).
2. The biggest difference between methods is on **unseen names** (recall 0.038 → 0.760).
3. **Structured decoding** (HMM, CRF) removes invalid IOB2 sequences. On the sentences where softmax was invalid, the CRF decoder raises F1 from 0.680 to 0.769 (Slide 87).
4. **Disease boundaries** are the hardest remaining problem, partly because of annotation choices such as "acute".
