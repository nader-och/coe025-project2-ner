# Disagreement examples (test split)

Chosen automatically by src/error_analysis.py (one rule per row; shortest sentence).

## baseline wrong, HMM right (label transitions help)  (sentence 641)

Sentence: Baboon syndrome induced by ketoconazole .

- **Gold**: [Baboon syndrome]Disease, [ketoconazole]Chemical  — tags: `Baboon/B-Disease syndrome/I-Disease ketoconazole/B-Chemical`
- 1. Frequent-tag baseline (wrong): [syndrome]Disease, [ketoconazole]Chemical  — tags: `syndrome/I-Disease ketoconazole/B-Chemical`
- 2. HMM + Viterbi (correct): [Baboon syndrome]Disease, [ketoconazole]Chemical
- 3. Feature CRF (wrong): [ketoconazole]Chemical  — tags: `ketoconazole/B-Chemical`
- 4. BERT + softmax (correct): [Baboon syndrome]Disease, [ketoconazole]Chemical
- 5. BERT + CRF (correct): [Baboon syndrome]Disease, [ketoconazole]Chemical

## HMM wrong, CRF right (rich features help)  (sentence 507)

Sentence: Fluconazole associated agranulocytosis and thrombocytopenia .

- **Gold**: [Fluconazole]Chemical, [agranulocytosis]Disease, [thrombocytopenia]Disease  — tags: `Fluconazole/B-Chemical agranulocytosis/B-Disease thrombocytopenia/B-Disease`
- 1. Frequent-tag baseline (wrong): [agranulocytosis]Disease, [thrombocytopenia]Disease  — tags: `agranulocytosis/B-Disease thrombocytopenia/B-Disease`
- 2. HMM + Viterbi (wrong): [agranulocytosis]Disease, [thrombocytopenia]Disease  — tags: `agranulocytosis/B-Disease thrombocytopenia/B-Disease`
- 3. Feature CRF (correct): [Fluconazole]Chemical, [agranulocytosis]Disease, [thrombocytopenia]Disease
- 4. BERT + softmax (correct): [Fluconazole]Chemical, [agranulocytosis]Disease, [thrombocytopenia]Disease
- 5. BERT + CRF (correct): [Fluconazole]Chemical, [agranulocytosis]Disease, [thrombocytopenia]Disease

## CRF wrong, BERT + softmax right (context from the encoder)  (sentence 219)

Sentence: Linezolid - induced optic neuropathy .

- **Gold**: [Linezolid]Chemical, [optic neuropathy]Disease  — tags: `Linezolid/B-Chemical optic/B-Disease neuropathy/I-Disease`
- 1. Frequent-tag baseline (wrong): [optic neuropathy]Disease  — tags: `optic/B-Disease neuropathy/I-Disease`
- 2. HMM + Viterbi (wrong): [optic neuropathy]Disease  — tags: `optic/B-Disease neuropathy/I-Disease`
- 3. Feature CRF (wrong): [optic neuropathy]Disease  — tags: `optic/B-Disease neuropathy/I-Disease`
- 4. BERT + softmax (correct): [Linezolid]Chemical, [optic neuropathy]Disease
- 5. BERT + CRF (correct): [Linezolid]Chemical, [optic neuropathy]Disease

## BERT + softmax invalid IOB2, BERT + CRF right  (sentence 2180)

Sentence: Oral procaterol 50 micrograms b .

- **Gold**: [procaterol]Chemical  — tags: `procaterol/B-Chemical`
- 1. Frequent-tag baseline (wrong): (none)  — tags: `(all O)`
- 2. HMM + Viterbi (wrong): (none)  — tags: `(all O)`
- 3. Feature CRF (correct): [procaterol]Chemical
- 4. BERT + softmax (wrong): [procaterol 50]Chemical  — tags: `procaterol/I-Chemical 50/I-Chemical`
- 5. BERT + CRF (correct): [procaterol]Chemical

## BERT + CRF wrong, BERT + softmax right (CRF is not always better)  (sentence 3623)

Sentence: The Olanzipine HGGW Study Group .

- **Gold**: [Olanzipine]Chemical  — tags: `Olanzipine/B-Chemical`
- 1. Frequent-tag baseline (wrong): (none)  — tags: `(all O)`
- 2. HMM + Viterbi (wrong): (none)  — tags: `(all O)`
- 3. Feature CRF (wrong): [Olanzipine HGGW]Chemical  — tags: `Olanzipine/B-Chemical HGGW/I-Chemical`
- 4. BERT + softmax (correct): [Olanzipine]Chemical
- 5. BERT + CRF (wrong): [Olanzipine HGGW]Chemical  — tags: `Olanzipine/B-Chemical HGGW/I-Chemical`

## unseen name: only BERT methods right  (sentence 1516)

Sentence: Hypertension in diabetics prone to hypoglycaemia

- **Gold**: [Hypertension]Disease, [diabetics]Disease, [hypoglycaemia]Disease  — tags: `Hypertension/B-Disease diabetics/B-Disease hypoglycaemia/B-Disease`
- 1. Frequent-tag baseline (wrong): [Hypertension]Disease  — tags: `Hypertension/B-Disease`
- 2. HMM + Viterbi (wrong): [Hypertension]Disease  — tags: `Hypertension/B-Disease`
- 3. Feature CRF (wrong): [Hypertension]Disease, [hypoglycaemia]Disease  — tags: `Hypertension/B-Disease hypoglycaemia/B-Disease`
- 4. BERT + softmax (correct): [Hypertension]Disease, [diabetics]Disease, [hypoglycaemia]Disease
- 5. BERT + CRF (correct): [Hypertension]Disease, [diabetics]Disease, [hypoglycaemia]Disease

## every method wrong  (sentence 3593)

Sentence: Patterns of sulfadiazine acute nephrotoxicity .

- **Gold**: [sulfadiazine]Chemical, [acute nephrotoxicity]Disease  — tags: `sulfadiazine/B-Chemical acute/B-Disease nephrotoxicity/I-Disease`
- 1. Frequent-tag baseline (wrong): [nephrotoxicity]Disease  — tags: `nephrotoxicity/B-Disease`
- 2. HMM + Viterbi (wrong): [nephrotoxicity]Disease  — tags: `nephrotoxicity/B-Disease`
- 3. Feature CRF (wrong): [sulfadiazine]Chemical, [nephrotoxicity]Disease  — tags: `sulfadiazine/B-Chemical nephrotoxicity/B-Disease`
- 4. BERT + softmax (wrong): [sulfadiazine]Chemical, [nephrotoxicity]Disease  — tags: `sulfadiazine/B-Chemical nephrotoxicity/B-Disease`
- 5. BERT + CRF (wrong): [sulfadiazine]Chemical, [nephrotoxicity]Disease  — tags: `sulfadiazine/B-Chemical nephrotoxicity/B-Disease`

## boundary error by the best method  (sentence 3705)

Sentence: Viracept and irregular heartbeat warning .

- **Gold**: [Viracept]Chemical, [irregular heartbeat]Disease  — tags: `Viracept/B-Chemical irregular/B-Disease heartbeat/I-Disease`
- 1. Frequent-tag baseline (wrong): (none)  — tags: `(all O)`
- 2. HMM + Viterbi (wrong): (none)  — tags: `(all O)`
- 3. Feature CRF (wrong): (none)  — tags: `(all O)`
- 4. BERT + softmax (wrong): [and]Disease, [irregular heartbeat]Disease  — tags: `and/I-Disease irregular/B-Disease heartbeat/I-Disease`
- 5. BERT + CRF (wrong): [Viracept and irregular heartbeat]Disease  — tags: `Viracept/B-Disease and/I-Disease irregular/I-Disease heartbeat/I-Disease`
