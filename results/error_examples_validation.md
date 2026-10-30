# Disagreement examples (validation split)

Chosen automatically by src/error_analysis.py (one rule per row; shortest sentence).

## baseline wrong, HMM right (label transitions help)  (sentence 753)

Sentence: 5 % had renal injury .

- **Gold**: [renal injury]Disease  — tags: `renal/B-Disease injury/I-Disease`
- 1. Frequent-tag baseline (wrong): [injury]Disease  — tags: `injury/I-Disease`
- 2. HMM + Viterbi (correct): [renal injury]Disease
- 3. Feature CRF (correct): [renal injury]Disease
- 4. BERT + softmax (correct): [renal injury]Disease
- 5. BERT + CRF (correct): [renal injury]Disease

## HMM wrong, CRF right (rich features help)  (sentence 977)

Sentence: Possible azithromycin - associated hiccups .

- **Gold**: [azithromycin]Chemical, [hiccups]Disease  — tags: `azithromycin/B-Chemical hiccups/B-Disease`
- 1. Frequent-tag baseline (correct): [azithromycin]Chemical, [hiccups]Disease
- 2. HMM + Viterbi (wrong): [azithromycin]Chemical  — tags: `azithromycin/B-Chemical`
- 3. Feature CRF (correct): [azithromycin]Chemical, [hiccups]Disease
- 4. BERT + softmax (correct): [azithromycin]Chemical, [hiccups]Disease
- 5. BERT + CRF (correct): [azithromycin]Chemical, [hiccups]Disease

## CRF wrong, BERT + softmax right (context from the encoder)  (sentence 786)

Sentence: 0 %; valdecoxib , 1 .

- **Gold**: [valdecoxib]Chemical  — tags: `valdecoxib/B-Chemical`
- 1. Frequent-tag baseline (wrong): (none)  — tags: `(all O)`
- 2. HMM + Viterbi (wrong): (none)  — tags: `(all O)`
- 3. Feature CRF (wrong): (none)  — tags: `(all O)`
- 4. BERT + softmax (correct): [valdecoxib]Chemical
- 5. BERT + CRF (correct): [valdecoxib]Chemical

## BERT + softmax invalid IOB2, BERT + CRF right  (sentence 3012)

Sentence: IBPB may cause transient auditory dysfunction

- **Gold**: [auditory dysfunction]Disease  — tags: `auditory/B-Disease dysfunction/I-Disease`
- 1. Frequent-tag baseline (wrong): [dysfunction]Disease  — tags: `dysfunction/I-Disease`
- 2. HMM + Viterbi (wrong): [transient auditory dysfunction]Disease  — tags: `transient/B-Disease auditory/I-Disease dysfunction/I-Disease`
- 3. Feature CRF (correct): [auditory dysfunction]Disease
- 4. BERT + softmax (wrong): [IBPB]Chemical, [auditory dysfunction]Disease  — tags: `IBPB/B-Chemical auditory/I-Disease dysfunction/I-Disease`
- 5. BERT + CRF (correct): [auditory dysfunction]Disease

## BERT + CRF wrong, BERT + softmax right (CRF is not always better)  (sentence 3543)

Sentence: o .) administration of IFS .

- **Gold**: [IFS]Chemical  — tags: `IFS/B-Chemical`
- 1. Frequent-tag baseline (wrong): (none)  — tags: `(all O)`
- 2. HMM + Viterbi (wrong): (none)  — tags: `(all O)`
- 3. Feature CRF (wrong): (none)  — tags: `(all O)`
- 4. BERT + softmax (correct): [IFS]Chemical
- 5. BERT + CRF (wrong): (none)  — tags: `(all O)`

## unseen name: only BERT methods right  (sentence 974)

Sentence: Administration of daidzein ( 4 .

- **Gold**: [daidzein]Chemical  — tags: `daidzein/B-Chemical`
- 1. Frequent-tag baseline (wrong): (none)  — tags: `(all O)`
- 2. HMM + Viterbi (wrong): (none)  — tags: `(all O)`
- 3. Feature CRF (wrong): (none)  — tags: `(all O)`
- 4. BERT + softmax (correct): [daidzein]Chemical
- 5. BERT + CRF (correct): [daidzein]Chemical

## every method wrong  (sentence 1065)

Sentence: A catch in the Reye .

- **Gold**: [Reye]Disease  — tags: `Reye/B-Disease`
- 1. Frequent-tag baseline (wrong): (none)  — tags: `(all O)`
- 2. HMM + Viterbi (wrong): (none)  — tags: `(all O)`
- 3. Feature CRF (wrong): (none)  — tags: `(all O)`
- 4. BERT + softmax (wrong): (none)  — tags: `(all O)`
- 5. BERT + CRF (wrong): (none)  — tags: `(all O)`

## boundary error by the best method  (sentence 374)

Sentence: The transient steroid - induced IOP rise

- **Gold**: [steroid]Chemical, [IOP rise]Disease  — tags: `steroid/B-Chemical IOP/B-Disease rise/I-Disease`
- 1. Frequent-tag baseline (wrong): [steroid]Chemical  — tags: `steroid/B-Chemical`
- 2. HMM + Viterbi (wrong): [steroid]Chemical  — tags: `steroid/B-Chemical`
- 3. Feature CRF (wrong): [steroid]Chemical  — tags: `steroid/B-Chemical`
- 4. BERT + softmax (wrong): [steroid]Chemical, [IOP]Disease  — tags: `steroid/B-Chemical IOP/B-Disease`
- 5. BERT + CRF (wrong): [steroid]Chemical, [IOP]Disease  — tags: `steroid/B-Chemical IOP/B-Disease`
