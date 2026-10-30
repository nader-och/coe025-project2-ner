# Test results (BC5CDR test split, used once)

| Method | Micro P | Micro R | Micro F1 | Macro F1 | Chemical F1 | Disease F1 | Token acc. | Invalid IOB2 | ms / sentence (CPU) | ms / sentence (GPU) |
|---|---|---|---|---|---|---|---|---|---|---|
| All-O (Slide 36 demo) | 0.000 | 0.000 | **0.000** | 0.000 | 0.000 | 0.000 | 0.891 | 0 | - | - |
| 1. Frequent-tag baseline | 0.690 | 0.578 | **0.629** | 0.630 | 0.730 | 0.529 | 0.944 | 1265 | 0.01 | - |
| 2. HMM + Viterbi | 0.822 | 0.592 | **0.689** | 0.688 | 0.717 | 0.659 | 0.948 | 0 | 0.21 | - |
| 3. Feature CRF | 0.842 | 0.777 | **0.808** | 0.804 | 0.855 | 0.752 | 0.964 | 0 | 0.58 | - |
| 4. BERT + softmax | 0.819 | 0.870 | **0.844** | 0.838 | 0.901 | 0.775 | 0.970 | 342 | 131.85 | 1.47 |
| 5. BERT + CRF | 0.854 | 0.859 | **0.857** | 0.852 | 0.903 | 0.801 | 0.970 | 0 | 125.41 | 1.81 |

Micro/macro F1 = entity-level exact match (Week 6, Slides 37-39). Zero-division rule: a score with denominator 0 is 0.
Speed = prediction time per sentence on the validation split, all methods on the same Colab machine (see results/timing/timing_colab.json): {'platform': 'Linux-6.6.122+-x86_64-with-glibc2.39', 'processor': 'x86_64', 'python': '3.13.16', 'cpu_count': 2, 'torch': '2.11.0+cu130', 'torch_threads': 1, 'gpu': 'Tesla T4'}
