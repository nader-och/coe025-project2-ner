"""Viterbi decoding, written from scratch, in log space / score space.

Week 6, Slide 56: delta_1(j) = pi(j) * b(j, x_1)
                  delta_t(j) = b(j, x_t) * max_i [delta_{t-1}(i) * a(i, j)]
                  store the predecessor that wins each max; backtrace at the end.
Week 6, Slide 65: in log space, products become sums (no underflow).
Week 6, Slide 74: CRF decoding uses the same algorithm with real-valued scores:
                  add unary and transition scores, keep the best per ending tag.
Week 6, Slide 58: this costs O(T K^2) instead of K^T complete paths.

The same function is used by the HMM (log probabilities), the feature CRF and
BERT + CRF (scores).
"""
import numpy as np


def viterbi(start, trans, unary, end=None):
    """Return (best_path, best_score).

    start: (K,)    log pi(j)      or CRF start score of label j
    trans: (K, K)  log a(i, j)    or CRF transition score i -> j
    unary: (T, K)  log b(j, x_t)  or CRF unary score u_t(j)
    end:   (K,)    optional end score of label j (CRF); None means 0
    """
    T, K = unary.shape
    delta = np.zeros((T, K))
    backpointer = np.zeros((T, K), dtype=int)

    delta[0] = start + unary[0]                         # initialization (Slide 56)
    for t in range(1, T):
        # candidates[i, j] = delta_{t-1}(i) + a(i, j): every predecessor i for each tag j
        candidates = delta[t - 1][:, None] + trans
        # Tie policy (Slide 56 notes): np.argmax keeps the smallest label index.
        backpointer[t] = np.argmax(candidates, axis=0)  # store the winning predecessor
        delta[t] = candidates[backpointer[t], np.arange(K)] + unary[t]

    final = delta[T - 1] + (end if end is not None else 0.0)
    best_last = int(np.argmax(final))                   # termination
    path = [best_last]
    for t in range(T - 1, 0, -1):                       # follow predecessors backward
        path.append(int(backpointer[t, path[-1]]))
    path.reverse()
    return path, float(final[best_last])
