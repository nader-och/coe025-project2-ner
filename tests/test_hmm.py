"""Unit tests: the HMM pieces must reproduce the worked examples on the slides."""
import math

import numpy as np

from src.hmm import smoothed_distribution
from src.viterbi import viterbi

# Week 6, Slides 52-53: "can fish", states N=0, V=1.
START = np.log([0.6, 0.4])
TRANS = np.log([[0.3, 0.7],
                [0.8, 0.2]])
EMIT_CAN = np.log([0.4, 0.7])
EMIT_FISH = np.log([0.6, 0.3])


def test_slides_54_57_viterbi_can_fish():
    # Slide 54: VN has the largest joint weight 0.1344. Slide 57: backtrace V -> N.
    path, score = viterbi(START, TRANS, np.vstack([EMIT_CAN, EMIT_FISH]))
    assert path == [1, 0]
    assert math.isclose(math.exp(score), 0.1344)


def test_slide_65_log_space():
    # Slide 65: ln P(X, VN) = ln 0.4 + ln 0.7 + ln 0.8 + ln 0.6 = -2.0069
    _, score = viterbi(START, TRANS, np.vstack([EMIT_CAN, EMIT_FISH]))
    assert round(score, 4) == -2.0069


def test_slides_88_89_changed_transition():
    # Slide 89: with V row (0.5, 0.5), VN still wins with joint weight 0.084.
    trans = np.log([[0.3, 0.7], [0.5, 0.5]])
    path, score = viterbi(START, trans, np.vstack([EMIT_CAN, EMIT_FISH]))
    assert path == [1, 0]
    assert math.isclose(math.exp(score), 0.084)


def test_slide_63_add_one():
    # Slide 63: counts N=2, V=0 -> 0.75 and 0.25 with add-one smoothing.
    assert np.allclose(smoothed_distribution([2, 0], 1.0), [0.75, 0.25])


def test_slide_64_unk():
    # Slide 64: can=4, fish=6, UNK=0 -> 5/13, 7/13, 1/13.
    assert np.allclose(smoothed_distribution([4, 6, 0], 1.0), [5 / 13, 7 / 13, 1 / 13])


def test_slide_58_matches_enumeration():
    # Slide 58: Viterbi finds the same best path as listing every path.
    rng = np.random.default_rng(0)
    K, T = 3, 6
    start, trans, unary = rng.normal(size=K), rng.normal(size=(K, K)), rng.normal(size=(T, K))
    best, best_score = None, -np.inf
    for code in range(K ** T):
        path = [(code // K ** t) % K for t in range(T)]
        score = start[path[0]] + sum(unary[t, path[t]] for t in range(T)) + \
            sum(trans[path[t - 1], path[t]] for t in range(1, T))
        if score > best_score:
            best, best_score = path, score
    path, score = viterbi(start, trans, unary)
    assert path == best and math.isclose(score, best_score)
