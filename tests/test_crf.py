"""Unit tests: the CRF layer must reproduce the worked examples on the slides."""
import math

import torch

from src.crf import LinearChainCRF, iob2_constraints, word_shape

LN2, LN8 = math.log(2), math.log(8)
N, V = 0, 1


def can_fish_crf(a_vn):
    """Week 6, Slides 70-71: 'can fish', unary u1=(ln2, 0), u2=(0, ln2),
    transition a(V,N)=a_vn, every other score 0."""
    crf = LinearChainCRF(2)
    with torch.no_grad():
        crf.trans[V, N] = a_vn
    unary = torch.tensor([[[LN2, 0.0], [0.0, LN2]]])
    mask = torch.tensor([[True, True]])
    return crf, unary, mask


def test_slide_75_partition_function():
    crf, unary, mask = can_fish_crf(LN8)
    assert math.isclose(crf.log_partition(unary, mask).item(), math.log(16), rel_tol=1e-6)


def test_slide_72_77_loss():
    # Gold VN: loss = ln Z - gold score = ln 16 - ln 8 = ln 2 = 0.6931.
    crf, unary, mask = can_fish_crf(LN8)
    loss = crf.neg_log_likelihood(unary, torch.tensor([[V, N]]), mask)
    assert round(loss.item(), 4) == 0.6931


def test_slide_74_viterbi():
    crf, unary, _ = can_fish_crf(LN8)
    assert crf.decode(unary[0]) == [V, N]


def test_slide_78_gradient():
    # d loss / d theta = expected count - gold count = 0.5 - 1 = -0.5
    crf, unary, mask = can_fish_crf(LN8)
    crf.neg_log_likelihood(unary, torch.tensor([[V, N]]), mask).backward()
    assert math.isclose(crf.trans.grad[V, N].item(), -0.5, rel_tol=1e-5)


def test_slide_91_changed_transition():
    # a(V,N) = ln 2: Z = 10, NV wins, loss for gold VN = ln 5 = 1.6094.
    crf, unary, mask = can_fish_crf(LN2)
    assert math.isclose(crf.log_partition(unary, mask).item(), math.log(10), rel_tol=1e-6)
    assert crf.decode(unary[0]) == [N, V]
    loss = crf.neg_log_likelihood(unary, torch.tensor([[V, N]]), mask)
    assert round(loss.item(), 4) == 1.6094


def test_padding_does_not_change_z():
    # A padded second sentence must give the same ln Z as the sentence alone.
    crf, unary, mask = can_fish_crf(LN8)
    padded = torch.cat([unary, torch.zeros(1, 1, 2)], dim=1)
    padded_mask = torch.tensor([[True, True, False]])
    assert math.isclose(crf.log_partition(padded, padded_mask).item(), math.log(16), rel_tol=1e-6)


def test_slide_80_81_mask_blocks_invalid_paths():
    # Labels O, B-PER, I-PER. Even with a huge unary score for I-PER at the
    # first position, decoding must not start with I-PER (START -> I-X forbidden).
    labels = ["O", "B-PER", "I-PER"]
    crf = LinearChainCRF(3, *iob2_constraints(labels))
    unary = torch.tensor([[0.0, 0.0, 50.0], [0.0, 0.0, 50.0]])
    path = [labels[j] for j in crf.decode(unary)]
    assert path[0] != "I-PER"
    assert path == ["B-PER", "I-PER"]


def test_word_shape():
    assert word_shape("Naloxone") == "Xx"
    assert word_shape("5-HT") == "d-X"
