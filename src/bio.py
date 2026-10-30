"""BIO / IOB2 label rules.

Week 6, Slide 23: BIO has structural rules (O -> I-PER is invalid).
Week 6, Slides 80-81: under IOB2, I-X may only follow B-X or I-X of the same
type X, and a sentence cannot start with I-X.
Week 6, Slide 21 notes: "At B create a new span; at I append the current
token; at O close an active span."
"""

START = "START"


def split_label(label):
    """'B-Chemical' -> ('B', 'Chemical'); 'O' -> ('O', None)."""
    if label == "O":
        return "O", None
    prefix, entity_type = label.split("-", 1)
    return prefix, entity_type


def is_allowed(prev_label, label):
    """Week 6, Slide 81: is the transition prev_label -> label legal in IOB2?

    prev_label may be START (the beginning of the sentence).
    """
    prefix, entity_type = split_label(label)
    if prefix in ("O", "B"):
        return True  # O and B-X may follow any label or START
    # prefix == "I": the previous label must be B-X or I-X with the same type X
    if prev_label == START:
        return False  # START -> I-X is forbidden
    prev_prefix, prev_type = split_label(prev_label)
    return prev_prefix in ("B", "I") and prev_type == entity_type


def count_invalid(tags):
    """Number of illegal IOB2 transitions in one sentence (Week 6, Slides 23, 80)."""
    count = 0
    prev = START
    for tag in tags:
        if not is_allowed(prev, tag):
            count += 1
        prev = tag
    return count


def get_spans(tags):
    """Turn BIO tags into entity spans (start, end, type); end is exclusive.

    Week 6, Slide 21 notes: B starts a span, I continues it, O closes it.
    Week 6, Slide 13: an entity is a span (boundaries) plus a type.
    """
    spans = []
    start, current_type = None, None
    for i, tag in enumerate(tags):
        prefix, entity_type = split_label(tag)
        continues = prefix == "I" and current_type == entity_type
        if not continues and current_type is not None:
            spans.append((start, i, current_type))  # close the active span
            start, current_type = None, None
        if prefix == "B" or (prefix == "I" and not continues):
            # Not in slides: the slides do not say how to read an invalid I-X
            # (for example O -> I-Disease). We start a new span here (the usual
            # "conlleval" rule), so predicted entity words are not silently lost.
            # Invalid transitions are counted separately with count_invalid().
            start, current_type = i, entity_type
    if current_type is not None:
        spans.append((start, len(tags), current_type))
    return spans
