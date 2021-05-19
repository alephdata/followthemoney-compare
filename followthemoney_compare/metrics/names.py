import itertools
import math

from fuzzywuzzy.fuzz import partial_token_set_ratio, WRatio

from .common import maybe_fingerprint
from ..lib.word_frequency import preprocess_text


def _compare_names_fuzzy_wf(t1, s1, t2, s2, frequencies):
    score = 0
    norm = 0
    n_tokens = 0
    tp1 = preprocess_text(t1)
    tp2 = preprocess_text(t2)
    for left, schema, right in ((tp1, s1, tp2), (tp2, s2, tp1)):
        right_str = " ".join(right)
        for token_left in left:
            n = math.log1p(frequencies.inv_tfidf(token_left, schema=schema))
            s = partial_token_set_ratio(
                token_left, right_str, force_ascii=False, full_process=False
            )
            score += n * s
            norm += n
            n_tokens += 1
    if not (score and norm):
        return 0.0
    min_n_tokens = min(len(tp1), len(tp2))
    dampening = 2 ** (-min_n_tokens + 3) + 1
    return (score / norm / 100.0) ** dampening


def compare_names_fuzzy_wf(group_type, proxy1, proxy2, frequencies):
    return max(
        _compare_names_fuzzy_wf(
            t1, proxy1.schema.name, t2, proxy2.schema.name, frequencies
        )
        for t1, t2 in itertools.product(proxy1.names, proxy2.names)
    )


def compare_names_fuzzy(group_type, left, right, max_names=200):
    result = 0.0
    left_list = list(itertools.islice(maybe_fingerprint(left.names), max_names))
    right_list = list(itertools.islice(maybe_fingerprint(right.names), max_names))
    if not left_list and not right_list:
        raise ValueError("At least one proxy must have name properties")
    elif not left_list or not right_list:
        return None
    for (left, right) in itertools.product(left_list, right_list):
        similarity = WRatio(left, right, full_process=False) / 100.0
        result = max(result, similarity)
        if result == 1.0:
            break
    # safeguard against datasets with bad keys where one entity will have
    # hundres of associated names
    result *= min(
        1.0, 2 ** (-len(left_list) * len(right_list) / (max_names * max_names))
    )
    return result
