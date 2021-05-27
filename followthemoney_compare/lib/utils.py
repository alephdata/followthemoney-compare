from itertools import chain
from collections import defaultdict, Counter, namedtuple
import random
import math
import json

import pandas as pd

from followthemoney import model
from followthemoney_compare import compare

from . import TARGETS


PairWeights = namedtuple("PairWeights", ("user_weight", "pair_weight"))


def stdin_to_proxies(stdin, exclude_schema=None, include_schema=None):
    for line in stdin:
        data = json.loads(line)
        proxy = model.get_proxy(data)
        if include_schema and not any(proxy.schema.is_a(s) for s in include_schema):
            continue
        if exclude_schema and any(proxy.schema.is_a(s) for s in exclude_schema):
            continue
        yield proxy


def pair_weight(e1, e2, plateau_start=0.25, plateau_end=0.7):
    """
    Piecewise weighting on two entities followthemoney.compare.compare score.

    The weight is a piecewise function that is linear from 0-1 for scores
    0 - plateau_start. Constant 1 for scores between plateau_start and
    plateau_end. Linear from 1-0 from plateau_end - 1.
    """
    scores = list(filter(None, compare.scores(model, e1, e2).values()))
    if not scores:
        return 0.0
    score = sum(scores) / len(scores)
    if score < plateau_start:
        return score / (plateau_start)
    elif score < plateau_end:
        return 1.0
    else:
        return (score - plateau_end) / (plateau_end - 1) + 1


def user_weight(judgement_counts):
    # max isn't necissary because the counts are always positive
    # which means tanh also will... but let's be safe
    w = max(0.0, math.tanh(sum(judgement_counts.values()) / 100))
    return w * int(len(judgement_counts) > 1)


def create_user_weights_lookup(profiles):
    user_decisions = defaultdict(Counter)
    for decision in profiles.iter_decisions():
        user = decision["added_by_id"]
        judgement = decision["judgement"]
        user_decisions[user][judgement] += 1
    return {uid: user_weight(c) for uid, c in user_decisions.items()}


def calculate_pair_weights(e1, e2, profile, user_weight):
    user_ids = [
        decision["added_by_id"]
        for decision in chain(
            profile.entity_decisions(e1), profile.entity_decisions(e2)
        )
    ]
    uw = min(map(user_weight.get, user_ids))
    pw = pair_weight(e1, e2)
    return PairWeights(uw, pw)


def max_or_none(s):
    try:
        return max(filter(None, s))
    except (ValueError, TypeError):
        return None


def profiles_to_pairs_pandas(profiles, judgements=None):
    pairs_scores = profiles.to_pairs_dict(judgements=judgements)
    columns = set(k for r in pairs_scores for k in r.keys())
    df = pd.DataFrame.from_records(pairs_scores, columns=columns)
    df = df[df.judgement != "unsure"]
    df["judgement"] = df.judgement.astype("category")
    df["y"] = df.judgement == "positive"
    df["weight"] = df.user_weight * df.pair_weight
    df["phase"] = random.choices(["train", "valid"], weights=[0.8, 0.2], k=df.shape[0])
    return df
