import math
import warnings
import os
from functools import partial

from followthemoney.exc import InvalidData
from followthemoney.types import registry

from .metrics import compare_names_fuzzy, compare_names_fuzzy_wf, compare_countries
from .lib import Frequencies


try:
    FREQUENCIES_DIR = os.environ.get(
        "FTM_COMPARE_FREQUENCIES_DIR", "./data/word_frequencies/"
    )
    frequencies = Frequencies.load_dir(FREQUENCIES_DIR)
    DEFAULT_OVERRIDES = {
        registry.name: partial(compare_names_fuzzy_wf, frequencies=frequencies),
        registry.country: compare_countries,
    }
except FileNotFoundError:
    warnings.warn(
        "Could not find frequencies data... using naive name matching", UserWarning
    )
    DEFAULT_OVERRIDES = {
        registry.name: compare_names_fuzzy,
        registry.country: compare_countries,
    }


def scores(model, left, right, overrides=DEFAULT_OVERRIDES):
    """Compare two entities and return a match score for each property."""
    left = model.get_proxy(left)
    right = model.get_proxy(right)
    try:
        common_schema = model.common_schema(left.schema, right.schema)
    except InvalidData:
        return {}
    values = dict()
    left_inv = left.get_type_inverted(matchable=True)
    right_inv = right.get_type_inverted(matchable=True)
    left_groups = set(left_inv.keys())
    right_groups = set(right_inv.keys())
    for group_name in left_groups.intersection(right_groups):
        group = registry.groups[group_name]
        try:
            if group in overrides:
                score = overrides[group](group, left, right)
            else:
                score = compare_group(
                    group, left_inv[group_name], right_inv[group_name]
                )
            values[group] = score
        except ValueError:
            values[group] = None
    for group_name in left_groups.symmetric_difference(right_groups):
        group = registry.groups[group_name]
        values[group] = None
    common_schema_types = {p.type for p in common_schema.properties.values()}
    N = len(common_schema_types)
    values.update(
        {
            "pct_share_prop": sum(s is not None for s in values.values()) / N,
            "pct_miss_prop": sum(s is None for s in values.values()) / N,
        }
    )
    return values


def apply_weights(scores, weights, n_std=1):
    if not scores or not any(scores.values()):
        return 0.0
    prob = 0
    for field, weight in weights.items():
        if field:
            prob += weight * (scores.get(field) or 0.0)
        else:
            prob += weight
    return 1.0 / (1.0 + math.exp(-prob))


def compare_group(group_type, left_values, right_values):
    if not left_values and not right_values:
        raise ValueError("At least one proxy must have property type: %s", group_type)
    elif not left_values or not right_values:
        return None
    return group_type.compare_sets(left_values, right_values)
