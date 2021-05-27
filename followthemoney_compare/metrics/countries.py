def compare_countries(group_type, left, right):
    left_countries = left.country_hints
    right_countries = right.country_hints
    if not left_countries and not right_countries:
        raise ValueError("At least one proxy must have country properties")
    elif not left_countries or not right_countries:
        return None
    intersection = left_countries.intersection(right_countries)
    union = left_countries.union(right_countries)
    return len(intersection) / len(union)
