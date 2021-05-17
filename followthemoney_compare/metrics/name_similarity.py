from fuzzywuzzy.fuzz import partial_token_set_ratio


def text_similarity(t1, t2, frequencies):
    score = 0
    norm = 0
    tp1 = preprocess_text(t1)
    tp2 = preprocess_text(t2)
    for left, right in ((tp1, tp2), (tp2, tp1)):
        right_str = " ".join(right)
        for token_left in left:
            n = frequencies.inv_tfidf(token)
            s = partial_token_set_ratio(token_left, right_str, force_ascii=False, full_process=False)
            score += n * s
            norm += n
    return score / norm / 100.0

