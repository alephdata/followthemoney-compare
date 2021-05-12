import copy
import pickle
import math
from collections import defaultdict

import numpy as np
import mmh3
from normality import normalize


def normalize_text(text):
    return normalize(text, ascii=True, latinize=True, lowercase=True, collapse=True)


def tokenize_text(text):
    return text.split(" ")


def preprocess_text(text):
    if not text:
        return []
    return tokenize_text(normalize_text(text))


class WordFrequency:
    def __init__(self, confidence, error_rate, bin_dtype="uint32"):
        self.n_items = 0
        self.confidence = confidence
        self.error_rate = error_rate

        self.width = math.ceil(2 / error_rate)
        numerator = -1 * math.log(1 - confidence)
        self.depth = math.ceil(numerator / 0.6931471805599453)

        self.dtype = np.dtype(bin_dtype)
        self.idtype = np.iinfo(self.dtype)
        self._bins = np.zeros(self.width * self.depth, dtype=self.dtype)

    def iter_idxs(self, key):
        k1, k2 = mmh3.hash64(key)
        for i in range(self.depth):
            idx = (k1 + i * k2) % self.width
            yield (idx % self.width) + (i * self.width)

    def add(self, key):
        idxs = list(self.iter_idxs(key))
        return self.add_idxs(idxs)

    def add_idxs(self, idxs):
        value = np.clip(self.get_idxs(idxs) + 1, 0, self.idtype.max)
        for idx in idxs:
            self._bins[idx] = np.maximum(value, self._bins[idx])
        self.n_items += 1
        return value

    def get_idxs(self, idxs):
        values = self._bins[idxs]
        return np.min(values)

    def get(self, key):
        idxs = list(self.iter_idxs(key))
        return self.get_idxs(idxs)

    def get_text(self, text):
        return {token: self.get(token) for token in preprocess_text(text)}

    def __getitem__(self, key):
        return self.get(key)

    def merge(self, *wfs):
        N = len(self._bins)
        assert all(
            len(wf._bins) == N for wf in wfs
        ), "All WordFrequency objects must be the same size"
        merged = copy.deepcopy(self)
        for wf in wfs:
            merged._bins += wf._bins
            merged.n_items += wf.n_items
        return merged

    def binarize(self):
        self._bins = self._bins.clip(max=1)
        return self

    def save(self, fd):
        fd.write(pickle.dumps(self))

    @classmethod
    def load(cls, fd):
        return pickle.load(fd)

    def __len__(self):
        return self.n_items

    def __repr__(self):
        error = int(self.error_rate * self.n_items)
        return f"<WordFrequency c:{self.confidence};e:{self.error_rate}@{self.n_items}={error}>"


def word_frequency_from_proxies(
    proxies,
    confidence,
    error_rate,
    document_frequency=False,
    schema_frequency=False,
    yield_freq=100_000,
):
    wf = WordFrequency(confidence=confidence, error_rate=error_rate)
    if document_frequency:
        idf = defaultdict(
            lambda: WordFrequency(confidence=confidence, error_rate=error_rate)
        )
    else:
        idf = None
    if schema_frequency:
        sf = defaultdict(
            lambda: WordFrequency(confidence=confidence, error_rate=error_rate)
        )
    else:
        sf = None
    for i, proxy in enumerate(proxies):
        collection = proxy.context.get("collection")
        for name in proxy.names:
            for token in preprocess_text(name):
                idxs = list(wf.iter_idxs(token))
                wf.add_idxs(idxs)
                if idf is not None and collection:
                    idf[collection].add_idxs(idxs)
                if sf is not None:
                    sf[proxy.schema.name].add_idxs(idxs)
        if yield_freq and (i + 1) % yield_freq == 0:
            print("Status report:", i)
            print(wf)
            print(idf)
            print(sf)
            yield wf, idf, sf
    yield wf, idf, sf
