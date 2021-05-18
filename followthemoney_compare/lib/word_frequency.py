import copy
import pickle
import math
import typing
import warnings
from pathlib import Path
from dataclasses import dataclass
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

    @staticmethod
    def merge(root, *wfs, binarize=False):
        N = len(root._bins)
        assert all(
            len(wf._bins) == N for wf in wfs
        ), "All WordFrequency objects must be the same size"
        merged = copy.deepcopy(root)
        if binarize:
            merged.binarize()
        for wf in wfs:
            if binarize:
                merged._bins += wf._bins.clip(max=1)
            else:
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


@dataclass
class Frequencies:
    token: WordFrequency
    schema: typing.Dict[str, WordFrequency]
    document: WordFrequency

    @classmethod
    def from_proxies(
        cls,
        proxies,
        confidence,
        error_rate,
        checkpoint_dir=None,
        checkpoint_freq=100_000,
    ):
        tf = WordFrequency(confidence=confidence, error_rate=error_rate)
        idf = defaultdict(
            lambda: WordFrequency(confidence=confidence, error_rate=error_rate)
        )
        sf = defaultdict(
            lambda: WordFrequency(confidence=confidence, error_rate=error_rate)
        )
        for i, proxy in enumerate(proxies):
            collection = proxy.context.get("collection")
            for name in proxy.names:
                for token in preprocess_text(name):
                    idxs = list(tf.iter_idxs(token))
                    tf.add_idxs(idxs)
                    idf[collection].add_idxs(idxs)
                    sf[proxy.schema.name].add_idxs(idxs)
            if checkpoint_freq and (i + 1) % checkpoint_freq == 0:
                idf_merge = WordFrequency.merge(*idf.values(), binarize=True)
                freq = cls(tf, idf_merge, sf)
                freq.summarize()
                if checkpoint_dir:
                    freq.save_dir(checkpoint_dir)
        idf_merge = WordFrequency.merge(*idf.values(), binarize=True)
        freq = cls(tf, idf_merge, sf)
        if checkpoint_dir:
            freq.save_dir(checkpoint_dir)
        return freq

    def summarize(self):
        print("Tokens:", self.token)
        print("Document:", self.document)
        print("Schmea:", self.schema)

    def token_frequency(self, token, schema=None):
        if schema is not None:
            try:
                tf = self.schema[str(schema)]
            except KeyError:
                warnings.warn(
                    f"Schema provided but it could not be found in frequencies lookup: {schema}",
                    UserWarning,
                )
                tf = self.token
        else:
            tf = self.token
        return tf[token]

    def document_frequency(self, token):
        return self.document[token]

    def tfidf(self, token, schema=None):
        return self.token_frequency(token, schema) / self.document_frequency(token)

    def inv_tfidf(self, token, schema=None):
        return self.document_frequency(token) / self.token_frequency(token, schema)

    @classmethod
    def load_dir(cls, directory):
        directory = Path(directory)
        with open(directory / "document_frequency.pro", "rb") as fd:
            df = WordFrequency.load(fd)
        with open(directory / "token_frequency.pro", "rb") as fd:
            tf = WordFrequency.load(fd)
        sf = {}
        for schema in directory.glob("schema_frequency/*.pro"):
            name = schema.stem
            with open(schema, "rb") as fd:
                sf[name] = WordFrequency.load(fd)
        return cls(token=tf, schema=sf, document=df)

    def save_dir(self, directory):
        directory = Path(directory)
        directory.mkdir(exist_ok=True, parents=True)
        with open(directory / "document_frequency.pro", "wb+") as fd:
            self.document.save(fd)
        with open(directory / "token_frequency.pro", "wb+") as fd:
            self.token.save(fd)
        schema_dir = directory / "schema_frequency"
        schema_dir.mkdir(exist_ok=True)
        for schema, f in self.schema.items():
            with open(schema_dir / f"{schema}.pro", "wb+") as fd:
                f.save(fd)


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
