from functools import singledispatchmethod
import pickle

import pandas as pd


class TrainerBase:
    def __init_subclass__(cls, /, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.name = getattr(cls, "name", None) or cls.__name__

    def __init__(self, targets):
        self.targets = targets

    def fit(self, data):
        raise NotImplementedError

    def setup(self, data):
        pass

    def set_evaluator(self, evaluator):
        self.evaluator = evaluator

    @property
    def has_evaluator(self):
        return self.evaluator is not None


class EvaluatorBase:
    def __init__(self, *args, targets, **kwargs):
        self.targets = targets

    def predict_std(self, data, n_samples=None):
        raise NotImplementedError

    def predict_proba(self, data):
        raise NotImplementedError

    def predict(self, data):
        return self.predict_proba(data) > 0.5

    def to_pickles(self):
        return pickle.dumps(self)

    @classmethod
    def from_pickles(self, p):
        return pickle.loads(p)

    @singledispatchmethod
    def create_vector(self, _):
        raise NotImplementedError

    @create_vector.register(pd.DataFrame)
    @create_vector.register(pd.Series)
    def create_vector_dataframe(self, df):
        return df[self.targets].fillna(0).values

    @create_vector.register(dict)
    def create_vector_ftm_scores(self, scores):
        df = pd.Dataframe.from_records(scores)
        N = len(self.targets)
        df["pct_full"] = sum(s is not None for s in scores.values()) / N
        df["pct_partial"] = sum(s is None for s in scores.values()) / N
        df["pct_empty"] = len(scores) / N
        return self.create_vector_dataframe(df)
