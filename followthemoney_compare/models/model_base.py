try:
    import numpy as np
    import seaborn as sns
    from sklearn import metrics

    DEVMODE = True
except ImportError:
    DEVMODE = False

from functools import singledispatchmethod
import pickle

import pandas as pd


class EvaluatorBase:
    def __init__(self, *args, features, **kwargs):
        self.features = features

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
        return df[self.features].fillna(0).values

    @create_vector.register(dict)
    def create_vector_ftm_scores(self, scores):
        if not isinstance(scores, (list, tuple)):
            scores = tuple(scores)
        df = pd.Dataframe.from_records(scores)
        N = len(self.features)
        df["pct_full"] = [
            sum(s is not None for s in score.values()) / N for score in scores
        ]
        df["pct_partial"] = [
            sum(s is None for s in score.values()) / N for score in scores
        ]
        df["pct_empty"] = [len(score) / N for score in scores]
        return self.create_vector_dataframe(df)


class TrainerBase:
    def __init_subclass__(cls, /, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.name = getattr(cls, "name", None) or cls.__name__

    def __init__(self, features):
        self.features = features

    def fit(self, data):
        raise NotImplementedError

    def summarize(self):
        raise NotImplementedError

    def setup(self, data):
        pass

    def set_evaluator(self, evaluator):
        self.evaluator = evaluator

    @property
    def has_evaluator(self):
        return self.evaluator is not None

    def precision_recall_accuracy_curve(self, data, ax=None):
        estimate = self.evaluator.predict_proba(data)
        precision, recall, threshold = metrics.precision_recall_curve(
            data.y, estimate, sample_weight=data.weight
        )
        T = np.arange(0, 1, 0.05)
        accuracy = [
            metrics.accuracy_score(data.y, estimate > t, sample_weight=data.weight)
            for t in T
        ]
        max_accuracy = max(accuracy)
        sns.lineplot(x=threshold, y=recall[:-1], label="recall", ax=ax)
        sns.lineplot(x=threshold, y=precision[:-1], label="precision", ax=ax)
        ax = sns.lineplot(
            x=T, y=accuracy, label=f"accuracy ({max_accuracy*100:0.1f}%)", ax=ax
        )
        ax.hlines(y=max_accuracy, xmin=0, xmax=1, alpha=0.5)
        ax.set_xlim(0, 1)
        return ax

    def estimate_distribution(self, data, ax=None):
        data["estimate"] = self.evaluator.predict_proba(data)
        fg = sns.kdeplot(
            data=data, x="estimate", hue="judgement", cumulative=True, ax=ax
        )
        return fg
