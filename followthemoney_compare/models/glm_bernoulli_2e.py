import numpy as np

from . import pm_model_base as pmb
from .model_base import EvaluatorBase


DEFAULT_FEATURES_2E = (
    ("pct_full", "pct_full"),
    ("name", "pct_full"),
    ("name", "name"),
    ("pct_full", "pct_partial"),
    ("pct_partial", "pct_partial"),
    ("name", "identifier"),
    ("country", "pct_full"),
    ("identifier", "identifier"),
    ("identifier", "pct_partial"),
    ("date", "date"),
    ("address", "address"),
)


def separate_2e_features(features_2e):
    return map(list, zip(*features_2e))


class GLMBernoulli2EEvaluator(EvaluatorBase):
    def __init__(self, features, features_2e, weights):
        self.features = features
        self.features_2e = features_2e
        self._features_2e_left, self._features_2e_right = separate_2e_features(
            features_2e
        )
        self.weights = weights
        self.weights_mean = {f: w.mean(axis=0) for f, w in weights.items()}

    def _calc_2e_terms(self, df):
        return np.nan_to_num(
            df[self._features_2e_left].values * df[self._features_2e_right].values
        )

    def predict_proba(self, df):
        data = self.create_vector(df)
        data_2e = self._calc_2e_terms(df)
        score = (
            np.inner(self.weights_mean["coef"], data)
            + np.inner(self.weights_mean["coef_2e"], data_2e)
            + self.weights_mean["intercept"]
        )
        return 1.0 / (1.0 + np.exp(-score))

    def predict_std(self, df, n_samples=None):
        data = self.create_vector(df)
        data_2e = self._calc_2e_terms(df)
        score = (
            np.inner(self.weights["coef"][:n_samples], data)
            + np.inner(self.weights["coef_2e"][:n_samples], data_2e)
            + self.weights["intercept"][:n_samples, np.newaxis]
        )
        proba = 1.0 / (1.0 + np.exp(-score))
        return proba.std(axis=-1)


class GLMBernoulli2E(pmb.PMModelBase):
    name = "glm_bernoulli_2e"

    def __init__(self, *args, features_2e=DEFAULT_FEATURES_2E, **kwargs):
        super().__init__(*args, **kwargs)
        self.features_2e = features_2e
        self._features_2e_left, self._features_2e_right = separate_2e_features(
            self.features_2e
        )

    def create_evaluator(self, trace):
        weights = {
            "coef": trace.get_values("coef"),
            "coef_2e": trace.get_values("coef_2e"),
            "intercept": trace.get_values("intercept"),
        }
        return GLMBernoulli2EEvaluator(self.features, self.features_2e, weights)

    def setup(self, data):
        data[self.features] = data[self.features].fillna(0)
        data_2e = (
            data[self._features_2e_left].values * data[self._features_2e_right].values
        )
        with self.model:
            weights = {
                "coef": pmb.pm.Normal("coef", 0, 5, shape=len(self.features)),
                "coef_2e": pmb.pm.Normal("coef_2e", 0, 5, shape=len(self.features_2e)),
                "intercept": pmb.pm.Normal("intercept", 0, 5),
            }
            score = (
                pmb.T.sum(weights["coef"] * data[self.features].values, axis=-1)
                + pmb.T.sum(weights["coef_2e"] * data_2e, axis=-1)
                + weights["intercept"]
            )
            mu = pmb.pm.math.invlogit(score)
            weights["error"] = self.weighted_bernoili_logli(data, mu)
            self._weights = weights
            return weights

