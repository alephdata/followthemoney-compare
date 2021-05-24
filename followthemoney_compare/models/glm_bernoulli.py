import numpy as np

from . import pm_model_base as pmb
from .model_base import EvaluatorBase


class GLMBernoulliEvaluator(EvaluatorBase):
    """
    TODO: vectorize this in the same way we did for glm_bernoulli_2e
    """

    def __init__(self, features, weights):
        self.features = features
        self.weights = weights
        self.weights_mean = {f: w.mean(axis=0) for f, w in weights.items()}

    def predict_proba(self, data):
        data = self.create_vector(data)
        score = (
            np.inner(self.weights_mean["coef"], data) + self.weights_mean["intercept"]
        )
        return 1.0 / (1.0 + np.exp(-score))

    def predict_std(self, data, n_samples=None):
        data = self.create_vector(data)
        score = (
            np.inner(self.weights["coef"][:n_samples], data)
            + self.weights["intercept"][:n_samples, np.newaxis]
        )
        proba = 1.0 / (1.0 + np.exp(-score))
        return proba.std(0)


class GLMBernoulli(pmb.PMModelBase):
    name = "glm_bernoulli"

    def create_evaluator(self, trace):
        weights = {
            "coef": trace.get_values("coef"),
            "intercept": trace.get_values("intercept"),
        }
        return GLMBernoulliEvaluator(self.features, weights)

    def setup(self, data):
        data[self.features] = data[self.features].fillna(0)
        with self.model:
            weights = {
                "coef": pmb.pm.Normal("coef", 0, 5, shape=len(self.features)),
                "intercept": pmb.pm.Normal("intercept", 0, 5),
            }
            score = (weights["coef"] * data[self.features].values).sum(
                axis=-1
            ) + weights["intercept"]
            mu = pmb.pm.math.invlogit(score)
            weights["error"] = self.weighted_bernoili_logli(data, mu)
            self._weights = weights
            return weights
