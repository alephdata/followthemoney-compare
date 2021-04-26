import numpy as np

from . import pm_model_base as pmb
from .model_base import EvaluatorBase


class GLMBernoulliEvaluator(EvaluatorBase):
    def __init__(self, targets, weights):
        self.targets = targets
        self.weights = weights
        self.weights_mean = {f: w.mean(axis=0) for f, w in weights.items()}

    def predict_proba(self, data):
        data = self.create_vector(data)
        score = (
            np.inner(self.weights_mean["coef"], data) + self.weights_mean["intercept"]
        )
        return 1.0 / (1.0 + np.exp(-score))

    def predict_std(self, data):
        data = self.create_vector(data)
        score = (
            np.inner(self.weights["coef"], data)
            + self.weights["intercept"][:, np.newaxis]
        )
        proba = 1.0 / (1.0 + np.exp(-score))
        return proba.std(axis=-1)


class GLMBernoulli(pmb.PMModelBase):
    name = "glm_bernoulli"

    def create_evaluator(self, trace):
        weights = {
            "coef": trace.get_values("coef"),
            "intercept": trace.get_values("intercept"),
        }
        return GLMBernoulliEvaluator(self.targets, weights)

    def setup(self, data):
        data[self.targets] = data[self.targets].fillna(0)
        with self.model:
            weights = {
                "coef": pmb.pm.Normal("coef", 0, 5, shape=len(self.targets)),
                "intercept": pmb.pm.Normal("intercept", 0, 5),
            }
            score = (weights["coef"] * data[self.targets].values).sum(
                axis=-1
            ) + weights["intercept"]
            mu = pmb.pm.math.invlogit(score)
            weights["error"] = self.weighted_bernoili_logli(data, mu)
            self._weights = weights
            return weights
