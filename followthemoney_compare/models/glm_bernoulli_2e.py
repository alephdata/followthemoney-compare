import numpy as np

from . import pm_model_base as pmb
from .model_base import EvaluatorBase


DEFAULT_FEATURES_2E = (
    ("pct_share_prop", "pct_share_prop"),
    ("name", "pct_share_prop"),
    ("name", "name"),
    ("pct_share_prop", "pct_miss_prop"),
    ("pct_miss_prop", "pct_miss_prop"),
    ("name", "identifier"),
    ("country", "pct_share_prop"),
    ("identifier", "identifier"),
    ("identifier", "pct_miss_prop"),
    ("date", "date"),
    ("address", "address"),
)


def separate_2e_features(features_1e, features_2e):
    left, right = [], []
    for feat_left, feat_right in features_2e:
        left.append(features_1e.index(feat_left))
        right.append(features_1e.index(feat_right))
    return left, right


class GLMBernoulli2EEvaluator(EvaluatorBase):
    def __init__(self, features, features_2e, weights):
        self.features = features
        self.features_2e = features_2e
        self._features_2e_left, self._features_2e_right = separate_2e_features(
            features, features_2e
        )
        W = np.vstack(
            [
                weights["coef"].T,
                weights["coef_2e"].T,
                weights["intercept"].T,
            ]
        )
        self.weights_mean = W.mean(axis=1)
        self.weights_cov = np.cov(W, bias=True)

    def create_vector(self, X):
        vec1 = super().create_vector(X)
        vec2 = vec1[:, self._features_2e_left] * vec1[:, self._features_2e_right]
        return np.vstack([vec1.T, vec2.T, [1] * vec1.shape[0]]).T

    def predict_proba(self, df):
        data = self.create_vector(df)
        score = np.inner(self.weights_mean, data)
        return 1.0 / (1.0 + np.exp(-score))

    def predict_std(self, df):
        return self.predict_proba_std(df)[1]

    def predict_proba_std(self, df):
        data = self.create_vector(df)
        var_W = self._var(data)
        E_W = np.inner(self.weights_mean, data)
        f_E_W = 1.0 / (1.0 + np.exp(-E_W))
        fp_E_W = f_E_W * (1 - f_E_W)
        # Use a one term taylor expansion of var[f(W)] to approximate the
        # variance of logit(sumexp)
        var = var_W * (fp_E_W) ** 2
        return f_E_W, np.sqrt(var)

    def _var(self, X):
        if X.shape[0] > 125:
            return np.einsum("ij,jk,ik->i", X, self.weights_cov, X)
        return (X @ self.weights_cov @ X.T).diagonal()


class GLMBernoulli2E(pmb.PMModelBase):
    name = "glm_bernoulli_2e"

    def __init__(self, *args, features_2e=DEFAULT_FEATURES_2E, **kwargs):
        super().__init__(*args, **kwargs)
        self.features_2e = features_2e
        self._features_2e_left, self._features_2e_right = map(
            list, zip(*self.features_2e)
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
