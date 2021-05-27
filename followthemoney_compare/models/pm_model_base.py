try:
    import pymc3 as pm
    import arviz as ar
    import theano.tensor as T

    DEVMODE = True
except ImportError:
    DEVMODE = False

from followthemoney_compare.lib import TARGETS
from followthemoney_compare.models.model_base import TrainerBase


DEFAULT_FEATURES = TARGETS + ["pct_share_prop", "pct_miss_prop"]


class PMModelBase(TrainerBase):
    can_sample = True

    def __init__(self, features=DEFAULT_FEATURES):
        assert DEVMODE
        self.model = pm.Model()
        self.features = features
        super().__init__(features)

    def create_evaluator(self, trace):
        raise NotImplementedError

    def weighted_bernoili_logli(self, data, mu):
        with self.model:
            logp = data.weight.values * pm.Bernoulli.dist(p=mu).logp(data.y.values)
            error = pm.Potential("error", logp)
        return error

    def fit(self, data, n_samples=1_000, n_tune=1_000, n_chains=4):
        if not self.can_sample:
            return self
        self.setup(data)
        with self.model:
            self._trace = pm.sample(
                n_samples,
                tune=n_tune,
                progressbar=True,
                return_inferencedata=False,
                chains=n_chains,
                cores=n_chains,
            )
            evaluator = self.create_evaluator(self._trace)
            self.set_evaluator(evaluator)
        return self

    def summarize(self):
        if not self.can_sample:
            return
        assert hasattr(self, "_trace")
        with self.model:
            return ar.summary(self._trace)

    def traceplot(self, ax=None):
        if not self.can_sample:
            return
        with self.model:
            return ar.plot_trace(self._trace, axes=ax)
