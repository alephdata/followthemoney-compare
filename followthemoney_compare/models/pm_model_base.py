try:
    import pymc3 as pm
    import numpy as np
    import arviz as ar
    import seaborn as sns
    from sklearn import metrics

    DEVMODE = True
except ImportError:
    DEVMODE = False

from followthemoney_compare.lib.utils import TARGETS, HAS_TARGETS
from followthemoney_compare.models.model_base import TrainerBase


class PMModelBase(TrainerBase):
    can_sample = True

    def __init__(self, data, targets=TARGETS, has_targets=HAS_TARGETS):
        assert DEVMODE
        self.model = pm.Model()
        self.targets = targets
        self.has_targets = has_targets
        self.weights = self.setup(data)
        super().__init__(targets)

    def create_evaluator(self, trace):
        raise NotImplementedError

    def weighted_bernoili_logli(self, data, mu):
        with self.model:
            logp = data.weight.values * pm.Bernoulli.dist(p=mu).logp(data.y.values)
            error = pm.Potential("error", logp)
        return error

    def fit(self, n_samples=2_500, n_tune=1_000):
        if not self.can_sample:
            return
        with self.model:
            self._trace = pm.sample(
                n_samples, tune=n_tune, progressbar=True, return_inferencedata=False
            )
            evaluator = self.create_evaluator(self._trace)
            self.set_evaluator(evaluator)

    def summarize(self, ax=None):
        if not self.can_sample:
            return
        assert hasattr(self, "_trace")
        with self.model:
            print(ar.summary(self._trace))
        return self.traceplot(ax)

    def traceplot(self, ax=None):
        if not self.can_sample:
            return
        with self.model:
            return ar.plot_trace(self._trace, axes=ax)

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
