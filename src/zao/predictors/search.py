"""Building the two tuned learners the ready-made models vote between."""

import warnings
from contextlib import contextmanager

import numpy as np
import optuna
from optuna.exceptions import ExperimentalWarning
from optuna_integration.sklearn import OptunaSearchCV
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _drop_constant() -> VarianceThreshold:
    """Drop the columns that never vary across the data set.

    Runs before the imputation, so that a column of all NaN goes here rather
    than being filled in with a constant.
    """
    return VarianceThreshold(0.0)


def _kernel_pipeline(estimator, selector) -> Pipeline:
    """Blocks, drop constants, impute, scale, fit."""
    return Pipeline(
        [
            ("blocks", selector),
            ("drop_constant", _drop_constant()),
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", estimator),
        ]
    )


def _tree_pipeline(estimator, selector) -> Pipeline:
    """Blocks, drop constants, fit. Trees need neither imputation nor scaling."""
    return Pipeline(
        [("blocks", selector), ("drop_constant", _drop_constant()), ("model", estimator)]
    )


def _search_spaces(classifier: bool):
    """What each learner's settings are searched over, keyed without the step prefix."""
    from optuna.distributions import FloatDistribution, IntDistribution

    kernel = {
        "C": FloatDistribution(1e-2, 1e3, log=True),
        "gamma": FloatDistribution(1e-5, 1e0, log=True),
    }
    if not classifier:
        kernel["epsilon"] = FloatDistribution(1e-3, 1e0, log=True)
    trees = {
        "n_estimators": IntDistribution(100, 1000),
        "learning_rate": FloatDistribution(1e-2, 3e-1, log=True),
        "num_leaves": IntDistribution(7, 127, log=True),
        "min_child_samples": IntDistribution(2, 50),
        "colsample_bytree": FloatDistribution(0.2, 1.0),
        "subsample": FloatDistribution(0.5, 1.0),
        "reg_lambda": FloatDistribution(1e-3, 1e2, log=True),
    }
    return kernel, trees


def _lightgbm(classifier: bool, random_state: int | None, threads: int):
    """Gradient-boosted trees."""
    from lightgbm import LGBMClassifier, LGBMRegressor

    cls = LGBMClassifier if classifier else LGBMRegressor
    return cls(random_state=random_state, n_jobs=threads, verbose=-1, subsample_freq=1)


def _calibration_folds(y, folds) -> int:
    """Folds for calibrating the support vector machine.

    Calibration happens inside each outer fold and needs an example of every
    class in every one of its folds, so the bound is the rarest class in the
    smallest training subset.
    """
    y = np.asarray(y)
    smallest = min(int(np.unique(y[train], return_counts=True)[1].min()) for train, _ in folds)
    if smallest < 2:
        classes = np.unique(y)
        raise ValueError(
            f"a training fold holds {smallest} example(s) of one of the {len(classes)} classes, "
            "which is too few to calibrate the support vector machine on. Use fewer folds, or "
            "more molecules of the rarer class"
        )
    return int(min(3, smallest))


def _within_fold(cv: int, budget: int) -> int:
    """How much concurrency each fold gets, once the folds themselves run side by side."""
    return max(1, budget // max(1, cv))


def _require_a_completed_trial(vote) -> None:
    """Refuse a fit in which a search never scored anything.

    A study whose every trial scored NaN has no best trial, and the refit path
    then fits the pipeline with the parameters it started with, which looks
    fitted and tuned and is neither.
    """
    for name, learner in zip([n for n, _ in vote.estimators], vote.estimators_, strict=True):
        try:
            _ = learner.best_params_
        except ValueError as e:
            raise ValueError(
                f"the {name} search scored no setting at all, so the model would be left at "
                "whatever it started with. Every trial failed or scored undefined, usually "
                "because the folds are too small for the scoring to mean anything: with cv "
                "folds a regression fold of one molecule has no R2. Use fewer folds, more "
                "molecules, or a scoring that is defined on your folds"
            ) from e


class _ClassifierSearch(ClassifierMixin, OptunaSearchCV):
    """``OptunaSearchCV``, told that it predicts labels.

    A voting ensemble refuses any member that does not say which it does. Written
    out rather than made on demand so that a fitted model can be pickled.
    """


class _RegressorSearch(RegressorMixin, OptunaSearchCV):
    """``OptunaSearchCV``, told that it predicts numbers. See ``_ClassifierSearch``."""


@contextmanager
def _quiet_search():
    """Keep the search's own commentary out of the caller's output, for this call only."""
    verbosity = optuna.logging.get_verbosity()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ExperimentalWarning)
            warnings.filterwarnings(
                "ignore", "`optuna.terminator` module has been deprecated", FutureWarning
            )
            # An all-NaN descriptor column is dropped by VarianceThreshold; silence only
            # that module's complaint about it, so the same warning from the caller's code
            # still reaches them.
            for complaint in ("All-NaN slice encountered", "Degrees of freedom <= 0"):
                warnings.filterwarnings(
                    "ignore",
                    complaint,
                    RuntimeWarning,
                    module=r"sklearn\.feature_selection\._variance_threshold",
                )
            yield
    finally:
        optuna.logging.set_verbosity(verbosity)
