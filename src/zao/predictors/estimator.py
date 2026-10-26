"""What ZAORegressor and ZAOClassifier share: the feature set, the block search,
the scaffold folds and the vote."""

import warnings

import joblib
import numpy as np
from sklearn.base import BaseEstimator, MetaEstimatorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion
from sklearn.utils.validation import check_consistent_length, check_is_fitted

from .._input import as_smiles_list
from ..parallel import native_threads
from ..scaffold import scaffold_groups
from ..transformers import (
    AtomPairTransformer,
    AvalonTransformer,
    ErGTransformer,
    MACCSTransformer,
    MordredTransformer,
    MorganTransformer,
    RDKitDescriptorTransformer,
    TopologicalTorsionTransformer,
    ZAOTransformer,
)
from .search import (
    _calibration_folds,
    _ClassifierSearch,
    _kernel_pipeline,
    _lightgbm,
    _quiet_search,
    _RegressorSearch,
    _require_a_completed_trial,
    _search_spaces,
    _tree_pipeline,
    _within_fold,
)

_MIN_TRIALS = 5


def default_features(
    zao=True, mordred: bool = True, n_jobs: int = -1, **zao_params
) -> FeatureUnion:
    """The feature set the ready-made models use, as a ``FeatureUnion`` you can edit.

    The ZAO embedding plus nine classical descriptor families.

    Parameters
    ----------
    zao : bool or transformer, default=True
        Include the ZAO embedding. Pass a configured ``ZAOTransformer`` to
        control the number of conformers, the device, the conformer backend or
        the weights; False leaves it out, giving the classical-descriptor
        baseline.
    mordred : bool, default=True
        Include Mordred's descriptors, the slowest block.
    n_jobs : int, default=-1
        Worker processes for the classical descriptors; -1 uses every CPU this
        process may use. The ZAO transformer has its own parallelism and is not
        affected.
    **zao_params
        Passed to ``ZAOTransformer`` when ``zao`` is True.

    Returns
    -------
    FeatureUnion
        Edit it like any other: ``features.set_params(...)``, or build your own
        from the transformers in ``zao.transformers``.
    """
    blocks = []
    if zao is not False:
        if zao is True:
            zao = ZAOTransformer(**zao_params)
        elif zao_params:
            raise TypeError(
                "pass either a configured ZAOTransformer or keyword arguments, not both"
            )
        blocks.append(("zao", zao))
    blocks += [
        ("ecfp", MorganTransformer(n_jobs=n_jobs)),
        ("fcfp", MorganTransformer(features=True, n_jobs=n_jobs)),
        ("atom_pair", AtomPairTransformer(n_jobs=n_jobs)),
        ("torsion", TopologicalTorsionTransformer(n_jobs=n_jobs)),
        ("avalon", AvalonTransformer(n_jobs=n_jobs)),
        ("maccs", MACCSTransformer(n_jobs=n_jobs)),
        ("erg", ErGTransformer(n_jobs=n_jobs)),
        ("rdkit", RDKitDescriptorTransformer(n_jobs=n_jobs)),
    ]
    if mordred:
        blocks.append(("mordred", MordredTransformer(n_jobs=n_jobs)))
    return FeatureUnion(blocks)


def _drop_featureless(smiles: list, Xf: np.ndarray, y, groups):
    """Leave out the molecules nothing could be computed for, and say how many.

    A molecule that lost only some of its columns is kept.
    """
    usable = ~np.isnan(Xf).all(axis=1)
    if usable.all():
        return smiles, Xf, y, groups
    dropped = int((~usable).sum())
    warnings.warn(
        f"{dropped} of {len(smiles)} molecules gave no features at all and are left out of "
        "the fit; the first is " + repr(smiles[int(np.flatnonzero(~usable)[0])]),
        stacklevel=3,
    )
    keep = np.flatnonzero(usable)
    smiles = [smiles[i] for i in keep]
    y = np.asarray(y)[keep]
    if groups is not None:
        groups = np.asarray(groups)[keep]
    return smiles, Xf[keep], y, groups


def _block_spans(features: FeatureUnion) -> list:
    """Each block that contributes columns, as ``(name, transformer, slice)``."""
    spans, start = [], 0
    for name, transformer in features.transformer_list:
        if transformer == "drop":
            continue
        width = len(transformer.get_feature_names_out())
        spans.append((name, transformer, slice(start, start + width)))
        start += width
    return spans


def _warn_unembedded(features: FeatureUnion, Xf: np.ndarray, smiles: list, stacklevel: int):
    """Say how many molecules lost the ZAO block, which the imputation would otherwise hide.

    Molecules with no features at all are reported separately and not counted here.
    """
    for _, transformer, columns in _block_spans(features):
        if not isinstance(transformer, ZAOTransformer):
            continue
        missing = np.isnan(Xf[:, columns]).all(axis=1)
        missing &= ~np.isnan(Xf).all(axis=1)
        if missing.any():
            warnings.warn(
                f"ZAO could not embed {int(missing.sum())} of {len(smiles)} molecules; "
                "their ZAO columns are imputed. The first is "
                + repr(smiles[int(np.flatnonzero(missing)[0])]),
                stacklevel=stacklevel + 1,
            )
        return


def _scaffold_groups_for_fitting(smiles: list) -> np.ndarray:
    """Scaffolds for the folds; anything unreadable joins the group without a ring."""
    return np.asarray([_scaffold_or_empty(s) for s in smiles], dtype=object)


def _scaffold_or_empty(smiles: str) -> str:
    try:
        return scaffold_groups([smiles])[0]
    except ValueError:
        return ""


def _run_featurization(features: FeatureUnion, smiles: list):
    """The fitted transformers and the matrix they produced, as one result.

    Both are cached together on purpose. The matrix alone is not enough: the
    transformers have to be fitted for the model to featurize anything later, and
    a cache hit skips the call that would have fitted them.
    """
    return features, features.fit_transform(smiles)


def _featurize(features: FeatureUnion, smiles: list, cache_dir: str | None):
    """The whole feature matrix, computed once, optionally remembered on disk."""
    if cache_dir is None:
        return _run_featurization(features, smiles)
    return joblib.Memory(cache_dir, verbose=0).cache(_run_featurization)(features, smiles)


class ZAOEstimator(MetaEstimatorMixin, BaseEstimator):
    """What ZAORegressor and ZAOClassifier have in common.

    Not used directly: it leaves the learners and the kind of prediction to the
    two subclasses and does everything else. Subclass it to put a different set
    of learners behind the same feature set, block search and scaffold folds.

    Parameters
    ----------
    features : "default" or FeatureUnion, default="default"
        What turns SMILES into numbers. "default" calls ``default_features``.
        Pass a ``FeatureUnion`` of your own to add or remove blocks; each member
        is a block the search can switch off, except the first, which is always
        kept. Every member must produce dense output and report its width
        through ``get_feature_names_out``, and must only compute, since the
        blocks are fitted once on everything before the folds are scored.
    n_trials : int, default=100
        Settings tried per learner. Fewer than 5 is refused.
    search_blocks : bool, default=True
        Let the search decide which descriptor blocks each learner sees.
        False fixes every block on.
    timeout : float, optional
        Seconds after which a learner's search stops starting new trials; trials
        already running are allowed to finish. The search stops at whichever of
        ``n_trials`` and this comes first.
    cv : int or splitter, default=5
        How to score each candidate setting. An integer means that many scaffold
        folds: the molecules are grouped by scaffold and each group is kept to
        one side. With fewer distinct groups than folds the fit refuses. Pass a
        scikit-learn splitter to decide for yourself; it is given the same
        groups.
    scoring : str or callable, optional
        What the search maximizes. None uses the estimator's own score (R² for
        regression, accuracy for classification).
    random_state : int, optional
        Makes the fit reproducible, at the cost of the kernel learner trying one
        setting at a time. Conformer generation on the GPU is not bit-reproducible
        between runs, so the features can still move slightly.
    n_jobs : int, default=-1
        How much of the machine the whole fit may use: the descriptors, the folds
        of each search, and the threads inside LightGBM. -1 uses every CPU this
        process is allowed, 1 keeps everything in this process. The two learners
        are always fitted one after the other.
    cache_dir : str, optional
        Directory in which to remember the computed features between fits of the
        same molecules. Cross-validating over this model gains nothing from it:
        every outer fold is a different list of molecules and so a different
        cache entry.
    zao : bool or transformer, default=True
        Passed to ``default_features``; ignored when ``features`` is given.
    """

    _classifier = False

    def __init__(
        self,
        features="default",
        n_trials: int = 100,
        timeout: float | None = None,
        cv: int = 5,
        scoring=None,
        search_blocks: bool = True,
        random_state: int | None = None,
        n_jobs: int = -1,
        cache_dir: str | None = None,
        zao=True,
    ):
        self.features = features
        self.search_blocks = search_blocks
        self.n_trials = n_trials
        self.timeout = timeout
        self.cv = cv
        self.scoring = scoring
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.cache_dir = cache_dir
        self.zao = zao

    def _budget(self) -> int:
        """Threads this fit may keep busy, from ``n_jobs`` and what the machine allows."""
        allowed = native_threads()
        return allowed if self.n_jobs < 1 else min(self.n_jobs, allowed)

    def _feature_union(self) -> FeatureUnion:
        features = self.features
        if isinstance(features, str):
            if features != "default":
                raise ValueError(f"features must be 'default' or a FeatureUnion, got {features!r}")
            features = default_features(zao=self.zao, n_jobs=self.n_jobs)
        if not isinstance(features, FeatureUnion):
            raise TypeError(
                "features must be a FeatureUnion, so that each block can be named and switched "
                f"on and off by the search; got {type(features).__name__}. Wrap a single "
                "transformer as FeatureUnion([(name, transformer)])"
            )
        return clone(features)

    def _selector(self, features: FeatureUnion) -> ColumnTransformer:
        """The blocks as columns a learner can be given or denied.

        Each entry names a block and a span of columns; setting it to "drop"
        takes that block away, with nothing recomputed.
        """
        return ColumnTransformer(
            [(name, "passthrough", columns) for name, _, columns in _block_spans(features)]
        )

    def _block_space(self, selector: ColumnTransformer, step: str) -> dict:
        """One choice per block after the first, which is always kept. Empty when off."""
        from optuna.distributions import CategoricalDistribution

        if not self.search_blocks:
            return {}
        return {
            f"{step}__{name}": CategoricalDistribution(["passthrough", "drop"])
            for name, _, _ in selector.transformers[1:]
        }

    def _check_enough_groups(self, groups) -> None:
        if not isinstance(self.cv, int):
            return
        n_groups = len(set(groups))
        if n_groups < self.cv:
            raise ValueError(
                f"{n_groups} distinct groups for cv={self.cv}: there are not enough to hold "
                "one out at a time. Lower cv, pass your own groups to fit, or pass a splitter "
                "as cv if you accept that a molecule's analogues may fall on both sides"
            )

    def _folds(self, y, groups):
        """The folds, as a fixed list of index pairs, so every trial is scored alike."""
        if not isinstance(self.cv, int):
            return list(self.cv.split(np.zeros(len(y)), y, groups))
        self._check_enough_groups(groups)
        # Stratified for labels, so a rare class is not confined to one fold.
        splitter = (
            StratifiedGroupKFold(self.cv, shuffle=True, random_state=self.random_state)
            if self._classifier
            else GroupKFold(self.cv)
        )
        return list(splitter.split(np.zeros(len(y)), y, groups))

    def _learners(self, selector: ColumnTransformer, folds, y):
        kernel_space, tree_space = _search_spaces(self._classifier)
        if self._classifier:
            from sklearn.calibration import CalibratedClassifierCV
            from sklearn.svm import SVC

            # Soft voting needs probabilities, which calibration turns the
            # distance from the boundary into.
            svm = CalibratedClassifierCV(
                SVC(random_state=self.random_state),
                ensemble=False,
                cv=_calibration_folds(y, folds),
            )
            kernel = _kernel_pipeline(svm, clone(selector))
            kernel_space = {f"model__estimator__{k}": v for k, v in kernel_space.items()}
        else:
            from sklearn.svm import SVR

            kernel = _kernel_pipeline(SVR(), clone(selector))
            kernel_space = {f"model__{k}": v for k, v in kernel_space.items()}
        within = _within_fold(len(folds), self._budget())
        trees = _tree_pipeline(
            _lightgbm(self._classifier, self.random_state, within), clone(selector)
        )
        tree_space = {f"model__{k}": v for k, v in tree_space.items()}

        kernel_space |= self._block_space(selector, "blocks")
        tree_space |= self._block_space(selector, "blocks")

        tuned = _ClassifierSearch if self._classifier else _RegressorSearch

        def search(estimator, space, n_jobs):
            return tuned(
                estimator,
                space,
                cv=folds,
                scoring=self.scoring,
                n_trials=self.n_trials,
                timeout=self.timeout,
                random_state=self.random_state,
                n_jobs=n_jobs,
            )

        # The kernel is single-threaded, so its share goes into parallel trials;
        # LightGBM spends its share on threads. Parallel trials cannot repeat,
        # so a caller who asked for a seed gets them one at a time.
        kernel_jobs = 1 if self.random_state is not None else within
        return [
            ("kernel", search(kernel, kernel_space, kernel_jobs)),
            ("trees", search(trees, tree_space, 1)),
        ]

    def fit(self, X, y, groups=None):
        """Fit on SMILES and their measured values.

        Parameters
        ----------
        X : sequence of str
            SMILES, one per molecule.
        y : array-like
            The measured values.
        groups : array-like, optional
            What the search's folds are grouped by, one entry per molecule.
            The default groups the molecules by scaffold, so that the settings
            are chosen on scaffolds the fit has not seen. Pass your own to group
            by something else, such as the assay a measurement came from.
        """
        if self.n_trials < _MIN_TRIALS:
            raise ValueError(
                f"n_trials must be at least {_MIN_TRIALS}, got {self.n_trials}; "
                "a search this small does worse than fixed defaults"
            )
        smiles = as_smiles_list(X)
        if groups is None:
            check_consistent_length(smiles, y)
        else:
            check_consistent_length(smiles, y, groups)
            # Before the conformers, not after.
            self._check_enough_groups(groups)
        self.features_, Xf = _featurize(self._feature_union(), smiles, self.cache_dir)
        _warn_unembedded(self.features_, Xf, smiles, stacklevel=2)
        selector = self._selector(self.features_)
        smiles, Xf, y, groups = _drop_featureless(smiles, Xf, y, groups)
        if groups is None:
            groups = _scaffold_groups_for_fitting(smiles)
        folds = self._folds(y, groups)
        with _quiet_search(), joblib.parallel_config(n_jobs=min(len(folds), self._budget())):
            # The folds run side by side; n_jobs=1 keeps the two learners sequential.
            pairs = self._learners(selector, folds, y)
            self.model_ = (
                VotingClassifier(pairs, voting="soft", n_jobs=1)
                if self._classifier
                else VotingRegressor(pairs, n_jobs=1)
            )
            self.model_.fit(Xf, y)
        _require_a_completed_trial(self.model_)
        return self

    @property
    def blocks_(self) -> dict:
        """Which feature blocks each learner ended up using."""
        return {
            name: [
                block
                for block, kind, _ in learner.best_estimator_["blocks"].transformers
                if kind != "drop"
            ]
            for name, learner in zip(
                [n for n, _ in self.model_.estimators], self.model_.estimators_, strict=True
            )
        }

    def _features_for(self, X):
        """The feature matrix for SMILES, and which rows have any features at all."""
        check_is_fitted(self)
        smiles = as_smiles_list(X)
        Xf = self.features_.transform(smiles)
        usable = ~np.isnan(Xf).all(axis=1)
        if not usable.all():
            warnings.warn(
                f"{int((~usable).sum())} of {len(smiles)} molecules gave no features at all and "
                "get no prediction; the first is " + repr(smiles[int(np.flatnonzero(~usable)[0])]),
                stacklevel=3,
            )
        _warn_unembedded(self.features_, Xf, smiles, stacklevel=3)
        return Xf, usable

    def predict(self, X):
        """Predict for SMILES. A molecule nothing could be computed for gets NaN."""
        Xf, usable = self._features_for(X)
        out = np.full(len(Xf), np.nan)
        if usable.any():
            out[usable] = self.model_.predict(Xf[usable])
        return out

    def _score_predicted(self, metric, X, y, sample_weight):
        """``metric`` over the molecules a prediction could be made for."""
        pred = self.predict(X)
        known = np.array([p is not None and p == p for p in pred], dtype=bool)
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight)[known]
        return metric(np.asarray(y)[known], pred[known], sample_weight=sample_weight)

    def __sklearn_is_fitted__(self):
        return hasattr(self, "model_")

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        # The input is SMILES: a sequence of strings, not a numeric matrix.
        tags.input_tags.string = True
        tags.input_tags.one_d_array = True
        tags.input_tags.two_d_array = False
        tags.target_tags.required = True
        return tags
