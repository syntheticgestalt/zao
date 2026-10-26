"""The ready-made models, built from two descriptor blocks so they run without the weights.

Leaving out the embedding leaves everything else the default builds: the block
search, the scaffold folds, the two learners and the vote.
"""

import numpy as np
import pytest
from sklearn.pipeline import FeatureUnion

from zao import (
    ErGTransformer,
    MACCSTransformer,
    ZAOClassifier,
    ZAORegressor,
    scaffold_groups,
)

# Distinct scaffolds, so that scaffold folds are possible, and a range of sizes
# so that a model has something to fit.
SMILES = [
    "CC(=O)Oc1ccccc1C(=O)O",
    "c1ccc2[nH]ccc2c1",
    "CCCCCCCCN",
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    "CC(=O)Nc1ccc(O)cc1",
    "OC(=O)c1ccccc1O",
    "c1ccc(cc1)S(=O)(=O)N",
    "CCOC(=O)c1ccccc1N",
    "CN1CCC[C@H]1c1cccnc1",
    "COc1ccc2cc(ccc2c1)C(C)C(=O)O",
    "Clc1ccc(cc1)C(c1ccccc1)N1CCN(C)CC1",
    "CCN(CC)CCNC(=O)c1ccc(N)cc1",
    "CC(C)NCC(O)COc1ccccc1",
    "OCC1OC(O)C(O)C(O)C1O",
]
Y = np.arange(len(SMILES), dtype=float)
LABELS = np.array([0, 1] * 7 + [0])

# The cost of fitting follows the number of columns, and the default's 10,503
# would make every test here a minute long. Two small blocks exercise the same
# wiring: the block search, the folds, the vote and the predictions.
TWO_BLOCKS = [("maccs", MACCSTransformer()), ("erg", ErGTransformer())]
FAST = {"n_trials": 5, "cv": 3, "random_state": 0}


def _fast(cls, **kw):
    return cls(features=FeatureUnion(TWO_BLOCKS), **FAST, **kw)


def test_the_search_picks_which_blocks_to_use():
    """The first block is kept whatever happens; the rest are searched."""
    model = _fast(ZAORegressor).fit(SMILES, Y)
    for kept in model.blocks_.values():
        assert "maccs" in kept
        assert set(kept) <= {"maccs", "erg"}
    assert [k for k in model.model_.estimators_[0].best_params_ if k.startswith("blocks__")] == [
        "blocks__erg"
    ]


def test_block_search_can_be_turned_off():
    model = _fast(ZAORegressor, search_blocks=False).fit(SMILES, Y)
    assert all(kept == ["maccs", "erg"] for kept in model.blocks_.values())
    assert not [k for k in model.model_.estimators_[0].best_params_ if k.startswith("blocks__")]


def test_the_folds_are_scaffold_folds():
    """A molecule's scaffold mates must not be split across the fold boundary."""
    groups = scaffold_groups(SMILES)
    folds = _fast(ZAORegressor)._folds(Y, groups)
    assert len(folds) == 3
    for train, test in folds:
        assert not (set(groups[train]) & set(groups[test]))


def test_too_few_scaffolds_is_refused():
    """One scaffold throughout: falling back to ordinary folds is the thing these folds prevent."""
    one_scaffold = ["c1ccccc1" + tail for tail in ("C", "CC", "CCC", "N", "O", "Cl")]
    with pytest.raises(ValueError, match="distinct groups"):
        _fast(ZAORegressor).fit(one_scaffold, np.arange(6, dtype=float))


def test_your_own_splitter_is_accepted_as_cv():
    """cv is scikit-learn's parameter, so a splitter belongs there as much as a number."""
    from sklearn.model_selection import GroupKFold

    model = ZAORegressor(
        features=FeatureUnion(TWO_BLOCKS),
        cv=GroupKFold(3),
        n_trials=5,
        random_state=0,
    ).fit(SMILES, Y)
    assert np.isfinite(model.predict(SMILES)).all()


def test_classifier_fits_and_gives_probabilities():
    # Grouped by hand: which scaffolds these 15 molecules share is RDKit's
    # business and changes with its version, and the calibration needs both
    # classes in every training fold.
    groups = np.repeat(np.arange(5), 3)
    model = _fast(ZAOClassifier).fit(SMILES, LABELS, groups=groups)
    proba = model.predict_proba(SMILES)
    assert proba.shape == (len(SMILES), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    # Soft voting over a calibrated SVM has to give more than vote fractions.
    assert len(np.unique(proba[:, 1])) > 2
    assert set(model.predict(SMILES)) <= {0, 1}
    assert list(model.classes_) == [0, 1]

    with pytest.warns(UserWarning, match="no prediction"):
        label = model.predict([SMILES[0], "not_a_smiles"])
    assert label[0] in (0, 1) and np.isnan(label[1])
    with pytest.warns(UserWarning, match="no prediction"):
        assert model.score([*SMILES, "not_a_smiles"], [*LABELS, 0]) == model.score(SMILES, LABELS)
    with pytest.warns(UserWarning, match="no prediction"):
        proba = model.predict_proba([SMILES[0], "not_a_smiles"])
    assert np.isfinite(proba[0]).all() and np.isnan(proba[1]).all()


def test_a_half_empty_molecule_is_kept_and_an_empty_one_is_left_out():
    """A SMILES that does not parse has nothing to learn from and must not stop the fit.

    Losing only some columns, the ZAO block say, is not losing the molecule: the
    rest of its descriptors are real.
    """
    with pytest.warns(UserWarning, match="no features"):
        model = _fast(ZAORegressor).fit([*SMILES, "not_a_smiles"], [*Y, 0.0])
    with pytest.warns(UserWarning, match="no prediction"):
        pred = model.predict([SMILES[0], "not_a_smiles"])
    assert np.isfinite(pred[0]) and np.isnan(pred[1])
    with pytest.warns(UserWarning, match="no prediction"):
        assert model.score([*SMILES, "not_a_smiles"], [*Y, 0.0]) == model.score(SMILES, Y)
    with pytest.raises(TypeError, match="single string"):
        model.predict(SMILES[0])

    import zao.predictors.estimator as r

    half_empty = np.ones((len(SMILES), 4))
    half_empty[0, :2] = np.nan
    kept_smiles, kept_X, _, _ = r._drop_featureless(list(SMILES), half_empty, Y, None)
    assert len(kept_smiles) == len(SMILES)
    assert kept_X.shape == half_empty.shape


def test_molecules_are_featurized_once_for_the_whole_fit(monkeypatch):
    """Not per trial, not per fold, and the two learners share the one matrix."""
    import zao.transformers.descriptors as d

    calls = []
    original = d._Descriptor.transform
    monkeypatch.setattr(
        d._Descriptor, "transform", lambda self, X: calls.append(len(X)) or original(self, X)
    )
    _fast(ZAORegressor).fit(SMILES, Y)
    assert calls == [len(SMILES), len(SMILES)]  # one call per block, nothing more


def test_a_fitted_model_can_be_saved_and_loaded():
    """Saving a model is the normal way to use one, so nothing in it may be unpicklable."""
    import pickle

    model = _fast(ZAORegressor).fit(SMILES, Y)
    reloaded = pickle.loads(pickle.dumps(model))
    assert np.allclose(reloaded.predict(SMILES), model.predict(SMILES))


def test_the_search_keeps_its_commentary_to_itself(capsys, recwarn):
    """A library that prints a line per trial is unusable in a notebook or a script."""
    import optuna

    before = optuna.logging.get_verbosity()
    _fast(ZAORegressor).fit(SMILES, Y)
    assert capsys.readouterr().err == ""
    assert optuna.logging.get_verbosity() == before


def test_a_seed_makes_the_whole_fit_repeat_itself():
    """A random_state that does not reproduce is a trap, so the search gives up its
    trial-level parallelism to keep its promise."""
    first = _fast(ZAORegressor).fit(SMILES, Y)
    second = _fast(ZAORegressor).fit(SMILES, Y)
    assert np.allclose(first.predict(SMILES), second.predict(SMILES))


def test_a_second_fit_with_the_same_cache_works():
    """The cache has to carry the fitted transformers, not only the matrix they produced."""
    import tempfile

    with tempfile.TemporaryDirectory() as cache:
        first = _fast(ZAORegressor, cache_dir=cache).fit(SMILES, Y)
        second = _fast(ZAORegressor, cache_dir=cache).fit(SMILES, Y)
        assert np.allclose(first.predict(SMILES), second.predict(SMILES))


def test_mismatched_lengths_are_refused():
    """Dropping a featureless row must not be allowed to make a malformed input look consistent."""
    with pytest.raises(ValueError):
        _fast(ZAORegressor).fit([*SMILES, "not_a_smiles"], Y)  # one molecule too many


# Folds too small for the scoring to be defined on: what the search says about
# that is the caller's to read, so this test ignores it rather than the fit.
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.filterwarnings("ignore::sklearn.exceptions.UndefinedMetricWarning")
def test_a_search_that_scored_nothing_is_refused():
    """Optuna refits with its starting parameters when every trial is NaN, which looks fitted."""
    groups = np.arange(len(SMILES))  # one molecule per group, so a fold holds one
    with pytest.raises(ValueError, match="scored no setting"):
        ZAORegressor(features=FeatureUnion(TWO_BLOCKS), n_trials=5, cv=len(SMILES)).fit(
            SMILES, Y, groups=groups
        )


def test_calibration_folds_follow_the_smallest_training_fold():
    """Calibration happens inside an outer fold, so the whole set's counts are the wrong bound."""
    from zao.predictors.search import _calibration_folds

    y = np.array([0, 0, 0, 0, 1, 1])
    plenty = [(np.array([0, 1, 4, 5]), np.array([2, 3]))]
    assert _calibration_folds(y, plenty) == 2
    starved = [(np.array([0, 1, 2, 4]), np.array([3, 5]))]  # one positive in training
    with pytest.raises(ValueError, match="too few to calibrate"):
        _calibration_folds(y, starved)


def test_an_impossible_split_is_refused_before_the_conformers_are_generated(monkeypatch):
    """Finding this out after an hour of featurization is the expensive way to learn it."""
    import zao.predictors.estimator as b

    called = []
    original = b._featurize
    monkeypatch.setattr(b, "_featurize", lambda *a, **k: called.append(1) or original(*a, **k))
    with pytest.raises(ValueError, match="distinct groups"):
        _fast(ZAORegressor).fit(SMILES, Y, groups=np.zeros(len(SMILES)))
    assert not called


def test_a_descriptor_that_is_never_defined_does_not_warn_the_caller():
    """A column of all NaN is dropped, and numpy's complaint about it belongs to us.

    Only that one step is silenced: the same complaint from a scorer of the
    caller's own is theirs to see.
    """
    import warnings

    from sklearn.feature_selection import VarianceThreshold

    from zao.predictors.search import _quiet_search

    all_nan_column = np.array([[1.0, np.nan], [2.0, np.nan]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with _quiet_search():
            VarianceThreshold(0.0).fit(all_nan_column)

            with pytest.raises(RuntimeWarning, match="Degrees of freedom"):
                warnings.warn(
                    "Degrees of freedom <= 0 in my own scorer", RuntimeWarning, stacklevel=1
                )


def test_the_search_restores_the_warning_filters_it_found():
    """Whatever the search silences is taken down again when the fit returns."""
    import warnings

    from zao.predictors.search import _quiet_search

    before = list(warnings.filters)
    with _quiet_search():
        assert warnings.filters != before
    assert warnings.filters == before


def test_a_molecule_zao_cannot_embed_is_reported():
    """The imputation fills its ZAO columns, so without a warning nobody would know."""
    from zao import ZAOTransformer

    class NoEmbedding(ZAOTransformer):
        def fit(self, X=None, y=None):
            self.embedding_dim_ = 3
            return self

        def get_feature_names_out(self, input_features=None):
            return np.array(["zao0", "zao1", "zao2"], dtype=object)

        def transform(self, X):
            out = np.array([[len(s), s.count("C"), s.count("O")] for s in X], dtype=float)
            out[[s == SMILES[0] for s in X]] = np.nan
            return out

    features = FeatureUnion([("zao", NoEmbedding()), *TWO_BLOCKS])
    with pytest.warns(UserWarning, match="ZAO could not embed 1 of"):
        model = ZAORegressor(features=features, **FAST).fit(SMILES, Y)
    with pytest.warns(UserWarning, match="ZAO could not embed 1 of 2"):
        pred = model.predict(SMILES[:2])
    assert np.isfinite(pred).all()
