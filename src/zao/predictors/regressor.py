"""ZAORegressor: predict a measured quantity from SMILES."""

from sklearn.base import RegressorMixin
from sklearn.metrics import r2_score

from .estimator import ZAOEstimator


class ZAORegressor(RegressorMixin, ZAOEstimator):
    """Predict a measured quantity from SMILES.

    For properties measured on a continuous scale: solubility, permeability,
    clearance, binding affinity as pIC50 or pKi, and so on. For yes/no outcomes
    use ``ZAOClassifier``.

    The ZAO embedding concatenated with a set of classical 2D descriptors. If you
    have features of your own, add them to ``features``.

    Examples
    --------
    >>> from zao import ZAORegressor
    >>> model = ZAORegressor(n_trials=50).fit(smiles, pIC50)   # doctest: +SKIP
    >>> model.predict(["CC(=O)Oc1ccccc1C(=O)O"])               # doctest: +SKIP

    Scored on scaffolds it has never seen:

    >>> from sklearn.model_selection import GroupKFold, cross_val_score
    >>> from zao import scaffold_groups
    >>> cross_val_score(ZAORegressor(), smiles, y,                    # doctest: +SKIP
    ...                 cv=GroupKFold(5), groups=scaffold_groups(smiles))

    Notes
    -----
    A molecule ZAO cannot embed keeps its classical descriptors: only the ZAO
    columns go missing and the imputation step fills them. A molecule RDKit
    cannot parse at all has no descriptors either: it is left out of the fit,
    and ``predict`` gives it NaN, each with a warning.
    """

    def score(self, X, y, sample_weight=None):
        """R² over the molecules a prediction could be made for."""
        return self._score_predicted(r2_score, X, y, sample_weight)
