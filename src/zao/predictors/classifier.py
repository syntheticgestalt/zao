"""ZAOClassifier: predict a yes/no outcome from SMILES."""

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score

from .estimator import ZAOEstimator


class ZAOClassifier(ClassifierMixin, ZAOEstimator):
    """Predict a yes/no outcome from SMILES.

    For properties recorded as a label: active or not, toxic or not, a blood
    brain barrier crossed or not. Everything said about ``ZAORegressor`` applies
    here.

    On imbalanced data, set ``scoring`` to something that reflects that, such as
    "roc_auc" or "average_precision"; the default, accuracy, is satisfied by
    predicting the majority class.
    """

    _classifier = True

    @property
    def classes_(self):
        return self.model_.classes_

    def predict(self, X):
        """Predict a label for SMILES.

        A molecule nothing could be computed for gets NaN when the labels are
        numbers and None otherwise, so numeric labels always come back as floats.
        """
        Xf, usable = self._features_for(X)
        if self.classes_.dtype.kind in "biuf":
            out = np.full(len(Xf), np.nan)
        else:
            out = np.full(len(Xf), None, dtype=object)
        if usable.any():
            out[usable] = self.model_.predict(Xf[usable])
        return out

    def predict_proba(self, X):
        """Probability of each class, averaged over the two learners; NaN where nothing
        could be computed.
        """
        Xf, usable = self._features_for(X)
        out = np.full((len(Xf), len(self.classes_)), np.nan)
        if usable.any():
            out[usable] = self.model_.predict_proba(Xf[usable])
        return out

    def score(self, X, y, sample_weight=None):
        """Accuracy over the molecules a prediction could be made for."""
        return self._score_predicted(accuracy_score, X, y, sample_weight)
