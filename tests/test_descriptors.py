"""The classical descriptor transformers. None of these need the weights."""

import numpy as np
import pytest

from zao import (
    AtomPairTransformer,
    AvalonTransformer,
    ErGTransformer,
    MACCSTransformer,
    MordredTransformer,
    MorganTransformer,
    RDKitDescriptorTransformer,
    TopologicalTorsionTransformer,
)

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
INDOLE = "c1ccc2[nH]ccc2c1"
SMILES = [ASPIRIN, INDOLE]

ALL = [
    (MorganTransformer, 1024),
    (AtomPairTransformer, 2048),
    (TopologicalTorsionTransformer, 1024),
    (AvalonTransformer, 1024),
    (MACCSTransformer, 167),
    (ErGTransformer, 315),
    # RDKit's descriptor list grows with the RDKit version; only its agreement
    # with the names this package reports is ours to keep.
    (RDKitDescriptorTransformer, None),
    (MordredTransformer, 1613),
]


@pytest.mark.parametrize("cls,width", ALL)
def test_shape_and_names(cls, width):
    f = cls()
    X = f.fit_transform(SMILES)
    assert X.shape[0] == len(SMILES)
    assert X.dtype == np.float32
    assert X.shape[1] == len(f.get_feature_names_out())
    if width is not None:
        assert X.shape[1] == width


@pytest.mark.parametrize("cls,_", ALL)
def test_unparsable_smiles_gives_a_nan_row(cls, _):
    X = cls().fit_transform([ASPIRIN, "not_a_smiles"])
    assert np.isnan(X[1]).all()
    assert not np.isnan(X[0]).all()


@pytest.mark.parametrize("cls,_", ALL)
def test_no_infinities_survive(cls, _):
    """An inf would break any scaler downstream, so descriptors that overflow become NaN."""
    X = cls().fit_transform(["C1=CC2=CC=C3C=CC4=CC=C5C=CC1=C1C2=C3C4=C51", ASPIRIN])
    assert not np.isinf(X).any()


def test_standardization_removes_the_counter_ion():
    """The default agrees with the molecule ZAO embeds: a salt matches its free base."""
    f = MorganTransformer()
    X = f.fit_transform(["CCCCCCCCN.Cl", "CCCCCCCCN"])
    assert np.array_equal(X[0], X[1])
    raw = MorganTransformer(standardize=False).fit_transform(["CCCCCCCCN.Cl", "CCCCCCCCN"])
    assert not np.array_equal(raw[0], raw[1])


def test_parallel_matches_in_process():
    """The worker processes change who computes the numbers, and nothing else."""
    smiles = [ASPIRIN, INDOLE] * 150
    assert np.array_equal(
        MorganTransformer(n_jobs=1).fit_transform(smiles),
        MorganTransformer(n_jobs=2).fit_transform(smiles),
    )


def test_empty_input():
    assert MorganTransformer().fit_transform([]).shape == (0, 1024)


def test_atom_map_numbers_change_nothing():
    """They label which atom came from where in a reaction and say nothing about the molecule."""
    plain, mapped = "c1ccccc1CCCCCC", "[cH:1]1[cH:2][cH:3][cH:4][cH:5][c:6]1CCCCCC"
    for cls, _ in ALL:
        X = cls().fit_transform([plain, mapped])
        # equal_nan: a descriptor undefined for this molecule is NaN in both rows,
        # and NaN does not equal itself.
        assert np.array_equal(X[0], X[1], equal_nan=True), cls.__name__
