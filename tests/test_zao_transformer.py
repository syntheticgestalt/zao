"""ZAOTransformer, which needs the pretrained weights; skipped without them."""

import os
from pathlib import Path

import numpy as np
import pytest

from zao import ZAOTransformer


def _weights_at_hand() -> bool:
    """Whether $ZAO_WEIGHTS names a local model.safetensors, or a directory holding it.

    Without one these tests are skipped, so that `pytest` never downloads the
    weights behind the user's back.
    """
    local = os.environ.get("ZAO_WEIGHTS")
    if not local:
        return False
    path = Path(local)
    return (path.is_file() and path.suffix == ".safetensors") or (
        path / "model.safetensors"
    ).is_file()


pytestmark = pytest.mark.skipif(
    not _weights_at_hand(), reason="set ZAO_WEIGHTS to a local copy of model.safetensors"
)


@pytest.fixture(scope="session")
def weights():
    """Where to load the weights from: $ZAO_WEIGHTS."""
    return os.environ["ZAO_WEIGHTS"]


SMILES = [
    "CC(=O)Oc1ccccc1C(=O)O",
    "Cn1c(CN2CCN(c3ccc(Cl)cc3)CC2)nc2ccccc21",
    "c1ccc2[nH]ccc2c1",
    "COc1ccc(CCN)cc1OC",
    "O=C(O)c1ccccc1O",
    "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
]


@pytest.fixture(scope="module")
def zao(weights):
    return ZAOTransformer(weights=weights).fit()


def test_transform_shape_and_failures(zao):
    X = zao.transform(SMILES + ["not_a_smiles", "CO"])
    assert X.shape == (8, 2048)
    assert np.isfinite(X[:6]).all()
    assert np.isnan(X[6:]).all()


def test_duplicates_and_salts_share_one_embedding(zao):
    X = zao.transform([SMILES[0], SMILES[4], SMILES[0], SMILES[4] + ".[Na+]"])
    assert np.array_equal(X[0], X[2])
    assert np.array_equal(X[1], X[3])


def test_stream_matches_input_order(zao):
    out = list(zao.stream(s for s in SMILES + ["not_a_smiles"]))
    assert [s for s, _ in out] == SMILES + ["not_a_smiles"]
    assert all(e is not None and e.shape == (2048,) for _, e in out[:-1])
    assert out[-1][1] is None


def test_refit_drops_a_model_loaded_under_earlier_settings(weights):
    """set_params + fit must not keep serving the previous weights or device."""
    zao = ZAOTransformer(weights=weights, device="cpu")
    zao.fit()
    zao._ensure_model()
    assert zao._model is not None
    zao.set_params(device="cpu").fit()
    assert zao._model is None and zao._device is None


def test_relative_weights_path_survives_chdir_and_pickling(weights, tmp_path, monkeypatch):
    """fit() pins the file it found; a later chdir or unpickle must not lose it."""
    import pickle

    src = Path(weights)
    monkeypatch.chdir(src.parent)
    zao = ZAOTransformer(weights=src.name, device="cpu").fit()
    monkeypatch.chdir(tmp_path)
    restored = pickle.loads(pickle.dumps(zao))
    restored._ensure_model()
    assert restored._model is not None


def test_num_conformers_sets_the_size_of_the_set(weights, monkeypatch):
    """Four conformers reach the model."""
    four = ZAOTransformer(weights=weights, num_conformers=4).fit()
    forward = four._forward
    sizes = []

    def recording(samples):
        sizes.extend(s["coors"].shape[0] for s in samples)
        return forward(samples)

    monkeypatch.setattr(four, "_forward", recording)
    X4 = four.transform(SMILES[:2])
    assert np.isfinite(X4).all()
    assert sizes == [4, 4]
    with pytest.raises(ValueError, match="num_conformers"):
        ZAOTransformer(weights=weights, num_conformers=0).fit()


def test_a_molecule_short_of_conformers_keeps_the_ones_it_has(weights, monkeypatch):
    """Embedded from its own conformers, beside molecules that have the full set."""
    import zao.transformers.zao as module
    from zao.preprocess import mol_to_features

    generate = module.generate_conformers
    short = []

    def three_for_the_first(mols, *args, **kwargs):
        out = generate(mols, *args, **kwargs)
        for conf_id in [c.GetId() for c in out[0].GetConformers()][3:]:
            out[0].RemoveConformer(conf_id)
        short.append(out[0])
        return out

    monkeypatch.setattr(module, "generate_conformers", three_for_the_first)
    zao = ZAOTransformer(weights=weights).fit()
    X = zao.transform(SMILES[:3])
    assert np.isfinite(X).all()
    assert short[0].GetNumConformers() == 3
    alone = zao._forward([mol_to_features(short[0], zao.max_atoms_)])
    assert np.allclose(X[0], alone[0], atol=1e-2)
