"""Conformer generation on the CPU, which needs neither a GPU nor the weights."""

import warnings

import pytest
from rdkit import Chem

from zao.preprocess import conformers, generate_conformers, prepare_mol, standardize_smiles
from zao.preprocess.conformers import gpu_available

SMILES = [
    "CC(=O)Oc1ccccc1C(=O)O",
    "c1ccc2[nH]ccc2c1",
    "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
    "COc1ccc(CCN)cc1OC",
]


@pytest.fixture(scope="module")
def mols():
    return [prepare_mol(standardize_smiles(s)) for s in SMILES]


def test_cpu_backend_produces_usable_conformers(mols):
    """Energies sorted, charges assigned, one Mol per input."""
    out = generate_conformers(mols, num_conformers=10, backend="cpu")
    assert len(out) == len(mols)
    for mol in out:
        assert mol is not None
        assert mol.GetNumConformers() == 10
        energies = [float(c.GetProp("Energy")) for c in mol.GetConformers()]
        assert energies == sorted(energies)  # lowest energy first
        assert mol.GetAtomWithIdx(0).HasProp("_MMFFPartialCharge")


def test_cpu_backend_is_deterministic(mols):
    a, b = (generate_conformers(mols, 10, backend="cpu") for _ in range(2))
    first = [float(c.GetProp("Energy")) for c in a[0].GetConformers()]
    second = [float(c.GetProp("Energy")) for c in b[0].GetConformers()]
    assert first == second


def test_cpu_backend_keeps_a_short_set(mols, monkeypatch):
    """Embedding fewer than asked for keeps those conformers, with an energy each."""
    from zao.preprocess import conformers

    embed = conformers.rdDistGeom.EmbedMultipleConfs

    def three(mol, numConfs, params):
        return embed(mol, numConfs=3, params=params)

    monkeypatch.setattr(conformers.rdDistGeom, "EmbedMultipleConfs", three)
    (mol,) = generate_conformers(mols[:1], 10, backend="cpu")
    assert mol is not None and mol.GetNumConformers() == 3
    energies = [float(c.GetProp("Energy")) for c in mol.GetConformers()]
    assert energies == sorted(energies)


def test_backend_auto_and_bad_backend(mols):
    assert generate_conformers([], backend="cpu") == []
    with pytest.raises(ValueError, match="backend must be"):
        generate_conformers(mols, backend="tpu")
    if not gpu_available():
        with pytest.raises(RuntimeError, match="nvMolKit"):
            generate_conformers(mols, backend="gpu")


def test_a_molecule_it_cannot_handle_gives_none():
    # Elemental iron has no MMFF94 parameters.
    out = generate_conformers([Chem.AddHs(Chem.MolFromSmiles("[Fe]"))], 10, backend="cpu")
    assert out == [None]


def test_falling_back_to_the_cpu_is_announced_once(monkeypatch):
    """pip can install the GPU extra as nothing, so the fallback has to say so itself."""
    monkeypatch.setattr(conformers, "gpu_available", lambda: False)
    monkeypatch.setattr(conformers, "_CPU_BACKEND_ANNOUNCED", False)
    mols = [conformers.prepare_mol(Chem.MolFromSmiles("CCO"))]
    with pytest.warns(UserWarning, match="CPU"):
        conformers.generate_conformers(mols, 2)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the second call stays quiet
        conformers.generate_conformers(mols, 2)


def test_asking_for_the_cpu_is_not_second_guessed(monkeypatch):
    monkeypatch.setattr(conformers, "gpu_available", lambda: False)
    monkeypatch.setattr(conformers, "_CPU_BACKEND_ANNOUNCED", False)
    mols = [conformers.prepare_mol(Chem.MolFromSmiles("CCO"))]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        conformers.generate_conformers(mols, 2, backend="cpu")
