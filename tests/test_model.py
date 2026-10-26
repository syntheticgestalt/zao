"""CPU checks of feature extraction and of the model wiring, with a small random model."""

import numpy as np
import pytest
import torch
from rdkit import Chem
from rdkit.Geometry import Point3D

from zao.model import ZAOModel
from zao.preprocess import collate, mol_to_features, prepare_mol, standardize_smiles
from zao.preprocess.features import ATOM_FEAT_DIM

SMALL_CONFIG = {
    "general": {
        "conformer_embed_dim": 32,
        "mol_embed_dim": 48,
        "max_atoms": 40,
        "num_conformers": 4,
        "dropout": 0.0,
    },
    "encoder": {
        "input_dim": ATOM_FEAT_DIM,
        "hidden_dim": 16,
        "depth": 1,
        "num_nearest_neighbors": 4,
        "norm_coors": True,
        "coor_weights_clamp_value": 2.0,
        "num_queries": 4,
        "substructure_embed_dim": 16,
        "attention_temperature_init": 1.0,
        "edge_dim": 5,
        "mol_graph_num_layers": 1,
        "conformer_pooling": {
            "layers": 1,
            "nhead": 2,
            "dim_feedforward": 32,
            "num_rbf": 4,
            "rbf_max_distance": 20.0,
        },
    },
    "ensemble": {
        "nhead": 2,
        "dim_feedforward_ratio": 2,
        "num_layers": 1,
        "use_boltzmann_aggregation": True,
        "boltzmann_temperature": 1.0,
    },
}


def _mol_with_conformers(smiles, n_confs, seed=0):
    """Explicit-H Mol with random (not physically generated) coordinates and energies."""
    mol = prepare_mol(standardize_smiles(smiles))
    rng = np.random.default_rng(seed)
    for k in range(n_confs):
        conf = Chem.Conformer(mol.GetNumAtoms())
        for i, xyz in enumerate(rng.normal(scale=2.0, size=(mol.GetNumAtoms(), 3))):
            conf.SetAtomPosition(i, Point3D(*map(float, xyz)))
        conf.SetProp("Energy", str(float(10 + k)))
        mol.AddConformer(conf, assignId=True)
    return mol


def test_features_shapes_and_padding():
    mol = _mol_with_conformers("CC(=O)Oc1ccccc1C(=O)O", 3)
    s = mol_to_features(mol, max_atoms=40)
    n = mol.GetNumAtoms()
    assert s["feats"].shape == (40, ATOM_FEAT_DIM)
    assert s["coors"].shape == (3, 40, 3)
    assert s["mask"].sum() == n
    assert np.all(s["feats"][n:] == 0)
    assert (s["bond_types"] == s["bond_types"].T).all()
    assert s["bond_types"].max() == 4  # aromatic ring present

    assert mol_to_features(mol, max_atoms=n - 1) is None


def test_energy_window_rejects():
    mol = _mol_with_conformers("CC(=O)Oc1ccccc1C(=O)O", 2)
    mol.GetConformer(1).SetProp("Energy", "5000")
    assert mol_to_features(mol, max_atoms=40) is None


def test_model_forward_and_conformer_order_invariance():
    torch.manual_seed(0)
    model = ZAOModel(SMALL_CONFIG).eval()
    mols = [
        _mol_with_conformers(s, 4, seed=i)
        for i, s in enumerate(["CC(=O)Oc1ccccc1C(=O)O", "c1ccc2[nH]ccc2c1"])
    ]
    samples = [mol_to_features(m, 40) for m in mols]
    batch = collate(samples)
    k = batch.pop("num_conformers")
    with torch.no_grad():
        emb = model(
            batch["feats"], batch["coors"], batch["mask"], batch["bond_types"], batch["energy"], k
        )
    assert emb.shape == (2, 48)
    assert torch.isfinite(emb).all()

    # No positional encoding over conformers: reordering them leaves the embedding unchanged.
    perm = [2, 0, 3, 1]
    shuffled = [dict(s, coors=s["coors"][perm], energy=s["energy"][perm]) for s in samples]
    b2 = collate(shuffled)
    b2.pop("num_conformers")
    with torch.no_grad():
        emb2 = model(b2["feats"], b2["coors"], b2["mask"], b2["bond_types"], b2["energy"], k)
    assert torch.allclose(emb, emb2, atol=1e-5)


def test_load_ignores_pretraining_heads_but_requires_embedding_keys():
    model = ZAOModel(SMALL_CONFIG)
    state = dict(model.state_dict())
    state["multi_task_predictor.weight"] = torch.zeros(1)
    model.load_pretrained_state_dict(state)
    del state[next(k for k in state if k.startswith("encoder."))]
    with pytest.raises(RuntimeError):
        model.load_pretrained_state_dict(state)


def test_default_config_is_the_released_architecture():
    """ZAOModel() builds the released network, so no config file ships with the weights."""
    model = ZAOModel()
    assert sum(p.numel() for p in model.parameters()) == 222_692_486


def test_bfloat16_only_on_gpus_that_run_it_natively(monkeypatch):
    """Capability 8.0+ only, asked of the model's own GPU, never of the current one."""
    from zao.transformers.zao import _native_bf16

    capability = {0: (7, 5), 1: (8, 0)}  # GPU 0 a T4, GPU 1 an A100
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda d: capability[d.index])
    assert not _native_bf16(torch.device("cuda:0"))
    assert _native_bf16(torch.device("cuda:1"))
    assert not _native_bf16(torch.device("cpu"))
