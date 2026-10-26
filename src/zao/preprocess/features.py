"""Model inputs computed from a Mol that carries conformers."""

import numpy as np
from rdkit import Chem
from rdkit.Chem import BondType, HybridizationType

# Must match the pretraining featurization for the released weights to mean anything.
ATOM_FEAT_DIM = 7

_HYBRIDIZATION_CODE = {
    HybridizationType.SP: 1,
    HybridizationType.SP2: 2,
    HybridizationType.SP3: 3,
    HybridizationType.SP3D: 4,
    HybridizationType.SP3D2: 5,
}

# 0 means no bond; unknown bond types are treated as single.
_BOND_TYPE_CODE = {
    BondType.SINGLE: 1,
    BondType.DOUBLE: 2,
    BondType.TRIPLE: 3,
    BondType.AROMATIC: 4,
}

# Conformer sets whose MMFF94 energies leave this range (kcal/mol) are rejected.
ENERGY_MIN = -300.0
ENERGY_MAX = 3000.0


def mol_to_features(mol: Chem.Mol, max_atoms: int) -> dict[str, np.ndarray] | None:
    """Atom features, coordinates, energies and bond matrix, padded to ``max_atoms``.

    Returns None if the molecule has more than ``max_atoms`` atoms (hydrogens
    included), has no conformers, or has a conformer energy outside
    [ENERGY_MIN, ENERGY_MAX].
    """
    n = mol.GetNumAtoms()
    confs = mol.GetConformers()
    if n > max_atoms or len(confs) == 0:
        return None

    feats = np.zeros((max_atoms, ATOM_FEAT_DIM), dtype=np.float32)
    for i, atom in enumerate(mol.GetAtoms()):
        try:
            charge = float(atom.GetProp("_MMFFPartialCharge"))
        except KeyError:
            charge = 0.0
        try:
            num_hs = float(atom.GetTotalNumHs())
        except RuntimeError:
            num_hs = 0.0
        feats[i] = [
            atom.GetAtomicNum(),
            charge,
            atom.IsInRing(),
            atom.GetIsAromatic(),
            atom.GetDegree(),
            num_hs,
            _HYBRIDIZATION_CODE.get(atom.GetHybridization(), 0),
        ]

    coors = np.zeros((len(confs), max_atoms, 3), dtype=np.float32)
    energy = np.zeros((len(confs), 1), dtype=np.float32)
    for k, conf in enumerate(confs):
        coors[k, :n] = conf.GetPositions()
        energy[k, 0] = float(conf.GetProp("Energy")) if conf.HasProp("Energy") else 0.0
    if (energy > ENERGY_MAX).any() or (energy < ENERGY_MIN).any():
        return None

    bond_types = np.zeros((max_atoms, max_atoms), dtype=np.int64)
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_types[i, j] = bond_types[j, i] = _BOND_TYPE_CODE.get(bond.GetBondType(), 1)

    mask = np.zeros(max_atoms, dtype=bool)
    mask[:n] = True
    return {
        "feats": feats,
        "coors": coors,
        "energy": energy,
        "mask": mask,
        "bond_types": bond_types,
    }


def collate(samples: list[dict[str, np.ndarray]]):
    """Stack samples into flattened (B*K, ...) torch tensors, molecule-major."""
    import torch

    k = samples[0]["coors"].shape[0]
    b = len(samples)

    def per_conformer(key):
        # Molecule-level arrays are shared by all K conformers of a molecule.
        arr = np.stack([s[key] for s in samples])  # (B, ...)
        return (
            torch.from_numpy(arr)
            .unsqueeze(1)
            .expand(b, k, *arr.shape[1:])
            .reshape(b * k, *arr.shape[1:])
        )

    return {
        "feats": per_conformer("feats"),
        "mask": per_conformer("mask"),
        "bond_types": per_conformer("bond_types"),
        "coors": torch.from_numpy(np.stack([s["coors"] for s in samples])).reshape(b * k, -1, 3),
        "energy": torch.from_numpy(np.stack([s["energy"] for s in samples])).reshape(b * k, 1),
        "num_conformers": k,
    }
