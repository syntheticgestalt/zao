"""SMILES standardization, identical to the procedure used to build ZAO's pretraining corpus."""

from rdkit import Chem, rdBase
from rdkit.Chem import Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

# Molecular weight window (exclusive) accepted by ZAO.
MIN_MOL_WEIGHT = 100.0
MAX_MOL_WEIGHT = 1000.0

_uncharger = None


def _get_uncharger():
    # Created lazily: the Uncharger cannot be pickled to worker processes.
    global _uncharger
    if _uncharger is None:
        _uncharger = rdMolStandardize.Uncharger()
    return _uncharger


def standardized_mol(mol: Chem.Mol) -> Chem.Mol:
    """The structural steps of ZAO's standardization, without the weight window.

    Keeps RDKit's parent fragment, neutralizes, drops atom map numbers, sanitizes
    and checks the result can be kekulized. Raises if any step fails.
    """
    mol = rdMolStandardize.FragmentParent(mol)
    mol = _get_uncharger().uncharge(mol)
    # Atom map numbers would otherwise change the canonical SMILES.
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    Chem.SanitizeMol(mol)
    # Part of the procedure that built the pretraining corpus.
    Chem.Kekulize(Chem.RWMol(mol), clearAromaticFlags=True)
    return mol


def standardize_smiles(smiles: str) -> str | None:
    """Return the standardized canonical SMILES, or None if the molecule is rejected."""
    try:
        with rdBase.BlockLogs():  # scoped, so the caller keeps RDKit's messages
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            mol = standardized_mol(mol)
            if not (MIN_MOL_WEIGHT < Descriptors.MolWt(mol) < MAX_MOL_WEIGHT):
                return None
            return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        return None
