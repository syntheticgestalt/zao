"""Scaffold grouping, for splits that score a model on scaffolds it has never seen.

A random split puts close analogues of a test molecule in the training set.
Grouping the molecules by scaffold and keeping each group on one side of the
split removes that. Two scaffold definitions are available:

- ``"murcko"``: the Bemis-Murcko framework (ring systems and the linkers
  between them, with atom elements and bond orders kept).
- ``"generic"``: the cyclic skeleton of the Murcko framework, with every atom
  made carbon and every bond single. Only ring sizes, how the rings are fused
  and how many atoms the linkers between them have remain. Pyridine and
  benzene, or a piperidine and a cyclohexane, are the same scaffold.

``"generic"`` puts more molecules in each group than ``"murcko"`` and is the
harder of the two.
"""

from collections.abc import Sequence

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold

from ._input import as_smiles_list
from .preprocess.standardize import standardized_mol

SCAFFOLD_TYPES = ("murcko", "generic")


def _scaffold_of(smiles: str, scaffold: str, largest_fragment: bool) -> str:
    """Canonical SMILES of the scaffold of one molecule; ``""`` if it has no ring.

    With ``largest_fragment`` (the default) the scaffold is taken from the
    molecule as ZAO standardizes it. Isotopes and stereochemistry are removed.
    """
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"cannot parse SMILES: {smiles!r}")
        try:
            if largest_fragment:
                mol = standardized_mol(mol)
            else:
                # The same normalization and neutralization, every fragment kept.
                mol = rdMolStandardize.Uncharger().uncharge(rdMolStandardize.Cleanup(mol))
            # Re-read, so equal standardized SMILES give equal scaffolds.
            mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol, isomericSmiles=True))
        except Exception as e:
            raise ValueError(f"cannot standardize SMILES: {smiles!r}") from e
        for atom in mol.GetAtoms():
            atom.SetIsotope(0)
        Chem.RemoveStereochemistry(mol)
        core = MurckoScaffold.GetScaffoldForMol(mol)
        if scaffold == "generic":
            # Extracted twice: made generic, a double-bonded oxygen becomes a side chain.
            core = MurckoScaffold.GetScaffoldForMol(MurckoScaffold.MakeScaffoldGeneric(core))
        return Chem.MolToSmiles(core)


def scaffold_groups(
    smiles: Sequence[str], scaffold: str = "generic", largest_fragment: bool = True
) -> np.ndarray:
    """Scaffold of each molecule, as canonical SMILES; acyclic molecules give ``""``.

    The result is the ``groups`` argument of any scikit-learn group splitter:

    >>> from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
    >>> groups = scaffold_groups(smiles)
    >>> GroupKFold(n_splits=5).split(smiles, y, groups)            # doctest: +SKIP
    >>> StratifiedGroupKFold(n_splits=5).split(smiles, y, groups)  # doctest: +SKIP

    Keeping the scaffold itself rather than an opaque code lets you count or
    inspect the scaffolds:

    >>> import numpy as np
    >>> np.unique(scaffold_groups(smiles), return_counts=True)  # doctest: +SKIP

    Parameters
    ----------
    smiles : list, array, Series or single-column DataFrame of SMILES
    scaffold : {"murcko", "generic"}, default="generic"
        See the module docstring.
    largest_fragment : bool, default=True
        Start from the molecule as ZAO standardizes it (RDKit's parent fragment,
        neutralized), so that the scaffold always agrees with what ZAO embeds.
        If False, every fragment is kept.

    Returns
    -------
    ndarray of dtype object
        One entry per input molecule.

    Raises
    ------
    ValueError
        If ``scaffold`` is unknown, or a SMILES cannot be parsed.
    """
    if scaffold not in SCAFFOLD_TYPES:
        raise ValueError(f"scaffold must be one of {SCAFFOLD_TYPES}, got {scaffold!r}")
    keys = [_scaffold_of(s, scaffold, largest_fragment) for s in as_smiles_list(smiles)]
    return np.asarray(keys, dtype=object)
