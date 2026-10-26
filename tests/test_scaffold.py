import numpy as np
import pandas as pd
import pytest

from zao import scaffold_groups

# Small set with known scaffold relations.
SMILES = [
    "c1ccccc1CC(=O)Nc1ccccc1",  # 0  benzene-CH2-amide-benzene
    "c1ccncc1CC(=O)Nc1ccccc1",  # 1  same generic skeleton as 0 (pyridine)
    "C1CCCCC1CCNC1CCCCC1",  # 2  same generic skeleton as 0 (saturated, N linker)
    "c1ccccc1CCC(=O)Nc1ccccc1",  # 3  one linker atom more than 0
    "c1ccccc1Nc1ccccc1",  # 4  diphenylamine
    "Cc1ccccc1Nc1ccccc1",  # 5  same Murcko scaffold as 4
    "c1ccc2ccccc2c1",  # 6  naphthalene
    "c1ccc2ncccc2c1",  # 7  quinoline: same generic skeleton as 6
    "CCO",  # 8  acyclic
    "CCCCN",  # 9  acyclic
    "O=C1CCC(c2ccccc2)CC1",  # 10 cyclohexanone-phenyl
    "c1ccc(C2CCCCC2)cc1",  # 11 same generic skeleton as 10
]


def test_generic_scaffold_ignores_elements_and_bond_orders():
    g = scaffold_groups(SMILES)
    assert len(g) == len(SMILES)
    assert g[0] == g[1] == g[2]
    assert g[0] != g[3]  # linker length matters
    assert g[10] == g[11]  # exocyclic carbonyl does not survive
    assert g[8] == g[9] == ""  # acyclic molecules
    # The entries are the scaffolds themselves, not opaque codes.
    assert g[6] == g[7] == "C1CCC2CCCCC2C1"


def test_murcko_scaffold_keeps_elements():
    m = scaffold_groups(SMILES, scaffold="murcko")
    assert m[0] != m[1]
    assert m[4] == m[5]  # the methyl substituent is removed
    assert m[6] != m[7]


def test_largest_fragment():
    # A second component with a ring of its own (p-toluenesulfonic acid) would
    # otherwise become part of the scaffold.
    salt, acid = "c1ccc(cc1)C(=O)Nc1ccccc1.Cc1ccc(S(=O)(=O)O)cc1", "c1ccc(cc1)C(=O)Nc1ccccc1"
    assert scaffold_groups([salt])[0] == scaffold_groups([acid])[0]
    assert scaffold_groups([salt], largest_fragment=False)[0] != scaffold_groups([acid])[0]


def test_invalid_input():
    with pytest.raises(ValueError, match="cannot parse SMILES"):
        scaffold_groups(["CCO", "not_a_smiles"])
    with pytest.raises(ValueError, match="scaffold must be one of"):
        scaffold_groups(["CCO"], scaffold="unknown")
    with pytest.raises(ValueError):
        scaffold_groups(pd.DataFrame({"a": SMILES, "b": SMILES}))
    with pytest.raises(TypeError, match="single string"):
        scaffold_groups("CCO")


def test_accepts_series_and_dataframe():
    ref = scaffold_groups(SMILES)
    assert np.array_equal(scaffold_groups(pd.Series(SMILES)), ref)
    assert np.array_equal(scaffold_groups(pd.DataFrame({"smiles": SMILES})), ref)


def test_rdkit_logging_is_left_as_it_was():
    """Silencing RDKit inside a call must not silence it for the caller afterwards."""
    from rdkit import RDLogger, rdBase

    from zao import standardize_smiles

    RDLogger.EnableLog("rdApp.error")
    scaffold_groups(["c1ccccc1"])
    standardize_smiles("not_a_smiles")
    assert rdBase.LogStatus().find("rdApp.error:enabled") != -1


def test_scaffold_depends_only_on_the_standardized_molecule():
    """Whatever ZAO standardizes to one SMILES gets one scaffold, by construction."""
    from zao import standardize_smiles

    raw = SMILES + [
        "O=S(Cc1ccc(-n2cccn2)cc1)c1ccc(CO)cc1",  # neutral sulfoxide
        "[O-][S+](Cc1ccc(-n2cccn2)cc1)c1ccc(CO)cc1",  # the same, charge-separated
        "CCc1cc[nH+]cc1.[Cl-]",
        "OC(=O)c1ccccc1Oc1ccccc1.[Na+]",
    ]
    for scaffold in ("generic", "murcko"):
        for smi in raw:
            std = standardize_smiles(smi)
            if std is not None:
                assert scaffold_groups([smi], scaffold)[0] == scaffold_groups([std], scaffold)[0]
