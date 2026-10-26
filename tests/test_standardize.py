from zao import standardize_smiles


def test_keeps_the_largest_fragment():
    """A small counter-ion and a solvent are dropped, so they do not change the result."""
    acid = "OC(=O)c1ccccc1Oc1ccccc1"
    assert standardize_smiles(acid + ".[Na+]") == standardize_smiles(acid)
    assert standardize_smiles(acid + ".O") == standardize_smiles(acid)


def test_neutralizes_charges():
    assert standardize_smiles("CCCCCCCC[NH3+]") == standardize_smiles("CCCCCCCCN")


def test_collapses_equivalent_spellings_but_keeps_stereochemistry():
    assert standardize_smiles("c1ccccc1CCCCCCN") == standardize_smiles("NCCCCCCc1ccccc1")
    # Stereochemistry survives, because conformer generation downstream needs it.
    assert standardize_smiles("C[C@H](N)C(=O)Nc1ccccc1CCCC") != standardize_smiles(
        "C[C@@H](N)C(=O)Nc1ccccc1CCCC"
    )
    assert standardize_smiles(r"C/C=C\CCCCCCCCC") != standardize_smiles(r"C/C=C/CCCCCCCCC")


def test_molecular_weight_window():
    assert standardize_smiles("CO") is None  # 32, below the lower bound of 100
    assert standardize_smiles("C" * 80) is None  # above the upper bound of 1000
    assert standardize_smiles("CCCCCCCCN") is not None


def test_returns_none_for_what_it_cannot_parse():
    assert standardize_smiles("not_a_smiles") is None
    assert standardize_smiles("") is None
