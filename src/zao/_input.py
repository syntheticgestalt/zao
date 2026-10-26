"""Reading the SMILES a caller hands over, in whichever container they hold them."""


def as_smiles_list(X) -> list:
    """Accept a list, array, Series or single-column DataFrame of SMILES."""
    if isinstance(X, str):
        # list() would split it into one-character "molecules".
        raise TypeError("pass a list of SMILES, not a single string; one molecule is [smiles]")
    if not hasattr(X, "__len__"):
        raise TypeError(
            "transform takes a list, array or Series; for an iterator, which is what a file "
            "larger than memory is read as, call stream() instead"
        )
    if hasattr(X, "iloc") and getattr(X, "ndim", 1) == 2:
        if X.shape[1] != 1:
            raise ValueError("pass a single column of SMILES, not a DataFrame with several columns")
        X = X.iloc[:, 0]
    return list(X)
