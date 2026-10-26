"""Embeddings for a list of SMILES, in one call, with no scikit-learn in sight.

The shortest thing you can do with ZAO: hand it molecules, get a row of 2,048
numbers for each one.

Usage:
    python examples/encode.py                       # writes embeddings.npy here
    python examples/encode.py --weights /path/to/model.safetensors
    python examples/encode.py --library library.smi # writes library_embeddings.tsv here
"""

import argparse

import numpy as np

from zao import ZAOTransformer, encode
from zao.model import DEFAULT_WEIGHTS

SMILES = [
    "CC(=O)Oc1ccccc1C(=O)O",  # aspirin
    "CC(C)Cc1ccc(C(C)C(=O)O)cc1",  # ibuprofen
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # caffeine
    "CC(=O)Nc1ccc(O)cc1",  # paracetamol
    "not_a_smiles",  # cannot be read: its row comes back as NaN
]


def a_handful(weights: str, out_path: str) -> None:
    # What the one call does to each molecule: standardize it, generate ten
    # conformers, and pass the whole set through the model once. Out comes a row
    # of 2,048 numbers, or a row of NaN where the molecule could not be
    # processed. The weights are downloaded on the first call and cached.
    X = encode(SMILES, weights=weights)
    print(f"{X.shape[0]} molecules, {X.shape[1]} numbers each\n")

    for smiles, row in zip(SMILES, X, strict=True):
        if np.isnan(row).all():
            print(f"  {smiles:32s} could not be processed")
        else:
            print(f"  {smiles:32s} first three: {row[:3].round(3)}")

    # The array is ordinary numpy, ready for scikit-learn.
    np.save(out_path, X)
    print(f"\nsaved {out_path}")

    # How close two molecules are, as the model sees them. Ibuprofen and
    # aspirin are both carboxylic acids; caffeine is not.
    usable = X[:4] / np.linalg.norm(X[:4], axis=1, keepdims=True)
    print("\ncosine similarity to aspirin:")
    for smiles, similarity in zip(SMILES[1:4], usable[1:] @ usable[0], strict=True):
        print(f"  {smiles:32s} {similarity:.3f}")


def a_whole_library(path: str, out_path: str, weights: str) -> None:
    """A file too large to hold in memory, one molecule at a time.

    ``stream`` takes an iterator and yields one molecule at a time, so nothing
    bigger than one chunk is ever held.
    """

    kept = 0
    zao = ZAOTransformer(weights=weights, show_progress=True)
    with open(path) as handle, open(out_path, "w") as out:
        molecules = (line.split()[0] for line in handle if line.strip())
        for smiles, embedding in zao.stream(molecules):
            if embedding is None:
                continue
            out.write(smiles + "\t" + "\t".join(f"{v:.9g}" for v in embedding) + "\n")
            kept += 1
    print(f"{kept} molecules from {path} written to {out_path}")
    if kept == 0:
        print(
            "none of them could be processed: ZAO takes a molecular weight strictly "
            "between 100 and 1,000, and at most 100 atoms including hydrogens"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights", default=DEFAULT_WEIGHTS, help="local model.safetensors, or a Hub id"
    )
    parser.add_argument("--library", default=None, help="a .smi file to stream instead")
    parser.add_argument("--out", default=None, help="where to write; the default depends on which")
    args = parser.parse_args()

    if args.library:
        a_whole_library(args.library, args.out or "library_embeddings.tsv", args.weights)
    else:
        a_handful(args.weights, args.out or "embeddings.npy")


if __name__ == "__main__":
    main()
