"""ZAOTransformer with the rest of scikit-learn.

The embedding is a feature set like any other, so everything scikit-learn
already does works on molecules unchanged. This fits a ridge regression on the
embedding and scores it on scaffolds the fit has never seen, which is the number
that says whether a model will hold up on new chemistry.

ZAOTransformer is called once, up front, and the folds index into the array it
returns. It can also go inside the pipeline handed to cross_val_score, which
reads better and costs more: every molecule's conformers are then generated
again for every fold.

Usage:
    python examples/transformer.py measurements.csv
    python examples/transformer.py measurements.csv --smiles-column smiles --target-column y

The CSV needs a column of SMILES and a numeric column to predict: solubility,
permeability, clearance, an affinity as pIC50, whatever you have.
"""

import argparse
import csv

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from zao import ZAOTransformer, scaffold_groups
from zao.model import DEFAULT_WEIGHTS


def read_csv(path: str, smiles_column: str, target_column: str):
    """SMILES and their measured values, with the standard library: no pandas needed."""
    smiles, y = [], []
    with open(path, newline="", encoding="utf-8-sig") as handle:
        rows = csv.DictReader(handle)
        missing = {smiles_column, target_column} - set(rows.fieldnames or ())
        if missing:
            raise SystemExit(
                f"{path} has no column {' or '.join(sorted(missing))}. Its columns are "
                f"{', '.join(rows.fieldnames or [])}; name yours with --smiles-column "
                "and --target-column"
            )
        for row in rows:
            molecule = (row[smiles_column] or "").strip()
            measurement = (row[target_column] or "").strip()
            if not molecule or not measurement:
                continue  # nothing measured for this molecule
            value = float(measurement)
            # float() takes "NaN" and "inf" quietly, and the fold check below reads
            # neither as a problem; both would surface only after the embedding.
            if not np.isfinite(value):
                raise SystemExit(
                    f"{path} line {rows.line_num}: {measurement!r} is not a value to fit on"
                )
            smiles.append(molecule)
            y.append(value)
    if not smiles:
        raise SystemExit(f"{path} has no rows with both a {smiles_column} and a {target_column}")
    return np.array(smiles, dtype=object), np.array(y)


def folds_for(y, groups, n_folds: int):
    """The folds, with the ones an R2 cannot be read from refused.

    GroupKFold says for itself when there are fewer scaffolds than folds. What it
    will not say is that a fold cannot be scored: R2 compares a model against
    predicting the mean of the fold, so a fold of one molecule has no R2, and a
    fold whose measurements never vary gets 0.0 or 1.0, which would be averaged
    in with the rest as if it meant something.
    """
    folds = list(GroupKFold(n_folds).split(np.zeros(len(y)), y, groups))
    for _, scored in folds:
        if len(scored) < 2 or np.ptp(y[scored]) == 0:
            raise SystemExit(
                f"one fold holds {len(scored)} molecule(s), or one measurement repeated, "
                "which leaves no R2. Use fewer folds, or more molecules"
            )
    return folds


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("csv")
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--target-column", default="y")
    parser.add_argument(
        "--weights", default=DEFAULT_WEIGHTS, help="local model.safetensors, or a Hub id"
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--scaffold", choices=("generic", "murcko"), default="generic")
    args = parser.parse_args()

    smiles, y = read_csv(args.csv, args.smiles_column, args.target_column)
    print(f"{len(smiles)} molecules from {args.csv}")

    # Conformer generation is the expensive part, so the folds are checked first;
    # the ones that are used are made again below, after the failures are dropped.
    groups = scaffold_groups(smiles, scaffold=args.scaffold)
    folds_for(y, groups, args.folds)

    # fit() only locates the weights; the transformer learns nothing from the
    # molecules, which is why fitting it on everything before the folds are
    # scored leaks nothing. transform() is where the time goes: conformers for
    # every molecule, then the forward passes. One row of 2,048 numbers each.
    zao = ZAOTransformer(weights=args.weights, show_progress=True)
    X = zao.fit_transform(smiles)

    # A molecule ZAO cannot process comes back as a row of NaN, which a ridge
    # regression cannot read.
    usable = ~np.isnan(X).any(axis=1)
    if not usable.all():
        print(f"{(~usable).sum()} of {len(smiles)} molecules could not be embedded, left out")
    X, y = X[usable], y[usable]
    smiles, groups = smiles[usable], groups[usable]

    # Molecules sharing a scaffold stay on one side of every split.
    folds = folds_for(y, groups, args.folds)
    print(f"{len(set(groups))} distinct scaffolds among {len(smiles)} molecules\n")

    # Any estimator can read the embedding. Ridge is a plain one that makes the
    # point without tuning: RidgeCV picks its own regularisation strength, and
    # standardizing first is what a linear model wants of 2,048 columns.
    model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 13)))
    scores = []
    for fold, (train, test) in enumerate(folds, start=1):
        model.fit(X[train], y[train])
        scores.append(r2_score(y[test], model.predict(X[test])))
        print(f"fold {fold}: {len(train)} train, {len(test)} test, R2 {scores[-1]:.3f}")

    print(f"\nR2 over {args.folds} scaffold folds: {np.mean(scores):.3f} ± {np.std(scores):.3f}")


if __name__ == "__main__":
    main()
