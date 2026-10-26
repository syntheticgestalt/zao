"""ZAORegressor on your own measurements: fit, score, predict, keep.

The ready-made model. It concatenates the ZAO embedding with nine families of
classical descriptors, searches each of two learners' settings on scaffold folds
of the training set, and averages them. Nothing to assemble.

The molecules are split by scaffold before anything else happens, so the score
printed at the end is on chemistry the fit never saw. A random split would put
close analogues of a test molecule in the training set and flatter the model.

Usage:
    python examples/regressor.py measurements.csv
    python examples/regressor.py measurements.csv --trials 100 --test-fraction 0.2

The CSV needs a column of SMILES and a numeric column to predict. The run takes
minutes to hours: it generates conformers for every molecule, then searches.
"""

import argparse
import csv
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from zao import ZAORegressor, ZAOTransformer, default_features, scaffold_groups
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
            # float() takes "NaN" and "inf" quietly, and nothing downstream catches
            # either until the molecules have been through the model.
            if not np.isfinite(value):
                raise SystemExit(
                    f"{path} line {rows.line_num}: {measurement!r} is not a value to fit on"
                )
            smiles.append(molecule)
            y.append(value)
    if not smiles:
        raise SystemExit(f"{path} has no rows with both a {smiles_column} and a {target_column}")
    return np.array(smiles, dtype=object), np.array(y)


def arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("csv")
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--target-column", default="y")
    parser.add_argument(
        "--weights", default=DEFAULT_WEIGHTS, help="local model.safetensors, or a Hub id"
    )
    parser.add_argument("--trials", type=int, default=50, help="settings tried per learner")
    parser.add_argument(
        "--cv", type=int, default=5, help="scaffold folds the search scores each setting on"
    )
    parser.add_argument(
        "--scoring",
        default=None,
        help="what the search maximizes; the default is R2, which small folds can leave "
        "undefined. neg_mean_absolute_error is defined on any fold",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.2,
        help="share of scaffolds held out, not of molecules: scaffolds differ in size",
    )
    parser.add_argument("--save", default="model.joblib", help="where to write the fitted model")
    return parser


def hold_out_scaffolds(smiles, y, groups, fraction: float):
    """Molecules to fit on and molecules to score on, with no scaffold in both.

    GroupShuffleSplit divides the scaffolds, and scaffolds hold different numbers
    of molecules, so the share of molecules held out only approximates
    ``fraction``. One split of one data set is a noisy number; for a figure worth
    reporting, cross-validate over several.
    """
    if len(set(groups)) < 2:
        raise SystemExit(
            f"all {len(smiles)} molecules share one scaffold, so nothing can be held out that "
            "the fit has not seen. Fit on a set with more than one scaffold"
        )
    return next(
        GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=0).split(smiles, y, groups)
    )


def refuse_folds_without_an_r2(y, groups, cv: int) -> None:
    """The folds the search scores settings on, checked before the hours begin.

    A fold of one molecule has no R2 at all, and the search averages over the
    folds that scored, so a setting would be chosen on fewer folds than asked
    for. A fold whose measurements never vary is worse: scikit-learn answers 0.0
    or 1.0 there, and that counts towards the average as if it meant something.
    """
    for _, scored in GroupKFold(cv).split(np.zeros(len(y)), y, groups):
        if len(scored) < 2 or np.ptp(y[scored]) == 0:
            raise SystemExit(
                f"one of the {cv} folds the search scores on holds {len(scored)} molecule(s), "
                "or measurements that never vary, which leaves no R2. Lower --cv, fit on more "
                "molecules, or pass --scoring neg_mean_absolute_error, which is defined on any "
                "fold"
            )


def main() -> None:
    args = arguments().parse_args()

    # Settled now: the fit is minutes to hours, and the saving is after it.
    if args.cv < 2:
        raise SystemExit(f"--cv must be 2 or more, got {args.cv}")
    save = Path(args.save)
    if save.is_dir() or not save.parent.is_dir():
        raise SystemExit(f"--save {args.save} is not a file this can write to")

    smiles, y = read_csv(args.csv, args.smiles_column, args.target_column)
    groups = scaffold_groups(smiles)
    train, test = hold_out_scaffolds(smiles, y, groups, args.test_fraction)
    if args.scoring is None:
        refuse_folds_without_an_r2(y[train], groups[train], args.cv)
    print(f"{len(train)} molecules to fit on, {len(test)} held out by scaffold\n")

    # default_features(zao=False) instead gives the classical-descriptor
    # baseline, which is what the ZAO columns have to beat to be worth their time.
    features = default_features(zao=ZAOTransformer(weights=args.weights))
    model = ZAORegressor(features=features, n_trials=args.trials, cv=args.cv, scoring=args.scoring)

    # One call does the lot. Minutes to hours, most of it conformers.
    model.fit(smiles[train], y[train], groups=groups[train])

    # SMILES in, predictions out: the fitted model carries its own featurization.
    predicted = model.predict(smiles[test])
    print("held-out scaffolds:")
    print(f"  R2  {r2_score(y[test], predicted):.3f}")
    print(f"  MAE {mean_absolute_error(y[test], predicted):.3f}")

    # Which descriptor families each learner kept. The search switches them on
    # and off; the ZAO embedding is always kept.
    print("\nfeature blocks each learner chose:")
    for learner, blocks in model.blocks_.items():
        print(f"  {learner:7s} {', '.join(blocks)}")

    # The saved file carries the featurization with it, so predict() on new
    # SMILES needs only the file and the ZAO weights.
    joblib.dump(model, save)
    print(f"\nsaved {args.save}; load it with joblib.load and call predict on new SMILES")


if __name__ == "__main__":
    main()
