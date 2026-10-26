"""The ``zao`` command: embeddings for a file of SMILES, without writing Python."""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from . import __version__
from .model.config import DEFAULT_WEIGHTS, ZAO_CONFIG

NPY, CSV = ".npy", ".csv"

# float32 takes 9 significant digits to be read back as the same number.
_number = "{:.9g}".format


def _positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be 1 or more, got {value}")
    return number


def _jobs(value: str) -> int:
    number = int(value)
    if number == 0 or number < -1:
        raise argparse.ArgumentTypeError(f"must be -1 (every CPU) or 1 or more, got {value}")
    return number


def _from_lines(lines) -> list[str]:
    """One SMILES per line; anything after the first space is a name and is ignored."""
    out = []
    for line in lines:
        line = line.lstrip("﻿").strip()
        if line and not line.startswith("#"):
            out.append(line.split()[0])
    return out


def _from_csv(handle, column: str | None, source: str) -> list[str]:
    """One column of a .csv, named by ``--column`` or called ``smiles``.

    Every data row gives an entry, empty if that row is short, so the output
    keeps the input's row order and length.
    """
    reader = csv.reader(handle)
    header = next(reader, None)
    if header is None:
        return []
    names = [h.strip().lstrip("﻿").lower() for h in header]
    wanted = (column or "smiles").strip().lower()
    if wanted not in names:
        raise SystemExit(
            f"zao: no column {wanted!r} in {source}. Its columns are "
            f"{', '.join(header)}; name the one holding the SMILES with --column"
        )
    index = names.index(wanted)
    return [row[index].strip() if len(row) > index else "" for row in reader if row]


def read_smiles(source: str, column: str | None) -> list[str]:
    """SMILES from a file, or from standard input when ``source`` is ``-``."""
    if source == "-":
        if column is not None:
            return _from_csv(sys.stdin, column, "the input")
        return _from_lines(sys.stdin)
    path = Path(source)
    if not path.is_file():
        raise SystemExit(f"zao: no such file: {source}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        if path.suffix.lower() == CSV or column is not None:
            return _from_csv(handle, column, source)
        return _from_lines(handle)


def write_npy(out: Path, X: np.ndarray) -> None:
    """The matrix alone. Written through a handle, so the name is used as given."""
    with out.open("wb") as handle:
        np.save(handle, X)


def write_csv(out: Path, pairs, width: int) -> tuple[int, int]:
    """The SMILES and their columns, written as they arrive. Returns (rows, failures)."""
    rows = failed = 0
    with out.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["smiles"] + [f"zao{i}" for i in range(width)])
        for smiles, embedding in pairs:
            rows += 1
            if embedding is None:
                failed += 1
                writer.writerow([smiles] + [""] * width)
            else:
                writer.writerow([smiles, *(_number(v) for v in embedding)])
    return rows, failed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="zao", description="ZAO embeddings for a file of SMILES.")
    parser.add_argument("--version", action="version", version=f"zao {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    encode = sub.add_parser(
        "encode",
        help="embed a file of SMILES",
        description=(
            "Embed a file of SMILES. A .smi or .txt file holds one SMILES per line, with "
            "anything after a space taken as its name; blank lines and lines starting with "
            "# are skipped. A .csv file must have a header row and a column called smiles, "
            "or one named with --column; every data row gives a row of output, so the two "
            "files line up. A molecule that cannot be processed gives an empty row."
        ),
    )
    encode.add_argument(
        "input", help="file of SMILES, one per line, or a .csv; - reads standard input"
    )
    encode.add_argument(
        "-o",
        "--output",
        required=True,
        help=(
            "where to write: .npy for the matrix alone, .csv for the SMILES and their "
            "columns. An existing file is replaced"
        ),
    )
    encode.add_argument("--column", help="which column of a .csv input holds the SMILES")
    encode.add_argument("--weights", default=DEFAULT_WEIGHTS, help="Hub repository or local file")
    encode.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    encode.add_argument(
        "--conformer-backend",
        default="auto",
        choices=("auto", "gpu", "cpu"),
        help="where conformers are generated",
    )
    encode.add_argument(
        "--conformers",
        type=_positive,
        default=ZAO_CONFIG["general"]["num_conformers"],
        help="conformers generated per molecule",
    )
    encode.add_argument(
        "--batch-size", type=_positive, default=256, help="molecules per forward pass"
    )
    encode.add_argument(
        "--chunk-size",
        type=_positive,
        default=10000,
        help="molecules per conformer-generation call",
    )
    encode.add_argument(
        "--jobs", type=_jobs, default=-1, help="CPUs to use; -1 is every CPU this process may use"
    )
    encode.add_argument("-q", "--quiet", action="store_true", help="no progress bar, no summary")
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)

    out = Path(args.output)
    if out.suffix.lower() not in (NPY, CSV):
        raise SystemExit(f"zao: --output must end in {NPY} or {CSV}, got {args.output!r}")
    if args.input != "-" and out.resolve() == Path(args.input).resolve():
        raise SystemExit(f"zao: --output is the input file: {args.input}")

    smiles = read_smiles(args.input, args.column)
    if not any(smiles):
        raise SystemExit(f"zao: no SMILES in {args.input}")

    from .transformers import ZAOTransformer

    model = ZAOTransformer(
        weights=args.weights,
        num_conformers=args.conformers,
        device=args.device,
        conformer_backend=args.conformer_backend,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        n_jobs=args.jobs,
        show_progress=not args.quiet and sys.stderr.isatty(),
    ).fit()

    if out.suffix.lower() == CSV:
        # Streamed, so a library larger than memory never becomes one array.
        rows, failed = write_csv(out, model.stream(iter(smiles)), model.embedding_dim_)
        shape = (rows, model.embedding_dim_)
    else:
        X = model.transform(smiles)
        write_npy(out, X)
        rows, failed, shape = len(X), int(np.isnan(X).all(axis=1).sum()), X.shape

    if not args.quiet:
        note = f", {failed} could not be processed and are blank" if failed else ""
        print(f"{rows} molecules -> {out} {shape}{note}", file=sys.stderr)
    return 0
