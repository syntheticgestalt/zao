"""The ``zao`` command and the one-call ``encode``, both weight-free here."""

import numpy as np
import pytest

import zao.transformers as transformers
from zao.cli import main, read_smiles, write_csv, write_npy

SMILES = ["CC(=O)Oc1ccccc1C(=O)O", "c1ccc2[nH]ccc2c1"]


class _FakeTransformer:
    """Stands in for ZAOTransformer: records its settings, fails on an empty SMILES."""

    seen = {}
    embedding_dim_ = 4

    def __init__(self, **kwargs):
        _FakeTransformer.seen = kwargs

    def fit(self, X=None, y=None):
        return self

    def _row(self, i, smiles):
        if not smiles:
            return None
        return np.full(4, i, dtype=np.float32)

    def stream(self, smiles):
        return ((s, self._row(i, s)) for i, s in enumerate(smiles))

    def transform(self, smiles):
        smiles = list(smiles)
        out = np.full((len(smiles), 4), np.nan, dtype=np.float32)
        for i, s in enumerate(smiles):
            row = self._row(i, s)
            if row is not None:
                out[i] = row
        return out


@pytest.fixture
def fake(monkeypatch):
    monkeypatch.setattr(transformers, "ZAOTransformer", _FakeTransformer)
    return _FakeTransformer


def _input(tmp_path, text, name="in.smi"):
    path = tmp_path / name
    path.write_text(text)
    return str(path)


def test_a_smiles_file_keeps_only_the_smiles(tmp_path):
    """One per line, a name after a space, blanks and comments."""
    source = _input(tmp_path, f"{SMILES[0]} aspirin\n\n# a comment\n{SMILES[1]}\n")
    assert read_smiles(source, None) == SMILES


def test_a_csv_input_finds_the_smiles_column(tmp_path):
    source = _input(tmp_path, f"id,SMILES,y\n1,{SMILES[0]},3.4\n2,{SMILES[1]},5.6\n", "in.csv")
    assert read_smiles(source, None) == SMILES
    assert read_smiles(source, "smiles") == SMILES


def test_a_csv_without_a_smiles_column_is_refused(tmp_path):
    """Falling back to the first column would embed whatever happens to be there."""
    source = _input(tmp_path, f"id,structure\n1,{SMILES[0]}\n", "in.csv")
    with pytest.raises(SystemExit, match="no column 'smiles'"):
        read_smiles(source, None)
    with pytest.raises(SystemExit, match="no column 'name'"):
        read_smiles(source, "name")
    assert read_smiles(source, "structure") == SMILES[:1]


def test_a_short_csv_row_keeps_its_place(tmp_path):
    """Dropping it would move every later molecule up a row in the output."""
    source = _input(tmp_path, f"id,smiles\n1,{SMILES[0]}\n2\n3,{SMILES[1]}\n", "in.csv")
    assert read_smiles(source, None) == [SMILES[0], "", SMILES[1]]


def test_a_byte_order_mark_is_not_part_of_the_smiles(tmp_path):
    """What a spreadsheet writes as UTF-8 starts with one."""
    source = _input(tmp_path, f"﻿{SMILES[0]}\n")
    assert read_smiles(source, None) == SMILES[:1]
    csv_source = _input(tmp_path, f"﻿smiles,id\n{SMILES[0]},1\n", "in.csv")
    assert read_smiles(csv_source, "smiles") == SMILES[:1]


def test_npy_holds_the_matrix_under_the_name_given(tmp_path):
    """np.save appends .npy to anything not ending in exactly that."""
    X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    out = tmp_path / "out.NPY"
    write_npy(out, X)
    assert out.is_file()
    assert np.array_equal(np.load(out), X)


def test_csv_values_are_written_exactly(tmp_path):
    """A CSV that does not round-trip float32 would disagree with the .npy."""
    X = np.array([[123456.78, 1.2345678], [-3.5e-8, 2.0]], dtype=np.float32)
    out = tmp_path / "out.csv"
    rows, failed = write_csv(out, zip(SMILES, X, strict=True), 2)

    assert (rows, failed) == (2, 0)
    lines = out.read_text().splitlines()
    assert lines[0] == "smiles,zao0,zao1"
    written = np.array([line.split(",")[1:] for line in lines[1:]], dtype=np.float32)
    assert np.array_equal(written, X)


def test_a_molecule_that_failed_leaves_an_empty_row(tmp_path):
    out = tmp_path / "out.csv"
    rows, failed = write_csv(out, [(SMILES[0], None)], 2)
    assert (rows, failed) == (1, 1)
    assert out.read_text().splitlines()[1] == f"{SMILES[0]},,"


def test_the_command_writes_what_it_was_asked_for(tmp_path, fake, capsys):
    source = _input(tmp_path, "\n".join(SMILES) + "\n")
    out = tmp_path / "out.npy"

    assert main(["encode", source, "-o", str(out), "--device", "cpu", "--quiet"]) == 0
    assert np.load(out).shape == (2, 4)
    assert fake.seen["device"] == "cpu"
    assert capsys.readouterr().err == ""


def test_a_csv_output_is_written_as_the_molecules_arrive(tmp_path, fake, capsys):
    """The streaming path, so a library larger than memory never becomes one array."""
    source = _input(tmp_path, f"smiles\n{SMILES[0]}\n\n{SMILES[1]}\n", "in.csv")
    out = tmp_path / "out.csv"

    main(["encode", source, "-o", str(out)])
    assert len(out.read_text().splitlines()) == 3
    assert "2 molecules" in capsys.readouterr().err


def test_encode_is_the_transformer_in_one_call(monkeypatch):
    """The short path runs the same code the long one does."""
    monkeypatch.setattr(transformers.zao, "ZAOTransformer", _FakeTransformer)

    X = transformers.encode(SMILES, device="cpu", batch_size=8)
    assert X.shape == (2, 4)
    assert _FakeTransformer.seen == {"device": "cpu", "batch_size": 8}

    streamed = list(_FakeTransformer().stream(iter(SMILES)))
    assert [smiles for smiles, _ in streamed] == SMILES
