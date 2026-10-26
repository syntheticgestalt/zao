"""The ZAO embedding as a scikit-learn transformer."""

from collections.abc import Iterable, Iterator, Sequence

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin

from .._input import as_smiles_list
from ..model.config import DEFAULT_WEIGHTS, ZAO_CONFIG
from ..parallel import process_workers
from ..preprocess.conformers import (
    default_hardware_options,
    generate_conformers,
    standardize_and_prepare,
)
from ..preprocess.features import collate, mol_to_features

# Below this many molecules per chunk, standardization runs in-process.
_PARALLEL_STANDARDIZE_MIN = 2000


def _native_bf16(device) -> bool:
    """Whether ``device`` executes bfloat16 natively, i.e. compute capability 8.0 or newer."""
    import torch

    return device.type == "cuda" and torch.cuda.get_device_capability(device) >= (8, 0)


class ZAOTransformer(TransformerMixin, BaseEstimator):
    """Turns SMILES into 2,048-dimensional ZAO embeddings.

    Each molecule is standardized, given ``num_conformers`` conformers (ETKDGv3 +
    MMFF94, on the GPU with nvMolKit or on the CPU with RDKit) and passed through
    the pretrained model in one forward pass. Nothing is learned: ``fit`` only
    locates the weights, downloading them if needed, and the model is built on
    the first ``transform``. The transformer can therefore sit at the front of a
    scikit-learn ``Pipeline``.

    Parameters
    ----------
    weights : str, default="syntheticgestalt/zao"
        Hugging Face Hub repository id, or a local path to ``model.safetensors``
        (or the directory holding it).
    revision : str, optional
        Hub revision (branch, tag or commit) to download.
    num_conformers : int, default=10
        Conformers generated per molecule. A molecule for which embedding yields
        fewer is embedded from the conformers it has.
    device : str, default="auto"
        Device for the model forward pass: "auto", "cuda", "cuda:N" or "cpu".
        Where conformers are generated is set by ``conformer_backend``.
    batch_size : int, default=256
        Molecules per forward pass.
    chunk_size : int, default=10000
        Molecules per conformer-generation call. Bounds memory on large inputs.
    amp : bool, default=True
        Run the forward pass in bfloat16 autocast on CUDA. Applies only on GPUs
        that execute bfloat16 natively (compute capability 8.0 or newer);
        elsewhere the pass runs in float32.
    conformer_gpu_ids : list of int, optional
        GPUs for conformer generation. None picks automatically.
    conformer_backend : {"auto", "gpu", "cpu"}, default="auto"
        Where conformers are generated. "auto" uses the GPU when nvMolKit and a CUDA device are both
        present, and the CPU otherwise. The CPU backend is orders of magnitude slower, so it suits
        trying the model out and small sets.
    n_jobs : int, default=-1
        Worker processes for SMILES standardization. -1 uses every CPU this
        process may use; 1 keeps the work here. Conformer generation is not
        affected: the CPU backend spreads each molecule's conformers over its
        own threads.
    show_progress : bool, default=False
        Show a progress bar over chunks, for both array and streaming input.

    Notes
    -----
    Output rows are NaN for molecules that cannot be processed: unparsable
    SMILES, molecular weight outside (100, 1000), more than ``max_atoms`` atoms
    including hydrogens (100 for the released weights), no MMFF94 parameters,
    failed conformer generation, or a conformer energy outside [-300, 3000]
    kcal/mol. Duplicate molecules (after standardization) are computed once
    within each chunk of ``chunk_size`` molecules.

    Examples
    --------
    >>> from zao import ZAOTransformer
    >>> zao = ZAOTransformer()
    >>> X = zao.transform(["CC(=O)Oc1ccccc1C(=O)O", "c1ccc2[nH]ccc2c1"])
    >>> X.shape
    (2, 2048)
    """

    def __init__(
        self,
        weights: str = DEFAULT_WEIGHTS,
        revision: str | None = None,
        num_conformers: int = ZAO_CONFIG["general"]["num_conformers"],
        device: str = "auto",
        batch_size: int = 256,
        chunk_size: int = 10000,
        amp: bool = True,
        conformer_gpu_ids: Sequence[int] | None = None,
        conformer_backend: str = "auto",
        n_jobs: int = -1,
        show_progress: bool = False,
    ):
        self.weights = weights
        self.revision = revision
        self.num_conformers = num_conformers
        self.device = device
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.amp = amp
        self.conformer_gpu_ids = conformer_gpu_ids
        self.conformer_backend = conformer_backend
        self.n_jobs = n_jobs
        self.show_progress = show_progress

    def fit(self, X=None, y=None):
        """Locate the weights. Nothing is learned; the model loads on first use."""
        # Drop a model loaded under earlier settings.
        self._model = None
        self._device = None
        for name in ("num_conformers", "batch_size", "chunk_size"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        from ..model import resolve_weights

        self._weights_path = resolve_weights(self.weights, self.revision)
        general = ZAO_CONFIG["general"]
        self.embedding_dim_ = general["mol_embed_dim"]
        self.max_atoms_ = general["max_atoms"]
        return self

    def get_feature_names_out(self, input_features=None):
        return np.array([f"zao{i}" for i in range(self.embedding_dim_)], dtype=object)

    def __sklearn_is_fitted__(self):
        return hasattr(self, "embedding_dim_")

    def transform(self, X) -> np.ndarray:
        """Embed molecules.

        Parameters
        ----------
        X : list, array, Series or single-column DataFrame of SMILES

        Returns
        -------
        ndarray of shape (n_samples, 2048)
            Rows of failed molecules are NaN. More molecules than fit in memory
            at once go through ``stream`` instead.
        """
        if not self.__sklearn_is_fitted__():
            self.fit()
        smiles = as_smiles_list(X)
        out = np.full((len(smiles), self.embedding_dim_), np.nan, dtype=np.float32)
        starts = range(0, len(smiles), self.chunk_size)
        if self.show_progress:
            from tqdm import tqdm

            starts = tqdm(starts, desc="ZAO", unit="chunk")
        for start in starts:
            chunk = smiles[start : start + self.chunk_size]
            out[start : start + len(chunk)] = self._embed_chunk(chunk)
        return out

    def stream(self, X: Iterable[str]) -> Iterator[tuple[str, np.ndarray | None]]:
        """Embed molecules from an iterator, yielding ``(smiles, embedding)`` in input order.

        The embedding is None for a molecule that could not be processed. Nothing
        larger than ``chunk_size`` molecules is held at once, so a file larger
        than memory can be read a line at a time.
        """
        if not self.__sklearn_is_fitted__():
            self.fit()
        return self._stream(X)

    def _stream(self, X: Iterable[str]) -> Iterator[tuple[str, np.ndarray | None]]:
        progress = None
        if self.show_progress:
            from tqdm import tqdm

            progress = tqdm(desc="ZAO", unit="chunk")
        chunk: list[str] = []
        try:
            for smi in X:
                chunk.append(smi)
                if len(chunk) >= self.chunk_size:
                    yield from self._emit(chunk, progress)
                    chunk = []
            if chunk:
                yield from self._emit(chunk, progress)
        finally:
            if progress is not None:
                progress.close()

    def _emit(self, chunk, progress=None):
        emb = self._embed_chunk(chunk)
        if progress is not None:
            progress.update(1)
        for smi, row in zip(chunk, emb, strict=True):
            yield smi, (None if np.isnan(row[0]) else row)

    def _embed_chunk(self, smiles: list[str]) -> np.ndarray:
        """Embeddings for one chunk, NaN rows for failures, in input order."""
        out = np.full((len(smiles), self.embedding_dim_), np.nan, dtype=np.float32)
        if not smiles:
            return out
        positions, unique_mols = self._deduplicate(smiles)
        if not unique_mols:
            return out
        conf_mols = self._conformers_for([mol for _, mol in unique_mols])
        groups = self._group_by_conformer_count(unique_mols, conf_mols)
        self._write_embeddings(groups, positions, out)
        return out

    def _deduplicate(self, smiles: list[str]) -> tuple[dict, list]:
        """Standardize and add hydrogens, keeping one molecule per standardized SMILES.

        Returns the input positions each standardized SMILES came from, and the
        molecules to embed. Whatever could not be standardized is left out.
        """
        n_workers = process_workers(len(smiles), self.n_jobs)
        if n_workers == 1 or len(smiles) < _PARALLEL_STANDARDIZE_MIN:
            prepared = [standardize_and_prepare(s) for s in smiles]
        else:
            prepared = Parallel(n_jobs=n_workers, backend="loky", batch_size=256)(
                delayed(standardize_and_prepare)(s) for s in smiles
            )
        positions: dict[str, list[int]] = {}
        unique_mols = []
        for i, (std, mol) in enumerate(prepared):
            if std is None or mol is None:
                continue
            if std not in positions:
                positions[std] = []
                unique_mols.append((std, mol))
            positions[std].append(i)
        return positions, unique_mols

    def _conformers_for(self, mols: list) -> list:
        return generate_conformers(
            mols,
            self.num_conformers,
            hardware_options=default_hardware_options(len(mols), self.conformer_gpu_ids),
            backend=self.conformer_backend,
        )

    def _group_by_conformer_count(self, unique_mols: list, conf_mols: list) -> dict:
        """Model inputs keyed by how many conformers they hold, since a batch holds one size."""
        groups: dict[int, tuple[list, list]] = {}
        for (std, _), mol in zip(unique_mols, conf_mols, strict=True):
            if mol is None:
                continue
            sample = mol_to_features(mol, self.max_atoms_)
            if sample is None:
                continue
            samples, keys = groups.setdefault(sample["coors"].shape[0], ([], []))
            samples.append(sample)
            keys.append(std)
        return groups

    def _write_embeddings(self, groups: dict, positions: dict, out: np.ndarray) -> None:
        """Run the forward passes and put each embedding on every row it came from."""
        for samples, keys in groups.values():
            for start in range(0, len(samples), self.batch_size):
                emb = self._forward(samples[start : start + self.batch_size])
                for std, row in zip(keys[start : start + self.batch_size], emb, strict=True):
                    out[positions[std]] = row

    def _ensure_model(self):
        if getattr(self, "_model", None) is not None:
            return
        import torch

        from ..model import load_model

        if self.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(self.device)
        self._device = device
        self._model = load_model(self._weights_path, device)

    def _forward(self, samples) -> np.ndarray:
        import torch

        self._ensure_model()
        batch = collate(samples)
        k = batch.pop("num_conformers")
        batch = {key: v.to(self._device, non_blocking=True) for key, v in batch.items()}
        use_amp = self.amp and _native_bf16(self._device)
        with (
            torch.no_grad(),
            torch.autocast(device_type=self._device.type, dtype=torch.bfloat16, enabled=use_amp),
        ):
            emb = self._model(
                batch["feats"],
                batch["coors"],
                batch["mask"],
                bond_types=batch["bond_types"],
                energy=batch["energy"],
                num_conformers=k,
            )
        return emb.float().cpu().numpy()

    def __getstate__(self):
        # The loaded model is rebuilt on demand rather than pickled.
        state = self.__dict__.copy()
        state.pop("_model", None)
        state.pop("_device", None)
        return state


def encode(smiles, **kwargs) -> np.ndarray:
    """SMILES in, ZAO embeddings out, in one call.

    Builds a ``ZAOTransformer`` and calls ``transform``.

    >>> from zao import encode
    >>> X = encode(["CC(=O)Oc1ccccc1C(=O)O", "c1ccc2[nH]ccc2c1"])  # doctest: +SKIP
    >>> X.shape                                                    # doctest: +SKIP
    (2, 2048)

    Parameters
    ----------
    smiles : list, array, Series or single-column DataFrame of SMILES
    **kwargs
        Any ``ZAOTransformer`` parameter.

    Returns
    -------
    ndarray of shape (n_samples, 2048)
        One row per molecule, NaN where the molecule could not be processed.

    Notes
    -----
    Each call loads the weights again, so a loop over batches should build one
    ``ZAOTransformer`` and call ``transform`` instead. More molecules than fit in
    memory at once go through ``ZAOTransformer.stream``.
    """
    return ZAOTransformer(**kwargs).transform(smiles)
