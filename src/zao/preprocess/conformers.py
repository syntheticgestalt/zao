"""Conformer generation: ETKDGv3 embedding followed by MMFF94 optimization.

Two backends: nvMolKit on the GPU, which is what ZAO's pretraining corpus was
generated with, and RDKit on the CPU, which is far slower.
"""

import logging
import warnings
from collections.abc import Sequence

from rdkit import Chem, rdBase
from rdkit.Chem import AllChem, rdDistGeom
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.rdDistGeom import ETKDGv3

from ..model.config import ZAO_CONFIG
from ..parallel import native_threads
from .standardize import standardize_smiles

logger = logging.getLogger(__name__)

MAX_ITERS = 1000
RANDOM_SEED = 42
_CPU_BACKEND_ANNOUNCED = False


def gpu_available() -> bool:
    """Whether nvMolKit is installed and a CUDA device is visible."""
    try:
        import nvmolkit.embedMolecules  # noqa: F401
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


def _warn_about_the_cpu_backend() -> None:
    """Say, once, that conformers are being generated on the CPU.

    ``pip install "zao[gpu]"`` installs nothing on a platform nvMolKit has no
    wheel for, so someone who asked for the GPU can end up here believing they
    did not.
    """
    global _CPU_BACKEND_ANNOUNCED
    if _CPU_BACKEND_ANNOUNCED:
        return
    _CPU_BACKEND_ANNOUNCED = True
    warnings.warn(
        "Generating conformers on the CPU with RDKit: nvMolKit and a visible CUDA device are "
        "what the GPU backend needs, and one of them is missing. The CPU backend is orders of "
        "magnitude slower. Pass conformer_backend='cpu' to say you meant it and silence this.",
        stacklevel=3,
    )


def default_hardware_options(n_mols: int, gpu_ids: Sequence[int] | None = None) -> dict:
    """nvMolKit settings scaled to the number of molecules in one call."""
    if n_mols <= 20:
        opts = {
            "batch_size": 100,
            "batches_per_gpu": 1,
            "preprocessing_threads": 32,
            "gpu_ids": [0],
        }
    elif n_mols <= 100:
        opts = {
            "batch_size": 500,
            "batches_per_gpu": 1,
            "preprocessing_threads": 64,
            "gpu_ids": [0],
        }
    else:
        opts = {
            "batch_size": 1250,
            "batches_per_gpu": 4,
            "preprocessing_threads": 192,
            "gpu_ids": [],
        }
    # nvMolKit's own CPU threads, bounded by what this process may use.
    opts["preprocessing_threads"] = native_threads(cap=opts["preprocessing_threads"])
    if gpu_ids is not None:
        opts["gpu_ids"] = list(gpu_ids)
    return opts


def _sort_by_energy(mol: Chem.Mol, energies: list[float]) -> Chem.Mol | None:
    """Copy of ``mol`` with conformers sorted by energy, each tagged with an ``Energy`` property."""
    if mol.GetNumConformers() != len(energies):
        return None
    pairs = sorted(zip(mol.GetConformers(), energies, strict=True), key=lambda x: x[1])
    out = Chem.Mol(mol)
    out.RemoveAllConformers()
    for conf, energy in pairs:
        new_conf = Chem.Conformer(conf)
        new_conf.SetProp("Energy", str(energy))
        out.AddConformer(new_conf, assignId=True)
    return out


def assign_mmff_charges(mol: Chem.Mol) -> None:
    """Store MMFF94 partial charges on each atom as ``_MMFFPartialCharge``.

    Does nothing when MMFF94 has no parameters for the molecule; ``mol_to_features``
    then reads a charge of 0 for every atom.
    """
    props = AllChem.MMFFGetMoleculeProperties(mol)
    if props:
        for atom in mol.GetAtoms():
            atom.SetDoubleProp("_MMFFPartialCharge", props.GetMMFFPartialCharge(atom.GetIdx()))


def _generate_one_cpu(
    mol: Chem.Mol, num_conformers: int, random_seed: int, threads: int
) -> Chem.Mol | None:
    """One molecule on the CPU, its conformers spread over ``threads`` C++ threads."""
    try:
        params = ETKDGv3()
        params.randomSeed = random_seed
        params.useRandomCoords = True
        params.maxIterations = MAX_ITERS
        params.numThreads = threads
        mol = Chem.Mol(mol)
        if not rdDistGeom.EmbedMultipleConfs(mol, numConfs=num_conformers, params=params):
            return None
        if AllChem.MMFFGetMoleculeProperties(mol) is None:
            return None
        results = AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=threads, maxIters=MAX_ITERS)
        out = _sort_by_energy(mol, [energy for _, energy in results])
        if out is not None:
            assign_mmff_charges(out)
        return out
    except Exception:
        return None


def _generate_cpu(
    mols: list[Chem.Mol], num_conformers: int, random_seed: int
) -> list[Chem.Mol | None]:
    """Every molecule on the CPU, one after another, parallel within a molecule."""
    # RDKit's own C++ threads, one per conformer at most.
    threads = native_threads(cap=num_conformers)
    logger.info(f"Generating conformers on the CPU for {len(mols)} molecules ({threads} threads)")
    return [_generate_one_cpu(m, num_conformers, random_seed, threads) for m in mols]


def generate_conformers(
    mols: list[Chem.Mol],
    num_conformers: int = ZAO_CONFIG["general"]["num_conformers"],
    hardware_options: dict | None = None,
    random_seed: int = RANDOM_SEED,
    backend: str = "auto",
) -> list[Chem.Mol | None]:
    """Embed and optimize ``num_conformers`` conformers per molecule.

    Parameters
    ----------
    mols : list of Mol
        Explicit-hydrogen molecules (see ``prepare_mol``). The GPU backend
        modifies them in place.
    num_conformers : int
        Conformers per molecule. A molecule for which fewer embed keeps the ones
        that did.
    hardware_options : dict, optional
        nvMolKit settings for the GPU backend (keys ``batch_size``,
        ``batches_per_gpu``, ``preprocessing_threads``, ``gpu_ids``). Defaults to
        ``default_hardware_options(len(mols))``.
    random_seed : int
        ETKDG random seed.
    backend : {"auto", "gpu", "cpu"}, default="auto"
        "auto" uses the GPU when nvMolKit and a CUDA device are both present and
        the CPU otherwise.

    Returns
    -------
    list of (Mol or None)
        One entry per input: a Mol whose conformers are sorted by MMFF94 energy
        and carry MMFF94 partial charges, or None when that molecule could not be
        processed (embedding gave no conformer, MMFF94 has no parameters for it,
        or an exception was raised). A conformer whose MMFF94 optimization has not
        converged is kept.

    Raises
    ------
    ValueError
        For an unknown ``backend``.
    RuntimeError
        If ``backend="gpu"`` but nvMolKit or a CUDA device is missing, or if
        nvMolKit returns an unexpected number of results.
    """
    if backend not in ("auto", "gpu", "cpu"):
        raise ValueError(f"backend must be 'auto', 'gpu' or 'cpu', got {backend!r}")
    if not mols:
        return []
    if backend == "auto":
        backend = "gpu" if gpu_available() else "cpu"
        if backend == "cpu":
            _warn_about_the_cpu_backend()
    if backend == "cpu":
        return _generate_cpu(mols, num_conformers, random_seed)
    if not gpu_available():
        raise RuntimeError(
            "backend='gpu' needs nvMolKit and a visible CUDA device; neither was found"
        )

    from nvmolkit.embedMolecules import EmbedMolecules
    from nvmolkit.mmffOptimization import MMFFOptimizeMoleculesConfs
    from nvmolkit.types import HardwareOptions

    opts = hardware_options or default_hardware_options(len(mols))
    hw = HardwareOptions(
        preprocessingThreads=opts["preprocessing_threads"],
        batchSize=opts["batch_size"],
        batchesPerGpu=opts["batches_per_gpu"],
        gpuIds=list(opts.get("gpu_ids") or []),
    )

    params = ETKDGv3()
    params.randomSeed = random_seed
    params.useRandomCoords = True

    EmbedMolecules(
        molecules=mols,
        params=params,
        confsPerMolecule=num_conformers,
        maxIterations=MAX_ITERS,
        hardwareOptions=hw,
    )
    energies_list = MMFFOptimizeMoleculesConfs(
        molecules=mols, maxIters=MAX_ITERS, hardwareOptions=hw
    )
    if not energies_list or len(energies_list) != len(mols):
        raise RuntimeError("nvMolKit MMFF optimization returned an unexpected number of results")

    out: list[Chem.Mol | None] = []
    for mol, energies in zip(mols, energies_list, strict=True):
        if mol.GetNumConformers() == 0 or not energies:
            out.append(None)
            continue
        sorted_mol = _sort_by_energy(mol, list(energies))
        if sorted_mol is not None:
            assign_mmff_charges(sorted_mol)
        out.append(sorted_mol)
    return out


def prepare_mol(smiles: str) -> Chem.Mol | None:
    """Turn a standardized SMILES into an explicit-hydrogen Mol ready for embedding.

    Returns None if the molecule fails cleanup or has no MMFF94 parameters.
    """
    with rdBase.BlockLogs():  # RDKit narrates Cleanup on every call
        try:
            mol = Chem.MolFromSmiles(smiles)
        except Exception:
            return None
        if mol is None:
            return None
        try:
            mol = rdMolStandardize.Cleanup(mol)
            Chem.SanitizeMol(mol)
            mol = Chem.AddHs(mol)
        except Exception:
            return None
        if AllChem.MMFFGetMoleculeProperties(mol) is None:
            return None
        return mol


def standardize_and_prepare(smiles: str) -> tuple[str | None, Chem.Mol | None]:
    """Worker entry point: (standardized SMILES, prepared Mol); either may be None."""
    std = standardize_smiles(smiles)
    if std is None:
        return None, None
    return std, prepare_mol(std)
