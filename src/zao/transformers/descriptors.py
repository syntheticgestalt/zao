"""The classical molecular descriptors, as scikit-learn transformers."""

import numpy as np
from joblib import Parallel, delayed
from rdkit import Chem, rdBase
from sklearn.base import BaseEstimator, TransformerMixin

from .._input import as_smiles_list
from ..parallel import process_workers
from ..preprocess.standardize import standardized_mol


class _Descriptor(TransformerMixin, BaseEstimator):
    """Shared machinery: parse, optionally standardize, featurize, collect.

    A subclass provides ``feature_names_`` and ``_values(mol)``. Whatever cannot
    be pickled is built on demand by ``_prepare`` and left out of the state.

    Parameters
    ----------
    standardize : bool, default=True
        Keep RDKit's parent fragment and neutralize it, the way ZAO does, so that
        a salt and its free base give the same features. ZAO's molecular weight
        window is not applied.
    n_jobs : int, default=-1
        Worker processes. -1 uses every CPU this process may use; 1 keeps
        everything in this process.
    """

    def __init__(self, standardize: bool = True, n_jobs: int = -1):
        self.standardize = standardize
        self.n_jobs = n_jobs

    def _prepare(self):
        """Build this featurizer's RDKit object, once per process."""
        if getattr(self, "_obj", None) is None:
            self._obj = self._build()

    def _build(self):
        """The RDKit object ``_values`` reads, or None for a featurizer that needs none."""
        return None

    def _values(self, mol) -> np.ndarray:
        raise NotImplementedError

    def fit(self, X=None, y=None):
        """Nothing is learned; this records the output width."""
        self.n_features_out_ = len(self.feature_names_)
        return self

    def __sklearn_is_fitted__(self):
        return hasattr(self, "n_features_out_")

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_, dtype=object)

    def transform(self, X) -> np.ndarray:
        """Features for each molecule; a row is NaN where the SMILES cannot be read."""
        if not self.__sklearn_is_fitted__():
            self.fit()
        smiles = as_smiles_list(X)
        if not smiles:
            return np.empty((0, self.n_features_out_), dtype=np.float32)
        rows = Parallel(
            n_jobs=process_workers(len(smiles), self.n_jobs), backend="loky", batch_size="auto"
        )(delayed(self._one)(s) for s in smiles)
        return np.vstack(rows)

    def _one(self, smiles: str) -> np.ndarray:
        self._prepare()
        blank = np.full((1, self.n_features_out_), np.nan, dtype=np.float32)
        try:
            with rdBase.BlockLogs():  # scoped, so the caller keeps RDKit's messages
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    return blank
                if self.standardize:
                    mol = standardized_mol(mol)
                values = np.asarray(self._values(mol), dtype=np.float32).reshape(1, -1)
        except Exception:
            return blank
        # inf comes from descriptors that overflow; it is a missing value here.
        return np.where(np.isfinite(values), values, np.nan)

    def __getstate__(self):
        # Rebuilt by _prepare in whichever process needs it.
        return {k: v for k, v in self.__dict__.items() if k != "_obj"}


class _GeneratorFP(_Descriptor):
    """A fingerprint from one of RDKit's fingerprint generators."""

    def _values(self, mol) -> np.ndarray:
        if self.counts:
            return self._obj.GetCountFingerprintAsNumPy(mol)
        return self._obj.GetFingerprintAsNumPy(mol)

    @property
    def feature_names_(self):
        return [f"{self.prefix}{i}" for i in range(self.n_bits)]


class MorganTransformer(_GeneratorFP):
    """Morgan (circular) fingerprint; radius 2 is what the literature calls ECFP4.

    Parameters
    ----------
    radius : int, default=2
        How many bonds outward from each atom to look.
    n_bits : int, default=1024
        Number of columns.
    counts : bool, default=True
        Count how often each substructure occurs rather than only whether it
        occurs. Gradient-boosted trees can use the counts; set it to False for a
        plain presence/absence fingerprint.
    features : bool, default=False
        Describe each atom by the role it plays (donor, acceptor, aromatic,
        halogen, basic, acidic) instead of by its element: the fingerprint known
        as FCFP.
    """

    def __init__(
        self,
        radius: int = 2,
        n_bits: int = 1024,
        counts: bool = True,
        features: bool = False,
        standardize: bool = True,
        n_jobs: int = -1,
    ):
        super().__init__(standardize=standardize, n_jobs=n_jobs)
        self.radius = radius
        self.n_bits = n_bits
        self.counts = counts
        self.features = features

    def _build(self):
        from rdkit.Chem import rdFingerprintGenerator as rfg

        invariants = rfg.GetMorganFeatureAtomInvGen() if self.features else None
        return rfg.GetMorganGenerator(
            radius=self.radius, fpSize=self.n_bits, atomInvariantsGenerator=invariants
        )

    @property
    def prefix(self):
        return "fcfp" if self.features else "ecfp"


class AtomPairTransformer(_GeneratorFP):
    """Atom pair fingerprint (Carhart et al., 1985).

    Every pair of atoms, recorded with the number of bonds between them.
    """

    prefix = "ap"

    def __init__(
        self,
        n_bits: int = 2048,
        counts: bool = True,
        standardize: bool = True,
        n_jobs: int = -1,
    ):
        super().__init__(standardize=standardize, n_jobs=n_jobs)
        self.n_bits = n_bits
        self.counts = counts

    def _build(self):
        from rdkit.Chem import rdFingerprintGenerator as rfg

        return rfg.GetAtomPairGenerator(fpSize=self.n_bits)


class TopologicalTorsionTransformer(_GeneratorFP):
    """Topological torsion fingerprint (Nilakantan et al., 1987).

    Runs of four consecutive bonded heavy atoms. Despite the name this is a 2D
    fingerprint: the "torsion" is four atoms in a row, not a dihedral angle.
    """

    prefix = "tt"

    def __init__(
        self,
        n_bits: int = 1024,
        counts: bool = True,
        standardize: bool = True,
        n_jobs: int = -1,
    ):
        super().__init__(standardize=standardize, n_jobs=n_jobs)
        self.n_bits = n_bits
        self.counts = counts

    def _build(self):
        from rdkit.Chem import rdFingerprintGenerator as rfg

        return rfg.GetTopologicalTorsionGenerator(fpSize=self.n_bits)


class AvalonTransformer(_Descriptor):
    """Avalon fingerprint: a fixed catalogue of substructure queries."""

    def __init__(
        self,
        n_bits: int = 1024,
        counts: bool = True,
        standardize: bool = True,
        n_jobs: int = -1,
    ):
        super().__init__(standardize=standardize, n_jobs=n_jobs)
        self.n_bits = n_bits
        self.counts = counts

    @property
    def feature_names_(self):
        return [f"avalon{i}" for i in range(self.n_bits)]

    def _values(self, mol) -> np.ndarray:
        from rdkit.Avalon import pyAvalonTools

        if self.counts:
            return np.asarray(pyAvalonTools.GetAvalonCountFP(mol, nBits=self.n_bits).ToList())
        return np.asarray(pyAvalonTools.GetAvalonFP(mol, nBits=self.n_bits).ToList())


class MACCSTransformer(_Descriptor):
    """The 166 MACCS structural keys, in RDKit's 167-column vector.

    Each column asks one fixed yes/no question about the molecule.
    """

    @property
    def feature_names_(self):
        return [f"maccs{i}" for i in range(167)]

    def _values(self, mol) -> np.ndarray:
        from rdkit.Chem import MACCSkeys

        return np.asarray(MACCSkeys.GenMACCSKeys(mol))


class ErGTransformer(_Descriptor):
    """ErG: the extended reduced graph fingerprint (Stiefl et al., 2006).

    The molecule is collapsed to the features that matter for binding, and the
    fingerprint records which of them occur at which separation.
    """

    @property
    def feature_names_(self):
        return [f"erg{i}" for i in range(315)]

    def _values(self, mol) -> np.ndarray:
        from rdkit.Chem import rdReducedGraphs

        return rdReducedGraphs.GetErGFingerprint(mol)


class RDKitDescriptorTransformer(_Descriptor):
    """RDKit's 2D whole-molecule descriptors.

    Molecular weight, lipophilicity, counts of rotatable bonds and rings,
    surface area terms and so on. ``Ipc`` is excluded: it overflows to infinity
    on molecules with many rings.
    """

    @property
    def feature_names_(self):
        return [name for name, _ in self._descriptors()]

    @staticmethod
    def _descriptors():
        from rdkit.Chem import Descriptors

        return [(n, f) for n, f in Descriptors.descList if n != "Ipc"]

    def _build(self):
        return self._descriptors()

    def _values(self, mol) -> np.ndarray:
        out = np.empty(len(self._obj), dtype=np.float64)
        for i, (_, fn) in enumerate(self._obj):
            try:
                out[i] = fn(mol)
            except Exception:
                # One descriptor failing is not a reason to lose the molecule;
                # it becomes a missing value like any other.
                out[i] = np.nan
        return out


class MordredTransformer(_Descriptor):
    """Mordred's 2D descriptors: 1,613 quantities, from the ``mordredcommunity`` package.

    The slow block, roughly 50 ms per molecule, and a tenth of its columns are
    undefined for any given small molecule and come back as NaN. The 3D
    descriptors are excluded.
    """

    def _build(self):
        from mordred import Calculator, descriptors

        return Calculator(descriptors, ignore_3D=True)

    @property
    def feature_names_(self):
        self._prepare()
        return [str(d) for d in self._obj.descriptors]

    def _values(self, mol) -> np.ndarray:
        return np.asarray(self._obj(mol).fill_missing(np.nan), dtype=np.float64)
