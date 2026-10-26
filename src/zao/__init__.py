"""ZAO: a molecular foundation model that reads each molecule as a set of conformers."""

from importlib.metadata import PackageNotFoundError, version

from .predictors import ZAOClassifier, ZAORegressor, default_features
from .preprocess import standardize_smiles
from .scaffold import scaffold_groups
from .transformers import (
    AtomPairTransformer,
    AvalonTransformer,
    ErGTransformer,
    MACCSTransformer,
    MordredTransformer,
    MorganTransformer,
    RDKitDescriptorTransformer,
    TopologicalTorsionTransformer,
    ZAOTransformer,
    encode,
)

try:
    __version__ = version("zao")
except PackageNotFoundError:  # running from a checkout that was never installed
    __version__ = "0.0.0"

__all__ = [
    "AtomPairTransformer",
    "AvalonTransformer",
    "ErGTransformer",
    "MACCSTransformer",
    "MordredTransformer",
    "MorganTransformer",
    "RDKitDescriptorTransformer",
    "TopologicalTorsionTransformer",
    "ZAOClassifier",
    "ZAORegressor",
    "ZAOTransformer",
    "default_features",
    "encode",
    "scaffold_groups",
    "standardize_smiles",
]
