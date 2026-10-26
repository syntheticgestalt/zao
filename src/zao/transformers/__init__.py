"""scikit-learn transformers from SMILES to features.

Every class here takes SMILES and returns a float array, so they are
interchangeable, combinable through ``FeatureUnion``, and usable as the first
step of a ``Pipeline``:

>>> from sklearn.pipeline import FeatureUnion
>>> from zao import ZAOTransformer, MorganTransformer, RDKitDescriptorTransformer
>>> features = FeatureUnion([
...     ("zao", ZAOTransformer()),
...     ("ecfp", MorganTransformer()),
...     ("desc", RDKitDescriptorTransformer()),
... ])

``ZAOTransformer``, in ``zao``, is the only one that needs PyTorch and the
pretrained weights; the classical descriptors in ``descriptors`` need neither.

Molecules that cannot be processed give a row of NaN rather than an error, and
individual descriptors that are undefined for a molecule are NaN too, so an
imputation step belongs between these transformers and any learner that cannot
read NaN itself.
"""

from .descriptors import (
    AtomPairTransformer,
    AvalonTransformer,
    ErGTransformer,
    MACCSTransformer,
    MordredTransformer,
    MorganTransformer,
    RDKitDescriptorTransformer,
    TopologicalTorsionTransformer,
)
from .zao import ZAOTransformer, encode

__all__ = [
    "AtomPairTransformer",
    "AvalonTransformer",
    "ErGTransformer",
    "MACCSTransformer",
    "MordredTransformer",
    "MorganTransformer",
    "RDKitDescriptorTransformer",
    "TopologicalTorsionTransformer",
    "ZAOTransformer",
    "encode",
]
