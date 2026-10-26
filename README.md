# ZAO

[![PyPI](https://img.shields.io/pypi/v/zao)](https://pypi.org/project/zao/)
[![Python](https://img.shields.io/pypi/pyversions/zao)](https://pypi.org/project/zao/)
[![Tests](https://github.com/syntheticgestalt/zao/actions/workflows/tests.yml/badge.svg)](https://github.com/syntheticgestalt/zao/actions/workflows/tests.yml)
[![Weights](https://img.shields.io/badge/weights-Hugging%20Face-yellow)](https://huggingface.co/syntheticgestalt/zao)
[![Licence](https://img.shields.io/badge/licence-SG--TVAR--1.0-blue)](https://github.com/syntheticgestalt/zao/blob/main/LICENSE.md)

ZAO is a molecular foundation model that represents each molecule as a set of conformers. It encodes 10 conformers per molecule by default in a single forward pass and returns one 2,048-dimensional embedding. It was pretrained on 11 billion conformers, and the released checkpoint has 223 million parameters.

This repository provides:

- `ZAORegressor` and `ZAOClassifier`: ready-made models. SMILES and measurements in, predictions out, with the feature set and the tuning already decided.
- `encode` and `zao encode`: SMILES in, embeddings out, from Python or the command line.
- `ZAOTransformer`: the same embedding as a scikit-learn transformer, alongside the classical descriptors in `zao.transformers`.
- `scaffold_groups`: groups molecules by scaffold for scikit-learn's group splitters, so a model is scored on scaffolds it has never seen.

The pretrained weights are hosted on the Hugging Face Hub at [`syntheticgestalt/zao`](https://huggingface.co/syntheticgestalt/zao), one file of about 890 MB, downloaded and cached on first use. Bugs and questions go to [the issue tracker](https://github.com/syntheticgestalt/zao/issues).

## Installation

```bash
pip install zao            # CPU
pip install "zao[gpu]"     # with nvMolKit, for conformer generation on the GPU
```

Python 3.10 or newer. The GPU extra needs Linux x86_64, Python 3.11 to 3.14, and an NVIDIA GPU and driver that [nvMolKit](https://github.com/NVIDIA-BioNeMo/nvMolKit) supports; see its installation guide.

The weights are released under a licence you accept on the Hugging Face Hub, so sign in once, with an account that has accepted it, before the first download:

```bash
hf auth login
```

Conformers are generated on the GPU with nvMolKit when it is installed and a CUDA device is visible, and on the CPU with RDKit otherwise. The CPU backend is orders of magnitude slower; it is there to try the model out and to handle small sets.

## Quick start

### Get the embedding

```python
from zao import encode

X = encode(["CC(=O)Oc1ccccc1C(=O)O", "c1ccc2[nH]ccc2c1"])  # the weights are downloaded on first use
X.shape  # (2, 2048)
```

`encode` builds a `ZAOTransformer` and calls it, so anything that class takes can be passed to it (`device`, `batch_size`, `conformer_backend`, and the rest of the table [below](#zaotransformer)). Loading the weights takes a few seconds, so a loop over batches should keep one transformer instead:

```python
from zao import ZAOTransformer

zao = ZAOTransformer()
X = zao.transform(["CC(=O)Oc1ccccc1C(=O)O", "c1ccc2[nH]ccc2c1"])
```

Without writing any Python, for a file with one SMILES per line or a `.csv` with a `smiles` column:

```bash
zao encode library.smi -o embeddings.npy     # the matrix alone
zao encode library.csv -o embeddings.csv     # the SMILES and their 2,048 columns
```

ZAO is meant to be used frozen, as features. If you already have features for your molecules (fingerprints, descriptors), keep them and concatenate the ZAO embedding to them.

### Fit a model on your own measurements

```python
from zao import ZAORegressor

# train_smiles: a list of SMILES; train_y: the measured value of each
model = ZAORegressor().fit(train_smiles, train_y)
model.predict(test_smiles)
```

The default is expensive: it generates conformers for every molecule and then searches 100 settings for each of two learners, which runs on a GPU for minutes on hundreds of molecules and for hours on tens of thousands. `ZAORegressor(n_trials=10)` is the way to see it work first. What it builds is described under [ZAORegressor](#zaoregressor-and-zaoclassifier).

### Score on scaffolds the fit has never seen

A random split puts close analogues of a test molecule in the training set and flatters the model. `scaffold_groups` turns SMILES into the `groups` array that scikit-learn's group splitters take, so they all work unchanged:

```python
from sklearn.model_selection import GroupKFold

from zao import scaffold_groups

groups = scaffold_groups(smiles)
split = GroupKFold(n_splits=5).split(smiles, y, groups)
```

The scaffold definitions and the other splitters are [below](#scaffold-splits).

## ZAOTransformer

`ZAOTransformer.transform` is what `encode` and `zao encode` call. It runs these steps for each molecule:

1. Standardize the SMILES the way the pretraining corpus was standardized (`zao.standardize_smiles`), which keeps one parent fragment, neutralizes it and requires a molecular weight strictly between 100 and 1,000 Da.
2. Generate `num_conformers` conformers (10 by default) with ETKDGv3 and optimize them with MMFF94.
3. Pass them through the model in one forward pass.

`transform` returns an array of shape `(n, 2048)`. A row is NaN when its molecule cannot be processed: the SMILES does not parse, the molecular weight is outside the window, the molecule has more than 100 atoms including hydrogens, MMFF94 has no parameters for it, embedding yields no conformer at all, or a conformer's MMFF94 energy falls outside [-300, 3000] kcal/mol. A failure of the GPU backend as a whole raises an exception instead.

`stream` takes an iterator and yields `(smiles, embedding)` in input order, with `None` for failures, so a file larger than memory can be read a line at a time. Memory stays bounded by `chunk_size`.

```python
zao = ZAOTransformer()
for smiles, emb in zao.stream(line.strip() for line in open("library.smi")):
    ...
```

Main parameters:

| Parameter | Default | Meaning |
|---|---|---|
| `weights` | `"syntheticgestalt/zao"` | Hub repository id, or a local path to `model.safetensors` (or the directory holding it) |
| `num_conformers` | 10 | Conformers generated per molecule |
| `device` | `"auto"` | Device for the forward pass (`"cuda"`, `"cuda:1"`, `"cpu"`) |
| `batch_size` | 256 | Molecules per forward pass |
| `chunk_size` | 10000 | Molecules per conformer-generation call |
| `amp` | True | bfloat16 for the forward pass on GPUs that support it natively |
| `conformer_backend` | `"auto"` | `"gpu"`, `"cpu"`, or `"auto"` (GPU when nvMolKit and a CUDA device are present) |
| `n_jobs` | -1 | Worker processes for standardization; -1 uses every CPU this process may use |

nvMolKit's GPU conformer generation is not bit-for-bit reproducible between runs, so two runs on the same molecule give slightly different embeddings. When downstream models are compared, embed each molecule once and reuse the array.

## ZAORegressor and ZAOClassifier

`ZAORegressor` predicts a measured quantity (solubility, permeability, clearance, binding affinity as pIC50); `ZAOClassifier` predicts a label (active or not, toxic or not). Both take SMILES directly.

A support vector machine and LightGBM are each tuned with Optuna, `n_trials` settings per learner, and the two predictions are averaged. The one reads the whole scaled vector at once, the other splits on individual columns, and they fail on different molecules. The features are computed once, before the search, so every setting and every fold reuses that matrix.

Candidate settings are scored on scaffold folds ([why](#scaffold-splits)): the molecules are grouped by scaffold and each group is kept to one side of every fold. Fewer distinct scaffolds than folds makes this impossible and the fit refuses; lower `cv`, or pass your own splitter as `cv` if you accept that. To group by something else, an assay or a series, pass `fit(X, y, groups=...)`.

Main parameters:

| Parameter | Default | Meaning |
|---|---|---|
| `n_trials` | 100 | Settings tried per learner |
| `timeout` | None | Seconds after which a learner's search stops starting new trials |
| `cv` | 5 | Scaffold folds used to score each candidate setting, or a scikit-learn splitter |
| `scoring` | None | What the search maximizes; worth setting to `"roc_auc"` on imbalanced classification |
| `search_blocks` | True | Let the search choose which feature blocks each learner sees |
| `features` | `"default"` | The `FeatureUnion`. Pass your own to add or remove blocks |
| `cache_dir` | None | Where to cache the computed features |
| `zao` | True | False leaves the embedding out, which gives the baseline to compare against |

A molecule ZAO cannot embed ([the reasons](#zaotransformer)) keeps its other features: only the ZAO columns go missing, the imputation step fills them, and a warning says how many. A molecule RDKit cannot parse at all has no features either: it is left out of the fit, `predict` gives it NaN (None for a `ZAOClassifier` with text labels), and `score` leaves it out.

## Scaffold splits

Beyond the `GroupKFold` of the [quick start](#score-on-scaffolds-the-fit-has-never-seen), the same `groups` array goes to every other group splitter:

```python
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from zao import scaffold_groups

groups = scaffold_groups(smiles)

# Folds that also keep each fold's class balance (for imbalanced classification)
folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0).split(smiles, y, groups)

# A single train/test split
train_idx, test_idx = next(
    GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0).split(smiles, y, groups)
)
```

`LeaveOneGroupOut` and `LeavePGroupsOut` take the same array. `train_test_split` has no `groups` parameter, so a single split is the first split of a group splitter, as above. Compute the array once and pass it to each splitter.

Scoring a model on those folds is scikit-learn's own call, with `groups` passed through. The embedding is computed once and the folds index into it, so no molecule's conformers are generated twice:

```python
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_score

from zao import encode

X = encode(smiles)
scores = cross_val_score(
    HistGradientBoostingRegressor(), X, y, cv=GroupKFold(n_splits=5), groups=groups, scoring="r2"
)
```

`ZAORegressor` and `ZAOClassifier` make these folds themselves; see [above](#zaoregressor-and-zaoclassifier).

Two scaffold definitions are available through `scaffold=`:

| `scaffold` | Group key | Same group |
|---|---|---|
| `"generic"` (default) | Cyclic skeleton of the Bemis-Murcko framework: all atoms carbon, all bonds single | Benzene and pyridine; piperidine and cyclohexane; an amide linker and an all-carbon linker of the same length |
| `"murcko"` | Bemis-Murcko framework with elements and bond orders | Molecules that differ only in side chains, stereochemistry or isotopes |

In both, ring sizes, ring fusion and the number of atoms in each linker are kept, so adding one atom to a linker makes a new scaffold. Acyclic molecules form one group (the empty scaffold). The scaffold is taken from the molecule as ZAO standardizes it, so two inputs that ZAO reads as one molecule, such as a hydrochloride and its free base, always share a group.

The array holds the scaffolds themselves, so counting them shows how many distinct scaffolds the set has and how large the biggest one is:

```python
scaffolds, counts = np.unique(scaffold_groups(smiles), return_counts=True)
```

A scaffold cannot be divided, so one scaffold holding a large share of the data distorts any scaffold-grouped split. With `GroupShuffleSplit`, `test_size` counts scaffolds rather than molecules, so the test set's share of molecules can be far from it; `GroupKFold` balances its folds by molecule count.

There is one entry per input molecule; a SMILES that RDKit cannot parse raises `ValueError`.

## Examples

The scripts in [`examples/`](https://github.com/syntheticgestalt/zao/tree/main/examples) are not installed with the package, so take them from a clone. Each is one file that needs nothing but the package.

| Script | What it shows |
|---|---|
| `encode.py` | embeddings for a handful of molecules, and for a library streamed a molecule at a time |
| `transformer.py` | the embedding computed once, then an ordinary scikit-learn pipeline cross-validated on held-out scaffolds |
| `regressor.py` | `ZAORegressor` fitted on your measurements, scored on scaffolds it never saw, saved to disk |

The two that take a CSV want a column of SMILES and a column to predict: a numeric one for `ZAORegressor`, a label column for `ZAOClassifier`.

## Licence

ZAO's code and its pretrained weights are released under the SyntheticGestalt Technical Verification and Academic Research License (SG-TVAR-1.0); the full text is in [LICENSE.md](https://github.com/syntheticgestalt/zao/blob/main/LICENSE.md). It covers the weights, this package and its documentation, together with the embeddings and predictions they produce and anything built from those.

Anyone may evaluate the model and decide from the results whether to take a commercial licence:

- On your own proprietary structures and data as well as on public ones.
- Benchmarking against your internal datasets, comparison with other models, measuring speed and hardware use, fitting it into your workflow, ablations and method development, reproducing published results, internal demonstrations.
- On compounds chosen to assess the model, compared with experimental or simulation data you already hold.
- Keeping the model, the embeddings and anything built from them inside your organization.
- Publishing aggregated statistics, such as benchmark scores and error analyses, with commentary, as long as no individual output can be reconstructed from them.

An academic institution may also use the model for scientific research on molecules, free of the limits on which compounds it is run on and on testing what the outputs predict (the third bullet). The rest, including keeping everything inside your organization, applies to academic and non-academic users alike. Working with a non-academic party on Licensed Materials puts those activities back under the terms above.

Everything else needs a commercial licence: info@syntheticgestalt.com. Some uses are forbidden outright, among them designing chemical or biological agents intended to harm human or animal life, unlawful use, and falsified scientific or regulatory evidence.
