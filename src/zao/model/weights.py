"""Locating and loading the pretrained ZAO weights, one file: model.safetensors."""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

WEIGHTS_NAME = "model.safetensors"


def resolve_weights(weights: str, revision: str | None = None) -> Path:
    """Return the local path to ``model.safetensors``.

    Parameters
    ----------
    weights : str
        The file itself, a local directory holding it, or a Hugging Face Hub
        repository id such as ``"syntheticgestalt/zao"``.
    revision : str, optional
        Hub revision (branch, tag or commit) when downloading.
    """
    # Absolute, so the path survives a chdir or an unpickle elsewhere.
    path = Path(weights).expanduser().resolve()
    if path.is_file():
        return path
    if path.is_dir():
        weights_path = path / WEIGHTS_NAME
        if not weights_path.is_file():
            raise FileNotFoundError(f"{weights_path} not found")
        return weights_path

    from huggingface_hub import hf_hub_download

    logger.info(f"Downloading ZAO weights from the Hugging Face Hub: {weights}")
    return Path(hf_hub_download(weights, WEIGHTS_NAME, revision=revision))


def load_model(weights_path: Path, device):
    """Build a ``ZAOModel`` in eval mode on ``device`` with pretrained weights."""
    from safetensors.torch import load_file

    from .zao import ZAOModel

    model = ZAOModel()
    model.load_pretrained_state_dict(load_file(str(weights_path)))
    model = model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"ZAO model loaded on {device} ({n_params:,} parameters)")
    return model
