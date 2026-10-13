"""Model utilities for ArcEFN."""
import torch.nn as nn


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model.

    Parameters
    ----------
    model : nn.Module
        PyTorch model.

    Returns
    -------
    int
        Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)