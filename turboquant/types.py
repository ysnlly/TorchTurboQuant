"""Typed containers for paper-aligned TurboQuant stages."""

from dataclasses import dataclass
import torch


@dataclass
class Stage1Compressed:
    """Output of Stage 1 (MSE) quantization."""

    mse_indices: torch.Tensor
    x_mse: torch.Tensor


@dataclass
class Stage2Compressed:
    """Output of Stage 2 (QJL) residual encoding."""

    qjl_signs: torch.Tensor
    residual_norm: torch.Tensor


@dataclass
class KeyCompressed:
    """Compressed key representation used by asymmetric estimator."""

    stage1: Stage1Compressed
    stage2: Stage2Compressed


@dataclass
class ValueCompressed:
    """Compressed value representation used for MSE reconstruction."""

    indices: torch.Tensor
