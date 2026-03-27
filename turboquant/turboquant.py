"""
TurboQuant core API.

This module keeps the legacy public classes (`TurboQuantMSE`, `TurboQuantProd`,
`TurboQuantKVCache`) but now implements them through the paper-first building
blocks (`stage1`, `stage2`, `estimator`) so the core path is no longer a
standalone duplicate implementation.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .lloyd_max import LloydMaxCodebook
from .stage1 import Stage1MSEQuantizer, generate_rotation_matrix as _generate_rotation_matrix
from .stage2 import Stage2QJLResidual, generate_qjl_matrix as _generate_qjl_matrix
from .types import Stage1Compressed, Stage2Compressed, KeyCompressed
from .estimator import estimate_inner_product


def generate_rotation_matrix(d: int, seed: Optional[int] = None, device: str = "cpu") -> torch.Tensor:
    """Backward-compatible wrapper for random rotation generation."""
    return _generate_rotation_matrix(d=d, seed=seed, device=device)


def generate_qjl_matrix(d: int, m: Optional[int] = None, seed: Optional[int] = None, device: str = "cpu") -> torch.Tensor:
    """Backward-compatible wrapper for QJL projection matrix generation."""
    return _generate_qjl_matrix(d=d, m=m, seed=seed, device=device)


class TurboQuantMSE(nn.Module):
    """Legacy Stage-1 API implemented via `Stage1MSEQuantizer`."""

    def __init__(self, d: int, bits: int, seed: int = 42, device: str = "cpu"):
        super().__init__()
        self.d = d
        self.bits = bits
        self.device = device

        self.stage1 = Stage1MSEQuantizer(d=d, bits=bits, seed=seed, device=device)
        self.codebook = LloydMaxCodebook(d, bits)

        # Keep legacy attributes for compatibility with existing callers.
        self.register_buffer("Pi", self.stage1.Pi)
        self.register_buffer("centroids", self.stage1.centroids)
        self.register_buffer("boundaries", self.codebook.boundaries.to(device))

    def rotate(self, x: torch.Tensor) -> torch.Tensor:
        return self.stage1.rotate(x)

    def unrotate(self, y: torch.Tensor) -> torch.Tensor:
        return self.stage1.unrotate(y)

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        return self.stage1.quantize_indices(x)

    def dequantize(self, indices: torch.Tensor) -> torch.Tensor:
        return self.stage1.reconstruct(indices)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        indices = self.quantize(x)
        x_hat = self.dequantize(indices)
        return x_hat, indices


class TurboQuantProd(nn.Module):
    """
    Legacy Stage1+Stage2 API implemented via paper-first modules.

    Storage format remains backward-compatible:
      - mse_indices
      - qjl_signs
      - residual_norm
    """

    def __init__(self, d: int, bits: int, qjl_dim: Optional[int] = None, seed: int = 42, device: str = "cpu"):
        super().__init__()
        self.d = d
        self.bits = bits
        self.mse_bits = max(bits - 1, 1)
        self.qjl_dim = qjl_dim or d
        self.device = device

        self.mse = TurboQuantMSE(d=d, bits=self.mse_bits, seed=seed, device=device)
        self.stage2 = Stage2QJLResidual(d=d, qjl_dim=self.qjl_dim, seed=seed + 1, device=device)

        # Legacy compatibility: expose S as a buffer-like tensor attr.
        self.register_buffer("S", self.stage2.S)

    def quantize(self, x: torch.Tensor) -> dict:
        x_hat, mse_indices = self.mse(x)
        residual = x - x_hat
        stage2_out = self.stage2.encode(residual)
        return {
            "mse_indices": mse_indices,
            "qjl_signs": stage2_out.qjl_signs,
            "residual_norm": stage2_out.residual_norm,
        }

    def dequantize(self, compressed: dict) -> torch.Tensor:
        return self.mse.dequantize(compressed["mse_indices"])

    def inner_product(self, y: torch.Tensor, compressed: dict) -> torch.Tensor:
        stage1_out = Stage1Compressed(
            mse_indices=compressed["mse_indices"],
            x_mse=self.mse.dequantize(compressed["mse_indices"]),
        )
        stage2_out = Stage2Compressed(
            qjl_signs=compressed["qjl_signs"],
            residual_norm=compressed["residual_norm"],
        )
        key = KeyCompressed(stage1=stage1_out, stage2=stage2_out)
        return estimate_inner_product(query=y, key=key, stage2=self.stage2)

    def forward(self, x: torch.Tensor) -> dict:
        return self.quantize(x)


class TurboQuantKVCache:
    """KV cache wrapper that preserves legacy behavior and interfaces."""

    def __init__(self, d_key: int, d_value: int, bits: int = 3, seed: int = 42, device: str = "cpu"):
        self.d_key = d_key
        self.d_value = d_value
        self.bits = bits
        self.device = device

        self.key_quantizer = TurboQuantProd(d_key, bits, seed=seed, device=device)
        self.value_quantizer = TurboQuantMSE(d_value, bits, seed=seed + 100, device=device)

        self.key_cache = []
        self.value_cache = []

    def append(self, keys: torch.Tensor, values: torch.Tensor):
        orig_shape = keys.shape
        flat_keys = keys.reshape(-1, self.d_key)
        flat_values = values.reshape(-1, self.d_value)

        compressed_keys = self.key_quantizer.quantize(flat_keys)
        value_indices = self.value_quantizer.quantize(flat_values)

        self.key_cache.append({
            "mse_indices": compressed_keys["mse_indices"],
            "qjl_signs": compressed_keys["qjl_signs"],
            "residual_norm": compressed_keys["residual_norm"],
            "shape": orig_shape,
        })
        self.value_cache.append({
            "indices": value_indices,
            "shape": values.shape,
        })

    def attention_scores(self, queries: torch.Tensor) -> torch.Tensor:
        scores = []
        for cached in self.key_cache:
            s = self.key_quantizer.inner_product(queries, cached)
            scores.append(s)
        return torch.cat(scores, dim=-1) if scores else torch.tensor([])

    def get_values(self) -> torch.Tensor:
        values = []
        for cached in self.value_cache:
            v = self.value_quantizer.dequantize(cached["indices"])
            values.append(v)
        return torch.cat(values, dim=0) if values else torch.tensor([])

    def memory_usage_bits(self) -> dict:
        n_keys = sum(c["mse_indices"].numel() for c in self.key_cache) if self.key_cache else 0
        n_qjl = sum(c["qjl_signs"].numel() for c in self.key_cache) if self.key_cache else 0
        n_norms = sum(c["residual_norm"].numel() for c in self.key_cache) if self.key_cache else 0
        n_values = sum(c["indices"].numel() for c in self.value_cache) if self.value_cache else 0

        key_bits = n_keys * self.key_quantizer.mse_bits + n_qjl * 1 + n_norms * 16
        value_bits = n_values * self.bits
        fp16_equivalent = (n_keys + n_values) * 16

        total_bits = key_bits + value_bits
        return {
            "key_bits": key_bits,
            "value_bits": value_bits,
            "total_bits": total_bits,
            "fp16_bits": fp16_equivalent,
            "compression_ratio": fp16_equivalent / total_bits if total_bits > 0 else 0,
        }

    def __len__(self):
        return sum(c["mse_indices"].shape[0] for c in self.key_cache) if self.key_cache else 0
