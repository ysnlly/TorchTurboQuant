"""Stage 2: QJL residual sign encoding and correction term."""

import math
from typing import Optional
import torch

from .types import Stage2Compressed


def generate_qjl_matrix(d: int, m: Optional[int] = None, seed: Optional[int] = None, device: str = "cpu") -> torch.Tensor:
    if m is None:
        m = d
    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(seed)
    return torch.randn(m, d, generator=gen).to(device)


class Stage2QJLResidual:
    """Encode residual r with sign(Sr), then estimate <q, r>."""

    def __init__(self, d: int, qjl_dim: Optional[int] = None, seed: int = 43, device: str = "cpu"):
        self.d = d
        self.qjl_dim = qjl_dim or d
        self.device = device
        self.S = generate_qjl_matrix(d, m=self.qjl_dim, seed=seed, device=device)

    def encode(self, residual: torch.Tensor) -> Stage2Compressed:
        projected = residual @ self.S.T
        signs = torch.sign(projected)
        signs[signs == 0] = 1.0
        residual_norm = torch.norm(residual, dim=-1)
        return Stage2Compressed(qjl_signs=signs, residual_norm=residual_norm)

    def correction(self, query: torch.Tensor, stage2: Stage2Compressed) -> torch.Tensor:
        query_proj = query @ self.S.T
        qjl_ip = (query_proj * stage2.qjl_signs).sum(dim=-1)
        scale = math.sqrt(math.pi / 2) / self.qjl_dim
        return stage2.residual_norm * scale * qjl_ip
