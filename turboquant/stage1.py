"""Stage 1: random rotation + Lloyd-Max quantization."""

from typing import Optional
import torch

from .lloyd_max import LloydMaxCodebook
from .types import Stage1Compressed


def generate_rotation_matrix(d: int, seed: Optional[int] = None, device: str = "cpu") -> torch.Tensor:
    """Generate a Haar-distributed orthogonal matrix via QR decomposition."""
    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(seed)
    g = torch.randn(d, d, generator=gen)
    q, r = torch.linalg.qr(g)
    diag_sign = torch.sign(torch.diag(r))
    diag_sign[diag_sign == 0] = 1.0
    q = q * diag_sign.unsqueeze(0)
    return q.to(device)


class Stage1MSEQuantizer:
    """Paper-aligned Stage 1 quantizer for unit vectors."""

    def __init__(self, d: int, bits: int, seed: int = 42, device: str = "cpu"):
        self.d = d
        self.bits = bits
        self.device = device
        self.Pi = generate_rotation_matrix(d, seed=seed, device=device)
        codebook = LloydMaxCodebook(d, bits)
        self.centroids = codebook.centroids.to(device)

    def rotate(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.Pi.T

    def unrotate(self, y: torch.Tensor) -> torch.Tensor:
        return y @ self.Pi

    def quantize_indices(self, x: torch.Tensor) -> torch.Tensor:
        y = self.rotate(x)
        diffs = y.unsqueeze(-1) - self.centroids
        return diffs.abs().argmin(dim=-1)

    def reconstruct(self, indices: torch.Tensor) -> torch.Tensor:
        y_hat = self.centroids[indices]
        return self.unrotate(y_hat)

    def compress(self, x: torch.Tensor) -> Stage1Compressed:
        idx = self.quantize_indices(x)
        return Stage1Compressed(mse_indices=idx, x_mse=self.reconstruct(idx))
