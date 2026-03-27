"""Tensor-oriented compressors with true bit-packing for KV-cache validation."""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch


# =========================
# Bit packing utilities
# =========================

def pack_bits(values: torch.Tensor, bits: int) -> Tuple[torch.Tensor, int]:
    """Pack non-negative integer values into a uint8 byte stream.

    Args:
        values: Tensor containing integers in [0, 2**bits).
        bits: Number of bits used by each value.

    Returns:
        (packed_bytes, n_values)
    """
    if bits < 1 or bits > 8:
        raise ValueError(f"bits must be in [1, 8], got {bits}")

    flat = values.reshape(-1).to(torch.int64)
    n_values = flat.numel()
    if n_values == 0:
        return torch.empty(0, dtype=torch.uint8, device=values.device), 0

    bit_positions = torch.arange(bits, device=flat.device, dtype=torch.int64)
    bit_slices = ((flat.unsqueeze(1) >> bit_positions) & 1).reshape(-1).to(torch.uint8)

    pad = (-bit_slices.numel()) % 8
    if pad:
        bit_slices = torch.cat([bit_slices, torch.zeros(pad, dtype=torch.uint8, device=flat.device)], dim=0)

    bit_slices = bit_slices.view(-1, 8)
    weights = (1 << torch.arange(8, device=flat.device, dtype=torch.int64)).to(torch.uint8)
    packed = (bit_slices * weights).sum(dim=1).to(torch.uint8)
    return packed, n_values


def unpack_bits(packed: torch.Tensor, bits: int, n_values: int) -> torch.Tensor:
    """Unpack uint8 byte stream into int64 tensor with n_values elements."""
    if bits < 1 or bits > 8:
        raise ValueError(f"bits must be in [1, 8], got {bits}")
    if n_values == 0:
        return torch.empty(0, dtype=torch.int64, device=packed.device)

    packed_u8 = packed.to(torch.uint8).reshape(-1)
    bit_positions = torch.arange(8, device=packed.device, dtype=torch.int64)
    all_bits = ((packed_u8.unsqueeze(1).to(torch.int64) >> bit_positions) & 1).reshape(-1)
    required = n_values * bits
    all_bits = all_bits[:required].view(n_values, bits)

    weights = (1 << torch.arange(bits, device=packed.device, dtype=torch.int64)).view(1, -1)
    return (all_bits.to(torch.int64) * weights).sum(dim=1)


def pack_signs(signs: torch.Tensor) -> Tuple[torch.Tensor, int]:
    """Pack sign tensor {-1,+1} into 1-bit stream (1 for +1, 0 for -1)."""
    bits = (signs.reshape(-1) > 0).to(torch.int64)
    return pack_bits(bits, bits=1)


def unpack_signs(packed: torch.Tensor, n_values: int) -> torch.Tensor:
    """Unpack 1-bit sign stream to {-1,+1} int8 tensor."""
    bits = unpack_bits(packed, bits=1, n_values=n_values)
    return (bits * 2 - 1).to(torch.int8)


# =========================
# TurboQuant compressors
# =========================

class TurboQuantCompressorV2:
    """Key compressor supporting asymmetric attention score estimation."""

    def __init__(self, head_dim: int, bits: int, seed: int, device: str = "cpu"):
        self.head_dim = head_dim
        self.bits = bits
        self.mse_bits = max(bits - 1, 1)
        self.device = device

        # Rotation matrix
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        g = torch.randn(head_dim, head_dim, generator=gen)
        q, r = torch.linalg.qr(g)
        diag_sign = torch.sign(torch.diag(r))
        diag_sign[diag_sign == 0] = 1.0
        self.Pi = (q * diag_sign.unsqueeze(0)).to(device)
        self.PiT = self.Pi.T.contiguous()

        # Lloyd-Max codebook
        self.centroids = self._solve_codebook(head_dim, self.mse_bits).to(device)

        # QJL matrix
        gen2 = torch.Generator(device="cpu")
        gen2.manual_seed(seed + 10000)
        self.S = torch.randn(head_dim, head_dim, generator=gen2).to(device)

    def _solve_codebook(self, d: int, bits: int) -> torch.Tensor:
        from scipy import integrate

        n_levels = 2 ** bits
        sigma = 1.0 / math.sqrt(d)

        def pdf(x):
            return (1.0 / math.sqrt(2 * math.pi * sigma ** 2)) * math.exp(-x * x / (2 * sigma ** 2))

        lo, hi = -3.5 * sigma, 3.5 * sigma
        centroids = [lo + (hi - lo) * (i + 0.5) / n_levels for i in range(n_levels)]

        for _ in range(200):
            boundaries = [(centroids[i] + centroids[i + 1]) / 2.0 for i in range(n_levels - 1)]
            edges = [lo * 3] + boundaries + [hi * 3]
            new_centroids = []
            for i in range(n_levels):
                a, b = edges[i], edges[i + 1]
                num, _ = integrate.quad(lambda x: x * pdf(x), a, b)
                den, _ = integrate.quad(pdf, a, b)
                new_centroids.append(num / den if den > 1e-15 else centroids[i])
            if max(abs(new_centroids[i] - centroids[i]) for i in range(n_levels)) < 1e-10:
                break
            centroids = new_centroids

        return torch.tensor(centroids, dtype=torch.float32)

    @torch.no_grad()
    def compress(self, states: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compress key states of shape (B, H_kv, S, D)."""
        bsz, n_heads, seq_len, dim = states.shape
        flat = states.reshape(-1, dim).float()

        vec_norms = torch.norm(flat, dim=-1, keepdim=True)
        flat_norm = flat / (vec_norms + 1e-8)

        rotated = flat_norm @ self.Pi.T
        mse_indices = (rotated.unsqueeze(-1) - self.centroids).abs().argmin(dim=-1).to(torch.int64)

        reconstructed_rot = self.centroids[mse_indices]
        k_mse = (reconstructed_rot @ self.Pi) * vec_norms

        residual = flat - k_mse
        residual_norm = torch.norm(residual, dim=-1)

        projected = residual @ self.S.T
        qjl_signs = (projected >= 0).to(torch.int8) * 2 - 1

        mse_packed, mse_count = pack_bits(mse_indices, bits=self.mse_bits)
        sign_packed, sign_count = pack_signs(qjl_signs)

        return {
            "mse_packed": mse_packed,
            "mse_count": torch.tensor(mse_count, device=states.device, dtype=torch.int64),
            "qjl_packed": sign_packed,
            "qjl_count": torch.tensor(sign_count, device=states.device, dtype=torch.int64),
            "vec_norms": vec_norms.squeeze(-1).to(torch.float16),
            "residual_norm": residual_norm.to(torch.float16),
            "shape": torch.tensor([bsz, n_heads, seq_len, dim], device=states.device, dtype=torch.int64),
        }

    @torch.no_grad()
    def _decode_k_mse(self, compressed: Dict[str, torch.Tensor]) -> torch.Tensor:
        bsz, n_heads, seq_len, dim = compressed["shape"].tolist()
        n_vecs = bsz * n_heads * seq_len

        mse_indices = unpack_bits(
            compressed["mse_packed"],
            bits=self.mse_bits,
            n_values=int(compressed["mse_count"].item()),
        ).view(n_vecs, dim)

        reconstructed = self.centroids[mse_indices.long()] @ self.Pi
        vec_norms = compressed["vec_norms"].float().unsqueeze(-1)
        return (reconstructed * vec_norms).view(bsz, n_heads, seq_len, dim)

    @torch.no_grad()
    def _decode_qjl_signs(self, compressed: Dict[str, torch.Tensor]) -> torch.Tensor:
        bsz, n_heads, seq_len, dim = compressed["shape"].tolist()
        signs = unpack_signs(compressed["qjl_packed"], int(compressed["qjl_count"].item()))
        return signs.view(bsz, n_heads, seq_len, dim).float()

    @torch.no_grad()
    def asymmetric_attention_scores(self, queries: torch.Tensor, compressed: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute <Q,K> estimate directly from compressed keys.

        queries shape:
          - (B, H_q, S_q, D), with either H_q == H_kv or H_q % H_kv == 0 (GQA case)
        """
        k_mse = self._decode_k_mse(compressed).float()
        signs = self._decode_qjl_signs(compressed).float()
        r_norm = compressed["residual_norm"].float().view(k_mse.shape[0], k_mse.shape[1], k_mse.shape[2])

        if queries.shape[1] != k_mse.shape[1]:
            if queries.shape[1] % k_mse.shape[1] != 0:
                raise ValueError(
                    f"Head mismatch: H_q={queries.shape[1]} not compatible with H_kv={k_mse.shape[1]}"
                )
            rep = queries.shape[1] // k_mse.shape[1]
            k_mse = k_mse.repeat_interleave(rep, dim=1)
            signs = signs.repeat_interleave(rep, dim=1)
            r_norm = r_norm.repeat_interleave(rep, dim=1)

        term1 = torch.matmul(queries.float(), k_mse.transpose(-2, -1))

        q_proj = torch.matmul(queries.float(), self.S.T)
        qjl_ip = torch.matmul(q_proj, signs.transpose(-2, -1))

        correction_scale = math.sqrt(math.pi / 2) / self.S.shape[0]
        term2 = correction_scale * qjl_ip * r_norm.unsqueeze(-2)
        return term1 + term2


class TurboQuantCompressorMSE:
    """MSE-only compressor for values with bit-packed indices."""

    def __init__(self, head_dim: int, bits: int, seed: int, device: str = "cpu"):
        self.head_dim = head_dim
        self.bits = bits
        self.device = device

        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        g = torch.randn(head_dim, head_dim, generator=gen)
        q, r = torch.linalg.qr(g)
        diag_sign = torch.sign(torch.diag(r))
        diag_sign[diag_sign == 0] = 1.0
        self.Pi = (q * diag_sign.unsqueeze(0)).to(device)
        self.centroids = self._solve_codebook(head_dim, bits).to(device)

    def _solve_codebook(self, d: int, bits: int) -> torch.Tensor:
        from scipy import integrate

        n_levels = 2 ** bits
        sigma = 1.0 / math.sqrt(d)

        def pdf(x):
            return (1.0 / math.sqrt(2 * math.pi * sigma ** 2)) * math.exp(-x * x / (2 * sigma ** 2))

        lo, hi = -3.5 * sigma, 3.5 * sigma
        centroids = [lo + (hi - lo) * (i + 0.5) / n_levels for i in range(n_levels)]

        for _ in range(200):
            boundaries = [(centroids[i] + centroids[i + 1]) / 2.0 for i in range(n_levels - 1)]
            edges = [lo * 3] + boundaries + [hi * 3]
            new_centroids = []
            for i in range(n_levels):
                a, b = edges[i], edges[i + 1]
                num, _ = integrate.quad(lambda x: x * pdf(x), a, b)
                den, _ = integrate.quad(pdf, a, b)
                new_centroids.append(num / den if den > 1e-15 else centroids[i])
            if max(abs(new_centroids[i] - centroids[i]) for i in range(n_levels)) < 1e-10:
                break
            centroids = new_centroids

        return torch.tensor(centroids, dtype=torch.float32)

    @torch.no_grad()
    def compress(self, states: torch.Tensor) -> Dict[str, torch.Tensor]:
        bsz, n_heads, seq_len, dim = states.shape
        flat = states.reshape(-1, dim).float()

        vec_norms = torch.norm(flat, dim=-1, keepdim=True)
        flat_norm = flat / (vec_norms + 1e-8)
        rotated = flat_norm @ self.Pi.T

        indices = (rotated.unsqueeze(-1) - self.centroids).abs().argmin(dim=-1).to(torch.int64)
        packed, n_values = pack_bits(indices, bits=self.bits)

        return {
            "packed_indices": packed,
            "index_count": torch.tensor(n_values, device=states.device, dtype=torch.int64),
            "vec_norms": vec_norms.squeeze(-1).to(torch.float16),
            "shape": torch.tensor([bsz, n_heads, seq_len, dim], device=states.device, dtype=torch.int64),
        }

    @torch.no_grad()
    def decompress(self, compressed: Dict[str, torch.Tensor]) -> torch.Tensor:
        bsz, n_heads, seq_len, dim = compressed["shape"].tolist()
        n_vecs = bsz * n_heads * seq_len

        indices = unpack_bits(
            compressed["packed_indices"],
            bits=self.bits,
            n_values=int(compressed["index_count"].item()),
        ).view(n_vecs, dim)

        reconstructed = self.centroids[indices.long()] @ self.Pi
        vec_norms = compressed["vec_norms"].float().unsqueeze(-1)
        return (reconstructed * vec_norms).view(bsz, n_heads, seq_len, dim)
