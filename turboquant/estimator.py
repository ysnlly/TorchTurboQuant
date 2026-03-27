"""Paper estimator: <q, k> ~= <q, k_mse> + qjl_correction."""

import torch

from .types import KeyCompressed
from .stage2 import Stage2QJLResidual


def estimate_inner_product(query: torch.Tensor, key: KeyCompressed, stage2: Stage2QJLResidual) -> torch.Tensor:
    """Estimate inner product with compressed key representation."""
    term1 = (query * key.stage1.x_mse).sum(dim=-1)
    term2 = stage2.correction(query, key.stage2)
    return term1 + term2
