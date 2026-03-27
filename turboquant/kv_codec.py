"""Paper-first KV codec API with explicit stage boundaries."""

from typing import List
import torch

from .stage1 import Stage1MSEQuantizer
from .stage2 import Stage2QJLResidual
from .types import KeyCompressed, ValueCompressed
from .estimator import estimate_inner_product


class PaperTurboQuantKVCodec:
    """KV cache codec that mirrors TurboQuant's two-stage idea explicitly."""

    def __init__(self, d_key: int, d_value: int, bits: int = 3, seed: int = 42, device: str = "cpu"):
        self.d_key = d_key
        self.d_value = d_value
        self.bits = bits
        self.device = device

        self.key_stage1 = Stage1MSEQuantizer(d_key, max(bits - 1, 1), seed=seed, device=device)
        self.key_stage2 = Stage2QJLResidual(d_key, qjl_dim=d_key, seed=seed + 1, device=device)
        self.value_stage1 = Stage1MSEQuantizer(d_value, bits, seed=seed + 100, device=device)

        self._keys: List[KeyCompressed] = []
        self._values: List[ValueCompressed] = []

    def append(self, keys: torch.Tensor, values: torch.Tensor):
        flat_keys = keys.reshape(-1, self.d_key)
        flat_values = values.reshape(-1, self.d_value)

        stage1 = self.key_stage1.compress(flat_keys)
        residual = flat_keys - stage1.x_mse
        stage2 = self.key_stage2.encode(residual)

        self._keys.append(KeyCompressed(stage1=stage1, stage2=stage2))

        value_indices = self.value_stage1.quantize_indices(flat_values)
        self._values.append(ValueCompressed(indices=value_indices))

    def attention_scores(self, query: torch.Tensor) -> torch.Tensor:
        scores = [estimate_inner_product(query, key, self.key_stage2) for key in self._keys]
        return torch.cat(scores, dim=-1) if scores else torch.tensor([], device=query.device)

    def get_values(self) -> torch.Tensor:
        values = [self.value_stage1.reconstruct(v.indices) for v in self._values]
        if not values:
            return torch.tensor([])
        return torch.cat(values, dim=0)
