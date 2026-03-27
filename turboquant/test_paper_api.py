"""Smoke tests for the paper-first API surface."""

import torch

from turboquant import PaperTurboQuantKVCodec, Stage1MSEQuantizer, Stage2QJLResidual


def test_stage_objects_shapes():
    d = 16
    x = torch.randn(8, d)
    x = x / (torch.norm(x, dim=-1, keepdim=True) + 1e-8)

    s1 = Stage1MSEQuantizer(d=d, bits=2)
    c1 = s1.compress(x)
    assert c1.mse_indices.shape == x.shape
    assert c1.x_mse.shape == x.shape

    s2 = Stage2QJLResidual(d=d)
    c2 = s2.encode(x - c1.x_mse)
    assert c2.qjl_signs.shape == x.shape
    assert c2.residual_norm.shape == (x.shape[0],)


def test_paper_codec_roundtrip_shapes():
    d = 16
    codec = PaperTurboQuantKVCodec(d_key=d, d_value=d, bits=3)

    keys = torch.randn(32, d)
    vals = torch.randn(32, d)
    codec.append(keys, vals)

    q = torch.randn(32, d)
    scores = codec.attention_scores(q)
    values = codec.get_values()

    assert scores.shape == (32,)
    assert values.shape == vals.shape


if __name__ == "__main__":
    test_stage_objects_shapes()
    test_paper_codec_roundtrip_shapes()
    print("paper api smoke tests passed")
