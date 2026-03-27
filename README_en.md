# TurboQuant (PyTorch)

A PyTorch implementation of TurboQuant for compressing LLM KV caches while preserving attention-score quality as much as possible.

> This README is organized in a usage-first order:
> 1) get it running first,
> 2) then dive into technical details and results.

For Chinese documentation, see [`README.md`](./README.md).

---

## 1. Quick Start

## 1.1 Requirements

- Python 3.10+
- PyTorch 2.0+
- scipy
- (Optional) transformers / accelerate / bitsandbytes: only required for real-model validation

Install dependencies:

```bash
pip install -r requirements.txt
```

If you need CUDA PyTorch (example):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

---

## 1.2 Minimal Usage (30 seconds)

### Option A: Paper-first API (recommended)

```python
import torch
from turboquant import PaperTurboQuantKVCodec

# Example: single-head, seq=256, dim=128
keys = torch.randn(256, 128)
values = torch.randn(256, 128)
queries = torch.randn(256, 128)

codec = PaperTurboQuantKVCodec(d_key=128, d_value=128, bits=3, device="cpu")
codec.append(keys, values)

# Estimate attention scores directly from compressed keys
scores = codec.attention_scores(queries)

# Values use the reconstruction path
reconstructed_values = codec.get_values()

print(scores.shape, reconstructed_values.shape)
```

---

## 1.3 Validation Commands (recommended order)

### 1) Smoke test for the new API

```bash
python -m turboquant.test_paper_api
```

### 2) Full synthetic validation (CPU-friendly)

```bash
python -m turboquant.test_turboquant
```

This includes:
- Lloyd-Max codebook properties
- MSE distortion vs. theoretical upper bounds
- QJL inner-product unbiasedness checks
- Needle-in-Haystack retrieval
- GPU benchmark (if CUDA is available)

### 3) Real-model validation (heavy)

```bash
python -m turboquant.validate
```

If CUDA is unavailable or HuggingFace download fails, run the fallback mode:

```bash
python -m turboquant.validate --synthetic
```

If you only want API limitation notes:

```bash
python -m turboquant.validate --api
```

---

## 1.4 Project Layout

```text
requirements.txt        # Python dependencies

turboquant/
  __init__.py           # Package exports
  lloyd_max.py          # Lloyd-Max codebook solver
  turboquant.py         # Internal legacy module (not exported as public API)
  compressors.py        # Tensor-oriented compressors with true bit-packing

  # Paper-first API
  stage1.py             # Stage 1: rotation + Lloyd-Max
  stage2.py             # Stage 2: QJL residual signs + correction
  estimator.py          # <q, k> estimator
  kv_codec.py           # PaperTurboQuantKVCodec
  types.py              # Typed compressed representations

  # Tests / scripts
  test_paper_api.py
  test_turboquant.py
  validate.py
```

---

## 2. Technical Details

TurboQuant handles key vectors in two stages:

1. **Stage 1 (MSE):** random orthogonal rotation + per-dimension Lloyd-Max quantization
2. **Stage 2 (QJL):** random projection of Stage-1 residuals, storing only sign bits (1-bit)

The asymmetric inner-product estimator is:

\[
\langle q, k \rangle \approx \langle q, k_{mse} \rangle + \|r\|\frac{\sqrt{\pi/2}}{m}\langle S q, \text{sign}(S r) \rangle
\]

Where:
- \(k_{mse}\): Stage-1 reconstruction
- \(r = k - k_{mse}\)
- \(S\): Gaussian random matrix

### Why this design?

- MSE-only quantization can leave systematic inner-product bias.
- The QJL residual term is not trying to recover vectors perfectly; it targets inner-product correction.
- For attention, preserving inner-product structure is usually more important than per-vector reconstruction fidelity.

---

## 3. API Surface

The public package API is unified around KVCodec / staged components:

- `Stage1MSEQuantizer`
- `Stage2QJLResidual`
- `PaperTurboQuantKVCodec`

`TurboQuantMSE` / `TurboQuantProd` / `TurboQuantKVCache` remain internal-only and are no longer exported at package level.

---

## 4. Results and Observations

Typical behavior on synthetic tests:

- higher bits -> lower MSE
- QJL correction drives inner-product bias close to zero
- 3-bit/4-bit usually provide a good compression vs. quality trade-off

From earlier reference runs (d=128):
- 4-bit: ~3.8x compression
- 3-bit: ~5.0x compression
- 2-bit: ~7.3x compression

Exact values can vary with model, context length, seed, and hardware. Treat local reruns as ground truth.

---

## 5. FAQ

### Q1: Why does `validate.py` fail sometimes?
Common causes:
- no HuggingFace/network access
- missing CUDA / bitsandbytes
- insufficient VRAM

Now the script auto-falls back to synthetic mode when real-model validation fails (unless `--no-fallback` is set).

### Q2: Which bit width should I start with?
Practical default:
- start with 3-bit
- try 2-bit for more aggressive compression
- use 4-bit for safer score fidelity

### Q3: Can I run this validation via hosted model APIs?
Usually no. This validation needs internal tensors (per-layer hidden states, `q_proj`, KV cache), which generic chat/completions APIs do not expose.

Practical options:
- run local/self-hosted open-weight models with internal-tensor access
- or run `python -m turboquant.validate --synthetic` for offline algorithm checks

---

## 6. References

- TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate (ICLR 2026)
- QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero Overhead
- PolarQuant: Quantizing KV Caches with Polar Transformation

---

## License

MIT
