# TurboQuant (PyTorch)

[中文](./README.md) | [English](./README_en.md)

TurboQuant is a PyTorch implementation for KV-cache compression in long-context LLM inference. The goal is to significantly reduce memory usage while preserving attention-score quality.

## 1. Quick Start

### Requirements

- Python 3.10+
- PyTorch 2.0+
- scipy
- (Optional) transformers / accelerate / bitsandbytes (only needed for real-model validation)

### Install dependencies

```bash
pip install -r requirements.txt
```

If you need CUDA PyTorch (example):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

### Validation commands (recommended order)

1) API smoke test

```bash
python -m turboquant.test_paper_api
```

2) Synthetic validation (CPU-friendly)

```bash
python -m turboquant.test_turboquant
```

3) Real-model validation (resource intensive)

```bash
python -m turboquant.validate
```

If CUDA is unavailable or model download fails:

```bash
python -m turboquant.validate --synthetic
```

## 2. Method Summary

TurboQuant compresses key vectors in two stages:

1. **Stage 1 (MSE)**: random orthogonal rotation + per-dimension Lloyd-Max quantization.
2. **Stage 2 (QJL)**: random projection on Stage-1 residuals, storing sign bits only (1-bit).

The inner-product estimator is:

$$
\langle q, k \rangle
\approx
\langle q, k_{\mathrm{mse}} \rangle
+
\lVert r \rVert \,\frac{\sqrt{\pi/2}}{m}\,\left\langle S q, \mathrm{sign}(S r) \right\rangle
$$

where：

- $k_{\mathrm{mse}}$: Stage 1 重建结果
- $r = k - k_{\mathrm{mse}}$
- $S$: 高斯随机矩阵
- $m$: Stage 2 投影维度

## 3. Results Snapshot (Synthetic)

Typical trends:

- higher bit width -> lower MSE
- QJL correction pushes inner-product bias close to zero
- 3-bit/4-bit usually provide a better quality-compression balance

Example Needle-in-Haystack results (d=128):

| Context | Bit | Compression | Score Cosine | Top-1 | Top-5 | Avg Needle Rank |
|---|---:|---:|---:|---:|---:|---:|
| 2065 tokens | 2-bit | 7.31x | 0.898703 | 26.0% | 56.4% | 489.0 |
| 2065 tokens | 3-bit | 5.02x | 0.957262 | 52.4% | 77.1% | 452.5 |
| 2065 tokens | 4-bit | 3.82x | 0.985677 | 69.6% | 88.5% | 412.3 |
| 4090 tokens | 2-bit | 7.31x | 0.895481 | 22.2% | 52.6% | 1204.2 |
| 4090 tokens | 3-bit | 5.02x | 0.956351 | 48.8% | 74.0% | 1105.3 |
| 4090 tokens | 4-bit | 3.82x | 0.985212 | 67.0% | 86.5% | 897.7 |
| 8221 tokens | 2-bit | 7.31x | 0.893167 | 20.1% | 45.8% | 2233.1 |
| 8221 tokens | 3-bit | 5.02x | 0.955408 | 42.7% | 72.7% | 1980.6 |
| 8221 tokens | 4-bit | 3.82x | 0.984929 | 60.2% | 85.2% | 1807.6 |

> Results vary with model, context length, random seed, and hardware. Treat local reruns as the source of truth.

## References

- TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate (ICLR 2026)
- QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero Overhead
- PolarQuant: Quantizing KV Caches with Polar Transformation

## License

MIT
