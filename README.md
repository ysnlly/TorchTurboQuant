# TurboQuant (PyTorch)

[中文](./README.md) | [English](./README_en.md)

TurboQuant 是一个基于 PyTorch 的 KV Cache 压缩实现，面向长上下文推理场景。目标是在显著降低显存占用的同时，尽量保持注意力分数质量。

## 1. 快速启动

### 环境要求

- Python 3.10+
- PyTorch 2.0+
- scipy
- （可选）transformers / accelerate / bitsandbytes（仅真实模型验证时需要）

### 安装依赖

```bash
pip install -r requirements.txt
```

如果需要 CUDA 版本 PyTorch（示例）：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

### 运行验证（建议顺序）

1) API 烟雾测试

```bash
python -m turboquant.test_paper_api
```

2) 合成数据验证（CPU 可跑）

```bash
python -m turboquant.test_turboquant
```

3) 真实模型验证（资源开销较高）

```bash
python -m turboquant.validate
```

无 CUDA 或模型下载失败时可使用：

```bash
python -m turboquant.validate --synthetic
```

## 2. 原理简介

TurboQuant 对 key 向量采用两阶段压缩：

1. **Stage 1（MSE）**：随机正交旋转 + 每维 Lloyd-Max 量化。
2. **Stage 2（QJL）**：对 Stage 1 残差做随机投影，仅存储投影符号（1-bit）。

内积估计器为：

\[
\langle q, k \rangle
\approx
\langle q, k_{\mathrm{mse}} \rangle
+
\|r\|\,\frac{\sqrt{\pi/2}}{m}\,\left\langle S q, \operatorname{sign}(S r) \right\rangle,
\]

其中：

- \(k_{\mathrm{mse}}\)：Stage 1 重建结果
- \(r = k - k_{\mathrm{mse}}\)
- \(S\)：高斯随机矩阵
- \(m\)：Stage 2 投影维度

## 3. 效果摘要（合成测试）

常见趋势：

- bit 数越高，MSE 越低。
- QJL 校正后，内积偏差接近 0。
- 3-bit/4-bit 通常在压缩率与分数保真之间更平衡。

示例结果（Needle-in-Haystack，d=128）：

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

> 具体数值会随模型、上下文长度、随机种子与硬件变化，请以本地复现实验为准。

## References

- TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate (ICLR 2026)
- QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero Overhead
- PolarQuant: Quantizing KV Caches with Polar Transformation

## License

MIT
