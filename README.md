# TurboQuant (PyTorch)

[中文](./README.md) | [English](./README_en.md)

一个基于 PyTorch 的 TurboQuant 实现：用于压缩 LLM 的 KV Cache，在尽量保持注意力分数质量的前提下，显著降低显存占用。

> 这份 README 按“先能用，再理解原理”的顺序组织：
> 1) 先讲怎么跑起来；
> 2) 再讲技术细节、实验结果和实现取舍。

英文文档见 [`README_en.md`](./README_en.md)。

---

## 1. 先用起来（Quick Start）

## 1.1 环境要求

- Python 3.10+
- PyTorch 2.0+
- scipy
- （可选）transformers / accelerate / bitsandbytes：仅用于真实模型验证脚本

安装依赖：

```bash
pip install -r requirements.txt
```

如果你要用 CUDA 版 PyTorch（示例）：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

---

## 1.2 30 秒最小用法（推荐从这里开始）

### 方案 A：paper-first API（更贴近 TurboQuant 论文思路）

```python
import torch
from turboquant import PaperTurboQuantKVCodec

# 例子：单头，seq=256，dim=128
keys = torch.randn(256, 128)
values = torch.randn(256, 128)
queries = torch.randn(256, 128)

codec = PaperTurboQuantKVCodec(d_key=128, d_value=128, bits=3, device="cpu")
codec.append(keys, values)

# 用压缩后的 key 直接估计注意力分数
scores = codec.attention_scores(queries)

# value 走重建路径
reconstructed_values = codec.get_values()

print(scores.shape, reconstructed_values.shape)
```

---

## 1.3 自测命令（建议按顺序）

### 1) 新 API 烟雾测试

```bash
python -m turboquant.test_paper_api
```

### 2) 合成数据完整验证（CPU 可跑）

```bash
python -m turboquant.test_turboquant
```

它会覆盖：
- Lloyd-Max 码本性质
- MSE 失真与理论上界对比
- QJL 内积无偏性检查
- Needle-in-Haystack 检索
- （有 CUDA 时）GPU benchmark

### 3) 真实模型验证（重）

```bash
python -m turboquant.validate
```

无 CUDA 或无法下载 HuggingFace 模型时，可直接跑 fallback：

```bash
python -m turboquant.validate --synthetic
```

若只想确认 API 限制说明：

```bash
python -m turboquant.validate --api
```

---

## 1.4 项目结构

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

## 2. 技术细节（How it works）

TurboQuant 的核心思想是“分两阶段处理 key 向量”：

1. **Stage 1（MSE）**：随机正交旋转 + 每维 Lloyd-Max 量化（低 bit）
2. **Stage 2（QJL）**：对 Stage 1 残差做随机投影，只存投影符号（1-bit）

最终对内积使用非对称估计器：

\[
\langle q, k \rangle \approx \langle q, k_{mse} \rangle + \|r\|\frac{\sqrt{\pi/2}}{m}\langle S q, \text{sign}(S r) \rangle
\]

其中：
- \(k_{mse}\) 是 Stage 1 的重建结果
- \(r = k - k_{mse}\)
- \(S\) 是高斯随机矩阵

### 为什么要这样做？

- 仅做 MSE 量化时，向量重建误差可能不小，内积会有系统偏差。
- QJL 残差项的目标不是“恢复原向量”，而是“修正内积估计偏差”。
- 对注意力来说，内积质量往往比逐向量重建质量更关键。

---

## 3. API 说明

当前公开 API 统一为 KVCodec / 分阶段风格：

- `Stage1MSEQuantizer`
- `Stage2QJLResidual`
- `PaperTurboQuantKVCodec`

`TurboQuantMSE` / `TurboQuantProd` / `TurboQuantKVCache` 仅保留在内部模块中，不再作为 package-level 公开入口。

---

## 4. 效果与实验结论（现有脚本可复现）

在合成向量测试中，典型现象是：

- bit 越高，MSE 越低；
- QJL 校正后，内积 bias 接近 0；
- 3-bit/4-bit 通常能在压缩比与分数保真间取得较好平衡。

以下是最新一组 Needle-in-Haystack 结果（来自本仓库验证截图，d=128）：

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

结论（与历史趋势一致）：
- 上下文越长，检索难度越高（Top-1/Top-5 下滑，Avg Needle Rank 上升）。
- 同一上下文下，bit 越高（2->3->4），分数余弦和检索指标越好。
- 3-bit 仍是较稳妥折中，4-bit 保真更高，2-bit 压缩最高但检索损失更明显。

具体数值会因模型、上下文长度、seed、硬件而变化，请以你本地跑出来的结果为准。

---

## 5. 常见问题（FAQ）

### Q1: 为什么 `validate.py` 跑不起来？
常见原因：
- 无法访问 HuggingFace（网络/代理问题）
- 没有 CUDA / bitsandbytes 环境
- 显存不足

当前脚本会在 real 模式失败后自动退化到 synthetic 模式；你也可以手动执行 `--synthetic`。

### Q2: 我应该用多少 bit？
经验上：
- 先从 3-bit 开始（通常是最稳妥折中）
- 追求极限压缩再试 2-bit
- 追求更稳分数可用 4-bit

### Q3: 可否直接用“云 API”做这套验证？
通常不行。因为本脚本需要拿到每层 hidden states、`q_proj` 路径和 KV cache 张量；通用 chat/completions API 不会暴露这些内部张量。

你可以：
- 本地/自托管 open-weight 模型（可拿内部张量）
- 或使用 `python -m turboquant.validate --synthetic` 先做离线算法校验

---

## 6. 参考资料

- TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate (ICLR 2026)
- QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero Overhead
- PolarQuant: Quantizing KV Caches with Polar Transformation

---

## License

MIT
