"""Validation for TurboQuant compressors.

Supports:
1) real-model validation (CUDA + local HF model load), and
2) synthetic fallback validation (CPU-friendly) when CUDA/HF is unavailable.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Iterable

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from turboquant.compressors import TurboQuantCompressorMSE, TurboQuantCompressorV2

# Allow `python turboquant/validate.py`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
NEEDLE = "The secret project code name is AURORA-7749."
QUESTION = "What is the secret project code name?"

FILLER = """The quarterly financial review meeting covered several topics including
budget allocations for the upcoming fiscal year, departmental spending reports, and projected
revenue streams from various business units. The committee discussed infrastructure upgrades
planned for the western regional offices and noted that maintenance schedules should be
coordinated with the facilities management team. Several action items were assigned to team
leads for follow-up before the next meeting cycle.\n\n"""


def build_prompt(tokenizer, target_tokens: int = 2048, needle_pos: float = 0.5) -> str:
    filler_len = len(tokenizer.encode(FILLER))
    n_reps = max(1, target_tokens // filler_len)
    needle_idx = int(n_reps * needle_pos)

    parts = []
    for i in range(n_reps):
        if i == needle_idx:
            parts.append(f"\n--- Memo ---\n{NEEDLE}\n--- End ---\n\n")
        parts.append(FILLER)

    haystack = "".join(parts)
    return f"<|im_start|>user\n{haystack}\nQuestion: {QUESTION}<|im_end|>\n<|im_start|>assistant\n"


def find_subsequence_start(sequence, subseq) -> int | None:
    for i in range(len(sequence) - len(subseq) + 1):
        if sequence[i : i + len(subseq)] == subseq:
            return i
    return None


def find_needle_token_start(tokenizer, input_ids: list[int], needle_text: str) -> int | None:
    """Handle tokenizers where the target span is encoded with leading whitespace."""
    candidates = [
        needle_text,
        f" {needle_text}",
        needle_text.rstrip("."),
        f" {needle_text.rstrip('.')}",
    ]
    for candidate in candidates:
        needle_tokens = tokenizer.encode(candidate, add_special_tokens=False)
        start = find_subsequence_start(input_ids, needle_tokens)
        if start is not None:
            return start
    return None


def get_decoder_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise RuntimeError("Unsupported model structure: cannot find model.model.layers")


def get_attention_head_dim(model, layer) -> tuple[int, int]:
    """Infer query-head count and head dimension across transformer variants."""
    attn = layer.self_attn

    head_dim = getattr(attn, "head_dim", None)
    if head_dim is None:
        hidden_size = getattr(attn, "hidden_size", None) or getattr(model.config, "hidden_size", None)
        num_heads = getattr(attn, "num_heads", None) or getattr(model.config, "num_attention_heads", None)
        if hidden_size is None or num_heads is None:
            raise RuntimeError("Cannot infer attention head dimension")
        head_dim = hidden_size // num_heads

    num_q_heads = (
        getattr(attn, "num_heads", None)
        or getattr(attn, "num_attention_heads", None)
        or getattr(getattr(attn, "config", None), "num_attention_heads", None)
        or getattr(model.config, "num_attention_heads", None)
    )
    if num_q_heads is None:
        raise RuntimeError("Cannot infer attention head count")

    return num_q_heads, head_dim


def gather_last_token_queries(outputs, model, layer_idx: int) -> torch.Tensor:
    """Use real q_proj path to get last-token queries for one layer.

    Returns:
        queries: (B, H_q, 1, D)
    """
    layer = get_decoder_layers(model)[layer_idx]

    # For decoder-only models, hidden_states[layer_idx] is the input to layer_idx.
    layer_input = outputs.hidden_states[layer_idx][:, -1:, :]  # (B, 1, hidden_size)

    q = layer.self_attn.q_proj(layer_input)
    bsz = q.shape[0]
    num_q_heads, head_dim = get_attention_head_dim(model, layer)
    return q.view(bsz, 1, num_q_heads, head_dim).transpose(1, 2).contiguous()  # (B, H_q, 1, D)


def evaluate_one_layer(keys: torch.Tensor, values: torch.Tensor, query: torch.Tensor, bits: int, seed: int, device: str):
    _, h_kv, _, d = keys.shape
    key_comp = TurboQuantCompressorV2(d, bits, seed=seed + 7, device=device)
    val_comp = TurboQuantCompressorMSE(d, bits, seed=seed + 503, device=device)

    compressed_k = key_comp.compress(keys)
    compressed_v = val_comp.compress(values)

    total_compressed_bytes = 0
    total_compressed_bytes += compressed_k["mse_packed"].numel()
    total_compressed_bytes += compressed_k["qjl_packed"].numel()
    total_compressed_bytes += compressed_k["vec_norms"].numel() * 2
    total_compressed_bytes += compressed_k["residual_norm"].numel() * 2
    total_compressed_bytes += compressed_v["packed_indices"].numel()
    total_compressed_bytes += compressed_v["vec_norms"].numel() * 2

    total_uncompressed_bytes = (keys.numel() + values.numel()) * 2  # fp16 baseline

    h_q = query.shape[1]
    if h_q % h_kv != 0:
        raise RuntimeError(f"Incompatible heads: H_q={h_q}, H_kv={h_kv}")
    rep = h_q // h_kv
    keys_for_q = keys.repeat_interleave(rep, dim=1)

    real_scores = torch.matmul(query.float(), keys_for_q.float().transpose(-2, -1)).squeeze(-2)
    tq_scores = key_comp.asymmetric_attention_scores(query, compressed_k).squeeze(-2)
    return real_scores, tq_scores, total_compressed_bytes, total_uncompressed_bytes


def summarize_scores(real_scores: torch.Tensor, tq_scores: torch.Tensor, needle_start: int | None):
    h_q = real_scores.shape[1]
    top1_matches = 0
    top5_matches = 0
    needle_rank_sum = 0
    cosine_sims = []

    for h in range(h_q):
        rs = real_scores[0, h]
        ts = tq_scores[0, h]

        cosine_sims.append(F.cosine_similarity(rs.unsqueeze(0), ts.unsqueeze(0)).item())

        real_top1 = rs.argmax().item()
        tq_top1 = ts.argmax().item()
        if real_top1 == tq_top1:
            top1_matches += 1
        if real_top1 in ts.topk(5).indices.tolist():
            top5_matches += 1

        if needle_start is not None:
            needle_rank = (ts.argsort(descending=True) == needle_start).nonzero()
            if len(needle_rank) > 0:
                needle_rank_sum += needle_rank[0].item()

    avg_cos = sum(cosine_sims) / len(cosine_sims)
    top1_pct = 100 * top1_matches / h_q
    top5_pct = 100 * top5_matches / h_q
    avg_needle_rank = (needle_rank_sum / h_q) if needle_start is not None else -1.0
    return avg_cos, top1_pct, top5_pct, avg_needle_rank, h_q


def run_real_model_validation(model_name: str, token_schedule: Iterable[int]):
    if not torch.cuda.is_available():
        raise RuntimeError("Real-model validate requires CUDA (not available).")

    print("Loading model for real validation...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        ),
        device_map="auto",
        dtype=torch.float16,
    )
    model.eval()
    print(f"Loaded. GPU memory: {torch.cuda.memory_allocated() // 1024 // 1024} MB\n")

    for target_tokens in token_schedule:
        prompt = build_prompt(tokenizer, target_tokens)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=target_tokens + 256).to("cuda")

        seq_len = inputs["input_ids"].shape[1]
        needle_start = find_needle_token_start(tokenizer, inputs["input_ids"][0].tolist(), "AURORA-7749")

        print(f"{'=' * 70}")
        print(f"[REAL] Context: {seq_len} tokens | Needle at token {needle_start}")
        print(f"{'=' * 70}")

        with torch.no_grad():
            outputs = model(**inputs, use_cache=True, output_attentions=False, output_hidden_states=True)

        cache = outputs.past_key_values
        n_layers = len(cache.layers)

        for bits in [2, 3, 4]:
            total_compressed_bytes = 0
            total_uncompressed_bytes = 0
            top1_sum = 0
            top5_sum = 0
            needle_rank_sum = 0.0
            cos_sum = 0.0
            total_heads = 0

            for layer_idx in range(n_layers):
                keys = cache.layers[layer_idx].keys
                values = cache.layers[layer_idx].values
                query = gather_last_token_queries(outputs, model, layer_idx)

                real_scores, tq_scores, cb, ub = evaluate_one_layer(
                    keys=keys,
                    values=values,
                    query=query,
                    bits=bits,
                    seed=layer_idx * 1000,
                    device="cuda",
                )
                avg_cos, top1_pct, top5_pct, avg_needle_rank, n_heads = summarize_scores(real_scores, tq_scores, needle_start)

                total_compressed_bytes += cb
                total_uncompressed_bytes += ub
                cos_sum += avg_cos * n_heads
                top1_sum += top1_pct * n_heads / 100.0
                top5_sum += top5_pct * n_heads / 100.0
                if needle_start is not None:
                    needle_rank_sum += avg_needle_rank * n_heads
                total_heads += n_heads

            ratio = total_uncompressed_bytes / total_compressed_bytes
            avg_cos = cos_sum / total_heads
            top1_pct = 100 * top1_sum / total_heads
            top5_pct = 100 * top5_sum / total_heads
            avg_needle_rank = (needle_rank_sum / total_heads) if needle_start is not None else -1.0

            print(f"\n  TQ-{bits}bit:")
            print(f"    Compression:       {ratio:.2f}x ({total_compressed_bytes / 1024 / 1024:.2f} MB vs {total_uncompressed_bytes / 1024 / 1024:.2f} MB)")
            print(f"    Score cosine sim:  {avg_cos:.6f}  (1.0 = perfect)")
            print(f"    Top-1 match:       {top1_pct:.1f}%")
            print(f"    Top-5 match:       {top5_pct:.1f}%")
            if needle_start is not None:
                print(f"    Avg needle rank:   {avg_needle_rank:.1f}")
        print()


def run_synthetic_validation(seq_len: int = 1024, n_layers: int = 8, h_q: int = 16, h_kv: int = 4, d: int = 128):
    """CPU-friendly fallback when no CUDA/HF is available."""
    if h_q % h_kv != 0:
        raise ValueError("Synthetic config requires h_q % h_kv == 0")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[SYNTHETIC] Running fallback validation on {device}. seq={seq_len}, layers={n_layers}, h_q={h_q}, h_kv={h_kv}, d={d}")

    for bits in [2, 3, 4]:
        total_compressed_bytes = 0
        total_uncompressed_bytes = 0
        cos_sum = 0.0
        top1_sum = 0.0
        top5_sum = 0.0
        total_heads = 0

        for layer_idx in range(n_layers):
            keys = torch.randn(1, h_kv, seq_len, d, device=device, dtype=torch.float16)
            values = torch.randn(1, h_kv, seq_len, d, device=device, dtype=torch.float16)
            query = torch.randn(1, h_q, 1, d, device=device, dtype=torch.float16)

            real_scores, tq_scores, cb, ub = evaluate_one_layer(
                keys=keys,
                values=values,
                query=query,
                bits=bits,
                seed=layer_idx * 1000 + 11,
                device=device,
            )
            avg_cos, top1_pct, top5_pct, _, n_heads = summarize_scores(real_scores, tq_scores, needle_start=None)

            total_compressed_bytes += cb
            total_uncompressed_bytes += ub
            cos_sum += avg_cos * n_heads
            top1_sum += top1_pct * n_heads / 100.0
            top5_sum += top5_pct * n_heads / 100.0
            total_heads += n_heads

        ratio = total_uncompressed_bytes / total_compressed_bytes
        avg_cos = cos_sum / total_heads
        top1_pct = 100 * top1_sum / total_heads
        top5_pct = 100 * top5_sum / total_heads

        print(f"\n  TQ-{bits}bit:")
        print(f"    Compression:       {ratio:.2f}x")
        print(f"    Score cosine sim:  {avg_cos:.6f}")
        print(f"    Top-1 match:       {top1_pct:.1f}%")
        print(f"    Top-5 match:       {top5_pct:.1f}%")


def parse_args():
    parser = argparse.ArgumentParser(description="Validate TurboQuant with real model or synthetic fallback")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--tokens", type=int, nargs="*", default=[2048, 4096, 8192], help="Target token schedule for real validation")
    parser.add_argument("--synthetic", action="store_true", help="Force synthetic fallback mode")
    parser.add_argument("--no-fallback", action="store_true", help="Disable synthetic fallback when real validation fails")
    parser.add_argument("--api", action="store_true", help="Show API-mode limitation note")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.api:
        print("[INFO] API mode limitation: hosted generation APIs do not expose per-layer hidden states/q_proj/KV tensors.")
        print("[INFO] So this validation (which needs internal KV/query tensors) cannot be run via generic chat/completions API.")
        print("[INFO] Use local/hosted open-weight model runtime that exposes internals, or run synthetic fallback.")
        return

    if args.synthetic:
        run_synthetic_validation()
        return

    try:
        run_real_model_validation(model_name=args.model_name, token_schedule=args.tokens)
    except Exception as exc:
        print(f"[WARN] Real-model validation failed: {type(exc).__name__}: {exc}")
        if args.no_fallback:
            raise
        print("[WARN] Falling back to synthetic validation (CPU-friendly).")
        run_synthetic_validation()


if __name__ == "__main__":
    main()
