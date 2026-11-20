#!/usr/bin/env python3
"""Módulo 2 (Lab 2): Quantização do KV Cache, Estabilidade do Softmax e QAT vs PTQ."""

from __future__ import annotations

import math
import pathlib
import random
import sys
import tempfile

import numpy as np
import torch
import torch.nn.functional as F

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MODULE_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = MODULE_DIR.parent.parent
TOOLS_DIR = REPO_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from finetune_tiny import (  # noqa: E402
    CTX,
    QuantizedTny2Runner,
    SEP_TOK,
    TinyTransformer,
    VOCAB,
    build_training_sample,
    collate_batch,
    select_device,
)


def demo_kv_cache_quantization() -> None:
    print("=" * 88)
    print("PARTE 1 — ECONOMIA E FIDELIDADE DO KV CACHE EM int8 (store() em tiny_engine.cpp)")
    print("=" * 88)
    d, layers, ctx = 96, 4, CTX
    fp32_bytes = layers * ctx * d * 2 * 4
    int8_bytes = layers * ctx * (d + 4) * 2
    print(f"KV Cache float32 puro : {fp32_bytes:,} bytes (~{fp32_bytes / 1024:.1f} KB)")
    print(f"KV Cache int8 (TNY2)  : {int8_bytes:,} bytes (~{int8_bytes / 1024:.1f} KB) -> {fp32_bytes / int8_bytes:.2f}x menor")

    # Captura ativações K reais do TINY.BIN para medir erro real de store()
    pt_model, _ = TinyTransformer.from_tny2(REPO_ROOT / "TINY.BIN")
    pt_model.eval()
    tokens = torch.tensor([[81, 75, SEP_TOK]], dtype=torch.long)
    with torch.no_grad():
        x = F.embedding(tokens, pt_model.emb) + pt_model.pos[:3].unsqueeze(0)
        h = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-5) * pt_model.g1[0]
        k_real = F.linear(h, pt_model.wk[0])[0, -1].numpy()

    q_i8, scale = QuantizedTny2Runner.store_kv(k_real)
    k_reconstructed = q_i8.astype(np.float32) * np.float32(scale)
    cos_sim = float(np.dot(k_real, k_reconstructed) / (np.linalg.norm(k_real) * np.linalg.norm(k_reconstructed)))
    signal_pow = float(np.mean(k_real**2))
    noise_pow = float(np.mean((k_real - k_reconstructed) ** 2))
    snr_db = 10.0 * math.log10(signal_pow / max(noise_pow, 1e-20))
    print(f"Fidelidade em vetor K real: Cosine Similarity = {cos_sim:.6f} | SNR = {snr_db:.2f} dB")
    assert cos_sim > 0.999


def demo_softmax_stability() -> None:
    print("\n" + "=" * 88)
    print("PARTE 2 — ESTABILIDADE DO SOFTMAX E CORTE safe_exp(x < -60.0f)")
    print("=" * 88)
    raw_scores = np.array([120.0, 118.5, 45.0, 10.0], dtype=np.float32)
    print(f"Logits brutos       : {raw_scores.tolist()}")
    print("expf(120.0) direto  : overflow em float32 (> 3.4e38) -> NaN no Softmax!")

    big = float(np.max(raw_scores))
    shifted = raw_scores - big
    safe_exps = np.array([0.0 if x < -60.0 else math.exp(float(x)) for x in shifted], dtype=np.float32)
    probs = safe_exps / float(np.sum(safe_exps))
    print(f"Após subtrair max   : {shifted.tolist()}")
    print(f"exp(-60.0) exato    : {math.exp(-60.0):.3e} (desprezível frente a 1.0!)")
    print(f"Probabilidades      : {[round(float(p), 5) for p in probs]}")
    assert math.isclose(float(np.sum(probs)), 1.0, rel_tol=1e-5)


def train_variant(qat: bool, device: torch.device, seed: int = 42) -> tuple[float, int, int]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    pairs = [
        ("sensor a", "temperatura normal."),
        ("sensor b", "alerta de calor."),
        ("sensor c", "umidade baixa."),
        ("sensor d", "sistema estavel."),
        ("status 1", "motor desligado."),
        ("status 2", "ventilador ativo."),
    ]
    model = TinyTransformer(96, 4, 4, 256).to(device)
    with torch.no_grad():
        for p in model.parameters():
            if p.ndim == 2:
                torch.nn.init.normal_(p, mean=0.0, std=0.02)

    samples = [build_training_sample(q, a, SEP_TOK) for q, a in pairs]
    x_t, y_t = collate_batch(samples, device)
    opt = torch.optim.AdamW(model.parameters(), lr=2.5e-3)

    model.train()
    for _ in range(140):
        logits = model(x_t, qat=qat)
        loss = F.cross_entropy(logits.view(-1, VOCAB), y_t.view(-1), ignore_index=-100)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    with tempfile.TemporaryDirectory() as td:
        blob = model.export_tny2(pathlib.Path(td) / "variant.bin")
    runner = QuantizedTny2Runner(blob)
    hits = sum(int(runner.ask(q, creativity=0) == a) for q, a in pairs)
    return float(loss.item()), hits, len(pairs)


def main() -> int:
    demo_kv_cache_quantization()
    demo_softmax_stability()

    print("\n" + "=" * 88)
    print("PARTE 3 — COMPARATIVO QAT (qat=True) vs PTQ (qat=False) AO EXPORTAR PARA int8")
    print("=" * 88)
    device = select_device("auto", 42)
    ptq_loss, ptq_hits, total = train_variant(qat=False, device=device)
    qat_loss, qat_hits, _ = train_variant(qat=True, device=device)
    print(f"PTQ (treina fp32, quantiza só no fim) -> Loss float32: {ptq_loss:.4f} | Exact Match int8: {ptq_hits}/{total}")
    print(f"QAT (simula ste_int8 durante treino)  -> Loss QAT    : {qat_loss:.4f} | Exact Match int8: {qat_hits}/{total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
