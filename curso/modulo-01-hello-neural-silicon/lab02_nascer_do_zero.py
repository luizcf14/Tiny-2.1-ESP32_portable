#!/usr/bin/env python3
"""Módulo 1 (Lab 1B): Vendo uma LLM nascer do ruído aleatório até o binário int8 (TNY2)."""

from __future__ import annotations

import argparse
import math
import pathlib
import random
import sys
import time

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
    QuantizedTny2Runner,
    SEP_TOK,
    TinyTransformer,
    VOCAB,
    build_training_sample,
    collate_batch,
    select_device,
)
from generate_esp32_assets import generate_weights_header  # noqa: E402

BASE_PAIRS_4: list[tuple[str, str]] = [
    ("oi", "ola! nasci do zero."),
    ("quem e voce?", "sou uma llm de 367 mil parametros."),
    ("qual sua placa?", "rodo no esp32-s3 de 16mb."),
    ("quanto e 2 mais 2?", "2 mais 2 e igual a 4."),
]

EXTRA_PAIRS_12: list[tuple[str, str]] = [
    ("qual seu objetivo?", "ensinar como uma llm funciona por dentro."),
    ("voce precisa de internet?", "nao, rodo 100% offline na flash."),
    ("quanta flash tem o s3?", "o esp32-s3-wroom-1 tem 16mb de flash."),
    ("onde ficam os pesos?", "os pesos ficam na flash em zero-copy."),
    ("onde fica o kv cache?", "o kv cache em int8 fica na sram."),
    ("o que e quantizacao?", "comprimir pesos float32 para int8 de 8 bits."),
    ("o que e qat?", "treinar simulando a quantizacao do chip."),
    ("qual a perda inicial?", "no inicio a loss fica perto de 4.57."),
    ("quantas camadas voce tem?", "tenho 4 camadas e 96 de largura."),
    ("qual o limite de contexto?", "meu limite atual e de 128 letras."),
    ("o que vem no modulo 2?", "vamos dissecar o transformer em c puro."),
    ("o que vem no modulo 3?", "vamos trocar letras por subpalavras bpe."),
]


def initialize_blank_brain(model: TinyTransformer) -> None:
    """Preenche matrizes 2D com N(0, 0.02); ganhos 1D (g1, g2, gf) permanecem 1.0."""
    with torch.no_grad():
        for p in model.parameters():
            if p.ndim == 2:
                torch.nn.init.normal_(p, mean=0.0, std=0.02)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cria um TinyTransformer do zero absoluto, treina com QAT e exporta TNY2."
    )
    parser.add_argument("--steps", type=int, default=180, help="Passos de otimização AdamW (padrão: 180)")
    parser.add_argument("--lr", type=float, default=3e-3, help="Taxa de aprendizado (padrão: 3e-3)")
    parser.add_argument(
        "--dataset-size",
        type=int,
        choices=[4, 16],
        default=4,
        help="Tamanho do mini-dataset do zero: 4 (padrão) ou 16 (Exercício 2)",
    )
    parser.add_argument(
        "--no-qat",
        action="store_true",
        help="Desativa QAT durante o treino para comparar PTQ vs QAT na saída int8",
    )
    parser.add_argument("--seed", type=int, default=42, help="Semente aleatória (padrão: 42)")
    parser.add_argument(
        "--generate-header",
        action="store_true",
        help="Também gera out/tiny_weights_do_zero.h (sem sobrescrever TinyESP32/tiny_weights.h)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = MODULE_DIR / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 84)
    print("LAB 02 — VENDO UMA LLM NASCER DO RUÍDO ALEATÓRIO ATÉ O BINÁRIO int8")
    print("=" * 84)
    device = select_device("auto", args.seed)

    # Etapa 2: Cérebro em branco (mesma geometria D=96, L=4, H=4, HID=256 do TINY.BIN)
    model = TinyTransformer(d=96, layers=4, heads=4, hid=256).to(device)
    initialize_blank_brain(model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[ETAPA 2] Modelo inicializado do zero com {n_params:,} parâmetros aleatórios N(0, 0.02).")

    before_path = out_dir / "lab02_antes_do_treino.bin"
    blob_before = model.export_tny2(before_path)
    runner_before = QuantizedTny2Runner(blob_before)
    print(
        f"[ANTES DO TREINO] quem e voce? -> {runner_before.ask('quem e voce?')!r}"
        f"  (loss teórica inicial ~ ln({VOCAB}) = {math.log(VOCAB):.2f})"
    )

    # Etapa 1 & 3: Tokenização ASCII + Dataset
    pairs = list(BASE_PAIRS_4)
    if args.dataset_size == 16:
        pairs.extend(EXTRA_PAIRS_12)
    samples = [build_training_sample(q, a, SEP_TOK) for q, a in pairs]
    x_t, y_t = collate_batch(samples, device)
    print(f"[ETAPA 3] Dataset carregado: {len(pairs)} pares | shape do lote = {tuple(x_t.shape)}")

    # Etapa 4 & 5: Motor de aprendizado (Forward causal + Answer-Only SFT + AdamW + QAT)
    qat = not args.no_qat
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    model.train()
    checkpoints = {1, max(1, args.steps // 6), max(1, args.steps // 3), max(1, (2 * args.steps) // 3), args.steps}

    t0 = time.perf_counter()
    print(f"[ETAPA 4/5] Treinando por {args.steps} passos (QAT={qat}, lr={args.lr:.1e})...")
    for step_idx in range(1, args.steps + 1):
        logits = model(x_t, qat=qat)
        loss = F.cross_entropy(logits.view(-1, VOCAB), y_t.view(-1), ignore_index=-100)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step_idx in checkpoints:
            print(f"  Passo {step_idx:4d}/{args.steps} | Cross-Entropy Loss: {loss.item():.4f}")

    # Etapa 6: Congelar em TNY2 (int8) e avaliar
    after_path = out_dir / "lab02_depois_do_treino.bin"
    blob_after = model.export_tny2(after_path)
    dt = time.perf_counter() - t0
    print(f"\n[ETAPA 6] Modelo exportado para {after_path} ({len(blob_after):,} bytes) em {dt:.2f}s")

    runner_after = QuantizedTny2Runner(blob_after)
    print("\n[AVALIAÇÃO int8 — PARES TREINADOS]")
    hits = 0
    for q, expected in pairs:
        got = runner_after.ask(q, creativity=0)
        ok = got == expected
        hits += int(ok)
        mark = "OK" if ok else "DIFF"
        print(f"  [{mark:4s}] Q: {q:<26s} -> A: {got}")
    print(f"  Taxa de acerto exato (quantizado int8): {hits}/{len(pairs)}")

    print("\n[TESTE INÉDITO — PERGUNTAS FORA DO DATASET]")
    for unseen_q in ("qual seu nome?", "voce roda na nuvem?"):
        print(f"  Q: {unseen_q:<26s} -> A: {runner_after.ask(unseen_q, creativity=0)!r}")

    if args.generate_header:
        hdr_path = out_dir / "tiny_weights_do_zero.h"
        generate_weights_header(after_path, hdr_path)
        print(f"\n[PROGMEM] Header C gerado em {hdr_path} (TinyESP32/tiny_weights.h intacto).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
