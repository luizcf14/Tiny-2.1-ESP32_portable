#!/usr/bin/env python3
"""Módulo 1 (Lab 3): Fine-tuning sobre TINY.BIN expondo os 3 gargalos do modelo char-level."""

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
    CTX,
    MAXQ,
    QuantizedTny2Runner,
    TinyTransformer,
    VOCAB,
    augment_pairs,
    build_training_sample,
    collate_batch,
    detect_eos_token,
    load_qa_file,
    normalize_question,
    run_chat,
    select_device,
    to_ascii,
)


def parse_raw_lines(path: pathlib.Path) -> list[tuple[str, str]]:
    """Lê os textos originais com acentos antes da conversão para ASCII."""
    rows: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "|" not in s:
            continue
        q_part, a_part = s.split("|", 1)
        first_q = q_part.split(";", 1)[0].strip()
        rows.append((first_q, a_part.strip()))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Treina sobre TINY.BIN com data/lab01_limites_char.txt e diagnostica as 3 Dores."
    )
    parser.add_argument("--epochs", type=int, default=35, help="Épocas de fine-tuning (padrão: 35)")
    parser.add_argument("--lr", type=float, default=8e-4, help="Taxa de aprendizado (padrão: 8e-4)")
    parser.add_argument("--seed", type=int, default=42, help="Semente aleatória (padrão: 42)")
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Abre chat interativo quantizado em int8 ao final do experimento",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset_path = MODULE_DIR / "data" / "lab01_limites_char.txt"
    base_bin = REPO_ROOT / "TINY.BIN"
    out_dir = MODULE_DIR / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_bin = out_dir / "lab03_limites.bin"

    print("=" * 88)
    print("LAB 03 — FINE-TUNING SOBRE TINY.BIN E DIAGNÓSTICO DAS 3 DORES DO CHAR-LEVEL")
    print("=" * 88)

    raw_accented = parse_raw_lines(dataset_path)
    raw_pairs = load_qa_file(dataset_path)
    train_pairs = augment_pairs(raw_pairs, enable=True)
    print(f"[DATA] {len(raw_pairs)} pares base -> {len(train_pairs)} variações para treino")

    device = select_device("auto", args.seed)
    model, _ = TinyTransformer.from_tny2(base_bin)
    model = model.to(device)
    eos_tok = detect_eos_token(model, device)

    samples = [build_training_sample(q, a, eos_tok) for q, a in train_pairs]
    ref_params = {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}

    batch_size = min(16, len(samples))
    steps_per_epoch = max(1, math.ceil(len(samples) / batch_size))
    total_steps = max(1, args.epochs * steps_per_epoch)
    warmup_steps = max(1, int(0.1 * total_steps))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        prog = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * prog))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    t0 = time.perf_counter()
    print(f"[TRAIN] Fine-tuning QAT int8 por {args.epochs} épocas (L2-SP=0.001, lr={args.lr:.1e})...")
    for epoch in range(1, args.epochs + 1):
        random.shuffle(samples)
        model.train()
        total_ce = 0.0
        n_batches = 0
        for i in range(0, len(samples), batch_size):
            x_t, y_t = collate_batch(samples[i : i + batch_size], device)
            logits = model(x_t, qat=True)
            ce = F.cross_entropy(logits.view(-1, VOCAB), y_t.view(-1), ignore_index=-100)
            l2_sp = sum((p - ref_params[n]).pow(2).mean() for n, p in model.named_parameters())
            loss = ce + 0.001 * (l2_sp / len(ref_params))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_ce += float(ce.item())
            n_batches += 1

        if epoch == 1 or epoch == args.epochs or epoch % max(1, args.epochs // 5) == 0:
            print(f"  Época {epoch:2d}/{args.epochs} | CE loss: {total_ce / max(1, n_batches):.4f}")

    blob = model.export_tny2(out_bin)
    print(f"[EXPORT] Salvo em {out_bin} ({len(blob):,} bytes) em {time.perf_counter() - t0:.2f}s")
    runner = QuantizedTny2Runner(blob)

    print("\n" + "-" * 88)
    print("RELATÓRIO DAS 3 DORES ESTRUTURAIS (INFERÊNCIA int8)")
    print("-" * 88)

    # DOR 1: Corte duro em CTX=128
    long_q, long_a = raw_accented[0]
    q_clean = normalize_question(long_q)[:MAXQ]
    max_fit = CTX - len(q_clean) - 2
    pred_long = runner.ask(long_q, creativity=0)
    print("\n[DOR 1 — Amputação de Contexto em CTX=128 letras]")
    print(f"  Pergunta ({len(q_clean)} chars): {q_clean}")
    print(f"  Resposta desejada ({len(long_a)} chars): {long_a}")
    print(f"  Espaço que sobrou na janela CTX={CTX}: apenas {max_fit} chars para a resposta!")
    print(f"  Saída do modelo   ({len(pred_long)} chars): {pred_long}")

    # DOR 2: Perda de acentos e fragilidade semântica
    acc_q, acc_a = raw_accented[1]
    print("\n[DOR 2 — Achatamento de Acentos + Desperdício com Ortografia]")
    print(f"  Entrada UTF-8 original   : {acc_q}")
    print(f"  Entrada vista pelo TINY  : {normalize_question(acc_q)}")
    print(f"  Resposta original (UTF-8): {acc_a}")
    print(f"  Resposta no modelo (ASCII): {runner.ask(acc_q, creativity=0)}")
    print("  Teste de paráfrase coloquial não treinada:")
    print(f"    Treinada : 'ligar ventilador da bancada'             -> {runner.ask('ligar ventilador da bancada')}")
    print(f"    Paráfrase: 'bota o ventilador da bancada pra girar'  -> {runner.ask('bota o ventilador da bancada pra girar')}")

    # DOR 3: Sem memória multi-turno na atenção
    print("\n[DOR 3 — Ausência de Diálogo Multi-Turno sem tiny_session.cpp]")
    print(f"  Turno 1: 'ligar ventilador da bancada' -> {runner.ask('ligar ventilador da bancada')}")
    print(f"  Turno 2: 'agora desliga ele'           -> {runner.ask('agora desliga ele')}")
    print("  Conclusão: como cada chamada começa do zero em 128 letras, sem as regras manuais")
    print("  de TinyESP32/tiny_session.cpp o modelo não sabe o que é 'ele' -> motivação dos Módulos 3 a 5!")

    if args.chat:
        return run_chat(out_bin)
    return 0


if __name__ == "__main__":
    sys.exit(main())
