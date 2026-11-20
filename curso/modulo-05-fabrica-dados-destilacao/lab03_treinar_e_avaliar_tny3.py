#!/usr/bin/env python3
"""Módulo 5 (Lab 3): Answer-Only SFT + QAT em TNY3 e Matriz de Avaliação Automatizada."""

from __future__ import annotations

import argparse
import json
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
COURSE_DIR = MODULE_DIR.parent
MOD3_DIR = COURSE_DIR / "modulo-03-tokenizador-bpe-embarcado"
MOD4_DIR = COURSE_DIR / "modulo-04-arquitetura-tny3-s3"
for p in (MOD3_DIR, MOD4_DIR, MODULE_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from bpe_tokenizer import ACT_ID, ASSISTANT_ID, EOS_ID, PAD_ID, TinyBPETokenizer  # noqa: E402
from lab01_gerar_dataset_sintetico import (  # noqa: E402
    build_synthetic_datasets,
    format_prompt_and_completion,
    validate_action_syntax,
)
from tny3_model import (  # noqa: E402
    PRESETS,
    QUANT_INT8,
    QUANT_Q4_0,
    QuantizedTny3Runner,
    Tny3Config,
    Tny3Transformer,
)


def make_sft_sample(tok: TinyBPETokenizer, item: dict[str, str], ctx_len: int) -> tuple[list[int], list[int]]:
    p_str, c_str = format_prompt_and_completion(item)
    p_ids = tok.encode(p_str)
    c_ids = tok.encode(c_str, add_eos=True)
    full = (p_ids + c_ids)[:ctx_len]
    x = full[:-1]
    y = full[1:]
    mask_until = max(0, len(p_ids) - 1)
    tgt = [-100 if i < mask_until else y[i] for i in range(len(y))]
    return x, tgt


def collate_sft(samples: list[tuple[list[int], list[int]]], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(x) for x, _ in samples)
    bsz = len(samples)
    x_np = np.full((bsz, max_len), PAD_ID, dtype=np.int64)
    y_np = np.full((bsz, max_len), -100, dtype=np.int64)
    for i, (inp, tgt) in enumerate(samples):
        x_np[i, : len(inp)] = inp
        y_np[i, : len(tgt)] = tgt
    return torch.from_numpy(x_np).to(device), torch.from_numpy(y_np).to(device)


def extract_act_and_reply(gen_ids: list[int], tok: TinyBPETokenizer) -> tuple[str, str, bool]:
    """Separa <|act|>...<|assistant|>... e checa formatação válida."""
    if ACT_ID not in gen_ids or ASSISTANT_ID not in gen_ids:
        return "", tok.decode(gen_ids, skip_special=True), False
    i_act = gen_ids.index(ACT_ID)
    i_ast = gen_ids.index(ASSISTANT_ID)
    if i_ast <= i_act:
        return "", tok.decode(gen_ids, skip_special=True), False
    act_str = tok.decode(gen_ids[i_act + 1 : i_ast]).strip()
    ast_str = tok.decode(gen_ids[i_ast + 1 :]).strip()
    return act_str, ast_str, validate_action_syntax(act_str)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Treina modelo TNY3 especialista e avalia métricas.")
    p.add_argument("--preset", choices=["micro", "full_s3"], default="micro")
    p.add_argument("--epochs", type=int, default=45, help="Épocas SFT + QAT (padrão: 45)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--q4", action="store_true", help="Treina e exporta em Q4_0 híbrido (4 bits)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = MODULE_DIR / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    train_jsonl = out_dir / "dataset_sft.jsonl"
    eval_jsonl = out_dir / "dataset_eval.jsonl"

    if train_jsonl.exists() and eval_jsonl.exists():
        train_rows = [json.loads(l) for l in train_jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]
        eval_rows = [json.loads(l) for l in eval_jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]
    else:
        train_rows, eval_rows = build_synthetic_datasets(seed=args.seed)

    base_cfg = PRESETS[args.preset]
    qmode = QUANT_Q4_0 if args.q4 else QUANT_INT8
    cfg = Tny3Config(
        vocab_size=base_cfg.vocab_size,
        ctx_len=base_cfg.ctx_len,
        d=base_cfg.d,
        layers=base_cfg.layers,
        q_heads=base_cfg.q_heads,
        kv_heads=base_cfg.kv_heads,
        hid=base_cfg.hid,
        rope_theta=base_cfg.rope_theta,
        quant_mode=qmode,
    )

    corpus_pt = MOD3_DIR / "data" / "corpus_ptbr_dominio.txt"
    corpus_texts = [
        ln.strip() for ln in corpus_pt.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.startswith("#")
    ]
    corpus_texts.extend(f"{r['ctx']} {r['user']} {r['act']} {r['assistant']}" for r in train_rows)
    tok = TinyBPETokenizer.train(corpus_texts, vocab_size=cfg.vocab_size, min_freq=2)
    vocab_path = out_dir / "tiny_vocab.bin"
    tok.save_bin(vocab_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 88)
    print(f"MÓDULO 5 — TREINO ANSWER-ONLY SFT + QAT TNY3 ({args.preset}, quant={'Q4_0' if args.q4 else 'int8'}, {device})")
    print("=" * 88)

    model = Tny3Transformer(cfg).to(device)
    samples = [make_sft_sample(tok, item, cfg.ctx_len) for item in train_rows]

    steps_per_epoch = max(1, math.ceil(len(samples) / args.batch_size))
    total_steps = args.epochs * steps_per_epoch
    warmup = max(1, int(0.1 * total_steps))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    def lr_fn(step: int) -> float:
        if step < warmup:
            return float(step + 1) / float(warmup)
        prog = float(step - warmup) / float(max(1, total_steps - warmup))
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * prog))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)

    t0 = time.perf_counter()
    for ep in range(1, args.epochs + 1):
        random.shuffle(samples)
        model.train()
        ce_sum = 0.0
        nb = 0
        for i in range(0, len(samples), args.batch_size):
            x_t, y_t = collate_sft(samples[i : i + args.batch_size], device)
            logits = model(x_t, qat=True)
            loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), y_t.view(-1), ignore_index=-100)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            ce_sum += float(loss.item())
            nb += 1
        if ep == 1 or ep == args.epochs or ep % max(1, args.epochs // 5) == 0:
            print(f"  Época {ep:2d}/{args.epochs} | SFT Loss: {ce_sum / max(1, nb):.4f}")

    model_path = out_dir / "tny3_agent.bin"
    blob = model.export_tny3(model_path)
    print(f"[EXPORT] Salvo em {model_path} ({len(blob):,} bytes) em {time.perf_counter() - t0:.2f}s")

    print("\n[MATRIZ DE AVALIAÇÃO QUANTIZADA — FRASES INÉDITAS]")
    runner = QuantizedTny3Runner(blob)
    act_hits = 0
    fmt_ok = 0
    refusal_total = 0
    refusal_hits = 0
    false_actuations = 0

    for item in eval_rows:
        prompt_str, _ = format_prompt_and_completion(item)
        p_ids = tok.encode(prompt_str)
        gen_ids = runner.generate(p_ids, eos_id=EOS_ID, max_new=64)
        pred_act, pred_ast, valid_fmt = extract_act_and_reply(gen_ids, tok)
        fmt_ok += int(valid_fmt)
        exact = pred_act == item["act"]
        act_hits += int(exact)

        if item["act"] == "NONE":
            refusal_total += 1
            if pred_act == "NONE":
                refusal_hits += 1
            elif pred_act:
                false_actuations += 1
        mark = "OK" if exact else "ERR"
        print(f"  [{mark:3s}] U: {item['user'][:40]:<40s} | Esp: {item['act']:<15s} | Got: {pred_act:<15s}")

    acc = 100.0 * act_hits / len(eval_rows)
    fmt_pct = 100.0 * fmt_ok / len(eval_rows)
    ref_pct = 100.0 * refusal_hits / max(1, refusal_total)
    print(
        f"\nMétricas Finais ({len(eval_rows)} prompts inéditos):\n"
        f"  Action Exact Match : {act_hits}/{len(eval_rows)} ({acc:.1f}%)\n"
        f"  Safe Refusal Rate  : {refusal_hits}/{refusal_total} ({ref_pct:.1f}% | falsos acionamentos={false_actuations})\n"
        f"  Formatação Válida  : {fmt_ok}/{len(eval_rows)} ({fmt_pct:.1f}%)"
    )
    assert acc >= 90.0, f"Action Exact Match abaixo de 90%: {acc:.1f}%"
    assert ref_pct >= 95.0, f"Safe Refusal abaixo de 95%: {ref_pct:.1f}%"
    return 0


if __name__ == "__main__":
    sys.exit(main())
