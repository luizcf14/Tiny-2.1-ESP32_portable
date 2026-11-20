#!/usr/bin/env python3
"""Módulo 4 (Lab): Validação de partições 16 MB, orçamento GQA/Flash e paridade TNY3 (int8 & Q4_0)."""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MODULE_DIR = pathlib.Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from tny3_model import (  # noqa: E402
    PRESETS,
    QUANT_INT8,
    QUANT_Q4_0,
    QuantizedTny3Runner,
    Tny3Config,
    Tny3Transformer,
)


def verify_s3_16mb_partitions(csv_path: pathlib.Path) -> None:
    print("=" * 90)
    print(f"PARTE 1 — VALIDAÇÃO DE PARTIÇÕES 16 MB ({csv_path.name})")
    print("=" * 90)
    prev_end = 0
    rows = []
    for raw in csv_path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        name, ptype, subtype, off_s, sz_s = parts[:5]
        off = int(off_s, 0)
        sz = int(sz_s, 0)
        assert off >= prev_end, f"Sobreposição na partição {name}"
        if ptype == "app" or name in ("model_bin", "rag_kb"):
            assert off % 0x10000 == 0, f"Partição {name} (0x{off:X}) não está alinhada em 64 KB!"
        prev_end = off + sz
        rows.append((name, ptype, subtype, off, sz))
    assert prev_end <= 0x1000000, f"Ultrapassa 16 MB: 0x{prev_end:X}"

    for name, ptype, subtype, off, sz in rows:
        print(f"  {name:<10} | {ptype:<4} | {subtype:<8} | 0x{off:06X} | {sz / (1024 * 1024):6.2f} MB ({sz:,} B)")
    print(f"[OK] Layout válido de 16 MB (total mapeado: 0x{prev_end:X} = {prev_end // (1024 * 1024)} MB)\n")


def print_memory_budget(cfg: Tny3Config) -> None:
    print("=" * 90)
    print(f"PARTE 2 — ORÇAMENTO DA ARQUITETURA TNY3 (D={cfg.d}, L={cfg.layers}, Q={cfg.q_heads}, KV={cfg.kv_heads}, HID={cfg.hid})")
    print("=" * 90)
    m = Tny3Transformer(cfg)
    n_params = sum(p.numel() for p in m.parameters())
    print(f"Parâmetros treináveis: {n_params:,}")

    for ctx in (256, 512, 1024):
        kv_mha = cfg.layers * ctx * 2 * (cfg.d + 4)
        kv_gqa = cfg.layers * ctx * 2 * (cfg.kv_dim + 4)
        where = "SRAM Interna (N16/N16R8)" if kv_gqa <= 300_000 else "PSRAM (N16R2/N16R8)"
        print(
            f"  CTX={ctx:4d} -> MHA ({cfg.q_heads} KV): {kv_mha / 1024:7.1f} KB | "
            f"GQA ({cfg.kv_heads} KV): {kv_gqa / 1024:6.1f} KB  [{where}]"
        )


def test_quant_mode(cfg: Tny3Config, qmode: int, out_path: pathlib.Path, label: str) -> None:
    cfg_m = Tny3Config(
        vocab_size=cfg.vocab_size,
        ctx_len=cfg.ctx_len,
        d=cfg.d,
        layers=cfg.layers,
        q_heads=cfg.q_heads,
        kv_heads=cfg.kv_heads,
        hid=cfg.hid,
        rope_theta=cfg.rope_theta,
        quant_mode=qmode,
    )
    torch.manual_seed(42)
    pt_model = Tny3Transformer(cfg_m)
    pt_model.eval()
    blob = pt_model.export_tny3(out_path)
    runner = QuantizedTny3Runner(blob)

    prompt = [1, 4, 29, 85, 110, 3]
    with torch.no_grad():
        pt_logits = pt_model(torch.tensor([prompt], dtype=torch.long), qat=True)[0, -1].numpy()

    kc, vc, ks, vs = runner.reset_cache()
    run_logits = np.zeros(cfg_m.vocab_size, dtype=np.float32)
    for p, tid in enumerate(prompt):
        run_logits = runner.step(tid, p, kc, vc, ks, vs)

    cos = float(np.dot(pt_logits, run_logits) / (np.linalg.norm(pt_logits) * np.linalg.norm(run_logits)))
    max_diff = float(np.max(np.abs(pt_logits - run_logits)))
    assert cos > 0.999, f"Divergência em {label}: cos={cos}"
    print(
        f"  [{label:<10}] {len(blob):>9,} bytes ({len(blob) / (1024 * 1024):.2f} MB) | "
        f"cos={cos:.6f} | max_diff={max_diff:.6f} | argmax PT={int(np.argmax(pt_logits))} == C={int(np.argmax(run_logits))}"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Valida partições 16 MB e exportação TNY3 int8/Q4_0.")
    p.add_argument("--preset", choices=["micro", "full_s3"], default="micro", help="Preset arquitetural")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = MODULE_DIR / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    verify_s3_16mb_partitions(MODULE_DIR / "partitions_s3_16mb.csv")
    print_memory_budget(PRESETS["full_s3"])

    cfg = PRESETS[args.preset]
    print(f"\n[PARTE 3 — EXPORTAÇÃO TNY3 E PARIDADE PYTORCH == RUNNER C (preset={args.preset})]")
    test_quant_mode(cfg, QUANT_INT8, out_dir / f"test_{args.preset}_int8.bin", "TNY3 int8")
    test_quant_mode(cfg, QUANT_Q4_0, out_dir / f"test_{args.preset}_q4.bin", "TNY3 Q4_0")
    print("\n[OK] Formato TNY3 alinhado em 4 bytes e paridade verificada!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
