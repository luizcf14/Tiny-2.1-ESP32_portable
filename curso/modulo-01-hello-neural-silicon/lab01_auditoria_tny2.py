#!/usr/bin/env python3
"""Módulo 1 (Lab 1A): Auditoria do binário TNY2 e Raio-X da tokenização ASCII."""

from __future__ import annotations

import argparse
import pathlib
import struct
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MODULE_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = MODULE_DIR.parent.parent
TOOLS_DIR = REPO_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from finetune_tiny import (  # noqa: E402
    CTX,
    PAD_TOK,
    SEP_TOK,
    VOCAB,
    build_training_sample,
    normalize_question,
    to_ascii,
)
from generate_esp32_assets import validate_tiny_bin  # noqa: E402


def inspect_table(
    blob: bytes, offset: int, name: str, rows: int, cols: int
) -> tuple[int, dict[str, object]]:
    """Lê uma tabela quantizada (scale float32 + cols int8 por linha) sem modificar o arquivo."""
    stride = 4 + cols
    total_bytes = rows * stride
    min_scale = float("inf")
    max_scale = 0.0
    min_w = 127
    max_w = -127

    for r in range(rows):
        pos = offset + r * stride
        (scale,) = struct.unpack_from("<f", blob, pos)
        if scale < min_scale:
            min_scale = scale
        if scale > max_scale:
            max_scale = scale
        weights = struct.unpack_from(f"<{cols}b", blob, pos + 4)
        lo = min(weights)
        hi = max(weights)
        if lo < min_w:
            min_w = lo
        if hi > max_w:
            max_w = hi

    info = {
        "name": name,
        "shape": f"{rows} x {cols}",
        "params": rows * cols,
        "scales": rows,
        "bytes": total_bytes,
        "scale_range": f"[{min_scale:.5f}, {max_scale:.5f}]",
        "weight_range": f"[{min_w}, {max_w}]",
    }
    return offset + total_bytes, info


def inspect_gains(blob: bytes, offset: int, name: str, d: int) -> tuple[int, dict[str, object]]:
    """Lê um vetor de ganhos RMSNorm em float32."""
    total_bytes = d * 4
    vals = struct.unpack_from(f"<{d}f", blob, offset)
    info = {
        "name": name,
        "shape": f"{d} (float32)",
        "params": d,
        "scales": 0,
        "bytes": total_bytes,
        "scale_range": f"[{min(vals):.4f}, {max(vals):.4f}]",
        "weight_range": "float32",
    }
    return offset + total_bytes, info


def audit_tny2(bin_path: pathlib.Path) -> None:
    if not bin_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {bin_path}")

    blob = bin_path.read_bytes()
    d, layers, heads, hid = validate_tiny_bin(blob)

    print("=" * 92)
    print(f"PARTE 1 — AUDITORIA DO ARQUIVO BINÁRIO TNY2: {bin_path.name}")
    print("=" * 92)
    print(
        f"Cabeçalho (12 bytes): magic='TNY2' | D={d} | L={layers} | H={heads} (HD={d // heads}) | HID={hid}"
    )
    print(f"Vocabulário fixo: VOCAB={VOCAB} | Janela de Contexto: CTX={CTX}\n")

    records: list[dict[str, object]] = [
        {
            "name": "header (TNY2)",
            "shape": "12 B",
            "params": 0,
            "scales": 0,
            "bytes": 12,
            "scale_range": "-",
            "weight_range": "-",
        }
    ]
    off = 12
    off, rec = inspect_table(blob, off, "emb (Weight Tying)", VOCAB, d)
    records.append(rec)
    off, rec = inspect_table(blob, off, "pos (Positional)", CTX, d)
    records.append(rec)

    for l in range(layers):
        off, rec = inspect_gains(blob, off, f"layer[{l}].g1", d)
        records.append(rec)
        for proj in ("wq", "wk", "wv", "wo"):
            off, rec = inspect_table(blob, off, f"layer[{l}].{proj}", d, d)
            records.append(rec)
        off, rec = inspect_gains(blob, off, f"layer[{l}].g2", d)
        records.append(rec)
        off, rec = inspect_table(blob, off, f"layer[{l}].f1", hid, d)
        records.append(rec)
        off, rec = inspect_table(blob, off, f"layer[{l}].f2", d, hid)
        records.append(rec)

    off, rec = inspect_gains(blob, off, "gf (Norma Final)", d)
    records.append(rec)

    assert off == len(blob), f"Offset final {off} != {len(blob)}"

    header_line = (
        f"{'Tensor':<22} | {'Dimensão':<14} | {'Parâmetros':>10} | "
        f"{'Escalas':>7} | {'Bytes .bin':>10} | {'Faixa Escala/Gain':<20}"
    )
    print(header_line)
    print("-" * len(header_line))
    total_params = 0
    total_scales = 0
    total_bytes = 0
    for r in records:
        total_params += int(r["params"])
        total_scales += int(r["scales"])
        total_bytes += int(r["bytes"])
        print(
            f"{str(r['name']):<22} | {str(r['shape']):<14} | {int(r['params']):>10,} | "
            f"{int(r['scales']):>7,} | {int(r['bytes']):>10,} | {str(r['scale_range']):<20}"
        )

    assert total_bytes == len(blob)
    fp32_bytes = total_params * 4
    ratio = fp32_bytes / total_bytes

    print("-" * len(header_line))
    print(
        f"{'TOTAL':<22} | {'TNY2 válido':<14} | {total_params:>10,} | "
        f"{total_scales:>7,} | {total_bytes:>10,} | economia {ratio:.2f}x vs fp32"
    )
    print(f"\nComparação de tamanho: float32 puro = {fp32_bytes:,} bytes (~{fp32_bytes / 1024:.1f} KB)")
    print(f"                       TNY2 int8    = {total_bytes:,} bytes (~{total_bytes / 1024:.1f} KB)")


def tok_label(tok: int) -> str:
    if tok == PAD_TOK:
        return "<PAD>"
    if tok == SEP_TOK:
        return "<SEP/EOS>"
    if tok == -100:
        return "<IGNORE>"
    return repr(chr(tok + 30))


def inspect_tokenization(question: str, answer: str) -> None:
    q_clean = normalize_question(question)
    a_clean = to_ascii(answer)
    inp, tgt = build_training_sample(q_clean, a_clean, SEP_TOK)

    print("\n" + "=" * 92)
    print("PARTE 2 — RAIO-X DA TOKENIZAÇÃO ASCII E MÁSCARA ANSWER-ONLY (-100)")
    print("=" * 92)
    print(f"Pergunta original   : {question!r}")
    print(f"Pergunta normalizada: {q_clean!r} (regra ASCII + minúsculas)")
    print(f"Resposta normalizada: {a_clean!r}\n")
    print(f"{'Pos p':>5} | {'Entrada':<10} | {'inp[p]':>6} | {'Alvo Esperado':<13} | {'tgt[p]':>6} | Papel no Treino")
    print("-" * 80)

    for p, (x_id, y_id) in enumerate(zip(inp, tgt)):
        if y_id == -100:
            role = "Prompt da pergunta (loss ignorada)"
        elif p == len(q_clean):
            role = "Separador dispara 1a letra da resposta"
        elif y_id == SEP_TOK:
            role = "Fim da resposta (<EOS>)"
        else:
            role = "Prevê próxima letra da resposta"
        print(
            f"{p:>5} | {tok_label(x_id):<10} | {x_id:>6} | "
            f"{tok_label(y_id):<13} | {y_id:>6} | {role}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audita pesos TNY2 e visualiza a tokenização ASCII de treinamento."
    )
    parser.add_argument(
        "--bin",
        type=pathlib.Path,
        default=REPO_ROOT / "TINY.BIN",
        help="Caminho para o binário TNY2 (padrão: TINY.BIN na raiz)",
    )
    parser.add_argument("--question", type=str, default="oi", help="Pergunta de exemplo para o Raio-X")
    parser.add_argument("--answer", type=str, default="ola!", help="Resposta de exemplo para o Raio-X")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bin_path = args.bin if args.bin.is_absolute() else (REPO_ROOT / args.bin)
    audit_tny2(bin_path)
    inspect_tokenization(args.question, args.answer)
    return 0


if __name__ == "__main__":
    sys.exit(main())
