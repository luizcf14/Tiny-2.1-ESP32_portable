#!/usr/bin/env python3
"""Módulo 2 (Lab 1): Espelhamento exato PyTorch <-> C (ZeroCopyModel) + Contagem de MACs."""

from __future__ import annotations

import argparse
import math
import pathlib
import sys

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
    encode_text,
    normalize_question,
    rms_norm,
    ste_int8,
)
from verify_port import ZeroCopyModel  # noqa: E402


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 1.0 if (na < 1e-12 and nb < 1e-12) else 0.0
    return float(np.dot(a, b) / (na * nb))


def trace_step_comparison(
    pt_model: TinyTransformer, zc_model: ZeroCopyModel, np_runner: QuantizedTny2Runner, prompt: str
) -> None:
    clean_q = normalize_question(prompt)
    q_toks = encode_text(clean_q) + [SEP_TOK]

    print("=" * 92)
    print(f"PARTE 1 — ESPELHO PASSO A PASSO: PyTorch (qat=True) vs C Zero-Copy ({clean_q!r})")
    print("=" * 92)

    # 1) Comparação em sequência no PyTorch (forward paralelo com máscara causal)
    pt_model.eval()
    tokens_t = torch.tensor([q_toks], dtype=torch.long)
    with torch.no_grad():
        pt_logits_all = pt_model(tokens_t, qat=True)[0].numpy()

    # 2) Comparação estágio por estágio no último token (SEP) entre PyTorch e ZeroCopyModel
    p = len(q_toks) - 1
    tok = q_toks[p]

    # Simula passo incremental no ZeroCopyModel para todo o prefixo
    d, layers, heads, hd = zc_model.d, zc_model.layers, zc_model.heads, zc_model.hd
    kc: list[list[list[int]]] = [[] for _ in range(layers)]
    vc: list[list[list[int]]] = [[] for _ in range(layers)]
    ks: list[list[float]] = [[] for _ in range(layers)]
    vs: list[list[float]] = [[] for _ in range(layers)]

    zc_stages: dict[str, np.ndarray] = {}
    for pos_idx, t_id in enumerate(q_toks):
        es, ew = zc_model.emb[t_id]
        ps, pw = zc_model.pos[pos_idx]
        bx = [ew[i] * es + pw[i] * ps for i in range(d)]
        if pos_idx == p:
            zc_stages["emb+pos"] = np.array(bx, dtype=np.float32)

        for l in range(layers):
            bh = zc_model.norm(bx, zc_model.g1[l])
            if pos_idx == p and l == 0:
                zc_stages["L0.norm1"] = np.array(bh, dtype=np.float32)
            kt = zc_model.matvec(zc_model.wk[l], bh)
            kq, kscale = zc_model.store(kt)
            kc[l].append(kq)
            ks[l].append(kscale)

            vt = zc_model.matvec(zc_model.wv[l], bh)
            vq, vscale = zc_model.store(vt)
            vc[l].append(vq)
            vs[l].append(vscale)

            bq = zc_model.matvec(zc_model.wq[l], bh)
            ba = [0.0] * d
            for h in range(heads):
                hoff = h * hd
                qh = bq[hoff : hoff + hd]
                scores = [
                    sum(a * b for a, b in zip(qh, kc[l][j][hoff : hoff + hd]))
                    * ks[l][j]
                    / zc_model.att_div
                    for j in range(pos_idx + 1)
                ]
                big = max(scores)
                exps = [zc_model.safe_exp(sc - big) for sc in scores]
                total = sum(exps)
                for j in range(pos_idx + 1):
                    if exps[j] == 0.0:
                        continue
                    w = exps[j] * vs[l][j] / total
                    vp = vc[l][j]
                    for i in range(hd):
                        ba[hoff + i] += w * vp[hoff + i]

            bt = zc_model.matvec(zc_model.wo[l], ba)
            bx = [bx[i] + bt[i] for i in range(d)]
            if pos_idx == p and l == 0:
                zc_stages["L0.after_att"] = np.array(bx, dtype=np.float32)

            bh = zc_model.norm(bx, zc_model.g2[l])
            bm = [v if v > 0.0 else 0.0 for v in zc_model.matvec(zc_model.f1[l], bh)]
            bt = zc_model.matvec(zc_model.f2[l], bm)
            bx = [bx[i] + bt[i] for i in range(d)]
            if pos_idx == p and l == 0:
                zc_stages["L0.after_ffn"] = np.array(bx, dtype=np.float32)

        bh = zc_model.norm(bx, zc_model.gf)
        zc_logits = np.array(zc_model.matvec(zc_model.emb, bh), dtype=np.float32)
        if pos_idx == p:
            zc_stages["final_norm"] = np.array(bh, dtype=np.float32)
            zc_stages["logits"] = zc_logits

    # Captura os mesmos estágios no PyTorch
    with torch.no_grad():
        emb_w = ste_int8(pt_model.emb)
        pos_w = ste_int8(pt_model.pos[: len(q_toks)])
        x = F.embedding(tokens_t, emb_w) + pos_w.unsqueeze(0)
        pt_stages: dict[str, np.ndarray] = {"emb+pos": x[0, p].numpy()}

        mask = pt_model.causal_mask[: len(q_toks), : len(q_toks)]
        att_div = math.sqrt(pt_model.hd)
        for l in range(pt_model.layers):
            h = rms_norm(x, pt_model.g1[l])
            if l == 0:
                pt_stages["L0.norm1"] = h[0, p].numpy()
            q = F.linear(h, ste_int8(pt_model.wq[l]))
            k = ste_int8(F.linear(h, ste_int8(pt_model.wk[l])))
            v = ste_int8(F.linear(h, ste_int8(pt_model.wv[l])))
            qh = q.view(1, len(q_toks), heads, hd).transpose(1, 2)
            kh = k.view(1, len(q_toks), heads, hd).transpose(1, 2)
            vh = v.view(1, len(q_toks), heads, hd).transpose(1, 2)
            att = (torch.matmul(qh, kh.transpose(-2, -1)) / att_div).masked_fill(
                mask.unsqueeze(0).unsqueeze(0), -1e9
            )
            probs = F.softmax(att, dim=-1)
            ba = torch.matmul(probs, vh).transpose(1, 2).contiguous().view(1, len(q_toks), d)
            x = x + F.linear(ba, ste_int8(pt_model.wo[l]))
            if l == 0:
                pt_stages["L0.after_att"] = x[0, p].numpy()
            h2 = rms_norm(x, pt_model.g2[l])
            bm = F.relu(F.linear(h2, ste_int8(pt_model.f1[l])))
            x = x + F.linear(bm, ste_int8(pt_model.f2[l]))
            if l == 0:
                pt_stages["L0.after_ffn"] = x[0, p].numpy()
        hf = rms_norm(x, pt_model.gf)
        pt_stages["final_norm"] = hf[0, p].numpy()
        pt_stages["logits"] = F.linear(hf, emb_w)[0, p].numpy()

    print(f"{'Estágio do Grafo':<16} | {'Max Abs Diff':>12} | {'Cosine Sim':>11} | Status")
    print("-" * 62)
    for key in ("emb+pos", "L0.norm1", "L0.after_att", "L0.after_ffn", "final_norm", "logits"):
        diff = float(np.max(np.abs(pt_stages[key] - zc_stages[key])))
        csim = cosine_sim(pt_stages[key], zc_stages[key])
        assert csim > 0.999, f"Divergência inesperada em {key}: csim={csim}"
        print(f"{key:<16} | {diff:>12.6f} | {csim:>11.7f} | OK")

    # Resposta completa gerada pelas duas engines independentes
    ans_zc = zc_model.ask_greedy(clean_q)
    ans_np = np_runner.ask(clean_q, creativity=0)
    assert ans_zc == ans_np, f"ZeroCopy ({ans_zc!r}) != QuantizedTny2Runner ({ans_np!r})"
    assert int(np.argmax(pt_logits_all[-1])) == int(np.argmax(zc_stages["logits"]))
    print(f"\nResposta greedy idêntica (C Zero-Copy == NumPy == PyTorch): {ans_zc!r}")


def print_macs_breakdown(d: int = 96, layers: int = 4, hid: int = 256, vocab: int = 97, p: int = 16) -> None:
    print("\n" + "=" * 92)
    print(f"PARTE 2 — ONDE ESTÃO OS CICLOS DE CPU? (1 token na posição p={p})")
    print("=" * 92)
    macs_qkv_wo = layers * 4 * (d * d)
    macs_ffn = layers * 2 * (d * hid)
    macs_head = vocab * d
    total_matvec = macs_qkv_wo + macs_ffn + macs_head

    ops_att = layers * (2 * (p + 1) * d)
    ops_norm_act = layers * (3 * d + hid + 3 * (p + 1)) + 3 * d + vocab
    total_ops = total_matvec + ops_att + ops_norm_act

    rows = [
        ("matvec: Q,K,V,Wo (4 camadas)", macs_qkv_wo),
        ("matvec: FFN f1+f2 (4 camadas)", macs_ffn),
        ("matvec: Head Weight Tying (emb)", macs_head),
        ("Atenção (Q.K + probs.V no KV)", ops_att),
        ("RMSNorm + ReLU + Softmax", ops_norm_act),
    ]
    print(f"{'Operação':<34} | {'MACs / Ops':>12} | {'Participação':>12}")
    print("-" * 66)
    for label, val in rows:
        pct = 100.0 * val / total_ops
        print(f"{label:<34} | {val:>12,} | {pct:>11.2f}%")
    print("-" * 66)
    print(
        f"{'TOTAL matvec()':<34} | {total_matvec:>12,} | "
        f"{100.0 * total_matvec / total_ops:>11.2f}%"
    )
    print("\nConclusão: >95% das multiplicações vivem em matvec() — justificando o SIMD no Módulo 6!")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Espelhamento PyTorch <-> C Zero-Copy do TINY 2.1.")
    parser.add_argument("--prompt", type=str, default="what is 7 times 8", help="Pergunta para rastrear")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bin_path = REPO_ROOT / "TINY.BIN"
    blob = bin_path.read_bytes()
    pt_model, _ = TinyTransformer.from_tny2(bin_path)
    zc_model = ZeroCopyModel(blob)
    np_runner = QuantizedTny2Runner(blob)

    trace_step_comparison(pt_model, zc_model, np_runner, args.prompt)
    print_macs_breakdown(zc_model.d, zc_model.layers, zc_model.hid, 97, p=16)
    return 0


if __name__ == "__main__":
    sys.exit(main())
