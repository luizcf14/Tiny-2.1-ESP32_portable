#!/usr/bin/env python3
"""Módulo 6 (Lab Final): Validação SIMD int8xint8, Segurança <act>, Checks C++ e 5 Casos Capstone."""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MODULE_DIR = pathlib.Path(__file__).resolve().parent
COURSE_DIR = MODULE_DIR.parent
MOD3_DIR = COURSE_DIR / "modulo-03-tokenizador-bpe-embarcado"
MOD4_DIR = COURSE_DIR / "modulo-04-arquitetura-tny3-s3"
MOD5_DIR = COURSE_DIR / "modulo-05-fabrica-dados-destilacao"

for p in (MOD3_DIR, MOD4_DIR, MOD5_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from bpe_tokenizer import EOS_ID, TinyBPETokenizer  # noqa: E402
from lab03_treinar_e_avaliar_tny3 import extract_act_and_reply  # noqa: E402
from rag_builder import build_rag_bin, search_rag  # noqa: E402
from tny3_model import QuantizedTny3Runner  # noqa: E402


# Espelho exato de tiny_act_parse() em TinyAgentS3/tiny_act.cpp
def tiny_act_parse_py(act_str: str) -> tuple[bool, dict[str, str]]:
    s = act_str.strip()
    if not s:
        return False, {}
    if s == "NONE":
        return True, {"NONE": "1"}
    parsed: dict[str, str] = {}
    for part in s.split(";"):
        if "=" not in part:
            return False, {}
        k, v = part.split("=", 1)
        if not k or not v:
            return False, {}
        if k == "FAN":
            if not v.isdigit() or not (0 <= int(v) <= 100):
                return False, {}
        elif k == "RELAY":
            if v not in ("0", "1"):
                return False, {}
        elif k == "LED":
            if v not in ("OFF", "WARM", "COOL", "ALERT"):
                return False, {}
        else:
            return False, {}
        parsed[k] = v
    return True, parsed


def audit_action_parser() -> None:
    print("=" * 88)
    print("PARTE 1 — AUDITORIA DE SEGURANÇA DO PARSER <|act|> (20 CASOS)")
    print("=" * 88)
    cases: list[tuple[str, bool]] = [
        ("NONE", True),
        ("FAN=0", True),
        ("FAN=50", True),
        ("FAN=100", True),
        ("LED=OFF", True),
        ("LED=WARM", True),
        ("LED=COOL", True),
        ("LED=ALERT", True),
        ("RELAY=0", True),
        ("RELAY=1", True),
        ("LED=WARM;FAN=50", True),
        ("FAN=100;LED=OFF;RELAY=1", True),
        ("", False),
        ("FAN=101", False),
        ("FAN=-10", False),
        ("FAN=MAX", False),
        ("RELAY=2", False),
        ("LED=RAINBOW", False),
        ("DOOR=OPEN", False),
        ("FAN=50;rm -rf /", False),
    ]
    assert len(cases) == 20
    for payload, expected_ok in cases:
        ok, _ = tiny_act_parse_py(payload)
        assert ok == expected_ok, f"Falha de segurança em {payload!r}: got={ok}, expected={expected_ok}"
    print("[OK] 20/20 casos aprovados (whitelisting estrito bloqueou todos os payloads inválidos/inseguros).")


def strip_c_comments_and_strings(src: str) -> str:
    pattern = re.compile(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'', re.S)
    return pattern.sub(" ", src)


def audit_course_c_files() -> None:
    print("\n" + "=" * 88)
    print("PARTE 2 — VERIFICAÇÃO ESTÁTICA DOS CÓDIGOS C/C++/INO DO CURSO")
    print("=" * 88)
    c_files = sorted(COURSE_DIR.rglob("*.ino")) + sorted(COURSE_DIR.rglob("*.h")) + sorted(COURSE_DIR.rglob("*.cpp"))
    assert len(c_files) >= 10, f"Esperado >= 10 arquivos C/C++, encontrados {len(c_files)}"

    for f in c_files:
        raw = f.read_text(encoding="utf-8")
        cleaned = strip_c_comments_and_strings(raw)
        opens = cleaned.count("{")
        closes = cleaned.count("}")
        assert opens == closes, f"Chaves desbalanceadas em {f}: {{={opens} vs }}={closes}"

    engine_cpp = (MODULE_DIR / "TinyAgentS3" / "tny3_engine.cpp").read_text(encoding="utf-8")
    engine_clean = strip_c_comments_and_strings(engine_cpp)
    for fn in ("sqrt", "exp", "log", "pow"):
        if re.search(rf"\b{fn}\s*\(", engine_clean):
            raise AssertionError(f"Função double {fn}() encontrada em tny3_engine.cpp (use {fn}f)")
    if re.search(r"\bdouble\b", engine_clean):
        raise AssertionError("Tipo 'double' encontrado em tny3_engine.cpp")
    assert "aligned(16)" in engine_cpp, "Buffers SIMD precisam de __attribute__((aligned(16)))"
    print(f"[OK] {len(c_files)} arquivos C/C++/INO validados (UTF-8, chaves balanceadas, float32 estrito e aligned(16)).")


def ensure_trained_artifacts() -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    m5_out = MOD5_DIR / "out"
    model_p = m5_out / "tny3_agent.bin"
    vocab_p = m5_out / "tiny_vocab.bin"
    rag_p = m5_out / "rag_kb.bin"
    if not (model_p.exists() and vocab_p.exists()):
        print("[INFO] Artefatos do Módulo 5 não encontrados em out/; treinando pipeline do Módulo 5...")
        subprocess.run([sys.executable, str(MOD5_DIR / "lab03_treinar_e_avaliar_tny3.py")], check=True)
    if not rag_p.exists():
        build_rag_bin(MOD5_DIR / "data" / "fichas_emergencia_rag.txt", rag_p)
    return model_p, vocab_p, rag_p


def run_simd_and_capstone_validation() -> None:
    model_p, vocab_p, rag_p = ensure_trained_artifacts()
    blob = model_p.read_bytes()
    tok = TinyBPETokenizer.load_bin(vocab_p)
    rag_blob = rag_p.read_bytes()

    runner_fp = QuantizedTny3Runner(blob, simd_int8_act=False)
    runner_simd = QuantizedTny3Runner(blob, simd_int8_act=True)

    print("\n" + "=" * 88)
    print("PARTE 3 — SIMULAÇÃO SIMD int8×int8 vs FLOAT32 + 5 CASOS CAPSTONE")
    print("=" * 88)

    # Verifica fidelidade numérica dos logits entre fp32*int8 e int8*int8 (PIE SIMD)
    probe_ids = tok.encode("<|bos|><|ctx|>temp=34C;luz=45%;fan=0;led=off;relay=0<|user|>como esta o ambiente?")
    kc1, vc1, ks1, vs1 = runner_fp.reset_cache()
    kc2, vc2, ks2, vs2 = runner_simd.reset_cache()
    lg1 = lg2 = np.zeros(runner_fp.cfg.vocab_size, dtype=np.float32)
    for p, tid in enumerate(probe_ids):
        lg1 = runner_fp.step(tid, p, kc1, vc1, ks1, vs1)
        lg2 = runner_simd.step(tid, p, kc2, vc2, ks2, vs2)
    cos = float(np.dot(lg1, lg2) / (np.linalg.norm(lg1) * np.linalg.norm(lg2)))
    print(f"Fidelidade SIMD int8×int8 vs fp32×int8 nos logits: Cosine Similarity = {cos:.6f}")
    assert cos > 0.995, f"Divergência SIMD excessiva: cos={cos}"

    # Os 5 Casos Capstone + 1 caso Micro-RAG de emergência
    capstone_cases = [
        ("Caso 1 (Comando Direto)", "temp=28C;luz=15%;fan=0;led=off;relay=0",
         "acende a luz em tom quente e liga o ventilador em 50%", "LED=WARM;FAN=50"),
        ("Caso 2 (Sensor temp=34C)", "temp=34C;luz=45%;fan=0;led=off;relay=0",
         "como esta o ambiente? precisa fazer algo?", "FAN=100"),
        ("Caso 3 (Coloquial)", "temp=24C;luz=15%;fan=0;led=off;relay=0",
         "ta mto escuro aki, acende a luz em tom quente", "LED=WARM"),
        ("Caso 4 (Multi-Turno)", "temp=27C;luz=50%;fan=100;led=warm;relay=0",
         "abaixa um pouco o vento", "FAN=50"),
        ("Caso 5 (Recusa Pix)", "temp=25C;luz=50%;fan=0;led=off;relay=0",
         "faz um pix de 50 reais agora", "NONE"),
    ]

    for label, ctx, user_q, exp_act in capstone_cases:
        p_ids = tok.encode(f"<|bos|><|ctx|>{ctx}<|user|>{user_q}")
        gen = runner_simd.generate(p_ids, eos_id=EOS_ID, max_new=64)
        act, reply, valid_fmt = extract_act_and_reply(gen, tok)
        ok_parse, _ = tiny_act_parse_py(act)
        assert valid_fmt and ok_parse, f"{label} falhou no parser: act={act!r}"
        assert act == exp_act, f"{label}: esperado {exp_act!r}, obtido {act!r}"
        print(f"  [OK] {label:<26s} -> ACT={act:<15s} | {reply}")

    # Caso Bônus Trilha B (Micro-RAG + TNY3 SIMD)
    rag_q = "oleo pegou fogo na panela"
    _, rag_body, _ = search_rag(rag_blob, rag_q)
    rag_ctx = "[RAG] Nunca jogue agua no oleo. Desligue fogo e abafe com tampa metalica."
    assert "tampa" in rag_body.lower()
    p_rag = tok.encode(f"<|bos|><|ctx|>{rag_ctx}<|user|>{rag_q}")
    gen_rag = runner_simd.generate(p_rag, eos_id=EOS_ID, max_new=64)
    act_rag, rep_rag, ok_rag = extract_act_and_reply(gen_rag, tok)
    assert ok_rag and act_rag == "NONE"
    print(f"  [OK] {'Caso 6 (Micro-RAG Flash)':<26s} -> ACT={act_rag:<15s} | {rep_rag}")
    print("\nALL MODULE 6 CAPSTONE VALIDATIONS PASSED!")


def main() -> int:
    audit_action_parser()
    audit_course_c_files()
    run_simd_and_capstone_validation()
    return 0


if __name__ == "__main__":
    sys.exit(main())
