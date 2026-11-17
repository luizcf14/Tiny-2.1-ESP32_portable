#!/usr/bin/env python3
"""Fine-tuning script for TINY 2.1 (TNY2 int8 Transformer) + ESP32 integration."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
import struct
import sys
import time
import unicodedata
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from generate_esp32_assets import generate_weights_header, validate_tiny_bin

VOCAB = 97
CTX = 128
MAXQ = 72
PAD_TOK = 0
SEP_TOK = 1

ANCHOR_QUESTIONS = [
    "what is the capital of peru",
    "what is the capital of france",
    "is a whale a fish",
    "what is 7 times 8",
    "what is 9 times 6",
    "what is 15 minus 6",
    "who invented the telephone",
    "spell elephant",
    "opposite of hot",
    "what is the symbol for gold",
    "how many legs does a spider have",
    "what sound does a cow make",
    "what is a baby kangaroo called",
    "how many moons does mars have",
    "how do you say thank you in spanish",
    "where is the eiffel tower",
    "what is the biggest ocean",
    "plural of mouse",
    "past tense of go",
    "what comes after 199",
    "which is bigger, 405 or 450",
    "is 31 even or odd",
    "what is 14 in roman numerals",
    "what is dos",
    "how many strings does a guitar have",
    "what is a group of lions called",
    "what is the fastest land animal",
    "which animal says hoot",
    "what is another word for big",
    "what is the superlative of good",
]

EVAL_CONTROLS = [
    "what is the capital of peru",
    "is a whale a fish",
    "what is 7 times 8",
    "who invented the telephone",
    "opposite of hot",
    "what is the symbol for gold",
]


def to_ascii(text: str) -> str:
    norm = unicodedata.normalize("NFKD", text)
    ascii_bytes = norm.encode("ascii", "ignore").decode("ascii")
    cleaned = "".join(ch for ch in ascii_bytes if 32 <= ord(ch) <= 126)
    return " ".join(cleaned.split())


def normalize_question(text: str) -> str:
    s = to_ascii(text).lower()
    out: list[str] = []
    for ch in s:
        if "a" <= ch <= "z" and len(out) >= 2 and out[-1] == ch and out[-2] == ch:
            continue
        out.append(ch)
    return "".join(out)


def perturb_question(q: str) -> str:
    if len(q) < 5:
        return q
    inner = [i for i in range(1, len(q) - 1) if "a" <= q[i] <= "z"]
    if not inner:
        return q
    idx = random.choice(inner)
    op = random.randrange(3)
    if op == 0:
        return q[: idx + 1] + q[idx] + q[idx + 1 :]
    if op == 1 and len(q) >= 6:
        return q[:idx] + q[idx + 1 :]
    if idx + 1 < len(q) - 1 and "a" <= q[idx + 1] <= "z":
        return q[:idx] + q[idx + 1] + q[idx] + q[idx + 2 :]
    return q[: idx + 1] + q[idx] + q[idx + 1 :]


def encode_text(text: str) -> list[int]:
    return [ord(ch) - 30 for ch in text if 32 <= ord(ch) <= 126]


def decode_tokens(tokens: Iterable[int]) -> str:
    return "".join(chr(t + 30) for t in tokens if 2 <= t < VOCAB)


def ste_int8(x: torch.Tensor) -> torch.Tensor:
    big = x.detach().abs().amax(dim=-1, keepdim=True).clamp(min=1e-20)
    scale = big / 127.0
    xq = (x / scale).round().clamp(-127.0, 127.0) * scale
    return x + (xq - x).detach()


def rms_norm(x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    inv = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-5)
    return x * inv * g


class TinyTransformer(nn.Module):
    def __init__(self, d: int, layers: int, heads: int, hid: int) -> None:
        super().__init__()
        self.d = d
        self.layers = layers
        self.heads = heads
        self.hd = d // heads
        self.hid = hid

        self.emb = nn.Parameter(torch.empty(VOCAB, d))
        self.pos = nn.Parameter(torch.empty(CTX, d))
        self.g1 = nn.ParameterList([nn.Parameter(torch.ones(d)) for _ in range(layers)])
        self.wq = nn.ParameterList([nn.Parameter(torch.empty(d, d)) for _ in range(layers)])
        self.wk = nn.ParameterList([nn.Parameter(torch.empty(d, d)) for _ in range(layers)])
        self.wv = nn.ParameterList([nn.Parameter(torch.empty(d, d)) for _ in range(layers)])
        self.wo = nn.ParameterList([nn.Parameter(torch.empty(d, d)) for _ in range(layers)])
        self.g2 = nn.ParameterList([nn.Parameter(torch.ones(d)) for _ in range(layers)])
        self.f1 = nn.ParameterList([nn.Parameter(torch.empty(hid, d)) for _ in range(layers)])
        self.f2 = nn.ParameterList([nn.Parameter(torch.empty(d, hid)) for _ in range(layers)])
        self.gf = nn.Parameter(torch.ones(d))

        causal = torch.triu(torch.ones(CTX, CTX, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", causal, persistent=False)

    def forward(self, tokens: torch.Tensor, qat: bool = True) -> torch.Tensor:
        bsz, seq_len = tokens.shape
        emb_w = ste_int8(self.emb) if qat else self.emb
        pos_w = ste_int8(self.pos[:seq_len]) if qat else self.pos[:seq_len]
        x = F.embedding(tokens, emb_w) + pos_w.unsqueeze(0)

        mask = self.causal_mask[:seq_len, :seq_len]
        att_div = math.sqrt(self.hd)

        for l in range(self.layers):
            h = rms_norm(x, self.g1[l])
            wq = ste_int8(self.wq[l]) if qat else self.wq[l]
            wk = ste_int8(self.wk[l]) if qat else self.wk[l]
            wv = ste_int8(self.wv[l]) if qat else self.wv[l]
            wo = ste_int8(self.wo[l]) if qat else self.wo[l]

            q = F.linear(h, wq)
            k = F.linear(h, wk)
            v = F.linear(h, wv)
            if qat:
                k = ste_int8(k)
                v = ste_int8(v)

            qh = q.view(bsz, seq_len, self.heads, self.hd).transpose(1, 2)
            kh = k.view(bsz, seq_len, self.heads, self.hd).transpose(1, 2)
            vh = v.view(bsz, seq_len, self.heads, self.hd).transpose(1, 2)

            att = torch.matmul(qh, kh.transpose(-2, -1)) / att_div
            att = att.masked_fill(mask.unsqueeze(0).unsqueeze(0), -1e9)
            probs = F.softmax(att, dim=-1)
            ba = torch.matmul(probs, vh).transpose(1, 2).contiguous().view(bsz, seq_len, self.d)
            x = x + F.linear(ba, wo)

            h2 = rms_norm(x, self.g2[l])
            f1 = ste_int8(self.f1[l]) if qat else self.f1[l]
            f2 = ste_int8(self.f2[l]) if qat else self.f2[l]
            bm = F.relu(F.linear(h2, f1))
            x = x + F.linear(bm, f2)

        hf = rms_norm(x, self.gf)
        return F.linear(hf, emb_w)

    @classmethod
    def from_tny2(cls, bin_path: pathlib.Path) -> tuple["TinyTransformer", bytes]:
        data = bin_path.read_bytes()
        d, layers, heads, hid = validate_tiny_bin(data)
        model = cls(d, layers, heads, hid)
        off = 12

        def read_table(rows: int, cols: int) -> torch.Tensor:
            nonlocal off
            stride = 4 + cols
            arr = np.empty((rows, cols), dtype=np.float32)
            for r in range(rows):
                (scale,) = struct.unpack_from("<f", data, off)
                w = np.frombuffer(data, dtype=np.int8, count=cols, offset=off + 4)
                arr[r] = w.astype(np.float32) * np.float32(scale)
                off += stride
            return torch.from_numpy(arr)

        def read_gains() -> torch.Tensor:
            nonlocal off
            arr = np.frombuffer(data, dtype=np.float32, count=d, offset=off).copy()
            off += 4 * d
            return torch.from_numpy(arr)

        with torch.no_grad():
            model.emb.copy_(read_table(VOCAB, d))
            model.pos.copy_(read_table(CTX, d))
            for l in range(layers):
                model.g1[l].copy_(read_gains())
                model.wq[l].copy_(read_table(d, d))
                model.wk[l].copy_(read_table(d, d))
                model.wv[l].copy_(read_table(d, d))
                model.wo[l].copy_(read_table(d, d))
                model.g2[l].copy_(read_gains())
                model.f1[l].copy_(read_table(hid, d))
                model.f2[l].copy_(read_table(d, hid))
            model.gf.copy_(read_gains())

        if off != len(data):
            raise ValueError(f"Offset mismatch reading {bin_path}: {off} != {len(data)}")
        return model, data

    def export_tny2(self, out_path: pathlib.Path) -> bytes:
        chunks: list[bytes] = [
            b"TNY2",
            struct.pack("<HHHH", self.d, self.layers, self.heads, self.hid),
        ]

        def write_table(param: torch.Tensor) -> None:
            arr = param.detach().cpu().float().numpy()
            big = np.maximum(np.max(np.abs(arr), axis=1), np.float32(1e-20)).astype(np.float32)
            scale = (big / np.float32(127.0)).astype(np.float32)
            v = arr * (np.float32(127.0) / big[:, None])
            q = np.where(v < 0.0, v - 0.5, v + 0.5)
            q_i8 = np.clip(np.trunc(q), -127, 127).astype(np.int8)
            for r in range(arr.shape[0]):
                chunks.append(struct.pack("<f", float(scale[r])))
                chunks.append(q_i8[r].tobytes())

        def write_gains(param: torch.Tensor) -> None:
            arr = param.detach().cpu().float().numpy().astype("<f4")
            chunks.append(arr.tobytes())

        write_table(self.emb)
        write_table(self.pos)
        for l in range(self.layers):
            write_gains(self.g1[l])
            write_table(self.wq[l])
            write_table(self.wk[l])
            write_table(self.wv[l])
            write_table(self.wo[l])
            write_gains(self.g2[l])
            write_table(self.f1[l])
            write_table(self.f2[l])
        write_gains(self.gf)

        blob = b"".join(chunks)
        validate_tiny_bin(blob)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(blob)
        return blob


class QuantizedTny2Runner:
    """Exact int8 quantized inference matching TINY.C / tiny_engine.cpp."""

    def __init__(self, blob: bytes) -> None:
        self.d, self.layers, self.heads, self.hid = validate_tiny_bin(blob)
        self.hd = self.d // self.heads
        self.att_div = math.sqrt(self.hd)
        off = 12

        def load_table(rows: int, cols: int) -> tuple[np.ndarray, np.ndarray]:
            nonlocal off
            scales = np.empty(rows, dtype=np.float32)
            weights = np.empty((rows, cols), dtype=np.int8)
            stride = 4 + cols
            for r in range(rows):
                scales[r] = struct.unpack_from("<f", blob, off)[0]
                weights[r] = np.frombuffer(blob, dtype=np.int8, count=cols, offset=off + 4)
                off += stride
            return scales, weights

        def load_gains() -> np.ndarray:
            nonlocal off
            g = np.frombuffer(blob, dtype=np.float32, count=self.d, offset=off).copy()
            off += 4 * self.d
            return g

        self.emb = load_table(VOCAB, self.d)
        self.pos = load_table(CTX, self.d)
        self.g1, self.wq, self.wk, self.wv, self.wo, self.g2, self.f1, self.f2 = (
            [], [], [], [], [], [], [], []
        )
        for _ in range(self.layers):
            self.g1.append(load_gains())
            self.wq.append(load_table(self.d, self.d))
            self.wk.append(load_table(self.d, self.d))
            self.wv.append(load_table(self.d, self.d))
            self.wo.append(load_table(self.d, self.d))
            self.g2.append(load_gains())
            self.f1.append(load_table(self.hid, self.d))
            self.f2.append(load_table(self.d, self.hid))
        self.gf = load_gains()

    @staticmethod
    def matvec(tbl: tuple[np.ndarray, np.ndarray], x: np.ndarray) -> np.ndarray:
        scales, w = tbl
        return (w.astype(np.float32) @ x) * scales

    def norm(self, x: np.ndarray, g: np.ndarray) -> np.ndarray:
        s = float(np.dot(x, x))
        inv = np.float32(1.0 / math.sqrt(s / self.d + 1e-5))
        return x * inv * g

    @staticmethod
    def store_kv(src: np.ndarray) -> tuple[np.ndarray, float]:
        big = float(np.max(np.abs(src)))
        if big < 1e-20:
            big = 1e-20
        v = src * np.float32(127.0 / big)
        q = np.where(v < 0.0, v - 0.5, v + 0.5)
        q_i8 = np.clip(np.trunc(q), -127, 127).astype(np.int8)
        return q_i8, big / 127.0

    def ask(self, question: str, creativity: int = 0) -> str:
        clean_q = normalize_question(question)[:MAXQ]

        kc = [np.zeros((CTX, self.d), dtype=np.int8) for _ in range(self.layers)]
        vc = [np.zeros((CTX, self.d), dtype=np.int8) for _ in range(self.layers)]
        ks = [np.zeros(CTX, dtype=np.float32) for _ in range(self.layers)]
        vs = [np.zeros(CTX, dtype=np.float32) for _ in range(self.layers)]

        emb_s, emb_w = self.emb
        pos_s, pos_w = self.pos

        def step(tok: int, p: int) -> np.ndarray:
            bx = emb_w[tok].astype(np.float32) * emb_s[tok] + pos_w[p].astype(np.float32) * pos_s[p]
            for l in range(self.layers):
                bh = self.norm(bx, self.g1[l])
                kc[l][p], ks[l][p] = self.store_kv(self.matvec(self.wk[l], bh))
                vc[l][p], vs[l][p] = self.store_kv(self.matvec(self.wv[l], bh))
                bq = self.matvec(self.wq[l], bh)

                ba = np.zeros(self.d, dtype=np.float32)
                for h in range(self.heads):
                    s0, s1 = h * self.hd, (h + 1) * self.hd
                    qh = bq[s0:s1]
                    k_mat = kc[l][: p + 1, s0:s1].astype(np.float32)
                    sc = ((k_mat @ qh) * ks[l][: p + 1]) / np.float32(self.att_div)
                    big = float(np.max(sc))
                    exps = np.where(sc - big < -60.0, 0.0, np.exp(sc - big)).astype(np.float32)
                    total = float(np.sum(exps))
                    if total > 0.0:
                        w = (exps * vs[l][: p + 1] / np.float32(total)).astype(np.float32)
                        v_mat = vc[l][: p + 1, s0:s1].astype(np.float32)
                        ba[s0:s1] = w @ v_mat

                bx = bx + self.matvec(self.wo[l], ba)
                bh = self.norm(bx, self.g2[l])
                bm = np.maximum(self.matvec(self.f1[l], bh), 0.0)
                bx = bx + self.matvec(self.f2[l], bm)

            bh = self.norm(bx, self.gf)
            return self.matvec(self.emb, bh)

        p = 0
        for ch in clean_q:
            tok = ord(ch) - 30
            if tok < 2 or tok >= VOCAB:
                tok = 2
            score = step(tok, p)
            p += 1
        score = step(SEP_TOK, p)
        p += 1

        out: list[str] = []
        while p < CTX:
            best = int(np.argmax(score))
            if creativity <= 0:
                tok = best
            else:
                t = max(1, min(9, creativity)) / 5.0
                diff = (score - score[best]) / t
                probs = np.where(diff < -60.0, 0.0, np.exp(diff))
                probs = np.where(probs < 0.22, 0.0, probs)
                total = float(np.sum(probs))
                if total <= 0.0:
                    tok = best
                else:
                    r = total * (random.randrange(10000) / 10000.0)
                    tok = best
                    for i in range(VOCAB):
                        r -= float(probs[i])
                        if r < 0.0 and probs[i] > 0.0:
                            tok = i
                            break
            if tok <= 1:
                break
            out.append(chr(tok + 30))
            score = step(tok, p)
            p += 1

        return "".join(out)


def detect_eos_token(model: TinyTransformer, device: torch.device) -> int:
    probe = encode_text("what is 7 times 8") + [SEP_TOK] + encode_text("7 times 8 is 56.")
    tokens = torch.tensor([probe], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(tokens, qat=False)
    l0 = float(logits[0, -1, 0].item())
    l1 = float(logits[0, -1, 1].item())
    eos = 0 if l0 >= l1 else 1
    print(f"[INFO] Detected base model stop token: {eos} (logit[0]={l0:.3f}, logit[1]={l1:.3f})")
    return eos


def load_qa_file(path: pathlib.Path) -> list[tuple[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    raw = path.read_text(encoding="utf-8")
    pairs: list[tuple[str, str]] = []
    suffix = path.suffix.lower()

    def add_pair(q_val: object, a_val: object) -> None:
        if not isinstance(a_val, str):
            return
        a_clean = to_ascii(a_val).strip()
        if not a_clean:
            return
        q_list = q_val if isinstance(q_val, list) else [q_val]
        for q_item in q_list:
            if isinstance(q_item, str):
                q_clean = normalize_question(q_item).strip()
                if q_clean:
                    pairs.append((q_clean, a_clean))

    if suffix == ".jsonl":
        for line in raw.splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            obj = json.loads(s)
            q_val = obj.get("q") or obj.get("question") or obj.get("prompt")
            a_val = obj.get("a") or obj.get("answer") or obj.get("completion")
            add_pair(q_val, a_val)
    elif suffix == ".json":
        data = json.loads(raw)
        if isinstance(data, dict):
            data = [data]
        for obj in data:
            q_val = obj.get("q") or obj.get("question") or obj.get("prompt")
            a_val = obj.get("a") or obj.get("answer") or obj.get("completion")
            add_pair(q_val, a_val)
    else:
        for line in raw.splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if "|" in s:
                q_s, a_s = s.split("|", 1)
            elif "\t" in s:
                q_s, a_s = s.split("\t", 1)
            else:
                continue
            for q_part in q_s.split(";"):
                add_pair(q_part, a_s)

    return pairs


def augment_pairs(pairs: list[tuple[str, str]], enable: bool) -> list[tuple[str, str]]:
    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, str]] = []

    def push(q: str, a: str) -> None:
        key = (q, a)
        if q and a and key not in seen:
            seen.add(key)
            out.append(key)

    for q, a in pairs:
        push(q, a)
        if enable:
            q_no_qmark = q[:-1].strip() if q.endswith("?") else q
            push(q_no_qmark, a)
            push(q_no_qmark + "?", a)
            push(q_no_qmark.lower(), a)
            push(q_no_qmark.lower() + "?", a)

    return out


def build_training_sample(q: str, a: str, eos_tok: int) -> tuple[list[int], list[int]]:
    q_toks = encode_text(q[:MAXQ])
    max_a = max(1, CTX - len(q_toks) - 2)
    a_toks = encode_text(a)[:max_a]
    inp = q_toks + [SEP_TOK] + a_toks
    tgt = ([-100] * len(q_toks)) + a_toks + [eos_tok]
    return inp, tgt


def collate_batch(
    samples: list[tuple[list[int], list[int]]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(x) for x, _ in samples)
    bsz = len(samples)
    x_np = np.full((bsz, max_len), PAD_TOK, dtype=np.int64)
    y_np = np.full((bsz, max_len), -100, dtype=np.int64)
    for i, (inp, tgt) in enumerate(samples):
        n = len(inp)
        x_np[i, :n] = inp
        y_np[i, :n] = tgt
    non_block = device.type == "cuda"
    return (
        torch.from_numpy(x_np).to(device, non_blocking=non_block),
        torch.from_numpy(y_np).to(device, non_blocking=non_block),
    )


def select_device(requested: str, seed: int) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            dev = torch.device("cuda")
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            dev = torch.device("xpu")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            dev = torch.device("mps")
        else:
            dev = torch.device("cpu")
    else:
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA solicitado, mas o PyTorch instalado ({torch.__version__}) nao tem suporte a GPU.\n"
                "Para ativar GPUs NVIDIA (incluindo RTX 5060 Ti Blackwell / sm_120), execute:\n"
                "  python -m pip install --upgrade --force-reinstall torch --index-url https://download.pytorch.org/whl/cu128"
            )
        if requested == "xpu" and not (hasattr(torch, "xpu") and torch.xpu.is_available()):
            raise RuntimeError("Dispositivo XPU nao disponivel neste ambiente.")
        if requested == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("Dispositivo MPS nao disponivel neste ambiente.")
        dev = torch.device(requested)

    if dev.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.set_float32_matmul_precision("high")
        print(f"[DEVICE] GPU ativa: {torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda})")
    elif dev.type == "xpu":
        print(f"[DEVICE] GPU XPU ativa: {torch.xpu.get_device_name(0)}")
    elif dev.type == "mps":
        print("[DEVICE] GPU Apple MPS ativa")
    else:
        if torch.version.cuda is None:
            print(
                f"[DEVICE] CPU ({torch.__version__} sem CUDA). Para habilitar GPU NVIDIA, reinstale torch com CUDA:\n"
                "         python -m pip install --upgrade --force-reinstall torch --index-url https://download.pytorch.org/whl/cu128"
            )
        else:
            print(f"[DEVICE] CPU ({torch.__version__})")
    return dev


def build_replay_pairs(base_blob: bytes) -> list[tuple[str, str]]:
    runner = QuantizedTny2Runner(base_blob)
    replay: list[tuple[str, str]] = []
    for q in ANCHOR_QUESTIONS:
        ans = runner.ask(q, creativity=0).strip()
        if len(ans) >= 2:
            replay.append((q, ans))
    return replay


def run_chat(model_path: pathlib.Path) -> int:
    blob = model_path.read_bytes()
    runner = QuantizedTny2Runner(blob)
    creativity = 0
    print(f"TINY 2.1 Chat (quantized int8) - {model_path}")
    print("Commands: /0 .. /9 (creativity), quit / exit")
    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            return 0
        if not line:
            continue
        if line.lower() in ("quit", "exit"):
            return 0
        if len(line) == 2 and line[0] == "/" and "0" <= line[1] <= "9":
            creativity = int(line[1])
            print(f"creativity = {creativity}")
            continue
        t0 = time.perf_counter()
        ans = runner.ask(line, creativity=creativity)
        dt = (time.perf_counter() - t0) * 1000.0
        print(f"  {ans}   [{len(ans)} chars in {dt:.0f} ms]")


def parse_args() -> argparse.Namespace:
    root = pathlib.Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description="Fine-tune TINY 2.1 (TNY2) and optionally update ESP32 weights.")
    p.add_argument("--data", type=pathlib.Path, action="append", help="Q&A dataset (.txt, .tsv, .jsonl, .json)")
    p.add_argument("--base", type=pathlib.Path, default=root / "TINY.BIN", help="Base TNY2 binary (default: TINY.BIN)")
    p.add_argument("--out", type=pathlib.Path, default=root / "models" / "tiny_finetuned.bin", help="Output TNY2 binary")
    p.add_argument("--epochs", type=int, default=40, help="Training epochs (default: 40)")
    p.add_argument("--batch-size", type=int, default=32, help="Mini-batch size (default: 32)")
    p.add_argument("--lr", type=float, default=4e-4, help="Peak learning rate (default: 4e-4)")
    p.add_argument("--l2-sp", type=float, default=0.005, help="L2-SP anchor regularization strength (default: 0.005)")
    p.add_argument("--replay-ratio", type=float, default=0.25, help="Replay ratio of base facts (default: 0.25)")
    p.add_argument("--freeze-layers", type=int, default=0, help="Freeze bottom K transformer layers (default: 0)")
    p.add_argument("--no-augment", action="store_true", help="Disable question punctuation/case augmentation")
    p.add_argument("--no-qat", action="store_true", help="Disable int8 quantization-aware training")
    p.add_argument("--char-noise", type=float, default=0.10, help="Fraction of samples per epoch with light typo noise (default: 0.10)")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p.add_argument("--device", choices=["auto", "cuda", "mps", "xpu", "cpu"], default="auto", help="Compute device (default: auto)")
    p.add_argument("--update-esp32", action="store_true", help="Write TinyESP32/tiny_weights.h from output model")
    p.add_argument("--restore-esp32", action="store_true", help="Restore TinyESP32/tiny_weights.h from --base (TINY.BIN)")
    p.add_argument("--chat", action="store_true", help="Interactive quantized chat in terminal")
    p.add_argument("--model", type=pathlib.Path, default=None, help="Model binary for --chat")
    return p.parse_args()


def main() -> int:
    root = pathlib.Path(__file__).resolve().parent.parent
    args = parse_args()

    def resolve(p: pathlib.Path) -> pathlib.Path:
        return p if p.is_absolute() else (root / p)

    base_path = resolve(args.base)

    if args.restore_esp32:
        hdr = root / "TinyESP32" / "tiny_weights.h"
        size, d, l, h, hid = generate_weights_header(base_path, hdr)
        print(f"[OK] Restored {hdr} from {base_path} ({size} bytes, D={d}, L={l}, H={h}, HID={hid})")
        return 0

    if args.chat:
        chat_model = resolve(args.model) if args.model else base_path
        return run_chat(chat_model)

    if not args.data:
        print("Error: specify --data <file> (e.g. --data data/custom_qa.txt), or use --chat / --restore-esp32.", file=sys.stderr)
        return 2

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    raw_pairs: list[tuple[str, str]] = []
    for dpath in args.data:
        raw_pairs.extend(load_qa_file(resolve(dpath)))
    if not raw_pairs:
        print("Error: no valid Q&A pairs found in dataset.", file=sys.stderr)
        return 2

    custom_pairs = augment_pairs(raw_pairs, enable=not args.no_augment)
    print(f"[DATA] Loaded {len(raw_pairs)} base Q&A pairs -> {len(custom_pairs)} training pairs (augment={not args.no_augment})")

    device = select_device(args.device, args.seed)
    model, base_blob = TinyTransformer.from_tny2(base_path)
    model = model.to(device)
    eos_tok = detect_eos_token(model, device)

    if args.freeze_layers > 0:
        k = min(args.freeze_layers, model.layers)
        for l in range(k):
            for p_list in (model.g1, model.wq, model.wk, model.wv, model.wo, model.g2, model.f1, model.f2):
                p_list[l].requires_grad_(False)
        print(f"[INFO] Froze bottom {k} of {model.layers} layers")

    custom_samples = [build_training_sample(q, a, eos_tok) for q, a in custom_pairs]
    replay_samples: list[tuple[list[int], list[int]]] = []
    if args.replay_ratio > 0.0:
        t_rep = time.perf_counter()
        replay_pairs = build_replay_pairs(base_blob)
        replay_samples = [build_training_sample(q, a, eos_tok) for q, a in replay_pairs]
        print(f"[REPLAY] Self-distilled {len(replay_samples)} anchor facts from {base_path.name} in {time.perf_counter() - t_rep:.2f}s")

    ref_params = {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}
    trainable = [p for p in model.parameters() if p.requires_grad]

    n_replay_per_epoch = int(round(len(custom_samples) * max(0.0, args.replay_ratio))) if replay_samples else 0
    steps_per_epoch = max(1, math.ceil((len(custom_samples) + n_replay_per_epoch) / args.batch_size))
    total_steps = max(1, args.epochs * steps_per_epoch)
    warmup_steps = max(1, int(0.1 * total_steps))

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    qat = not args.no_qat
    t_start = time.perf_counter()
    print(f"[TRAIN] Starting {args.epochs} epochs (QAT={qat}, L2-SP={args.l2_sp}, lr={args.lr})...")

    for epoch in range(1, args.epochs + 1):
        if args.char_noise > 0.0:
            epoch_samples = [
                build_training_sample(perturb_question(q), a, eos_tok)
                if (len(q) >= 5 and random.random() < args.char_noise)
                else custom_samples[idx]
                for idx, (q, a) in enumerate(custom_pairs)
            ]
        else:
            epoch_samples = list(custom_samples)
        if n_replay_per_epoch > 0:
            chosen = [replay_samples[i % len(replay_samples)] for i in range(n_replay_per_epoch)]
            random.shuffle(replay_samples)
            epoch_samples.extend(chosen)
        random.shuffle(epoch_samples)
        window = args.batch_size * 8
        bucketed: list[tuple[list[int], list[int]]] = []
        for w_start in range(0, len(epoch_samples), window):
            chunk = epoch_samples[w_start : w_start + window]
            chunk.sort(key=lambda item: len(item[0]))
            bucketed.extend(chunk)
        batches = [bucketed[i : i + args.batch_size] for i in range(0, len(bucketed), args.batch_size)]
        random.shuffle(batches)

        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in batches:
            x_t, y_t = collate_batch(batch, device)
            logits = model(x_t, qat=qat)
            ce_loss = F.cross_entropy(
                logits.view(-1, VOCAB),
                y_t.view(-1),
                ignore_index=-100,
                label_smoothing=0.005,
            )

            loss = ce_loss
            if args.l2_sp > 0.0:
                l2_sp = sum((p - ref_params[name]).pow(2).mean() for name, p in model.named_parameters() if p.requires_grad)
                loss = loss + args.l2_sp * (l2_sp / len(ref_params))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += float(ce_loss.item())
            n_batches += 1

        if epoch == 1 or epoch % max(1, args.epochs // 8) == 0 or epoch == args.epochs:
            avg_ce = total_loss / max(1, n_batches)
            print(f"  Epoch {epoch:3d}/{args.epochs} | CE loss: {avg_ce:.4f} | lr: {scheduler.get_last_lr()[0]:.2e}")

    out_path = resolve(args.out)
    out_blob = model.export_tny2(out_path)
    dt_train = time.perf_counter() - t_start
    print(f"[EXPORT] Saved {out_path} ({len(out_blob)} bytes) in {dt_train:.2f}s")

    print("\n[EVAL] Quantized int8 verification (identical to ESP32):")
    runner = QuantizedTny2Runner(out_blob)
    sample_eval = raw_pairs[: min(8, len(raw_pairs))]
    hits = 0
    for q, target in sample_eval:
        pred = runner.ask(q, creativity=0)
        ok = pred.strip().lower() == target.strip().lower()
        hits += int(ok)
        mark = "OK" if ok else "DIFF"
        print(f"  [{mark}] Q: {q}\n         Expected: {target}\n         Got:      {pred}")
    print(f"  Custom exact match: {hits}/{len(sample_eval)}")

    print("\n[CONTROL] Pretrained knowledge check:")
    for cq in EVAL_CONTROLS:
        print(f"  Q: {cq} -> {runner.ask(cq, creativity=0)}")

    if args.update_esp32:
        hdr = root / "TinyESP32" / "tiny_weights.h"
        size, d, l, h, hid = generate_weights_header(out_path, hdr)
        print(f"\n[ESP32] Updated {hdr} from {out_path.name} ({size} bytes, D={d}, L={l}, H={h}, HID={hid})")
        print("        Now upload TinyESP32/TinyESP32.ino in Arduino IDE!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
