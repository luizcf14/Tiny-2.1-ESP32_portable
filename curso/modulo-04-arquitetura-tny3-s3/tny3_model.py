#!/usr/bin/env python3
"""Módulo 4: Arquitetura TNY3-S3 (RoPE + GQA + SwiGLU + Weight Tying + int8/Q4_0 QAT)."""

from __future__ import annotations

import math
import pathlib
import random
import struct
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

TNY3_MAGIC = b"TNY3"
TNY3_VERSION = 1
TNY3_HEADER_FMT = "<4sBBHHHHHHHHHfI"  # 32 bytes alinhados
TNY3_HEADER_SIZE = struct.calcsize(TNY3_HEADER_FMT)
assert TNY3_HEADER_SIZE == 32

QUANT_INT8 = 1
QUANT_Q4_0 = 2


@dataclass(frozen=True)
class Tny3Config:
    vocab_size: int = 1024
    ctx_len: int = 128
    d: int = 128
    layers: int = 4
    q_heads: int = 4
    kv_heads: int = 2
    hid: int = 256
    rope_theta: float = 10000.0
    quant_mode: int = QUANT_INT8

    @property
    def hd(self) -> int:
        return self.d // self.q_heads

    @property
    def kv_dim(self) -> int:
        return self.kv_heads * self.hd

    def validate(self) -> None:
        if self.d < 16 or self.d % 16 != 0:
            raise ValueError(f"d={self.d} precisa ser múltiplo de 16 para SIMD PIE")
        if self.d % self.q_heads != 0:
            raise ValueError(f"d={self.d} deve ser divisível por q_heads={self.q_heads}")
        if self.q_heads % self.kv_heads != 0:
            raise ValueError(f"q_heads={self.q_heads} deve ser múltiplo de kv_heads={self.kv_heads}")
        if self.hd % 2 != 0:
            raise ValueError(f"head_dim={self.hd} precisa ser par para RoPE")
        if self.hid % 16 != 0:
            raise ValueError(f"hid={self.hid} precisa ser múltiplo de 16 para SIMD PIE")
        if self.quant_mode not in (QUANT_INT8, QUANT_Q4_0):
            raise ValueError(f"quant_mode inválido: {self.quant_mode}")


PRESETS: dict[str, Tny3Config] = {
    "micro": Tny3Config(
        vocab_size=512, ctx_len=128, d=128, layers=4, q_heads=4, kv_heads=2, hid=256
    ),
    "full_s3": Tny3Config(
        vocab_size=4096, ctx_len=256, d=256, layers=8, q_heads=8, kv_heads=2, hid=688
    ),
}


def ste_int8(x: torch.Tensor) -> torch.Tensor:
    big = x.detach().abs().amax(dim=-1, keepdim=True).clamp(min=1e-20)
    scale = big / 127.0
    xq = (x / scale).round().clamp(-127.0, 127.0) * scale
    return x + (xq - x).detach()


def ste_q4_0(x: torch.Tensor) -> torch.Tensor:
    """Straight-Through Estimator por blocos de 32 pesos (simétrico em [-7, +7])."""
    rows, cols = x.shape
    pad = (32 - (cols % 32)) % 32
    xp = F.pad(x, (0, pad)) if pad > 0 else x
    blocks = xp.view(rows, -1, 32)
    big = blocks.detach().abs().amax(dim=-1, keepdim=True).clamp(min=1e-20)
    scale = big / 7.0
    bq = (blocks / scale).round().clamp(-7.0, 7.0) * scale
    x_hat = bq.view(rows, -1)[:, :cols]
    return x + (x_hat - x).detach()


def rms_norm(x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    inv = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-5)
    return x * inv * g


def build_rope_cache(ctx_len: int, hd: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    idx = torch.arange(0, hd, 2, dtype=torch.float32)
    inv_freq = 1.0 / (theta ** (idx / float(hd)))
    pos = torch.arange(ctx_len, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)  # [ctx, hd//2]
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope_torch(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [B, H, T, HD], cos/sin: [T, HD//2] -> rotação em pares (2i, 2i+1)."""
    x0 = x[..., 0::2]
    x1 = x[..., 1::2]
    c = cos.unsqueeze(0).unsqueeze(0)
    s = sin.unsqueeze(0).unsqueeze(0)
    out0 = x0 * c - x1 * s
    out1 = x0 * s + x1 * c
    return torch.stack((out0, out1), dim=-1).flatten(-2)


def _pack_int8_table(arr: np.ndarray) -> bytes:
    rows, cols = arr.shape
    pad = (4 - (cols % 4)) % 4
    chunks: list[bytes] = []
    big = np.maximum(np.max(np.abs(arr), axis=1), np.float32(1e-20)).astype(np.float32)
    scale = (big / np.float32(127.0)).astype(np.float32)
    v = arr * (np.float32(127.0) / big[:, None])
    q = np.where(v < 0.0, v - 0.5, v + 0.5)
    q_i8 = np.clip(np.trunc(q), -127, 127).astype(np.int8)
    pad_bytes = b"\x00" * pad
    for r in range(rows):
        chunks.append(struct.pack("<f", float(scale[r])))
        chunks.append(q_i8[r].tobytes())
        if pad:
            chunks.append(pad_bytes)
    return b"".join(chunks)


def _pack_q4_table(arr: np.ndarray) -> bytes:
    """Empacota cada linha em blocos de 32 pesos: [float32 scale (4B)][16 bytes nibbles] = 20 B/bloco."""
    rows, cols = arr.shape
    n_blocks = (cols + 31) // 32
    pad_cols = n_blocks * 32 - cols
    if pad_cols > 0:
        arr = np.pad(arr, ((0, 0), (0, pad_cols)), mode="constant")
    blocks = arr.reshape(rows, n_blocks, 32)
    big = np.maximum(np.max(np.abs(blocks), axis=2), np.float32(1e-20)).astype(np.float32)
    scale = (big / np.float32(7.0)).astype(np.float32)
    v = blocks * (np.float32(7.0) / big[:, :, None])
    q = np.where(v < 0.0, v - 0.5, v + 0.5)
    q_i = np.clip(np.trunc(q), -7, 7).astype(np.int32)
    nib = (q_i + 8).astype(np.uint8)  # 0..15 (deslocado por +8)
    packed = nib[:, :, 0::2] | (nib[:, :, 1::2] << 4)  # [rows, n_blocks, 16]

    chunks: list[bytes] = []
    for r in range(rows):
        for b in range(n_blocks):
            chunks.append(struct.pack("<f", float(scale[r, b])))
            chunks.append(packed[r, b].tobytes())
    return b"".join(chunks)


class Tny3Transformer(nn.Module):
    """Transformer moderno (RoPE + GQA + SwiGLU + Weight Tying) para ESP32-S3."""

    def __init__(self, cfg: Tny3Config) -> None:
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        d, l, hid, kv_dim = cfg.d, cfg.layers, cfg.hid, cfg.kv_dim

        self.emb = nn.Parameter(torch.empty(cfg.vocab_size, d))
        self.g1 = nn.ParameterList([nn.Parameter(torch.ones(d)) for _ in range(l)])
        self.wq = nn.ParameterList([nn.Parameter(torch.empty(d, d)) for _ in range(l)])
        self.wk = nn.ParameterList([nn.Parameter(torch.empty(kv_dim, d)) for _ in range(l)])
        self.wv = nn.ParameterList([nn.Parameter(torch.empty(kv_dim, d)) for _ in range(l)])
        self.wo = nn.ParameterList([nn.Parameter(torch.empty(d, d)) for _ in range(l)])
        self.g2 = nn.ParameterList([nn.Parameter(torch.ones(d)) for _ in range(l)])
        self.w_gate = nn.ParameterList([nn.Parameter(torch.empty(hid, d)) for _ in range(l)])
        self.w_up = nn.ParameterList([nn.Parameter(torch.empty(hid, d)) for _ in range(l)])
        self.w_down = nn.ParameterList([nn.Parameter(torch.empty(d, hid)) for _ in range(l)])
        self.gf = nn.Parameter(torch.ones(d))

        causal = torch.triu(torch.ones(cfg.ctx_len, cfg.ctx_len, dtype=torch.bool), diagonal=1)
        cos, sin = build_rope_cache(cfg.ctx_len, cfg.hd, cfg.rope_theta)
        self.register_buffer("causal_mask", causal, persistent=False)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        with torch.no_grad():
            for p in self.parameters():
                if p.ndim == 2:
                    nn.init.normal_(p, mean=0.0, std=0.02)

    def _q_weight(self, w: torch.Tensor, qat: bool) -> torch.Tensor:
        if not qat:
            return w
        return ste_q4_0(w) if self.cfg.quant_mode == QUANT_Q4_0 else ste_int8(w)

    def forward(self, tokens: torch.Tensor, qat: bool = True) -> torch.Tensor:
        bsz, seq_len = tokens.shape
        if seq_len > self.cfg.ctx_len:
            raise ValueError(f"seq_len={seq_len} excede ctx_len={self.cfg.ctx_len}")

        emb_w = ste_int8(self.emb) if qat else self.emb  # emb sempre int8 (híbrido mesmo em Q4_0)
        x = F.embedding(tokens, emb_w)
        mask = self.causal_mask[:seq_len, :seq_len]
        cos = self.rope_cos[:seq_len]
        sin = self.rope_sin[:seq_len]
        hd = self.cfg.hd
        groups = self.cfg.q_heads // self.cfg.kv_heads
        att_div = math.sqrt(hd)

        for l in range(self.cfg.layers):
            h = rms_norm(x, self.g1[l])
            q = F.linear(h, self._q_weight(self.wq[l], qat))
            k = F.linear(h, self._q_weight(self.wk[l], qat))
            v = F.linear(h, self._q_weight(self.wv[l], qat))
            if qat:
                k = ste_int8(k)
                v = ste_int8(v)

            qh = q.view(bsz, seq_len, self.cfg.q_heads, hd).transpose(1, 2)
            kh = k.view(bsz, seq_len, self.cfg.kv_heads, hd).transpose(1, 2)
            vh = v.view(bsz, seq_len, self.cfg.kv_heads, hd).transpose(1, 2)

            qh = apply_rope_torch(qh, cos, sin)
            kh = apply_rope_torch(kh, cos, sin)

            if groups > 1:
                kh = kh.repeat_interleave(groups, dim=1)
                vh = vh.repeat_interleave(groups, dim=1)

            att = torch.matmul(qh, kh.transpose(-2, -1)) / att_div
            att = att.masked_fill(mask.unsqueeze(0).unsqueeze(0), -1e9)
            probs = F.softmax(att, dim=-1)
            ba = torch.matmul(probs, vh).transpose(1, 2).contiguous().view(bsz, seq_len, self.cfg.d)
            x = x + F.linear(ba, self._q_weight(self.wo[l], qat))

            h2 = rms_norm(x, self.g2[l])
            gate = F.silu(F.linear(h2, self._q_weight(self.w_gate[l], qat)))
            up = F.linear(h2, self._q_weight(self.w_up[l], qat))
            x = x + F.linear(gate * up, self._q_weight(self.w_down[l], qat))

        hf = rms_norm(x, self.gf)
        return F.linear(hf, emb_w)

    def export_tny3(self, out_path: pathlib.Path | None = None) -> bytes:
        c = self.cfg
        pack_layer = _pack_q4_table if c.quant_mode == QUANT_Q4_0 else _pack_int8_table
        body = bytearray()

        def add_gains(param: torch.Tensor) -> None:
            body.extend(param.detach().cpu().float().numpy().astype("<f4").tobytes())

        body.extend(_pack_int8_table(self.emb.detach().cpu().float().numpy()))
        for l in range(c.layers):
            add_gains(self.g1[l])
            body.extend(pack_layer(self.wq[l].detach().cpu().float().numpy()))
            body.extend(pack_layer(self.wk[l].detach().cpu().float().numpy()))
            body.extend(pack_layer(self.wv[l].detach().cpu().float().numpy()))
            body.extend(pack_layer(self.wo[l].detach().cpu().float().numpy()))
            add_gains(self.g2[l])
            body.extend(pack_layer(self.w_gate[l].detach().cpu().float().numpy()))
            body.extend(pack_layer(self.w_up[l].detach().cpu().float().numpy()))
            body.extend(pack_layer(self.w_down[l].detach().cpu().float().numpy()))
        add_gains(self.gf)

        total_bytes = TNY3_HEADER_SIZE + len(body)
        assert total_bytes % 4 == 0
        header = struct.pack(
            TNY3_HEADER_FMT,
            TNY3_MAGIC,
            TNY3_VERSION,
            c.quant_mode,
            c.vocab_size,
            c.ctx_len,
            c.d,
            c.layers,
            c.q_heads,
            c.kv_heads,
            c.hid,
            0,
            0,
            float(c.rope_theta),
            total_bytes,
        )
        blob = header + bytes(body)
        validate_tny3_bin(blob)
        if out_path is not None:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(blob)
        return blob


def _table_bytes(rows: int, cols: int, quant_mode: int) -> int:
    if quant_mode == QUANT_INT8:
        stride = 4 + ((cols + 3) // 4) * 4
        return rows * stride
    n_blocks = (cols + 31) // 32
    return rows * n_blocks * 20


def validate_tny3_bin(blob: bytes) -> Tny3Config:
    if len(blob) < TNY3_HEADER_SIZE or len(blob) % 4 != 0:
        raise ValueError(f"Tamanho inválido de TNY3: {len(blob)} bytes")
    (
        magic,
        ver,
        qmode,
        vocab,
        ctx,
        d,
        layers,
        q_heads,
        kv_heads,
        hid,
        _flags,
        _res,
        rope_theta,
        total_bytes,
    ) = struct.unpack_from(TNY3_HEADER_FMT, blob, 0)
    if magic != TNY3_MAGIC or ver != TNY3_VERSION:
        raise ValueError(f"Header TNY3 inválido: magic={magic!r}, ver={ver}")
    if total_bytes != len(blob):
        raise ValueError(f"total_bytes={total_bytes} != len(blob)={len(blob)}")

    cfg = Tny3Config(
        vocab_size=vocab,
        ctx_len=ctx,
        d=d,
        layers=layers,
        q_heads=q_heads,
        kv_heads=kv_heads,
        hid=hid,
        rope_theta=rope_theta,
        quant_mode=qmode,
    )
    cfg.validate()

    off = TNY3_HEADER_SIZE
    off += _table_bytes(vocab, d, QUANT_INT8)  # emb sempre int8
    for _ in range(layers):
        assert off % 4 == 0
        off += d * 4  # g1
        off += _table_bytes(d, d, qmode)  # wq
        off += _table_bytes(cfg.kv_dim, d, qmode)  # wk
        off += _table_bytes(cfg.kv_dim, d, qmode)  # wv
        off += _table_bytes(d, d, qmode)  # wo
        off += d * 4  # g2
        off += _table_bytes(hid, d, qmode)  # w_gate
        off += _table_bytes(hid, d, qmode)  # w_up
        off += _table_bytes(d, hid, qmode)  # w_down
    off += d * 4  # gf
    if off != len(blob):
        raise ValueError(f"Offset calculado {off} != tamanho real {len(blob)}")
    return cfg


def _unpack_int8_table(blob: bytes, off: int, rows: int, cols: int) -> tuple[tuple[np.ndarray, np.ndarray], int]:
    padded = ((cols + 3) // 4) * 4
    stride = 4 + padded
    scales = np.empty(rows, dtype=np.float32)
    weights = np.empty((rows, cols), dtype=np.int8)
    for r in range(rows):
        scales[r] = struct.unpack_from("<f", blob, off)[0]
        weights[r] = np.frombuffer(blob, dtype=np.int8, count=cols, offset=off + 4)
        off += stride
    return (scales, weights), off


def _unpack_q4_table(blob: bytes, off: int, rows: int, cols: int) -> tuple[np.ndarray, int]:
    """Desempacota Q4_0 em matriz float32 equivalente ao que o C desquantiza on-the-fly."""
    n_blocks = (cols + 31) // 32
    mat = np.empty((rows, n_blocks * 32), dtype=np.float32)
    for r in range(rows):
        for b in range(n_blocks):
            (scale,) = struct.unpack_from("<f", blob, off)
            raw = np.frombuffer(blob, dtype=np.uint8, count=16, offset=off + 4)
            lo = (raw & 0x0F).astype(np.int32) - 8
            hi = ((raw >> 4) & 0x0F).astype(np.int32) - 8
            blk = np.empty(32, dtype=np.float32)
            blk[0::2] = lo * np.float32(scale)
            blk[1::2] = hi * np.float32(scale)
            mat[r, b * 32 : (b + 1) * 32] = blk
            off += 20
    return mat[:, :cols].copy(), off


class QuantizedTny3Runner:
    """Executor autorregressivo TNY3 (RoPE + GQA + SwiGLU + KV Cache int8 + modo int8xint8 opcional)."""

    def __init__(self, blob: bytes, simd_int8_act: bool = False) -> None:
        self.cfg = validate_tny3_bin(blob)
        self.simd_int8_act = simd_int8_act
        c = self.cfg
        self.hd = c.hd
        self.groups = c.q_heads // c.kv_heads
        self.att_div = math.sqrt(self.hd)

        idx = np.arange(0, self.hd, 2, dtype=np.float32)
        inv_freq = 1.0 / (c.rope_theta ** (idx / float(self.hd)))
        pos = np.arange(c.ctx_len, dtype=np.float32)[:, None]
        freqs = pos * inv_freq[None, :]
        self.cos_tbl = np.cos(freqs).astype(np.float32)
        self.sin_tbl = np.sin(freqs).astype(np.float32)

        off = TNY3_HEADER_SIZE
        self.emb, off = _unpack_int8_table(blob, off, c.vocab_size, c.d)

        def read_layer_tbl(rows: int, cols: int) -> tuple[str, object]:
            nonlocal off
            if c.quant_mode == QUANT_INT8:
                tbl, off = _unpack_int8_table(blob, off, rows, cols)
                return ("int8", tbl)
            mat, off = _unpack_q4_table(blob, off, rows, cols)
            return ("q4", mat)

        def read_gains() -> np.ndarray:
            nonlocal off
            g = np.frombuffer(blob, dtype=np.float32, count=c.d, offset=off).copy()
            off += 4 * c.d
            return g

        self.g1, self.wq, self.wk, self.wv, self.wo = [], [], [], [], []
        self.g2, self.w_gate, self.w_up, self.w_down = [], [], [], []
        for _ in range(c.layers):
            self.g1.append(read_gains())
            self.wq.append(read_layer_tbl(c.d, c.d))
            self.wk.append(read_layer_tbl(c.kv_dim, c.d))
            self.wv.append(read_layer_tbl(c.kv_dim, c.d))
            self.wo.append(read_layer_tbl(c.d, c.d))
            self.g2.append(read_gains())
            self.w_gate.append(read_layer_tbl(c.hid, c.d))
            self.w_up.append(read_layer_tbl(c.hid, c.d))
            self.w_down.append(read_layer_tbl(c.d, c.hid))
        self.gf = read_gains()
        assert off == len(blob)

    def matvec(self, tbl: tuple[str, object], x: np.ndarray) -> np.ndarray:
        kind, payload = tbl
        if kind == "q4":
            mat: np.ndarray = payload  # type: ignore[assignment]
            return mat @ x
        scales, w = payload  # type: ignore[misc]
        if self.simd_int8_act:
            # Simula o kernel PIE int8xint8 -> int32 do Módulo 6
            big = float(np.max(np.abs(x)))
            if big < 1e-20:
                big = 1e-20
            x_scale = np.float32(big / 127.0)
            v = x * np.float32(127.0 / big)
            xq = np.clip(np.trunc(np.where(v < 0.0, v - 0.5, v + 0.5)), -127, 127).astype(np.int32)
            acc = w.astype(np.int32) @ xq
            return acc.astype(np.float32) * (scales * x_scale)
        return (w.astype(np.float32) @ x) * scales

    def norm(self, x: np.ndarray, g: np.ndarray) -> np.ndarray:
        s = float(np.dot(x, x))
        inv = np.float32(1.0 / math.sqrt(s / self.cfg.d + 1e-5))
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

    def apply_rope(self, vec: np.ndarray, n_heads: int, p: int) -> np.ndarray:
        cos = self.cos_tbl[p]
        sin = self.sin_tbl[p]
        out = vec.reshape(n_heads, self.hd).copy()
        x0 = out[:, 0::2].copy()
        x1 = out[:, 1::2].copy()
        out[:, 0::2] = x0 * cos[None, :] - x1 * sin[None, :]
        out[:, 1::2] = x0 * sin[None, :] + x1 * cos[None, :]
        return out.reshape(-1)

    def reset_cache(self) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        c = self.cfg
        kc = [np.zeros((c.ctx_len, c.kv_dim), dtype=np.int8) for _ in range(c.layers)]
        vc = [np.zeros((c.ctx_len, c.kv_dim), dtype=np.int8) for _ in range(c.layers)]
        ks = [np.zeros(c.ctx_len, dtype=np.float32) for _ in range(c.layers)]
        vs = [np.zeros(c.ctx_len, dtype=np.float32) for _ in range(c.layers)]
        return kc, vc, ks, vs

    def step(
        self,
        tok: int,
        p: int,
        kc: list[np.ndarray],
        vc: list[np.ndarray],
        ks: list[np.ndarray],
        vs: list[np.ndarray],
    ) -> np.ndarray:
        c = self.cfg
        emb_s, emb_w = self.emb
        bx = emb_w[tok].astype(np.float32) * emb_s[tok]

        for l in range(c.layers):
            bh = self.norm(bx, self.g1[l])
            q = self.apply_rope(self.matvec(self.wq[l], bh), c.q_heads, p)
            k = self.apply_rope(self.matvec(self.wk[l], bh), c.kv_heads, p)
            v = self.matvec(self.wv[l], bh)
            kc[l][p], ks[l][p] = self.store_kv(k)
            vc[l][p], vs[l][p] = self.store_kv(v)

            ba = np.zeros(c.d, dtype=np.float32)
            for qh_idx in range(c.q_heads):
                kvh_idx = qh_idx // self.groups
                qs0, qs1 = qh_idx * self.hd, (qh_idx + 1) * self.hd
                ks0, ks1 = kvh_idx * self.hd, (kvh_idx + 1) * self.hd
                qh = q[qs0:qs1]
                k_mat = kc[l][: p + 1, ks0:ks1].astype(np.float32)
                sc = ((k_mat @ qh) * ks[l][: p + 1]) / np.float32(self.att_div)
                big = float(np.max(sc))
                exps = np.where(sc - big < -60.0, 0.0, np.exp(sc - big)).astype(np.float32)
                total = float(np.sum(exps))
                if total > 0.0:
                    w = (exps * vs[l][: p + 1] / np.float32(total)).astype(np.float32)
                    v_mat = vc[l][: p + 1, ks0:ks1].astype(np.float32)
                    ba[qs0:qs1] = w @ v_mat

            bx = bx + self.matvec(self.wo[l], ba)
            h2 = self.norm(bx, self.g2[l])
            g_proj = self.matvec(self.w_gate[l], h2)
            silu = g_proj / (1.0 + np.exp(-np.clip(g_proj, -40.0, 40.0)))
            up_proj = self.matvec(self.w_up[l], h2)
            bx = bx + self.matvec(self.w_down[l], silu * up_proj)

        hf = self.norm(bx, self.gf)
        return self.matvec(("int8", self.emb), hf)

    def generate(self, prompt_ids: list[int], eos_id: int = 2, max_new: int = 64) -> list[int]:
        kc, vc, ks, vs = self.reset_cache()
        p = 0
        logits = np.zeros(self.cfg.vocab_size, dtype=np.float32)
        for tid in prompt_ids[: self.cfg.ctx_len - 1]:
            logits = self.step(tid, p, kc, vc, ks, vs)
            p += 1

        out: list[int] = []
        while p < self.cfg.ctx_len and len(out) < max_new:
            nxt = int(np.argmax(logits))
            if nxt == eos_id:
                break
            out.append(nxt)
            logits = self.step(nxt, p, kc, vc, ks, vs)
            p += 1
        return out
