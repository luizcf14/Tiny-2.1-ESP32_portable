#!/usr/bin/env python3
"""Verify the ESP32 port: bit-exact assets, partitions, static checks, and zero-copy inference."""

from __future__ import annotations

import csv
import gzip
import math
import operator
import pathlib
import re
import struct
import sys
import time

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from generate_esp32_assets import validate_tiny_bin

VOCAB = 97
CTX = 128


def verify_weights_header(weights_h: pathlib.Path, root: pathlib.Path) -> tuple[bytes, bool]:
    src = weights_h.read_text(encoding="utf-8")
    m = re.search(r"TINY_BIN_U32\s*\[\s*(\d+)\s*\]\s*PROGMEM\s*=\s*\{(.*?)\}\s*;", src, re.S)
    if not m:
        raise AssertionError("Could not find TINY_BIN_U32[...] PROGMEM array in tiny_weights.h")
    declared_len = int(m.group(1))
    hex_tokens = re.findall(r"0x([0-9A-Fa-f]{8})u?", m.group(2))
    if len(hex_tokens) != declared_len:
        raise AssertionError(f"Declared {declared_len} words, parsed {len(hex_tokens)}")

    words = [int(tok, 16) for tok in hex_tokens]
    reconstructed = struct.pack(f"<{len(words)}I", *words)
    validate_tiny_bin(reconstructed)

    src_m = re.search(r"Auto-generated from (\S+)", src)
    src_name = src_m.group(1) if src_m else "TINY.BIN"

    if src_name == "TINY.BIN":
        original = (root / "TINY.BIN").read_bytes()
        if reconstructed != original:
            raise AssertionError(
                f"Reconstructed weights ({len(reconstructed)} B) != TINY.BIN ({len(original)} B)"
            )
        print(f"[OK] tiny_weights.h matches TINY.BIN bit-for-bit ({len(reconstructed)} bytes)")
        return reconstructed, True

    matched_path = None
    for cand in (root / src_name, root / "models" / src_name):
        if cand.exists():
            if cand.read_bytes() != reconstructed:
                raise AssertionError(f"Reconstructed weights != {cand}")
            matched_path = cand
            break

    if matched_path is not None:
        print(f"[OK] tiny_weights.h matches {src_name} bit-for-bit ({len(reconstructed)} bytes)")
    else:
        print(f"[OK] tiny_weights.h is a valid TNY2 model ({src_name}, {len(reconstructed)} bytes)")
    return reconstructed, False


def verify_web_header(web_h: pathlib.Path, html_path: pathlib.Path) -> None:
    src = web_h.read_text(encoding="utf-8")
    m = re.search(r"INDEX_HTML_GZ\s*\[\s*(\d+)\s*\]\s*PROGMEM\s*=\s*\{(.*?)\}\s*;", src, re.S)
    if not m:
        raise AssertionError("Could not find INDEX_HTML_GZ[...] PROGMEM array in web_ui.h")
    declared_len = int(m.group(1))
    byte_tokens = re.findall(r"0x([0-9A-Fa-f]{2})", m.group(2))
    if len(byte_tokens) != declared_len:
        raise AssertionError(f"Declared {declared_len} bytes, parsed {len(byte_tokens)}")

    gz = bytes(int(b, 16) for b in byte_tokens)
    raw = gzip.decompress(gz)
    html_bytes = html_path.read_bytes()
    if raw != html_bytes:
        raise AssertionError("Decompressed INDEX_HTML_GZ != web/index.html")

    html = raw.decode("utf-8")
    for endpoint in ("/api/ask", "/api/status", "/api/wifi", "/api/session/clear"):
        if endpoint not in html:
            raise AssertionError(f"Missing endpoint {endpoint} in web/index.html")
    if re.search(r"<script[^>]+src=|https?://", html, re.I):
        raise AssertionError("External URL detected in web/index.html (must be 100% offline)")

    print(f"[OK] web_ui.h decompresses bit-for-bit to web/index.html ({len(gz)} B gz -> {len(raw)} B)")


def verify_partitions(part_csv: pathlib.Path) -> None:
    rows = []
    with part_csv.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 5:
                raise AssertionError(f"Malformed partition row: {line}")
            name, ptype, subtype, off_s, size_s = parts[:5]
            off = int(off_s, 0)
            size = int(size_s, 0)
            rows.append((name, ptype, subtype, off, size))

    prev_end = 0
    app0_size = 0
    for name, ptype, subtype, off, size in rows:
        if off < prev_end:
            raise AssertionError(f"Partition {name} overlaps previous partition (0x{off:X} < 0x{prev_end:X})")
        if ptype == "app" and off % 0x10000 != 0:
            raise AssertionError(f"App partition {name} offset 0x{off:X} is not 64KB aligned")
        if name == "app0":
            app0_size = size
        prev_end = off + size

    if prev_end > 0x400000:
        raise AssertionError(f"Partitions exceed 4MB flash: ends at 0x{prev_end:X}")
    if app0_size < 0x1D0000:
        raise AssertionError(f"Expected app0 size >= 0x1D0000, got 0x{app0_size:X}")

    print(f"[OK] partitions.csv valid (app0={app0_size // 1024} KB, total flash used=0x{prev_end:X})")


def verify_static_code(sketch_dir: pathlib.Path) -> None:
    engine_cpp = (sketch_dir / "tiny_engine.cpp").read_text(encoding="utf-8")
    engine_h = (sketch_dir / "tiny_engine.h").read_text(encoding="utf-8")
    session_cpp = (sketch_dir / "tiny_session.cpp").read_text(encoding="utf-8")
    session_h = (sketch_dir / "tiny_session.h").read_text(encoding="utf-8")
    ino = (sketch_dir / "TinyESP32.ino").read_text(encoding="utf-8")

    # No double-precision math functions in tiny_engine.cpp
    for fn in ("sqrt", "exp", "log", "pow"):
        if re.search(rf"\b{fn}\s*\(", engine_cpp):
            raise AssertionError(f"Double-precision {fn}() found in tiny_engine.cpp; use {fn}f()")
    if re.search(r"\bdouble\b", engine_cpp):
        raise AssertionError("'double' type found in tiny_engine.cpp")

    # KV cache must be pointers allocated via malloc, not static large arrays
    if "malloc(" not in engine_cpp or re.search(r"static\s+int8_t\s+kc\s*\[\s*TINY_MAXL\s*\]\s*\[", engine_cpp):
        raise AssertionError("KV cache must be malloc-allocated, not static 2D .bss array")

    # Check 4-arg send_P for gzipped HTML
    if not re.search(
        r'server\.send_P\s*\(\s*200\s*,\s*"[^"]*"\s*,.*?INDEX_HTML_GZ\s*,\s*INDEX_HTML_GZ_LEN\s*\)',
        ino,
        re.S,
    ):
        raise AssertionError("TinyESP32.ino must use 4-argument server.send_P for INDEX_HTML_GZ")

    if "/api/session/clear" not in ino:
        raise AssertionError("TinyESP32.ino must expose /api/session/clear")

    # Check only native Arduino-ESP32 headers are included
    allowed_includes = {
        "WiFi.h", "WebServer.h", "DNSServer.h", "Preferences.h", "ESPmDNS.h", "esp_system.h",
        "stdarg.h", "stdbool.h", "stdio.h", "stdlib.h", "string.h", "stddef.h", "stdint.h",
        "math.h", "time.h", "Arduino.h", "tiny_engine.h", "tiny_session.h", "tiny_weights.h",
        "web_ui.h", "pgmspace.h"
    }
    files_to_scan = [
        ("tiny_engine.h", engine_h),
        ("tiny_engine.cpp", engine_cpp),
        ("tiny_session.h", session_h),
        ("tiny_session.cpp", session_cpp),
        ("TinyESP32.ino", ino),
    ]
    for fname, text in files_to_scan:
        for inc in re.findall(r'#include\s*[<"]([^>"]+)[>"]', text):
            if inc not in allowed_includes:
                raise AssertionError(f"Unexpected include <{inc}> in {fname}")

    print("[OK] Static checks passed (float32 math, malloc KV cache, 4-arg send_P, native-only includes)")


# ---- Zero-copy inference simulator on reconstructed tiny_weights.h bytes ----

class ZeroCopyModel:
    def __init__(self, blob: bytes) -> None:
        if blob[:4] != b"TNY2":
            raise ValueError("Bad magic")
        self.d, self.layers, self.heads, self.hid = struct.unpack_from("<HHHH", blob, 4)
        self.hd = self.d // self.heads
        self.att_div = math.sqrt(self.hd)
        self.blob = blob

        off = 12

        def read_table(rows: int, cols: int) -> list[tuple[float, tuple[int, ...]]]:
            nonlocal off
            table = []
            for _ in range(rows):
                (scale,) = struct.unpack_from("<f", blob, off)
                weights = struct.unpack_from(f"<{cols}b", blob, off + 4)
                table.append((scale, weights))
                off += 4 + cols
            return table

        def read_gains() -> tuple[float, ...]:
            nonlocal off
            vals = struct.unpack_from(f"<{self.d}f", blob, off)
            off += 4 * self.d
            return vals

        self.emb = read_table(VOCAB, self.d)
        self.pos = read_table(CTX, self.d)
        self.g1 = []
        self.wq = []
        self.wk = []
        self.wv = []
        self.wo = []
        self.g2 = []
        self.f1 = []
        self.f2 = []
        for _ in range(self.layers):
            self.g1.append(read_gains())
            self.wq.append(read_table(self.d, self.d))
            self.wk.append(read_table(self.d, self.d))
            self.wv.append(read_table(self.d, self.d))
            self.wo.append(read_table(self.d, self.d))
            self.g2.append(read_gains())
            self.f1.append(read_table(self.hid, self.d))
            self.f2.append(read_table(self.d, self.hid))
        self.gf = read_gains()
        assert off == len(blob)

    @staticmethod
    def matvec(table: list[tuple[float, tuple[int, ...]]], x: list[float]) -> list[float]:
        mul = operator.mul
        return [sum(map(mul, w, x)) * scale for scale, w in table]

    def norm(self, x: list[float], g: tuple[float, ...]) -> list[float]:
        s = sum(v * v for v in x)
        inv = 1.0 / math.sqrt(s / self.d + 1e-5)
        return [xi * inv * gi for xi, gi in zip(x, g)]

    @staticmethod
    def safe_exp(x: float) -> float:
        return 0.0 if x < -60.0 else math.exp(x)

    @staticmethod
    def store(src: list[float]) -> tuple[list[int], float]:
        big = max((abs(v) for v in src), default=0.0)
        if big < 1e-20:
            big = 1e-20
        k = 127.0 / big
        q = []
        for v in src:
            sv = v * k
            q.append(int(sv - 0.5) if sv < 0.0 else int(sv + 0.5))
        return q, big / 127.0

    def ask_greedy(self, question: str) -> str:
        d, hd, heads = self.d, self.hd, self.heads
        kc: list[list[list[int]]] = [[] for _ in range(self.layers)]
        vc: list[list[list[int]]] = [[] for _ in range(self.layers)]
        ks: list[list[float]] = [[] for _ in range(self.layers)]
        vs: list[list[float]] = [[] for _ in range(self.layers)]
        mul = operator.mul

        def step(tok: int, p: int) -> list[float]:
            es, ew = self.emb[tok]
            ps, pw = self.pos[p]
            bx = [ew[i] * es + pw[i] * ps for i in range(d)]

            for l in range(self.layers):
                bh = self.norm(bx, self.g1[l])
                kt = self.matvec(self.wk[l], bh)
                kq, kscale = self.store(kt)
                kc[l].append(kq)
                ks[l].append(kscale)

                vt = self.matvec(self.wv[l], bh)
                vq, vscale = self.store(vt)
                vc[l].append(vq)
                vs[l].append(vscale)

                bq = self.matvec(self.wq[l], bh)
                ba = [0.0] * d
                for h in range(heads):
                    hoff = h * hd
                    qh = bq[hoff : hoff + hd]
                    scores = [
                        sum(map(mul, qh, kc[l][j][hoff : hoff + hd])) * ks[l][j] / self.att_div
                        for j in range(p + 1)
                    ]
                    big = max(scores)
                    exps = [self.safe_exp(sc - big) for sc in scores]
                    total = sum(exps)
                    for j in range(p + 1):
                        if exps[j] == 0.0:
                            continue
                        w = exps[j] * vs[l][j] / total
                        vp = vc[l][j]
                        for i in range(hd):
                            ba[hoff + i] += w * vp[hoff + i]

                bt = self.matvec(self.wo[l], ba)
                bx = [bx[i] + bt[i] for i in range(d)]

                bh = self.norm(bx, self.g2[l])
                bm = [v if v > 0.0 else 0.0 for v in self.matvec(self.f1[l], bh)]
                bt = self.matvec(self.f2[l], bm)
                bx = [bx[i] + bt[i] for i in range(d)]

            bh = self.norm(bx, self.gf)
            return self.matvec(self.emb, bh)

        p = 0
        scores: list[float] = []
        for ch in question:
            scores = step(ord(ch) - 30, p)
            p += 1
        scores = step(1, p)
        p += 1

        out = []
        while p < CTX:
            tok = max(range(VOCAB), key=scores.__getitem__)
            if tok <= 1:
                break
            out.append(chr(tok + 30))
            scores = step(tok, p)
            p += 1
        return "".join(out)


def verify_inference(blob: bytes, is_factory_model: bool) -> None:
    t0 = time.perf_counter()
    model = ZeroCopyModel(blob)
    if is_factory_model:
        tests = [
            ("what is the capital of peru", "lima"),
            ("is a whale a fish", "mammal"),
            ("what is 7 times 8", "56"),
            ("who invented the telephone", "bell"),
            ("spell elephant", "e-l-e-p-h-a-n-t"),
            ("opposite of hot", "cold"),
            ("what is the symbol for gold", "au"),
            ("how many legs does a spider have", "eight"),
        ]
        for q, expected_sub in tests:
            ans = model.ask_greedy(q)
            print(f"  Q: {q}  ->  A: {ans}")
            if expected_sub not in ans.lower():
                raise AssertionError(f"Unexpected answer for '{q}': got '{ans}', expected '{expected_sub}'")
        dt = time.perf_counter() - t0
        print(f"[OK] Zero-copy inference passed all {len(tests)} benchmark questions in {dt:.2f}s")
    else:
        smoke_qs = ["quem e voce?", "what is 7 times 8"]
        for q in smoke_qs:
            ans = model.ask_greedy(q)
            if not all(32 <= ord(ch) <= 126 for ch in ans):
                raise AssertionError(f"Non-ASCII character in answer for '{q}': {ans!r}")
            print(f"  Q: {q}  ->  A: {ans}")
        dt = time.perf_counter() - t0
        print(f"[OK] Zero-copy inference smoke test passed on fine-tuned model in {dt:.2f}s")


def main() -> int:
    root = pathlib.Path(__file__).resolve().parent.parent
    sketch = root / "TinyESP32"
    blob, is_factory_model = verify_weights_header(sketch / "tiny_weights.h", root)
    verify_web_header(sketch / "web_ui.h", root / "web" / "index.html")
    verify_partitions(sketch / "partitions.csv")
    verify_static_code(sketch)
    verify_inference(blob, is_factory_model)
    print("\nALL VERIFICATIONS PASSED!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
