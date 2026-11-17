#!/usr/bin/env python3
"""Generate TinyESP32/tiny_weights.h and TinyESP32/web_ui.h from TINY.BIN and web/index.html."""

from __future__ import annotations

import argparse
import gzip
import pathlib
import struct
import sys

VOCAB = 97
CTX = 128
MAXD = 128
MAXHID = 512
MAXL = 8


def validate_tiny_bin(data: bytes) -> tuple[int, int, int, int]:
    if len(data) < 12:
        raise ValueError(f"TINY.BIN too short: {len(data)} bytes")
    if len(data) % 4 != 0:
        raise ValueError(f"TINY.BIN length {len(data)} is not a multiple of 4")
    if data[:4] != b"TNY2":
        raise ValueError(f"Invalid magic {data[:4]!r}, expected b'TNY2'")

    d, layers, heads, hid = struct.unpack_from("<HHHH", data, 4)
    if d < 8 or d > MAXD or layers < 1 or layers > MAXL or heads < 1 or d % heads != 0 or hid < 8 or hid > MAXHID:
        raise ValueError(f"Unsupported model dimensions: D={d}, L={layers}, H={heads}, HID={hid}")

    offset = 12

    def check_table(name: str, rows: int, cols: int) -> None:
        nonlocal offset
        if offset % 4 != 0:
            raise ValueError(f"Table {name} starts at unaligned offset {offset}")
        stride = 4 + cols
        if stride % 4 != 0:
            raise ValueError(f"Table {name} row stride {stride} is not 4-byte aligned")
        offset += rows * stride

    def check_gains(name: str) -> None:
        nonlocal offset
        if offset % 4 != 0:
            raise ValueError(f"Gains {name} starts at unaligned offset {offset}")
        offset += d * 4

    check_table("emb", VOCAB, d)
    check_table("pos", CTX, d)
    for l in range(layers):
        check_gains(f"g1[{l}]")
        check_table(f"wq[{l}]", d, d)
        check_table(f"wk[{l}]", d, d)
        check_table(f"wv[{l}]", d, d)
        check_table(f"wo[{l}]", d, d)
        check_gains(f"g2[{l}]")
        check_table(f"f1[{l}]", hid, d)
        check_table(f"f2[{l}]", d, hid)
    check_gains("gf")

    if offset != len(data):
        raise ValueError(f"Size mismatch: parsed {offset} bytes, file has {len(data)} bytes")

    return d, layers, heads, hid


def generate_weights_header(bin_path: pathlib.Path, out_path: pathlib.Path) -> tuple[int, int, int, int, int]:
    data = bin_path.read_bytes()
    d, layers, heads, hid = validate_tiny_bin(data)
    words = struct.unpack(f"<{len(data) // 4}I", data)

    lines = [
        "#ifndef TINY_WEIGHTS_H",
        "#define TINY_WEIGHTS_H",
        "",
        "#include <stdint.h>",
        "#ifdef ARDUINO",
        "#include <pgmspace.h>",
        "#else",
        "#ifndef PROGMEM",
        "#define PROGMEM",
        "#endif",
        "#endif",
        "",
        f"/* Auto-generated from {bin_path.name} (D={d}, L={layers}, H={heads}, HID={hid}, {len(data)} bytes) */",
        f"#define TINY_BIN_SIZE {len(data)}UL",
        "",
        f"static const uint32_t TINY_BIN_U32[{len(words)}] PROGMEM = {{",
    ]

    per_line = 12
    for i in range(0, len(words), per_line):
        chunk = words[i : i + per_line]
        hex_vals = ", ".join(f"0x{w:08X}u" for w in chunk)
        comma = "," if (i + per_line < len(words)) else ""
        lines.append(f"    {hex_vals}{comma}")

    lines.extend([
        "};",
        "",
        "#define TINY_BIN_DATA ((const uint8_t *)TINY_BIN_U32)",
        "",
        "#endif /* TINY_WEIGHTS_H */",
        "",
    ])

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return len(data), d, layers, heads, hid


def generate_web_header(html_path: pathlib.Path, out_path: pathlib.Path) -> tuple[int, int]:
    raw = html_path.read_bytes()
    gz = gzip.compress(raw, compresslevel=9, mtime=0)

    lines = [
        "#ifndef TINY_WEB_UI_H",
        "#define TINY_WEB_UI_H",
        "",
        "#include <stddef.h>",
        "#include <stdint.h>",
        "#ifdef ARDUINO",
        "#include <pgmspace.h>",
        "#else",
        "#ifndef PROGMEM",
        "#define PROGMEM",
        "#endif",
        "#endif",
        "",
        f"/* Auto-generated from {html_path.name} ({len(raw)} bytes raw -> {len(gz)} bytes gzip) */",
        f"static const uint8_t INDEX_HTML_GZ[{len(gz)}] PROGMEM = {{",
    ]

    per_line = 16
    for i in range(0, len(gz), per_line):
        chunk = gz[i : i + per_line]
        hex_vals = ", ".join(f"0x{b:02X}" for b in chunk)
        comma = "," if (i + per_line < len(gz)) else ""
        lines.append(f"    {hex_vals}{comma}")

    lines.extend([
        "};",
        "",
        "static const size_t INDEX_HTML_GZ_LEN = sizeof(INDEX_HTML_GZ);",
        "",
        "#endif /* TINY_WEB_UI_H */",
        "",
    ])

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return len(raw), len(gz)


def main() -> int:
    root = pathlib.Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Generate ESP32 PROGMEM headers for TINY 2.1.")
    parser.add_argument(
        "--bin",
        type=pathlib.Path,
        default=root / "TINY.BIN",
        help="Path to TNY2 weights binary (default: TINY.BIN)",
    )
    args = parser.parse_args()

    bin_path = args.bin if args.bin.is_absolute() else (root / args.bin)
    html_path = root / "web" / "index.html"
    sketch_dir = root / "TinyESP32"
    sketch_dir.mkdir(parents=True, exist_ok=True)

    size, d, layers, heads, hid = generate_weights_header(bin_path, sketch_dir / "tiny_weights.h")
    raw_len, gz_len = generate_web_header(html_path, sketch_dir / "web_ui.h")

    print(f"Generated tiny_weights.h from {bin_path}: {size} bytes (D={d}, L={layers}, H={heads}, HID={hid})")
    print(f"Generated web_ui.h: {raw_len} bytes raw -> {gz_len} bytes gzip")
    return 0


if __name__ == "__main__":
    sys.exit(main())
