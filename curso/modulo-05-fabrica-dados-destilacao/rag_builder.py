#!/usr/bin/env python3
"""Módulo 5 (Lab 5B): Compilador da partição rag_kb.bin (RAG1) + busca léxica rápida."""

from __future__ import annotations

import pathlib
import struct
import sys
import time
import unicodedata

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MODULE_DIR = pathlib.Path(__file__).resolve().parent
RAG_MAGIC = b"RAG1"
RAG_HEADER_FMT = "<4sHH"  # 8 bytes
RAG_ENTRY_FMT = "<IHIHIHH"  # 20 bytes por ficha


def normalize_search_text(text: str) -> str:
    s = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii").lower()
    cleaned = "".join(ch if ("a" <= ch <= "z" or "0" <= ch <= "9") else " " for ch in s)
    return " ".join(cleaned.split())


def build_rag_bin(txt_path: pathlib.Path, out_path: pathlib.Path) -> bytes:
    entries_raw: list[tuple[str, str, str, str]] = []
    for raw in txt_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split("|", 3)]
        if len(parts) != 4:
            raise ValueError(f"Linha malformada em {txt_path}: {line}")
        entries_raw.append((parts[0], parts[1], parts[2], parts[3]))

    pool = bytearray()
    table = bytearray()

    def push_str(s: str) -> tuple[int, int]:
        b = s.encode("utf-8")
        off = len(pool)
        pool.extend(b)
        return off, len(b)

    for doc_id, title, keywords, body in entries_raw:
        kw_norm = normalize_search_text(doc_id + " " + title + " " + keywords)
        t_off, t_len = push_str(title)
        k_off, k_len = push_str(kw_norm)
        b_off, b_len = push_str(body)
        table.extend(struct.pack(RAG_ENTRY_FMT, t_off, t_len, k_off, k_len, b_off, b_len, 0))

    while len(pool) % 4 != 0:
        pool.append(0)

    header = struct.pack(RAG_HEADER_FMT, RAG_MAGIC, len(entries_raw), 0)
    blob = header + bytes(table) + bytes(pool)
    assert len(blob) % 4 == 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(blob)
    return blob


def search_rag(blob: bytes, query: str, last_topic: str | None = None) -> tuple[str, str, float]:
    magic, num_docs, _ = struct.unpack_from(RAG_HEADER_FMT, blob, 0)
    assert magic == RAG_MAGIC
    table_off = struct.calcsize(RAG_HEADER_FMT)
    entry_sz = struct.calcsize(RAG_ENTRY_FMT)
    pool_off = table_off + num_docs * entry_sz

    q_words = set(w for w in normalize_search_text(query).split() if len(w) >= 2)
    t_words = set(w for w in normalize_search_text(last_topic or "").split() if len(w) >= 2)

    best_title = ""
    best_body = ""
    best_score = 0.0

    for i in range(num_docs):
        t_off, t_len, k_off, k_len, b_off, b_len, _ = struct.unpack_from(
            RAG_ENTRY_FMT, blob, table_off + i * entry_sz
        )
        title = blob[pool_off + t_off : pool_off + t_off + t_len].decode("utf-8")
        kws = set(blob[pool_off + k_off : pool_off + k_off + k_len].decode("utf-8").split())
        body = blob[pool_off + b_off : pool_off + b_off + b_len].decode("utf-8")
        body_words = set(normalize_search_text(body).split())

        score = 0.0
        for qw in q_words:
            if qw in kws:
                score += 3.0
            elif any(len(qw) >= 4 and (qw in kw or kw in qw) for kw in kws):
                score += 2.0
            elif qw in body_words:
                score += 1.0
        if score == 0.0 and t_words:
            for tw in t_words:
                if tw in kws:
                    score += 1.0

        if score > best_score:
            best_score = score
            best_title = title
            best_body = body

    return best_title, best_body, best_score


def main() -> int:
    txt_path = MODULE_DIR / "data" / "fichas_emergencia_rag.txt"
    out_path = MODULE_DIR / "out" / "rag_kb.bin"
    blob = build_rag_bin(txt_path, out_path)
    print(f"[RAG1] Compilado {out_path} ({len(blob):,} bytes, alinhado em 4 bytes)")

    queries = [
        ("ele engoliu muita agua na piscina e esta tossindo", None),
        ("e o que nao devo fazer?", "afogamento"),
        ("oleo pegou fogo na frigideira", None),
        ("como fazer agua segura para beber na enchente", None),
        ("qual a cotacao do dolar hoje?", None),
    ]
    for q, prev in queries:
        t0 = time.perf_counter()
        title, body, sc = search_rag(blob, q, prev)
        us = (time.perf_counter() - t0) * 1e6
        print(f"  Q: {q!r} (prev={prev!r}) -> [{sc:.1f} | {us:.0f} us] {title}: {body[:72]}...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
