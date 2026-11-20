#!/usr/bin/env python3
"""Módulo 5 (Lab 5B): Compilador da partição rag_kb.bin (RAG1) + busca léxica rápida com IDF."""

from __future__ import annotations

import math
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

STOPWORDS_PT: set[str] = {
    "a", "o", "os", "as", "um", "uma", "de", "do", "da", "dos", "das",
    "em", "no", "na", "nos", "nas", "e", "ou", "que", "qual", "quais",
    "como", "onde", "quando", "por", "para", "pra", "com", "sem", "se",
    "ele", "ela", "isso", "esta", "estao", "foi", "ser", "fazer", "devo",
    "posso", "nao", "muito", "muita", "hoje", "agora",
}


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
        kw_norm = normalize_search_text(doc_id.replace("_", " ") + " " + title + " " + keywords)
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

    q_words = [
        w for w in normalize_search_text(query).split()
        if len(w) >= 3 and w not in STOPWORDS_PT
    ]
    t_words = [
        w for w in normalize_search_text(last_topic or "").split()
        if len(w) >= 3 and w not in STOPWORDS_PT
    ]
    terms = set(q_words) if q_words else set(t_words)
    if not terms:
        return "", "", 0.0

    docs: list[tuple[str, set[str], str, set[str]]] = []
    for i in range(num_docs):
        t_off, t_len, k_off, k_len, b_off, b_len, _ = struct.unpack_from(
            RAG_ENTRY_FMT, blob, table_off + i * entry_sz
        )
        title = blob[pool_off + t_off : pool_off + t_off + t_len].decode("utf-8")
        kws = set(blob[pool_off + k_off : pool_off + k_off + k_len].decode("utf-8").split())
        body = blob[pool_off + b_off : pool_off + b_off + b_len].decode("utf-8")
        body_words = set(w for w in normalize_search_text(body).split() if len(w) >= 3 and w not in STOPWORDS_PT)
        docs.append((title, kws, body, body_words))

    best_title = ""
    best_body = ""
    best_score = 0.0

    for title, kws, body, body_words in docs:
        score = 0.0
        for qw in terms:
            doc_freq = sum(
                1 for _, d_kws, _, d_body in docs
                if qw in d_kws or qw in d_body or (len(qw) >= 5 and any(kw[:5] == qw[:5] for kw in d_kws))
            )
            idf = math.log((num_docs + 1.0) / (doc_freq + 0.5)) + 0.5
            if qw in kws:
                score += 4.0 * idf
            elif len(qw) >= 5 and any(len(kw) >= 5 and qw[:5] == kw[:5] for kw in kws):
                score += 2.5 * idf
            elif qw in body_words:
                score += 1.0 * idf

        if score > best_score:
            best_score = score
            best_title = title
            best_body = body

    if best_score < 2.5:
        return "", "", 0.0
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
    results = []
    for q, prev in queries:
        t0 = time.perf_counter()
        title, body, sc = search_rag(blob, q, prev)
        us = (time.perf_counter() - t0) * 1e6
        results.append((title, body, sc))
        print(f"  Q: {q!r} (prev={prev!r}) -> [{sc:.1f} | {us:.0f} us] {title or 'SEM MATCH'}: {body[:72]}")

    assert "Afogamento" in results[0][0], f"Esperado Afogamento, obtido {results[0][0]}"
    assert "Afogamento" in results[1][0], f"Esperado Afogamento no follow-up, obtido {results[1][0]}"
    assert "panela de oleo" in results[2][0].lower(), f"Esperado panela de oleo, obtido {results[2][0]}"
    assert "Purificacao de agua" in results[3][0], f"Esperado Purificacao de agua, obtido {results[3][0]}"
    assert results[4][2] == 0.0, f"Pergunta fora de domínio deveria retornar 0.0, obtido {results[4]}"
    print("[OK] Todas as 5 buscas Micro-RAG passaram com IDF e filtro de stopwords!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
