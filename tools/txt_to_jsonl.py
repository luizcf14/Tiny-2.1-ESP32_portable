#!/usr/bin/env python3
"""Convert a TINY 2.1 Q&A .txt/.tsv file into .jsonl format."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import unicodedata


MAXQ = 72
MAX_TOTAL = 126


def clean_text(text: str, keep_accents: bool) -> str:
    if not keep_accents:
        norm = unicodedata.normalize("NFKD", text)
        text = norm.encode("ascii", "ignore").decode("ascii")
        text = "".join(ch for ch in text if 32 <= ord(ch) <= 126)
    return " ".join(text.split())


def normalize_question_text(text: str, keep_accents: bool) -> str:
    """Normalize question casing and collapse 3+ repeated ASCII letters to 2."""
    s = clean_text(text, keep_accents=keep_accents).lower()
    if keep_accents:
        return s
    out: list[str] = []
    for ch in s:
        if "a" <= ch <= "z" and len(out) >= 2 and out[-1] == ch and out[-2] == ch:
            continue
        out.append(ch)
    return "".join(out)


def convert_txt_to_jsonl(
    in_path: pathlib.Path,
    out_path: pathlib.Path,
    group: bool = True,
    keep_accents: bool = False,
) -> tuple[int, int]:
    raw_lines = in_path.read_text(encoding="utf-8").splitlines()
    grouped: dict[str, list[str]] = {}
    flat_pairs: list[tuple[str, str]] = []
    seen_pairs: set[tuple[str, str]] = set()

    for lineno, line in enumerate(raw_lines, start=1):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if "|" in s:
            q_part, a_part = s.split("|", 1)
        elif "\t" in s:
            q_part, a_part = s.split("\t", 1)
        else:
            print(f"[WARN] Ignorando linha {lineno} (esperado '|' ou TAB): {s}", file=sys.stderr)
            continue

        ans = clean_text(a_part, keep_accents=keep_accents)
        if not ans:
            continue

        for q_raw in q_part.split(";"):
            q = normalize_question_text(q_raw, keep_accents=keep_accents)
            if not q or (q, ans) in seen_pairs:
                continue
            if len(q) > MAXQ:
                print(
                    f"[WARN] Linha {lineno}: pergunta excede MAXQ={MAXQ} ({len(q)} chars): '{q}'",
                    file=sys.stderr,
                )
            elif len(q) + 1 + len(ans) > MAX_TOTAL:
                print(
                    f"[WARN] Linha {lineno}: par excede MAX_TOTAL={MAX_TOTAL} ({len(q) + 1 + len(ans)} chars): '{q}'",
                    file=sys.stderr,
                )
            seen_pairs.add((q, ans))
            flat_pairs.append((q, ans))
            grouped.setdefault(ans, []).append(q)

    records: list[dict[str, object]] = []
    if group:
        for ans, qs in grouped.items():
            q_val: object = qs[0] if len(qs) == 1 else qs
            records.append({"q": q_val, "a": ans})
    else:
        for q, ans in flat_pairs:
            records.append({"q": q, "a": ans})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_lines = [json.dumps(rec, ensure_ascii=False) for rec in records]
    out_path.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")
    return len(flat_pairs), len(records)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Converte arquivo pergunta | resposta (.txt/.tsv) para .jsonl do TINY 2.1."
    )
    p.add_argument("input", type=pathlib.Path, help="Arquivo .txt ou .tsv de entrada")
    p.add_argument("-o", "--out", type=pathlib.Path, default=None, help="Arquivo .jsonl de saida (padrao: mesmo nome .jsonl)")
    p.add_argument("--no-group", action="store_true", help="Nao agrupar perguntas com a mesma resposta")
    p.add_argument("--keep-accents", action="store_true", help="Manter acentos UTF-8 em vez de converter para ASCII")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    in_path = args.input
    if not in_path.exists():
        print(f"Erro: arquivo nao encontrado: {in_path}", file=sys.stderr)
        return 2
    out_path = args.out if args.out is not None else in_path.with_suffix(".jsonl")
    n_pairs, n_records = convert_txt_to_jsonl(
        in_path,
        out_path,
        group=not args.no_group,
        keep_accents=args.keep_accents,
    )
    print(f"[OK] {n_pairs} pares Q&A convertidos em {n_records} registros -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
