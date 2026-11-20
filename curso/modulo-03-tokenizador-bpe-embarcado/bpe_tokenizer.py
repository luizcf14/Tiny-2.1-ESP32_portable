#!/usr/bin/env python3
"""Módulo 3: Tokenizador Byte-Level BPE determinístico com paridade Python <-> C (TNYV)."""

from __future__ import annotations

import pathlib
import struct
import unicodedata
from collections import Counter

SPECIAL_TOKENS: list[str] = [
    "<|pad|>",        # 0
    "<|bos|>",        # 1
    "<|eos|>",        # 2
    "<|user|>",       # 3
    "<|ctx|>",        # 4
    "<|act|>",        # 5
    "<|assistant|>",  # 6
]
RESERVED_SLOTS = 16  # 0..6 especiais, 7..15 reservados
BYTE_OFFSET = RESERVED_SLOTS  # 16..271 = bytes 0x00..0xFF
MERGE_START = RESERVED_SLOTS + 256  # 272+ = subpalavras BPE

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
USER_ID = 3
CTX_ID = 4
ACT_ID = 5
ASSISTANT_ID = 6


def _is_word_byte(b: int) -> bool:
    return (
        (ord("a") <= b <= ord("z"))
        or (ord("A") <= b <= ord("Z"))
        or b == ord("_")
        or b >= 0x80
    )


def _is_digit_byte(b: int) -> bool:
    return ord("0") <= b <= ord("9")


def pretokenize_bytes(raw: bytes) -> list[tuple[int, bytes | int]]:
    """
    Segmenta bytes UTF-8 de forma 100% determinística e idêntica ao algoritmo em C:
      - tags especiais <|...|> -> ('S', special_id)
      - espaço + palavra UTF-8 -> ('B', bytes)
      - palavra UTF-8          -> ('B', bytes)
      - dígito ASCII isolado   -> ('B', 1 byte)
      - qualquer outro byte    -> ('B', 1 byte)
    """
    out: list[tuple[int, bytes | int]] = []
    n = len(raw)
    i = 0
    special_bytes = [(tok.encode("ascii"), idx) for idx, tok in enumerate(SPECIAL_TOKENS)]

    while i < n:
        matched_special = False
        if raw[i] == ord("<") and i + 1 < n and raw[i + 1] == ord("|"):
            for sbytes, sid in special_bytes:
                if raw.startswith(sbytes, i):
                    out.append((1, sid))
                    i += len(sbytes)
                    matched_special = True
                    break
        if matched_special:
            continue

        b = raw[i]
        if b == 0x20 and i + 1 < n and _is_word_byte(raw[i + 1]):
            j = i + 2
            while j < n and _is_word_byte(raw[j]):
                j += 1
            out.append((0, raw[i:j]))
            i = j
        elif _is_word_byte(b):
            j = i + 1
            while j < n and _is_word_byte(raw[j]):
                j += 1
            out.append((0, raw[i:j]))
            i = j
        else:
            out.append((0, raw[i : i + 1]))
            i += 1
    return out


def _apply_merge_to_seq(seq: tuple[int, ...], pair: tuple[int, int], new_id: int) -> tuple[int, ...]:
    a, b = pair
    res: list[int] = []
    i = 0
    n = len(seq)
    while i < n:
        if i + 1 < n and seq[i] == a and seq[i + 1] == b:
            res.append(new_id)
            i += 2
        else:
            res.append(seq[i])
            i += 1
    return tuple(res)


class TinyBPETokenizer:
    """Byte-Level BPE compacto para ESP32-S3 (compatível com C zero-malloc)."""

    def __init__(
        self,
        vocab_size: int,
        token_bytes: list[bytes],
        merges: list[tuple[int, int, int, int]],
    ) -> None:
        self.vocab_size = vocab_size
        self.token_bytes = token_bytes
        self.merges = merges  # [(left, right, merged, rank), ...]
        self.merge_map: dict[tuple[int, int], tuple[int, int]] = {
            (l, r): (m, rk) for l, r, m, rk in merges
        }

    @classmethod
    def train(cls, texts: list[str], vocab_size: int = 4096, min_freq: int = 2) -> "TinyBPETokenizer":
        if vocab_size < MERGE_START + 16:
            raise ValueError(f"vocab_size deve ser >= {MERGE_START + 16}")

        # Inicializa tokens 0..271
        token_bytes: list[bytes] = [b""] * vocab_size
        for idx, stok in enumerate(SPECIAL_TOKENS):
            token_bytes[idx] = stok.encode("ascii")
        for r_idx in range(len(SPECIAL_TOKENS), RESERVED_SLOTS):
            token_bytes[r_idx] = f"<|res_{r_idx}|>".encode("ascii")
        for b in range(256):
            token_bytes[BYTE_OFFSET + b] = bytes([b])

        word_counts: Counter[tuple[int, ...]] = Counter()
        for txt in texts:
            norm = unicodedata.normalize("NFC", txt).encode("utf-8")
            for kind, val in pretokenize_bytes(norm):
                if kind == 0:
                    seq = tuple(b + BYTE_OFFSET for b in val)  # type: ignore[union-attr]
                    word_counts[seq] += 1

        pair_counts: Counter[tuple[int, int]] = Counter()
        for seq, cnt in word_counts.items():
            for k in range(len(seq) - 1):
                pair_counts[(seq[k], seq[k + 1])] += cnt

        merges: list[tuple[int, int, int, int]] = []
        next_id = MERGE_START

        while next_id < vocab_size and pair_counts:
            best_pair, best_freq = max(pair_counts.items(), key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]))
            if best_freq < min_freq:
                break

            rank = next_id - MERGE_START
            merges.append((best_pair[0], best_pair[1], next_id, rank))
            token_bytes[next_id] = token_bytes[best_pair[0]] + token_bytes[best_pair[1]]

            # Atualiza apenas as sequências afetadas e recalcula seus pares
            new_word_counts: Counter[tuple[int, ...]] = Counter()
            for seq, cnt in word_counts.items():
                has_pair = False
                for k in range(len(seq) - 1):
                    if seq[k] == best_pair[0] and seq[k + 1] == best_pair[1]:
                        has_pair = True
                        break
                if not has_pair:
                    new_word_counts[seq] += cnt
                    continue

                for k in range(len(seq) - 1):
                    pair_counts[(seq[k], seq[k + 1])] -= cnt
                    if pair_counts[(seq[k], seq[k + 1])] <= 0:
                        pair_counts.pop((seq[k], seq[k + 1]), None)

                updated = _apply_merge_to_seq(seq, best_pair, next_id)
                new_word_counts[updated] += cnt
                for k in range(len(updated) - 1):
                    pair_counts[(updated[k], updated[k + 1])] += cnt

            word_counts = new_word_counts
            next_id += 1

        # Preenche slots não usados além dos merges ativos (se corpus for pequeno)
        for idx in range(next_id, vocab_size):
            token_bytes[idx] = f"<|unused_{idx}|>".encode("ascii")

        return cls(vocab_size=vocab_size, token_bytes=token_bytes, merges=merges)

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        raw = unicodedata.normalize("NFC", text).encode("utf-8")
        tokens: list[int] = [BOS_ID] if add_bos else []

        for kind, val in pretokenize_bytes(raw):
            if kind == 1:
                tokens.append(int(val))  # type: ignore[arg-type]
                continue

            seq: list[int] = [b + BYTE_OFFSET for b in val]  # type: ignore[union-attr]
            while len(seq) >= 2:
                best_rank = 1 << 30
                best_pair: tuple[int, int] | None = None
                best_merged = -1
                for k in range(len(seq) - 1):
                    pair = (seq[k], seq[k + 1])
                    hit = self.merge_map.get(pair)
                    if hit is not None and hit[1] < best_rank:
                        best_rank = hit[1]
                        best_pair = pair
                        best_merged = hit[0]
                if best_pair is None:
                    break
                seq = list(_apply_merge_to_seq(tuple(seq), best_pair, best_merged))
            tokens.extend(seq)

        if add_eos:
            tokens.append(EOS_ID)
        return tokens

    def decode(self, tokens: list[int], skip_special: bool = False) -> str:
        chunks: list[bytes] = []
        for tid in tokens:
            if tid < 0 or tid >= self.vocab_size:
                continue
            if skip_special and tid < RESERVED_SLOTS:
                continue
            chunks.append(self.token_bytes[tid])
        return b"".join(chunks).decode("utf-8", errors="replace")

    def save_bin(self, out_path: pathlib.Path) -> bytes:
        """Salva no formato binário TNYV (alinhado)."""
        pool = bytearray()
        slices = bytearray()
        for tid in range(self.vocab_size):
            tb = self.token_bytes[tid]
            if len(tb) > 255:
                tb = tb[:255]
            off = len(pool)
            pool.extend(tb)
            slices.extend(struct.pack("<IB", off, len(tb)))
        # Pad do pool até múltiplo de 4 bytes
        while len(pool) % 4 != 0:
            pool.append(0)

        sorted_merges = sorted(self.merges, key=lambda item: (item[0], item[1]))
        merges_bytes = bytearray()
        for l, r, m, rk in sorted_merges:
            merges_bytes.extend(struct.pack("<HHHH", l, r, m, rk))

        header = struct.pack(
            "<4sIIII",
            b"TNYV",
            self.vocab_size,
            len(sorted_merges),
            len(pool),
            len(slices),
        )
        blob = header + bytes(slices) + bytes(merges_bytes) + bytes(pool)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(blob)
        return blob

    @classmethod
    def load_bin(cls, path: pathlib.Path) -> "TinyBPETokenizer":
        blob = path.read_bytes()
        magic, vocab_size, num_merges, pool_len, slices_len = struct.unpack_from("<4sIIII", blob, 0)
        if magic != b"TNYV":
            raise ValueError(f"Magic inválido em {path}: {magic!r}")
        off = 20
        slice_rows = [struct.unpack_from("<IB", blob, off + i * 5) for i in range(vocab_size)]
        off += slices_len
        merges = [struct.unpack_from("<HHHH", blob, off + i * 8) for i in range(num_merges)]
        off += num_merges * 8
        pool = blob[off : off + pool_len]
        token_bytes = [pool[s_off : s_off + s_len] for s_off, s_len in slice_rows]
        return cls(vocab_size=vocab_size, token_bytes=token_bytes, merges=merges)

    def export_c_header(self, out_path: pathlib.Path) -> None:
        """Gera header PROGMEM ordenado por (left, right) para busca binária em C."""
        pool = bytearray()
        slices: list[tuple[int, int]] = []
        for tid in range(self.vocab_size):
            tb = self.token_bytes[tid]
            off = len(pool)
            pool.extend(tb)
            slices.append((off, len(tb)))

        sorted_merges = sorted(self.merges, key=lambda item: (item[0], item[1]))
        lines = [
            "#ifndef TINY_VOCAB_H",
            "#define TINY_VOCAB_H",
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
            f"#define TINY_BPE_VOCAB_SIZE {self.vocab_size}",
            f"#define TINY_BPE_NUM_MERGES {len(sorted_merges)}",
            f"#define TINY_BPE_POOL_BYTES {len(pool)}",
            "",
            "typedef struct {",
            "    uint32_t offset;",
            "    uint8_t  len;",
            "} TinyTokenSlice;",
            "",
            "typedef struct {",
            "    uint16_t left;",
            "    uint16_t right;",
            "    uint16_t merged;",
            "    uint16_t rank;",
            "} TinyBpeMerge;",
            "",
            f"static const uint8_t TINY_VOCAB_POOL[{len(pool)}] PROGMEM = {{",
        ]

        for i in range(0, len(pool), 16):
            chunk = pool[i : i + 16]
            vals = ", ".join(f"0x{b:02X}" for b in chunk)
            comma = "," if i + 16 < len(pool) else ""
            lines.append(f"    {vals}{comma}")
        lines.extend(["};", "", f"static const TinyTokenSlice TINY_VOCAB_SLICES[{self.vocab_size}] PROGMEM = {{"])

        for i in range(0, len(slices), 6):
            chunk = slices[i : i + 6]
            vals = ", ".join(f"{{{off}u,{ln}u}}" for off, ln in chunk)
            comma = "," if i + 6 < len(slices) else ""
            lines.append(f"    {vals}{comma}")
        lines.extend(["};", ""])

        merge_len = max(1, len(sorted_merges))
        lines.append(f"static const TinyBpeMerge TINY_BPE_MERGES[{merge_len}] PROGMEM = {{")
        if sorted_merges:
            for i in range(0, len(sorted_merges), 4):
                chunk = sorted_merges[i : i + 4]
                vals = ", ".join(f"{{{l}u,{r}u,{m}u,{rk}u}}" for l, r, m, rk in chunk)
                comma = "," if i + 4 < len(sorted_merges) else ""
                lines.append(f"    {vals}{comma}")
        else:
            lines.append("    {0u,0u,0u,0u}")
        lines.extend(["};", "", "#endif /* TINY_VOCAB_H */", ""])

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(lines), encoding="utf-8")
