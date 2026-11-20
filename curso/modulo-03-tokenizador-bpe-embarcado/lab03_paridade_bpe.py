#!/usr/bin/env python3
"""Módulo 3 (Lab): Treino BPE PT-BR (VOCAB=4096), exportação tiny_vocab.h e paridade Python == C."""

from __future__ import annotations

import argparse
import bisect
import pathlib
import re
import sys
import time
import unicodedata

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MODULE_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = MODULE_DIR.parent.parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from bpe_tokenizer import BYTE_OFFSET, SPECIAL_TOKENS, TinyBPETokenizer, pretokenize_bytes  # noqa: E402


TEST_SENTENCES_55: list[str] = [
    "",
    "Oi!",
    "Olá, tudo bem?",
    "A temperatura da sala está 29°C; ligar ventilação?",
    "Está muito abafado aqui na bancada, ligue a ventilação no máximo!",
    "Por favor, acenda a luz da estufa em tom quente com brilho médio.",
    "Apague todas as luzes e desligue o ventilador porque já vou dormir.",
    "Qual é a temperatura atual do ambiente e o nível de luminosidade?",
    "A ventilação automática renova o oxigênio, reduz o calor e evita condensação.",
    "O sensor DHT22 registrou 34°C na estufa, ligando exaustão de emergência agora.",
    "Está muito escuro para trabalhar; aumente a iluminação para 80%.",
    "Diminua a velocidade do cooler para 50% e mantenha o relé da bomba ligado.",
    "Quando a temperatura baixar de 22°C, desligue a ventilação imediatamente.",
    "Em caso de enchente ou alagamento, nunca atravesse ruas com correnteza a pé.",
    "Se houver cheiro de gás ou risco elétrico na inundação, desligue o disjuntor.",
    "A água da enchente transmite leptospirose; desinfete com água sanitária.",
    "Para purificar água em emergência, ferva por pelo menos 1 minuto.",
    "No método SODIS, deixe garrafas PET transparentes ao sol forte por 6 horas.",
    "Em caso de convulsão, proteja a cabeça da pessoa e nunca segure a língua.",
    "Se a crise convulsiva durar mais de 5 minutos, ligue para o SAMU 192.",
    "No afogamento, retire a vítima com segurança usando flutuador.",
    "Se a vítima não respira após afogamento, inicie compressões torácicas.",
    "Em picada de cobra peçonhenta, imobilize o membro e não corte nem chupe.",
    "Não faça torniquete em picada de serpente; leve ao hospital para soro.",
    "Sinais de deslizamento incluem postes inclinados, rachaduras e água barrenta.",
    "Ao notar estalos na encosta durante chuva forte, acione a Defesa Civil 199.",
    "Em incêndio em panela de óleo, nunca jogue água; abafe com tampa metálica.",
    "Se a roupa pegar fogo, deite no chão, cubra o rosto e role até apagar.",
    "O ESP32-S3-WROOM-1 possui 16 MB de Flash SPI e instruções vetoriais PIE.",
    "Este modelo roda 100% offline na borda, sem acesso à internet.",
    "Não tenho acesso a saldo bancário, transferências Pix ou senhas.",
    "Bom dia! O sistema embarcado está online com sensores calibrados.",
    "Boa tarde! Como posso ajudar com a automação da bancada?",
    "Boa noite! Posso apagar as luzes e reduzir o ventilador?",
    "Ligar relé da irrigação por 10 minutos e verificar umidade.",
    "Desligar luz alerta após temperatura voltar para 24 graus Celsius.",
    "Acentuação: coração, ação, atenção, geração, informação, solução, próximo.",
    "Símbolos: 0%, 100%, 12V, 5V, 3.3V, 16MB, 240MHz, @#$&*()!",
    "Oi!! Tudo bem???",
    "   espaços   múltiplos   e   pontuação...   ",
    "liga o led azul 50%",
    "247 vs 24 vs 7",
    "temperatura=31C;umid=78%;luz=20%;fan=0;led=off",
    "<|bos|><|ctx|>temp=31C;fan=0<|user|>tá quente<|act|>FAN=100<|assistant|>Ventilador ligado!<|eos|>",
    "<|user|>faz um pix de 100 reais<|act|>NONE<|assistant|>Não posso fazer Pix.<|eos|>",
    "Qual a capital do Pará? Belém!",
    "São Luís, João Pessoa, Maceió, Florianópolis e Brasília.",
    "Açaí, cupuaçu, pão de queijo, café e guaraná.",
    "É preciso verificar o código UTF-8: áéíóú âêô ãõ ç ÀÉÍÓÚ Ç.",
    "Quebra\nde\nlinha\te\ttabulação.",
    "Espaço final ",
    "1234567890",
    "a b c d e f g",
    "ventilacao sem acento e ventilação com acento",
    "Fim do teste de paridade Python <-> C!",
]


def parse_c_header_and_simulate(hdr_path: pathlib.Path) -> tuple[list[bytes], dict[tuple[int, int], tuple[int, int]], list[tuple[int, int]]]:
    """Lê tiny_vocab.h exatamente como o compilador C e valida a ordenação para busca binária."""
    src = hdr_path.read_text(encoding="utf-8")

    m_pool = re.search(r"TINY_VOCAB_POOL\[\d+\]\s+PROGMEM\s*=\s*\{(.*?)\};", src, re.S)
    m_slices = re.search(r"TINY_VOCAB_SLICES\[\d+\]\s+PROGMEM\s*=\s*\{(.*?)\};", src, re.S)
    m_merges = re.search(r"TINY_BPE_MERGES\[\d+\]\s+PROGMEM\s*=\s*\{(.*?)\};", src, re.S)
    assert m_pool and m_slices and m_merges, "Falha ao parsear arrays de tiny_vocab.h"

    pool = bytes(int(h, 16) for h in re.findall(r"0x([0-9A-Fa-f]{2})", m_pool.group(1)))
    slices = [(int(o), int(l)) for o, l in re.findall(r"\{(\d+)u,(\d+)u\}", m_slices.group(1))]
    merges = [
        (int(l), int(r), int(m), int(rk))
        for l, r, m, rk in re.findall(r"\{(\d+)u,(\d+)u,(\d+)u,(\d+)u\}", m_merges.group(1))
    ]
    # Remove placeholder {0,0,0,0} caso haja 0 merges
    if merges == [(0, 0, 0, 0)]:
        merges = []

    keys = [(l, r) for l, r, _, _ in merges]
    assert keys == sorted(keys), "TINY_BPE_MERGES não está ordenado para busca binária em C!"
    token_bytes = [pool[o : o + l] for o, l in slices]
    merge_lookup = {(l, r): (m, rk) for l, r, m, rk in merges}
    return token_bytes, merge_lookup, keys


def c_sim_encode(
    text: str, merge_keys: list[tuple[int, int]], merge_lookup: dict[tuple[int, int], tuple[int, int]]
) -> list[int]:
    """Simula busca binária em TINY_BPE_MERGES[] como em tiny_tokenizer.h."""
    raw = unicodedata.normalize("NFC", text).encode("utf-8")
    out: list[int] = []
    for kind, val in pretokenize_bytes(raw):
        if kind == 1:
            out.append(int(val))  # type: ignore[arg-type]
            continue
        seq = [b + BYTE_OFFSET for b in val]  # type: ignore[union-attr]
        while len(seq) >= 2:
            best_rank = 0xFFFFFFFF
            best_pos = -1
            best_pair = (0, 0)
            best_merged = 0
            for k in range(len(seq) - 1):
                pair = (seq[k], seq[k + 1])
                idx = bisect.bisect_left(merge_keys, pair)
                if idx < len(merge_keys) and merge_keys[idx] == pair:
                    m, rk = merge_lookup[pair]
                    if rk < best_rank:
                        best_rank = rk
                        best_pos = k
                        best_pair = pair
                        best_merged = m
            if best_pos < 0:
                break
            nxt: list[int] = []
            i = 0
            while i < len(seq):
                if i + 1 < len(seq) and (seq[i], seq[i + 1]) == best_pair:
                    nxt.append(best_merged)
                    i += 2
                else:
                    nxt.append(seq[i])
                    i += 1
            seq = nxt
        out.extend(seq)
    return out


def c_sim_decode(tokens: list[int], c_token_bytes: list[bytes]) -> str:
    buf = bytearray()
    for tid in tokens:
        buf.extend(c_token_bytes[tid])
    return bytes(buf).decode("utf-8", errors="replace")


def load_training_corpus() -> list[str]:
    texts: list[str] = []
    corpus_pt = MODULE_DIR / "data" / "corpus_ptbr_dominio.txt"
    texts.extend(
        ln.strip()
        for ln in corpus_pt.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    )
    custom_qa = REPO_ROOT / "data" / "custom_qa.txt"
    if custom_qa.exists():
        for ln in custom_qa.read_text(encoding="utf-8").splitlines():
            s = ln.strip()
            if not s or s.startswith("#") or "|" not in s:
                continue
            q, a = s.split("|", 1)
            texts.append(q.replace(";", " ") + " " + a)
    return texts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treina BPE PT-BR e testa paridade Python == C.")
    parser.add_argument("--vocab-size", type=int, default=4096, help="Tamanho do vocabulário (padrão: 4096)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = MODULE_DIR / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print(f"MÓDULO 3 — TREINO DE TOKENIZADOR BPE PT-BR (VOCAB={args.vocab_size}) E PARIDADE C")
    print("=" * 88)

    t0 = time.perf_counter()
    texts = load_training_corpus()
    tok = TinyBPETokenizer.train(texts, vocab_size=args.vocab_size, min_freq=2)
    bin_path = out_dir / "tiny_vocab.bin"
    hdr_path = out_dir / "tiny_vocab.h"
    tok.save_bin(bin_path)
    tok.export_c_header(hdr_path)

    # Recarrega TNYV para validar roundtrip binário
    tok_reloaded = TinyBPETokenizer.load_bin(bin_path)
    c_bytes, c_lookup, c_keys = parse_c_header_and_simulate(hdr_path)

    print(
        f"[TRAIN] {len(tok.merges)} merges aprendidos em {time.perf_counter() - t0:.2f}s | "
        f"tiny_vocab.bin = {bin_path.stat().st_size:,} B | tiny_vocab.h = {hdr_path.stat().st_size:,} B"
    )

    assert len(TEST_SENTENCES_55) == 55
    total_chars = 0
    total_toks = 0
    for idx, sent in enumerate(TEST_SENTENCES_55):
        py_ids = tok.encode(sent)
        bin_ids = tok_reloaded.encode(sent)
        c_ids = c_sim_encode(sent, c_keys, c_lookup)
        assert py_ids == bin_ids == c_ids, f"Divergência de IDs na frase #{idx}: {sent!r}"
        assert tok.decode(py_ids) == unicodedata.normalize("NFC", sent)
        assert c_sim_decode(c_ids, c_bytes) == unicodedata.normalize("NFC", sent)
        if 1 <= idx <= 35:  # frases naturais PT-BR
            total_chars += len(sent)
            total_toks += max(1, len(py_ids))

    ratio = total_chars / max(1, total_toks)
    print(f"[PARIDADE] 55/55 frases idênticas (Python == TNYV == tiny_vocab.h C) e 100% lossless UTF-8!")
    print(f"[COMPRESSÃO] {total_chars} chars -> {total_toks} tokens ({ratio:.2f} chars/token; meta >= 3.50)")
    assert ratio >= 3.5, f"Taxa de compressão abaixo da meta: {ratio:.2f}"

    sample_phrase = "A temperatura da sala está 29°C; ligar ventilação?"
    sample_ids = tok.encode(sample_phrase)
    pieces = [tok.decode([tid]) for tid in sample_ids]
    print(f"\nExemplo ({len(sample_phrase)} chars -> {len(sample_ids)} tokens): {pieces}")

    print("\n[EXERCÍCIO 1 — TRADE-OFF DE EMBEDDINGS EM D=256 (int8: 4 + 256 B/linha)]")
    for v in (1024, 4096, 16384):
        emb_bytes = v * (4 + 256)
        print(f"  VOCAB={v:>5} -> tabela emb = {emb_bytes:>10,} bytes ({emb_bytes / (1024 * 1024):.2f} MB na Flash)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
