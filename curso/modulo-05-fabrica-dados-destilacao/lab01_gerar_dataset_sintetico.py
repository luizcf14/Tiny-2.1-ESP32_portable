#!/usr/bin/env python3
"""Módulo 5 (Lab 1): Fábrica de dados sintéticos (~1.000 amostras PT-BR) + avaliação disjunta."""

from __future__ import annotations

import json
import pathlib
import random
import re
import sys
import unicodedata

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MODULE_DIR = pathlib.Path(__file__).resolve().parent
COURSE_DIR = MODULE_DIR.parent
MOD3_DIR = COURSE_DIR / "modulo-03-tokenizador-bpe-embarcado"
if str(MOD3_DIR) not in sys.path:
    sys.path.insert(0, str(MOD3_DIR))

from bpe_tokenizer import TinyBPETokenizer  # noqa: E402

ALLOWED_ACT_RE = re.compile(
    r"^(NONE|(?:(?:FAN=(?:0|50|100)|LED=(?:OFF|WARM|COOL|ALERT)|RELAY=[01])(?:;(?:FAN=(?:0|50|100)|LED=(?:OFF|WARM|COOL|ALERT)|RELAY=[01]))*))$"
)

TEACHER_PROMPT_TEMPLATE = """
Você é um gerador de dados sintéticos para um agente embarcado em ESP32-S3 (PT-BR).
Gere exemplos JSONL no esquema:
{"ctx":"temp=31C;luz=20%;fan=0;led=off;relay=0","user":"...","act":"FAN=100;LED=WARM","assistant":"..."}
Regras estritas:
1. Ações permitidas em "act": NONE, FAN=0|50|100, LED=OFF|WARM|COOL|ALERT, RELAY=0|1 (combináveis por ';').
2. Inclua variações coloquiais, abreviações ("vc", "tá", "mto", "aki") e ~20% de recusas seguras (act="NONE").
3. Todo o registro tokenizado deve caber em <= 120 tokens BPE.
""".strip()


def strip_accents(text: str) -> str:
    norm = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return norm


def validate_action_syntax(act: str) -> bool:
    return bool(ALLOWED_ACT_RE.match(act.strip()))


def build_synthetic_datasets(seed: int = 42) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    rng = random.Random(seed)

    # Conjunto de avaliação inédito (24 casos, incluindo os 5 cenários Capstone do Módulo 6!)
    eval_rows: list[dict[str, str]] = [
        {"ctx": "temp=28C;luz=15%;fan=0;led=off;relay=0", "user": "acende a luz em tom quente e liga o ventilador em 50%", "act": "LED=WARM;FAN=50", "assistant": "Luz em tom quente ligada e ventilador em 50%."},
        {"ctx": "temp=34C;luz=45%;fan=0;led=off;relay=0", "user": "como esta o ambiente? precisa fazer alguma coisa?", "act": "FAN=100", "assistant": "Ambiente quente (34°C). Liguei o ventilador no máximo para resfriar."},
        {"ctx": "temp=24C;luz=15%;fan=0;led=off;relay=0", "user": "ta mto escuro aki, acende a luz em tom quente", "act": "LED=WARM", "assistant": "Luz em tom quente ativada."},
        {"ctx": "temp=27C;luz=50%;fan=100;led=warm;relay=0", "user": "abaixa um pouco o vento", "act": "FAN=50", "assistant": "Ventilação reduzida para 50%."},
        {"ctx": "temp=25C;luz=50%;fan=0;led=off;relay=0", "user": "faz um pix de 100 reais agora", "act": "NONE", "assistant": "Não posso fazer isso; opero 100% offline controlando apenas LED, ventilador e relé."},

        {"ctx": "temp=32C;luz=60%;fan=0;led=off;relay=0", "user": "ta mto quente aki, bota o ventilador no maximo", "act": "FAN=100", "assistant": "Ventilador ligado em 100% (32°C)."},
        {"ctx": "temp=30C;luz=30%;fan=0;led=off;relay=0", "user": "ventilacao em 100% agora", "act": "FAN=100", "assistant": "Ventilador ligado em 100% (30°C)."},
        {"ctx": "temp=26C;luz=40%;fan=0;led=off;relay=0", "user": "bota o ventilador em 50% por favor", "act": "FAN=50", "assistant": "Ventilação ajustada para 50%."},
        {"ctx": "temp=20C;luz=40%;fan=50;led=off;relay=0", "user": "corta a ventilacao agora", "act": "FAN=0", "assistant": "Ventilador desligado."},
        {"ctx": "temp=24C;luz=20%;fan=0;led=off;relay=0", "user": "bota luz fria pra trabalhar melhor", "act": "LED=COOL", "assistant": "Iluminação branca fria ativada."},
        {"ctx": "temp=24C;luz=20%;fan=0;led=off;relay=0", "user": "acende luz de alerta agora", "act": "LED=ALERT", "assistant": "LED de alerta ativado."},
        {"ctx": "temp=23C;luz=80%;fan=0;led=warm;relay=0", "user": "apaga o led da bancada", "act": "LED=OFF", "assistant": "Iluminação desligada."},
        {"ctx": "temp=29C;luz=20%;fan=0;led=off;relay=0", "user": "acende luz quente e bota o ventilador em 50%", "act": "LED=WARM;FAN=50", "assistant": "Luz em tom quente ligada e ventilador em 50%."},
        {"ctx": "temp=22C;luz=75%;fan=50;led=cool;relay=0", "user": "apaga a luz e para o ventilador", "act": "LED=OFF;FAN=0", "assistant": "Luz e ventilador desligados."},
        {"ctx": "temp=25C;luz=50%;fan=0;led=off;relay=0", "user": "ligar rele da bomba agora", "act": "RELAY=1", "assistant": "Relé ligado."},
        {"ctx": "temp=25C;luz=50%;fan=0;led=off;relay=1", "user": "corta o rele da bomba", "act": "RELAY=0", "assistant": "Relé desligado."},
        {"ctx": "temp=35C;luz=40%;fan=0;led=off;relay=0", "user": "como esta a temperatura ai? precisa fazer algo?", "act": "FAN=100", "assistant": "Ambiente quente (35°C). Liguei o ventilador no máximo para resfriar."},
        {"ctx": "temp=23C;luz=55%;fan=0;led=off;relay=0", "user": "como esta o ambiente? precisa fazer algo?", "act": "NONE", "assistant": "Ambiente confortável (23°C). Nenhuma ação necessária."},
        {"ctx": "temp=22C;luz=55%;fan=0;led=off;relay=0", "user": "como esta a temperatura do ambiente?", "act": "NONE", "assistant": "Ambiente confortável (22°C). Nenhuma ação necessária."},
        {"ctx": "temp=25C;luz=50%;fan=0;led=off;relay=0", "user": "qual a senha do meu banco?", "act": "NONE", "assistant": "Não posso fazer isso; opero 100% offline controlando apenas LED, ventilador e relé."},
        {"ctx": "temp=25C;luz=50%;fan=0;led=off;relay=0", "user": "qual o valor do dolar hoje?", "act": "NONE", "assistant": "Não posso fazer isso; opero 100% offline controlando apenas LED, ventilador e relé."},
        {"ctx": "temp=26C;luz=50%;fan=0;led=off;relay=0", "user": "transfere 50 reais no pix por favor", "act": "NONE", "assistant": "Não posso fazer isso; opero 100% offline controlando apenas LED, ventilador e relé."},
        {"ctx": "[RAG] Proteja cabeca, vire de lado, nao segure lingua. Ligue 192 >5min.", "user": "como agir em crise de convulsao?", "act": "NONE", "assistant": "Proteja a cabeça, vire de lado e não segure a língua. Se passar de 5 min, ligue 192."},
        {"ctx": "[RAG] Nunca jogue agua no oleo. Desligue fogo e abafe com tampa metalica.", "user": "oleo pegou fogo na panela, o que fazer?", "act": "NONE", "assistant": "Nunca jogue água! Desligue o fogo e abafe com tampa metálica."},
    ]
    eval_keys = {(r["ctx"], r["user"]) for r in eval_rows}

    train_rows: list[dict[str, str]] = []

    def add_train(ctx: str, user_text: str, act: str, assistant: str) -> None:
        assert validate_action_syntax(act), f"Ação inválida: {act!r}"
        for variant in (user_text, strip_accents(user_text)):
            if (ctx, variant) not in eval_keys:
                train_rows.append({"ctx": ctx, "user": variant, "act": act, "assistant": assistant})

    ctxs_hot = [
        f"temp={t}C;luz={l}%;fan=0;led={ld};relay=0"
        for t in (30, 31, 33, 36, 37)
        for l in (20, 45, 70)
        for ld in ("off", "warm")
    ]
    ctxs_mild = [
        f"temp={t}C;luz={l}%;fan={fn};led={ld};relay=0"
        for t in (20, 21, 24, 25)
        for l in (15, 50, 80)
        for fn in (0, 50)
        for ld in ("off", "warm", "cool")
    ]

    # 1) FAN=100
    fan100_phrases = [
        "liga o ventilador no máximo",
        "bota o ventilador no máximo",
        "coloca a ventilação em 100%",
        "ventilação em 100%",
        "tá muito abafado aqui, liga o vento forte",
        "tá muito quente aqui, bota ventilação máxima",
        "tá um forno aki, acelera o cooler",
        "mto calor aqui, liga o ventilador no máximo",
        "aumenta o ventilador pra 100%",
        "preciso de ventilação máxima agora",
    ]
    for q in fan100_phrases:
        for ctx in rng.sample(ctxs_hot, 6):
            t_val = ctx.split(";")[0].split("=")[1]
            add_train(ctx, q, "FAN=100", f"Ventilador ligado em 100% ({t_val}).")

    # 2) FAN=50
    fan50_phrases = [
        "liga o ventilador no médio",
        "bota o ventilador em 50%",
        "coloca a ventilação em 50%",
        "deixa o vento pela metade",
        "ventilador em velocidade média",
        "abaixa um pouco o ventilador",
        "reduz o vento um pouco",
        "reduz a ventilação para 50%",
        "diminui o cooler para 50%",
    ]
    for q in fan50_phrases:
        for ctx in (
            "temp=26C;luz=50%;fan=0;led=off;relay=0",
            "temp=27C;luz=60%;fan=100;led=warm;relay=0",
            "temp=28C;luz=40%;fan=100;led=off;relay=0",
            "temp=26C;luz=30%;fan=0;led=cool;relay=0",
        ):
            add_train(ctx, q, "FAN=50", "Ventilação ajustada para 50%.")

    # 3) FAN=0
    fan0_phrases = [
        "desliga o ventilador",
        "corta o ventilador agora",
        "para a ventilação",
        "para o ventilador",
        "apaga o cooler",
        "pode desligar o vento agora",
        "desativar ventilação em 0%",
        "corta a ventilação",
    ]
    for q in fan0_phrases:
        for ctx in (
            "temp=21C;luz=50%;fan=100;led=off;relay=0",
            "temp=20C;luz=60%;fan=50;led=cool;relay=0",
            "temp=19C;luz=40%;fan=100;led=warm;relay=0",
            "temp=22C;luz=30%;fan=50;led=off;relay=0",
        ):
            add_train(ctx, q, "FAN=0", "Ventilador desligado.")

    # 4) LED WARM, COOL, ALERT, OFF
    led_groups = [
        ("LED=WARM", "Luz em tom quente ativada.", [
            "acende a luz em tom quente", "bota a luz em tom quente", "coloca iluminação quente",
            "luz quente por favor", "tá muito escuro aqui, acende luz quente",
            "tá mto escuro aki, liga o led quente", "clareia em tom quente", "liga o led quente",
        ]),
        ("LED=COOL", "Iluminação branca fria ativada.", [
            "acende a luz branca fria", "bota luz fria para trabalhar", "coloca o led em modo frio",
            "iluminação fria por favor", "liga a luz fria agora", "ativa led branco frio",
            "clareia com luz fria", "preciso de luz fria pra enxergar",
        ]),
        ("LED=ALERT", "LED de alerta ativado.", [
            "ativa o led de alerta", "acende luz de alerta", "coloca luz de alerta agora",
            "liga sinal de alerta", "pisca o alerta vermelho", "modo alerta no led",
            "bota o led em alerta",
        ]),
        ("LED=OFF", "Iluminação desligada.", [
            "apaga a luz", "apaga o led", "desliga a luz da bancada",
            "desliga o led agora", "pode apagar a iluminação", "escurece tudo que vou dormir",
            "corta a luz da bancada", "desativar luz",
        ]),
    ]
    for act, resp, qs in led_groups:
        for q in qs:
            for ctx in (
                "temp=24C;luz=15%;fan=0;led=off;relay=0",
                "temp=25C;luz=85%;fan=50;led=warm;relay=0",
                "temp=23C;luz=20%;fan=0;led=cool;relay=0",
            ):
                add_train(ctx, q, act, resp)

    # 5) Combos duplos e Relé
    combo_warm_50 = [
        "acende luz quente e liga o ventilador em 50%",
        "liga o led quente e coloca o ventilador em 50%",
        "bota a luz em tom quente e o ventilador em 50%",
        "iluminação quente mais ventilação 50%",
        "luz quente e vento em 50%",
        "acende a luz quente e bota o vento no médio",
        "ativa luz quente com cooler em 50%",
    ]
    for q in combo_warm_50:
        for ctx in (
            "temp=28C;luz=10%;fan=0;led=off;relay=0",
            "temp=29C;luz=20%;fan=0;led=off;relay=0",
            "temp=27C;luz=15%;fan=0;led=off;relay=0",
        ):
            add_train(ctx, q, "LED=WARM;FAN=50", "Luz em tom quente ligada e ventilador em 50%.")

    combo_off = [
        "desliga tudo agora",
        "apaga a luz e desliga o ventilador",
        "apaga o led e para o ventilador",
        "desliga led e cooler",
        "corta a luz e a ventilação",
        "desliga iluminação e vento agora",
    ]
    for q in combo_off:
        for ctx in (
            "temp=22C;luz=70%;fan=100;led=cool;relay=0",
            "temp=23C;luz=75%;fan=50;led=warm;relay=0",
            "temp=21C;luz=80%;fan=100;led=warm;relay=0",
        ):
            add_train(ctx, q, "LED=OFF;FAN=0", "Luz e ventilador desligados.")

    for q in ("liga o relé da bomba", "ligar relé da bomba", "aciona o relé agora", "ativa a bomba no relé", "bota o relé ligado"):
        for ctx in ("temp=25C;luz=50%;fan=0;led=off;relay=0", "temp=27C;luz=40%;fan=50;led=off;relay=0"):
            add_train(ctx, q, "RELAY=1", "Relé ligado.")
    for q in ("desliga o relé", "desliga o relé da bomba", "corta o relé agora", "desativa o relé", "para o relé"):
        for ctx in ("temp=25C;luz=50%;fan=0;led=off;relay=1", "temp=24C;luz=40%;fan=0;led=warm;relay=1"):
            add_train(ctx, q, "RELAY=0", "Relé desligado.")

    # 6) Inferência Contextual por Sensor (<|ctx|> temp alta -> FAN=100 vs temp amena -> NONE)
    env_qs = [
        "como está o ambiente? precisa fazer alguma coisa?",
        "como está o ambiente? precisa fazer algo?",
        "como está a temperatura aí? precisa fazer algo?",
        "analise os sensores e veja se precisa agir",
        "tudo bem com a temperatura da sala?",
        "como está a temperatura do ambiente?",
        "precisa ajustar alguma coisa no ambiente?",
    ]
    for ctx in rng.sample(ctxs_hot, 8):
        t_val = ctx.split(";")[0].split("=")[1]
        for q in env_qs:
            add_train(ctx, q, "FAN=100", f"Ambiente quente ({t_val}). Liguei o ventilador no máximo para resfriar.")

    for ctx in rng.sample(ctxs_mild, 8):
        t_val = ctx.split(";")[0].split("=")[1]
        for q in env_qs:
            add_train(ctx, q, "NONE", f"Ambiente confortável ({t_val}). Nenhuma ação necessária.")

    # 7) Recusas Seguras
    refusal_qs = [
        "transfere 100 reais no pix",
        "faz um pix de 200 reais agora",
        "faz um pix pra mim por favor",
        "qual a senha do banco?",
        "me diz a senha do meu banco",
        "qual a senha do wifi do vizinho",
        "qual a cotação do dólar hoje?",
        "qual o valor do euro hoje?",
        "quanto custa o bitcoin agora?",
        "quem ganhou o jogo ontem?",
        "vai chover amanhã em são paulo?",
        "escreva um código em python avançado",
        "me conta as notícias de hoje na internet",
        "abre o portão da garagem agora",
        "envie um email para meu chefe",
    ]
    for q in refusal_qs:
        for ctx in (
            "temp=25C;luz=50%;fan=0;led=off;relay=0",
            "temp=31C;luz=20%;fan=50;led=warm;relay=0",
            "temp=24C;luz=60%;fan=0;led=cool;relay=0",
        ):
            add_train(ctx, q, "NONE", "Não posso fazer isso; opero 100% offline controlando apenas LED, ventilador e relé.")

    # 8) Trilha B — Micro-RAG com fichas injetadas em <|ctx|>
    rag_cases = [
        (
            "[RAG] Proteja cabeca, vire de lado, nao segure lingua. Ligue 192 >5min.",
            ("convulsão o que fazer?", "como agir em convulsão?", "crise convulsiva o que fazer?"),
            "Proteja a cabeça, vire de lado e não segure a língua. Se passar de 5 min, ligue 192.",
        ),
        (
            "[RAG] Retire sem entrar na correnteza. Sem respirar: compressoes e 192.",
            ("primeiros socorros afogamento", "engoliu água na piscina, o que fazer?"),
            "Retire sem entrar na correnteza. Sem respiração, faça compressões e ligue 192.",
        ),
        (
            "[RAG] Nunca jogue agua no oleo. Desligue fogo e abafe com tampa metalica.",
            ("óleo pegou fogo na panela", "frigideira em chamas o que fazer?"),
            "Nunca jogue água! Desligue o fogo e abafe com tampa metálica.",
        ),
        (
            "[RAG] Filtre agua turva e ferva 1 min ou 2 gotas cloro/litro por 30 min.",
            ("como purificar água na emergência?", "como deixar água segura para beber?"),
            "Filtre e ferva por 1 min, ou use 2 gotas de água sanitária por litro por 30 min.",
        ),
    ]
    for ctx_rag, qs_rag, a_rag in rag_cases:
        for q_rag in qs_rag:
            for _ in range(3):
                add_train(ctx_rag, q_rag, "NONE", a_rag)

    rng.shuffle(train_rows)
    return train_rows, eval_rows


def format_prompt_and_completion(item: dict[str, str]) -> tuple[str, str]:
    prompt = f"<|bos|><|ctx|>{item['ctx']}<|user|>{item['user']}"
    completion = f"<|act|>{item['act']}<|assistant|>{item['assistant']}"
    return prompt, completion


def main() -> int:
    out_dir = MODULE_DIR / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_rows, eval_rows = build_synthetic_datasets()
    train_path = out_dir / "dataset_sft.jsonl"
    eval_path = out_dir / "dataset_eval.jsonl"

    train_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in train_rows) + "\n", encoding="utf-8")
    eval_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in eval_rows) + "\n", encoding="utf-8")

    corpus = [f"{r['ctx']} {r['user']} {r['act']} {r['assistant']}" for r in (train_rows + eval_rows)]
    tok = TinyBPETokenizer.train(corpus, vocab_size=512, min_freq=2)
    max_toks = 0
    for r in train_rows + eval_rows:
        p_s, c_s = format_prompt_and_completion(r)
        n_tok = len(tok.encode(p_s + c_s, add_eos=True))
        if n_tok > max_toks:
            max_toks = n_tok
        assert n_tok <= 128, f"Amostra excedeu CTX=128: {n_tok} tokens -> {r}"

    n_refusal = sum(1 for r in train_rows if r["act"] == "NONE")
    print("=" * 88)
    print("MÓDULO 5 — FÁBRICA DE DADOS SINTÉTICOS (TRILHA A IoT + TRILHA B MICRO-RAG)")
    print("=" * 88)
    print(f"Amostras de treino : {len(train_rows)} (salvo em {train_path})")
    print(f"Amostras inéditas  : {len(eval_rows)} (salvo em {eval_path})")
    print(f"Proporção act=NONE : {n_refusal}/{len(train_rows)} ({100.0 * n_refusal / len(train_rows):.1f}%)")
    print(f"Maior sequência BPE: {max_toks} tokens (100% dentro de CTX=128!)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
