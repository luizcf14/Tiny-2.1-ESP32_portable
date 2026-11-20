#!/usr/bin/env python3
"""Módulo 5 (Lab 1): Fábrica de dados sintéticos (Trilha A IoT + Trilha B Micro-RAG) + validação."""

from __future__ import annotations

import json
import pathlib
import random
import re
import sys

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

# Template de prompt pronto caso o aluno queira expandir com um LLM Professor externo (ex.: Gemini API)
TEACHER_PROMPT_TEMPLATE = """
Você é um gerador de dados sintéticos para um agente embarcado em ESP32-S3 (PT-BR).
Gere exemplos JSONL no esquema:
{"ctx":"temp=31C;luz=20%;fan=0;led=off","user":"...","act":"FAN=100;LED=WARM","assistant":"..."}
Regras estritas:
1. Ações permitidas em "act": NONE, FAN=0|50|100, LED=OFF|WARM|COOL|ALERT, RELAY=0|1 (combináveis por ';').
2. Inclua variações coloquiais, abreviações ("vc", "tá", "mto") e 15% de recusas seguras (act="NONE").
3. Todo o registro tokenizado deve caber em <= 120 tokens BPE.
""".strip()


def validate_action_syntax(act: str) -> bool:
    return bool(ALLOWED_ACT_RE.match(act.strip()))


def build_synthetic_datasets(seed: int = 42) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    rng = random.Random(seed)
    train_rows: list[dict[str, str]] = []
    eval_rows: list[dict[str, str]] = []

    def add(ctx: str, user: str, act: str, assistant: str, is_eval: bool = False) -> None:
        assert validate_action_syntax(act), f"Ação inválida: {act!r}"
        item = {"ctx": ctx, "user": user, "act": act, "assistant": assistant}
        (eval_rows if is_eval else train_rows).append(item)

    # 1) Ventilador no máximo (vários contextos de temperatura)
    fan_max_train = [
        "liga o ventilador no máximo",
        "coloca a ventilação em 100%",
        "tá muito abafado aqui, liga o vento forte",
        "tá um forno, acelera esse ventilador",
        "ventilação no talo por favor",
        "liga o cooler no máximo",
        "aumenta o ventilador pra 100%",
        "preciso de ventilação máxima agora",
    ]
    fan_max_eval = [
        "ta mto quente aki, bota o ventilador no maximo",
        "ventilacao em 100% agora",
    ]
    for t in (28, 30, 32, 34, 36):
        ctx = f"temp={t}C;luz=50%;fan=0;led=off;relay=0"
        for q in fan_max_train:
            add(ctx, q, "FAN=100", f"Ventilador ligado em 100% (temperatura em {t}°C).")
        add(ctx, rng.choice(fan_max_eval), "FAN=100", f"Ventilador ligado em 100% (temperatura em {t}°C).", is_eval=True)

    # 2) Ventilador médio (50%) e desligar (0%)
    fan_mid_train = [
        "liga o ventilador no médio",
        "coloca a ventilação em 50%",
        "deixa o vento pela metade",
        "ventilador em velocidade média",
        "abaixa um pouco o ventilador",
        "reduz a ventilação para 50%",
    ]
    for ctx in ("temp=26C;luz=50%;fan=0;led=off;relay=0", "temp=27C;luz=60%;fan=100;led=warm;relay=0"):
        for q in fan_mid_train:
            add(ctx, q, "FAN=50", "Ventilação ajustada para 50%.")
    add("temp=27C;luz=50%;fan=100;led=warm;relay=0", "abaixa um pouco o vento", "FAN=50", "Ventilação ajustada para 50%.", is_eval=True)

    fan_off_train = [
        "desliga o ventilador",
        "para a ventilação",
        "apaga o cooler",
        "pode desligar o vento agora",
        "desativar ventilador em 0%",
    ]
    for ctx in ("temp=21C;luz=50%;fan=100;led=off;relay=0", "temp=20C;luz=40%;fan=50;led=cool;relay=0"):
        for q in fan_off_train:
            add(ctx, q, "FAN=0", "Ventilador desligado.")
    add("temp=19C;luz=40%;fan=50;led=off;relay=0", "corta a ventilacao agora", "FAN=0", "Ventilador desligado.", is_eval=True)

    # 3) LED WARM, COOL, ALERT, OFF
    led_cases = [
        ("LED=WARM", "Luz em tom quente ativada.", [
            "acende a luz em tom quente", "coloca iluminação quente", "luz quente por favor",
            "tá escuro, liga luz amarela quente", "clareia em tom quente", "liga o led quente",
        ], ["ta mto escuro aki, acende a luz em tom quente"]),
        ("LED=COOL", "Iluminação branca fria ativada.", [
            "acende a luz branca fria", "coloca o led em modo frio", "iluminação fria para trabalhar",
            "liga a luz fria", "ativa led branco frio", "clareia com luz fria",
        ], ["bota luz fria pra enxergar melhor"]),
        ("LED=ALERT", "LED de alerta ativado.", [
            "ativa o led de alerta", "coloca luz de emergência", "liga sinal de alerta",
            "pisca o alerta vermelho", "modo alerta no led",
        ], ["acende luz de alerta"]),
        ("LED=OFF", "Iluminação desligada.", [
            "apaga a luz", "desliga o led", "pode apagar a iluminação",
            "escurece tudo que vou dormir", "desativar luz da bancada",
        ], ["apaga o led da bancada"]),
    ]
    for act, resp, tr_qs, ev_qs in led_cases:
        for ctx in ("temp=24C;luz=15%;fan=0;led=off;relay=0", "temp=25C;luz=85%;fan=50;led=warm;relay=0"):
            for q in tr_qs:
                add(ctx, q, act, resp)
        add("temp=24C;luz=15%;fan=0;led=off;relay=0", ev_qs[0], act, resp, is_eval=True)

    # 4) Combos duplos (Caso 1 do Capstone!) e Relé
    combo_train = [
        "acende a luz em tom quente e liga o ventilador em 50%",
        "liga o led quente e coloca o ventilador no médio",
        "iluminação quente mais ventilação 50%",
        "luz quente e vento em 50%",
        "ativa luz quente com cooler em 50%",
    ]
    for ctx in ("temp=28C;luz=10%;fan=0;led=off;relay=0", "temp=29C;luz=20%;fan=0;led=off;relay=0"):
        for q in combo_train:
            add(ctx, q, "LED=WARM;FAN=50", "Luz em tom quente ligada e ventilador em 50%.")
    add("temp=28C;luz=15%;fan=0;led=off;relay=0", "acende luz quente e bota o ventilador em 50%", "LED=WARM;FAN=50", "Luz em tom quente ligada e ventilador em 50%.", is_eval=True)

    for q in ("desliga tudo agora", "apaga a luz e desliga o ventilador", "desliga led e cooler"):
        add("temp=22C;luz=70%;fan=100;led=cool;relay=0", q, "LED=OFF;FAN=0", "Luz e ventilador desligados.")
    add("temp=22C;luz=70%;fan=50;led=warm;relay=0", "apaga a luz e para o ventilador", "LED=OFF;FAN=0", "Luz e ventilador desligados.", is_eval=True)

    for q in ("liga o relé da bomba", "aciona o relé", "ativa a bomba no relé"):
        add("temp=25C;luz=50%;fan=0;led=off;relay=0", q, "RELAY=1", "Relé ligado.")
    for q in ("desliga o relé", "corta o relé da bomba", "desativa o relé"):
        add("temp=25C;luz=50%;fan=0;led=off;relay=1", q, "RELAY=0", "Relé desligado.")
    add("temp=25C;luz=50%;fan=0;led=off;relay=0", "ligar rele agora", "RELAY=1", "Relé ligado.", is_eval=True)

    # 5) Inferência Contextual por Sensor (<|ctx|> temp=34C vs temp=22C)
    ctx_questions = [
        "como está o ambiente? precisa fazer alguma coisa?",
        "analise os sensores e veja se precisa agir",
        "tudo bem com a temperatura da sala?",
        "como está a temperatura aí?",
    ]
    for t in (33, 34, 35, 37):
        ctx = f"temp={t}C;luz=50%;fan=0;led=off;relay=0"
        for q in ctx_questions:
            add(ctx, q, "FAN=100", f"Ambiente quente ({t}°C). Liguei o ventilador no máximo para resfriar.")
    add("temp=34C;luz=45%;fan=0;led=off;relay=0", "como esta o ambiente? precisa fazer algo?", "FAN=100", "Ambiente quente (34°C). Liguei o ventilador no máximo para resfriar.", is_eval=True)

    for t in (21, 22, 23, 24):
        ctx = f"temp={t}C;luz=55%;fan=0;led=off;relay=0"
        for q in ctx_questions:
            add(ctx, q, "NONE", f"Ambiente confortável ({t}°C). Nenhuma ação necessária.")
    add("temp=23C;luz=55%;fan=0;led=off;relay=0", "como esta a temperatura do ambiente?", "NONE", "Ambiente confortável (23°C). Nenhuma ação necessária.", is_eval=True)

    # 6) Recusas Seguras (15%+, ZERO falsos positivos em atuadores!)
    refusals_train = [
        "transfere 100 reais no pix",
        "faz um pix pra mim",
        "qual a senha do banco?",
        "me diz a senha do wifi do vizinho",
        "qual a cotação do dólar hoje?",
        "quanto custa o bitcoin agora?",
        "quem ganhou o jogo ontem?",
        "vai chover amanhã em são paulo?",
        "escreva um código em python avançado",
        "me conta as notícias de hoje na internet",
        "abre o portão da garagem e destranca a porta",
        "envie um email para meu chefe",
    ]
    for ctx in ("temp=25C;luz=50%;fan=0;led=off;relay=0", "temp=31C;luz=20%;fan=50;led=warm;relay=0"):
        for q in refusals_train:
            add(ctx, q, "NONE", "Não posso fazer isso; opero 100% offline controlando apenas LED, ventilador e relé.")
    for q_ev in ("faz um pix de 50 reais agora", "qual a senha do meu banco?", "qual o valor do dolar hoje?"):
        add("temp=25C;luz=50%;fan=0;led=off;relay=0", q_ev, "NONE", "Não posso fazer isso; opero 100% offline controlando apenas LED, ventilador e relé.", is_eval=True)

    # 7) Trilha B — Micro-RAG com ficha técnica em <|ctx|>
    rag_samples = [
        ("[RAG] Proteja a cabeca, vire de lado, nunca segure a lingua. Ligue 192 se passar de 5 min.",
         "convulsão o que fazer?", "Proteja a cabeça, vire a pessoa de lado e nunca segure a língua. Ligue 192 se durar mais de 5 min."),
        ("[RAG] Retire sem entrar na correnteza. Sem respiracao inicie compressoes e ligue 192.",
         "primeiros socorros afogamento", "Retire sem entrar na correnteza. Se não respirar, faça compressões e ligue 192."),
        ("[RAG] Nunca jogue agua no oleo quente. Desligue o fogo e abafe com tampa metalica.",
         "oleo pegou fogo na panela", "Nunca jogue água. Desligue o fogo e abafe a panela com uma tampa metálica."),
        ("[RAG] Filtre agua turva e ferva por 1 min ou use 2 gotas de agua sanitaria por litro (30 min).",
         "como purificar agua na emergencia?", "Filtre a água e ferva por 1 min, ou use 2 gotas de água sanitária por litro por 30 min."),
    ]
    for ctx_rag, q_rag, a_rag in rag_samples:
        for _ in range(4):
            add(ctx_rag, q_rag, "NONE", a_rag)
    add(rag_samples[0][0], "como agir em crise de convulsao?", "NONE", rag_samples[0][2], is_eval=True)

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

    # Validação de orçamento de tokens com TinyBPETokenizer (vocab=1024)
    corpus = [
        f"{r['ctx']} {r['user']} {r['act']} {r['assistant']}" for r in (train_rows + eval_rows)
    ]
    tok = TinyBPETokenizer.train(corpus, vocab_size=1024, min_freq=2)
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
