#!/usr/bin/env python3
"""Generate an extensive Brazilian Q&A dataset for TINY 2.1 (custom_qa.txt + custom_qa.jsonl)."""

from __future__ import annotations

import argparse
import math
import pathlib
import sys
import unicodedata

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from txt_to_jsonl import convert_txt_to_jsonl

MAXQ = 72
MAX_TOTAL = 126  # len(q) + 1 (SEP) + len(a) <= 126 so EOS fits inside CTX=128

# (UF, nome, prep, capital, regiao, lat, lon)
ESTADOS = [
    ("AC", "Acre", "do", "Rio Branco", "Norte", -9.97499, -67.8243),
    ("AL", "Alagoas", "de", "Maceió", "Nordeste", -9.66599, -35.7350),
    ("AP", "Amapá", "do", "Macapá", "Norte", 0.03493, -51.0694),
    ("AM", "Amazonas", "do", "Manaus", "Norte", -3.11903, -60.0217),
    ("BA", "Bahia", "da", "Salvador", "Nordeste", -12.9714, -38.5014),
    ("CE", "Ceará", "do", "Fortaleza", "Nordeste", -3.71722, -38.5433),
    ("DF", "Distrito Federal", "do", "Brasília", "Centro-Oeste", -15.7801, -47.9292),
    ("ES", "Espírito Santo", "do", "Vitória", "Sudeste", -20.3155, -40.3128),
    ("GO", "Goiás", "de", "Goiânia", "Centro-Oeste", -16.6869, -49.2648),
    ("MA", "Maranhão", "do", "São Luís", "Nordeste", -2.53073, -44.3068),
    ("MT", "Mato Grosso", "de", "Cuiabá", "Centro-Oeste", -15.6014, -56.0979),
    ("MS", "Mato Grosso do Sul", "de", "Campo Grande", "Centro-Oeste", -20.4697, -54.6201),
    ("MG", "Minas Gerais", "de", "Belo Horizonte", "Sudeste", -19.9167, -43.9345),
    ("PA", "Pará", "do", "Belém", "Norte", -1.45583, -48.5044),
    ("PB", "Paraíba", "da", "João Pessoa", "Nordeste", -7.11532, -34.8610),
    ("PR", "Paraná", "do", "Curitiba", "Sul", -25.4284, -49.2733),
    ("PE", "Pernambuco", "de", "Recife", "Nordeste", -8.04756, -34.8770),
    ("PI", "Piauí", "do", "Teresina", "Nordeste", -5.08921, -42.8016),
    ("RJ", "Rio de Janeiro", "do", "Rio de Janeiro", "Sudeste", -22.9068, -43.1729),
    ("RN", "Rio Grande do Norte", "do", "Natal", "Nordeste", -5.79448, -35.2110),
    ("RS", "Rio Grande do Sul", "do", "Porto Alegre", "Sul", -30.0346, -51.2177),
    ("RO", "Rondônia", "de", "Porto Velho", "Norte", -8.76077, -63.8999),
    ("RR", "Roraima", "de", "Boa Vista", "Norte", 2.82350, -60.6758),
    ("SC", "Santa Catarina", "de", "Florianópolis", "Sul", -27.5954, -48.5480),
    ("SP", "São Paulo", "de", "São Paulo", "Sudeste", -23.5505, -46.6333),
    ("SE", "Sergipe", "de", "Aracaju", "Nordeste", -10.9472, -37.0731),
    ("TO", "Tocantins", "do", "Palmas", "Norte", -10.2491, -48.3243),
]

HUB_CAPITALS = {
    "Brasília",
    "Belém",
    "São Paulo",
    "Rio de Janeiro",
    "Belo Horizonte",
    "Salvador",
    "Recife",
    "Fortaleza",
    "Manaus",
    "Curitiba",
    "Porto Alegre",
}

RODOVIA_KM: dict[tuple[str, str], int] = {
    ("Belém", "Brasília"): 2120,
    ("Belém", "São Luís"): 806,
    ("Belém", "Palmas"): 1225,
    ("Belo Horizonte", "Brasília"): 716,
    ("Belo Horizonte", "Rio de Janeiro"): 434,
    ("Belo Horizonte", "São Paulo"): 586,
    ("Belo Horizonte", "Vitória"): 524,
    ("Brasília", "Curitiba"): 1366,
    ("Brasília", "Goiânia"): 209,
    ("Brasília", "Palmas"): 840,
    ("Brasília", "Rio de Janeiro"): 1148,
    ("Brasília", "Salvador"): 1446,
    ("Brasília", "São Paulo"): 1015,
    ("Campo Grande", "Cuiabá"): 694,
    ("Curitiba", "Florianópolis"): 300,
    ("Curitiba", "Porto Alegre"): 711,
    ("Curitiba", "São Paulo"): 408,
    ("Florianópolis", "Porto Alegre"): 476,
    ("Fortaleza", "Natal"): 537,
    ("Fortaleza", "Recife"): 800,
    ("Fortaleza", "Teresina"): 634,
    ("João Pessoa", "Natal"): 185,
    ("João Pessoa", "Recife"): 120,
    ("Maceió", "Recife"): 285,
    ("Maceió", "Aracaju"): 294,
    ("Rio de Janeiro", "São Paulo"): 430,
    ("Rio de Janeiro", "Vitória"): 521,
    ("Salvador", "Aracaju"): 356,
}


def to_ascii(text: str) -> str:
    norm = unicodedata.normalize("NFKD", text)
    ascii_bytes = norm.encode("ascii", "ignore").decode("ascii")
    cleaned = "".join(ch for ch in ascii_bytes if 32 <= ord(ch) <= 126)
    return " ".join(cleaned.split())


def normalize_q(text: str) -> str:
    s = to_ascii(text).lower()
    out: list[str] = []
    for ch in s:
        if "a" <= ch <= "z" and len(out) >= 2 and out[-1] == ch and out[-2] == ch:
            continue
        out.append(ch)
    return "".join(out)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> int:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    return int(round(2.0 * r * math.asin(math.sqrt(a))))


def validate_entry(questions: list[str], answer: str) -> None:
    a_ascii = to_ascii(answer)
    if not a_ascii:
        raise ValueError(f"Empty answer for {questions}")
    for q in questions:
        q_ascii = normalize_q(q)
        if not q_ascii:
            raise ValueError(f"Empty question in {questions}")
        if len(q_ascii) > MAXQ:
            raise ValueError(f"Question > {MAXQ} chars ({len(q_ascii)}): {q_ascii}")
        total = len(q_ascii) + 1 + len(a_ascii)
        if total > MAX_TOTAL:
            raise ValueError(
                f"Pair > {MAX_TOTAL} chars ({total}): Q='{q_ascii}' ({len(q_ascii)}) | A='{a_ascii}' ({len(a_ascii)})"
            )


def build_sections(all_pairs: bool = False) -> list[tuple[str, list[tuple[list[str], str]]]]:
    sections: list[tuple[str, list[tuple[list[str], str]]]] = []

    # 1. Conversação Cotidiana, Identidade AMEBA, Hardware ESP32 e Personalizados
    persona = [
        (["oi", "oii", "oi!", "ola", "olaa", "ola!", "opa", "e ai", "e ai?", "eae", "fala", "salve", "alo", "hi", "hello"], "Olá! Sou o AMEBA. Como posso ajudar?"),
        (["tudo bem?", "tudo bem", "oi tudo bem?", "ola tudo bem?", "tudo bom?", "tudo bom", "tudo certo?", "tudo joia?", "td bem?", "beleza?", "blz?"], "Tudo ótimo por aqui no ESP32! E com você?"),
        (["como vai?", "como vai", "como você está?", "como você esta?", "como vc ta?", "como anda?"], "Tudo ótimo por aqui no ESP32! E com você?"),
        (["tudo bem também", "estou bem", "tudo ótimo", "bem obrigado", "to bem", "tudo certo"], "Que bom! Pode mandar sua pergunta."),
        (["bom dia", "bom dia!", "bom dia ameba"], "Bom dia! Sou o AMEBA no ESP32. Em que posso ajudar?"),
        (["boa tarde", "boa tarde!", "boa tarde ameba"], "Boa tarde! Sou o AMEBA no ESP32. Pode perguntar!"),
        (["boa noite", "boa noite!", "boa noite ameba"], "Boa noite! Estou ligado aqui no ESP32. O que deseja saber?"),
        (["tchau", "até mais", "até logo", "falou", "adeus", "fui", "bye"], "Até mais! Quando precisar, estarei aqui no ESP32."),
        (["obrigado", "obrigada", "valeu", "muito obrigado", "vlw"], "De nada! Estou pronto para a próxima pergunta."),
        (["legal", "muito bom", "top", "show", "bacana", "maneiro"], "Valeu! Pode mandar a próxima pergunta."),
        (["kkk", "haha", "hehe", "rs"], "Haha! Gostou? Pergunta mais alguma coisa."),
        (["teste", "testando", "test", "123", "alou"], "Teste recebido! O AMEBA está online no ESP32."),
        (["quem é você?", "quem é você", "quem é vc?", "fale sobre você"], "Eu sou o AMEBA rodando em um ESP32."),
        (["qual seu hardware?", "onde você está rodando?", "onde você mora?"], "Estou rodando direto em um ESP32 sem PSRAM."),
        (["qual é o seu nome?", "qual seu nome?", "qual seu nome"], "Meu nome é AMEBA."),
        (["ameba?", "ameba"], "Sim e dai?"),
        (["como pode me ajudar?", "como você pode me ajudar?"], "Se nem você sabe, tá difícil."),
        (["o que você sabe fazer?", "sobre o que posso perguntar?", "o que você conhece?"], "Sei sobre estados, capitais, distâncias, biomas e presidentes do Brasil."),
        (["quem criou você?", "quem te treinou?"], "Fui portado para ESP32 e ajustado localmente em PyTorch."),
        (["você fala português?", "fala português?"], "Sim! Agora eu entendo e respondo em português no ESP32."),
        (["você tem internet?", "você acessa a internet?"], "Não, rodo 100% offline dentro da memória Flash do ESP32."),
        (["quantos parâmetros você tem?", "qual o seu tamanho?"], "Tenho 367 mil parâmetros em 8 bits (381 KB na Flash)."),
        (["me conta uma piada", "conta uma piada"], "Por que o ESP32 foi ao médico? Porque estava sem memória livre!"),
        (
            [
                "qual a cidade mais bonita do brasil?",
                "qual a cidade mais legal do brasil?",
                "qual a cidade mais charmosa do brasil?",
                "qual a cidade mais famosa do brasil?",
            ],
            "É Belém do Pará.",
        ),
        (["qual a cidade mais antiga do brasil?"], "Pergunta pra o Dom Pedro."),
        (["qual a cidade com mais tecnologia do brasil?", "qual a cidade mais tecnológica do brasil?"], "Talvez seja São Paulo."),
        (["qual a segunda cidade mais bonita do brasil?", "qual a segunda cidade a mais bonita do brasil?"], "Talvez seja o Rio de Janeiro."),
        (["qual a segunda cidade com melhor pão de queijo?"], "Talvez seja em Minas Gerais."),
        (["o que é esp32?", "o que é esp32"], "O ESP32 é um microcontrolador com Wi-Fi e Bluetooth da Espressif."),
        (["qual o endereço web do tiny?", "qual o endereço web do ameba?", "qual o ip do esp32?"], "Acesse http://192.168.4.1 na rede TINY-2.1-ESP32."),
        (["qual a velocidade da serial?", "qual baud rate da serial?"], "A serial do TINY roda a 115200 baud."),
        (["qual a capital do brasil?", "qual a capital do brasil"], "A capital do Brasil é Brasília."),
        (["qual a capital da franca?", "qual a capital da franca"], "A capital da França é Paris."),
        (["qual a capital do japao?", "qual a capital do japao"], "A capital do Japão é Tóquio."),
        (["quanto é 7 vezes 8?", "quanto é 7 vezes 8"], "7 vezes 8 é 56."),
        (["quanto é 12 mais 15?", "quanto é 12 mais 15"], "12 mais 15 é 27."),
        (["uma baleia é um peixe?", "uma baleia é um peixe"], "Não, uma baleia é um mamífero."),
        (["quantas patas tem uma aranha?"], "Uma aranha tem oito patas."),
        (["qual o símbolo do ouro?"], "O símbolo do ouro é Au."),
        (["quem inventou o telefone?"], "Alexander Graham Bell inventou o telefone."),
    ]
    sections.append(("CONVERSACAO COTIDIANA, IDENTIDADE AMEBA E PERSONALIZADOS", persona))

    # 1B. Perguntas Fora do Escopo / Desconhecidas (evita alucinar texto aleatório)
    unknown_msg = "Não sei responder isso; sou o AMEBA offline com 367 mil parâmetros."
    unknown_qa = [
        (["asdf", "qwerty", "blablabla", "??", "xyz", "abc", "nada"], unknown_msg),
        (["qual a cotação do dólar hoje?", "quanto está o dólar?", "qual a cotação do bitcoin?", "quanto está o bitcoin?"], unknown_msg),
        (["vai chover amanhã?", "qual a previsão do tempo?", "qual a temperatura hoje?", "que horas são agora?"], unknown_msg),
        (["quem ganhou o jogo ontem?", "qual o resultado da mega sena?", "quem vai ganhar a copa?", "qual a senha do wifi?"], unknown_msg),
        (["onde eu moro?", "qual o meu nome?", "quantos anos eu tenho?", "qual meu filme favorito?", "qual sua comida favorita?"], unknown_msg),
        (["faz um café pra mim", "qual a melhor linguagem de programação?", "me dá o número do pix", "qual o sentido da vida?"], unknown_msg),
        (["me ajuda com matemática avançada", "escreva um código em python", "traduza este texto grande", "o que você acha da política?"], unknown_msg),
    ]
    sections.append(("PERGUNTAS FORA DO ESCOPO (RECUSA SEGURA)", unknown_qa))

    # 2. Estados, Capitais, Siglas e Regiões do Brasil
    estados_qa: list[tuple[list[str], str]] = [
        (["quantos estados tem o Brasil?", "quantas unidades federativas tem o Brasil?"], "O Brasil tem 26 estados e o Distrito Federal, totalizando 27 UFs."),
        (["quais são as regiões do Brasil?", "quantas regiões tem o Brasil?"], "O Brasil tem 5 regiões: Norte, Nordeste, Centro-Oeste, Sudeste e Sul."),
        (["quais estados ficam na região Norte?"], "Norte: Acre, Amapá, Amazonas, Pará, Rondônia, Roraima e Tocantins."),
        (["quais estados ficam na região Sul?"], "Sul: Paraná, Rio Grande do Sul e Santa Catarina."),
        (["quais estados ficam na região Sudeste?"], "Sudeste: Espírito Santo, Minas Gerais, Rio de Janeiro e São Paulo."),
        (["quais estados ficam na região Centro-Oeste?"], "Centro-Oeste: Distrito Federal, Goiás, Mato Grosso e Mato Grosso do Sul."),
        (["quais estados ficam na região Nordeste?"], "Nordeste: AL, BA, CE, MA, PB, PE, PI, RN e SE."),
    ]
    for uf, nome, prep, capital, regiao, _, _ in ESTADOS:
        estados_qa.append((
            [f"qual a capital {prep} {nome}?", f"capital {prep} {nome}", f"qual a capital de {uf}?"],
            f"A capital {prep} {nome} ({uf}) é {capital}.",
        ))
        estados_qa.append((
            [f"{capital} é capital de qual estado?", f"onde fica {capital}?"],
            f"{capital} é a capital {prep} {nome} ({uf}), na região {regiao}.",
        ))
        estados_qa.append((
            [f"em qual região fica {nome}?", f"região {prep} {nome}", f"região de {uf}"],
            f"{nome} ({uf}) fica na região {regiao} do Brasil.",
        ))
    sections.append(("ESTADOS, CAPITAIS, SIGLAS E REGIOES DO BRASIL (27 UFs)", estados_qa))

    # 3. Biomas Brasileiros (IBGE)
    biomas_qa: list[tuple[list[str], str]] = [
        (["o que é um bioma?", "definição de bioma"], "Bioma é um grande conjunto de ecossistemas com clima, solo e vida típicos."),
        (["quais são os biomas brasileiros?", "quais os biomas do Brasil?"], "Amazônia, Cerrado, Mata Atlântica, Caatinga, Pampa e Pantanal."),
        (["quantos biomas existem no Brasil?"], "O Brasil possui 6 biomas terrestres principais reconhecidos pelo IBGE."),
        (["qual o maior bioma do Brasil?", "maior bioma brasileiro"], "A Amazônia é o maior bioma do Brasil, ocupando cerca de 49% do país."),
        (["qual o segundo maior bioma do Brasil?"], "O Cerrado é o segundo maior bioma brasileiro, com cerca de 24% do país."),
        (["qual o menor bioma do Brasil?", "menor bioma brasileiro"], "O Pantanal é o menor bioma do Brasil em área, com cerca de 1,8% do país."),
        (["qual bioma existe apenas no Brasil?", "qual o único bioma exclusivo do Brasil?"], "A Caatinga é o único bioma exclusivamente brasileiro."),
        (["qual a maior planície alagável do mundo?"], "O Pantanal é a maior planície inundável contínua do planeta."),
        (["qual o bioma de Belém?", "qual bioma fica no Pará?"], "Belém e o Pará ficam no bioma Amazônia."),
        (["qual o bioma de Brasília?", "qual bioma fica em Goiás?"], "Brasília e Goiás ficam no bioma Cerrado."),
        (["qual o bioma do Rio Grande do Sul?", "bioma exclusivo do Sul"], "O Pampa ocorre no Brasil na metade sul do Rio Grande do Sul."),

        # Amazônia
        (["o que é a Amazônia?", "características da Amazônia"], "É a maior floresta tropical do mundo, de clima equatorial quente e úmido."),
        (["qual o clima da Amazônia?", "vegetação da Amazônia"], "Clima equatorial muito chuvoso, com floresta densa e árvores altas."),
        (["animais da Amazônia", "fauna da Amazônia"], "Onça-pintada, arara, boto-cor-de-rosa, pirarucu, sucuri e preguiça."),
        (["plantas da Amazônia", "flora da Amazônia"], "Seringueira, castanheira-do-pará, açaizeiro, samaúma e vitória-régia."),
        (["onde fica a Amazônia no Brasil?"], "Na região Norte e partes do Mato Grosso e Maranhão."),

        # Cerrado
        (["o que é o Cerrado?", "características do Cerrado"], "É a savana brasileira, segundo maior bioma e chamado berço das águas."),
        (["qual a vegetação do Cerrado?", "clima do Cerrado"], "Clima tropical com seca e chuva; árvores de tronco torto e casca grossa."),
        (["animais do Cerrado", "fauna do Cerrado"], "Lobo-guará, tamanduá-bandeira, ema, seriema e tatu-canastra."),
        (["plantas do Cerrado", "flora do Cerrado"], "Ipê, pequi, buriti, baru, cagaita e capim-dourado."),
        (["onde fica o Cerrado?"], "No Centro-Oeste e partes de MG, BA, TO, MA, PI e SP."),

        # Mata Atlântica
        (["o que é a Mata Atlântica?", "características da Mata Atlântica"], "Floresta tropical úmida do litoral brasileiro com altíssima biodiversidade."),
        (["qual o estado atual da Mata Atlântica?"], "É um dos biomas mais ameaçados, restando cerca de 12% da mata original."),
        (["animais da Mata Atlântica", "fauna da Mata Atlântica"], "Mico-leão-dourado, muriqui, onça-pintada, tucano e jaguatirica."),
        (["plantas da Mata Atlântica", "flora da Mata Atlântica"], "Pau-brasil, palmito-juçara, ipê, bromélias e orquídeas."),
        (["onde fica a Mata Atlântica?"], "Ao longo da costa do RN ao RS e no interior de SP, MG e PR."),

        # Caatinga
        (["o que é a Caatinga?", "características da Caatinga"], "Bioma semiárido exclusivo do Brasil, com plantas adaptadas à seca."),
        (["qual a vegetação da Caatinga?", "clima da Caatinga"], "Clima semiárido quente; cactos e árvores que perdem folhas na seca."),
        (["animais da Caatinga", "fauna da Caatinga"], "Tatu-bola, ararinha-azul, asa-branca, mocó e veado-catingueiro."),
        (["plantas da Caatinga", "flora da Caatinga"], "Mandacaru, xique-xique, juazeiro, umbuzeiro e facheiro."),
        (["onde fica a Caatinga?"], "No interior do Nordeste e no norte de Minas Gerais."),

        # Pampa
        (["o que é o Pampa?", "características do Pampa"], "Bioma de campos sulinos com relevo de coxilhas e clima subtropical."),
        (["qual a vegetação do Pampa?", "clima do Pampa"], "Predomínio de gramíneas, arbustos baixos e quatro estações definidas."),
        (["animais do Pampa", "fauna do Pampa"], "Quero-quero, veado-campeiro, graxaim, joão-de-barro e tuco-tuco."),
        (["onde fica o Pampa no Brasil?"], "Fica restrito à metade sul do Rio Grande do Sul."),

        # Pantanal
        (["o que é o Pantanal?", "características do Pantanal"], "Maior planície alagável do mundo, com cheias e secas sazonais."),
        (["qual o clima do Pantanal?", "vegetação do Pantanal"], "Clima tropical com cheia no verão; mistura campos, cerrado e matas."),
        (["animais do Pantanal", "fauna do Pantanal"], "Tuiuiú, jacaré-do-pantanal, arara-azul, capivara e onça-pintada."),
        (["onde fica o Pantanal?"], "Fica no oeste de Mato Grosso e Mato Grosso do Sul, na bacia do Paraguai."),
    ]
    sections.append(("BIOMAS BRASILEIROS", biomas_qa))

    # 4. Distâncias entre Capitais Brasileiras
    dist_qa: list[tuple[list[str], str]] = []
    pair_distances: list[tuple[int, str, str]] = []

    for i in range(len(ESTADOS)):
        _, _, _, cap1, reg1, lat1, lon1 = ESTADOS[i]
        for j in range(i + 1, len(ESTADOS)):
            _, _, _, cap2, reg2, lat2, lon2 = ESTADOS[j]
            km = haversine_km(lat1, lon1, lat2, lon2)
            pair_distances.append((km, cap1, cap2))

            include = all_pairs or (cap1 in HUB_CAPITALS) or (cap2 in HUB_CAPITALS) or (reg1 == reg2)
            if not include:
                continue

            road = RODOVIA_KM.get((cap1, cap2)) or RODOVIA_KM.get((cap2, cap1))
            if road:
                full_ans = f"Entre {cap1} e {cap2}: ~{km} km em linha reta ({road} km de estrada)."
                short_ans = f"Cerca de {km} km em linha reta ({road} km de estrada)."
            else:
                full_ans = f"Entre {cap1} e {cap2} há cerca de {km} km em linha reta."
                short_ans = f"Cerca de {km} km em linha reta."

            qs = [
                f"qual a distância entre {cap1} e {cap2}?",
                f"distância de {cap2} a {cap1}",
            ]
            max_q_len = max(len(normalize_q(q)) for q in qs)
            ans = full_ans if (max_q_len + 1 + len(to_ascii(full_ans)) <= MAX_TOTAL) else short_ans
            dist_qa.append((qs, ans))

    pair_distances.sort()
    min_km, c1_min, c2_min = pair_distances[0]
    max_km, c1_max, c2_max = pair_distances[-1]
    dist_qa.insert(
        0,
        (
            ["quais são as capitais mais distantes do Brasil?"],
            f"{c1_max} e {c2_max}, separadas por cerca de {max_km} km em linha reta.",
        ),
    )
    dist_qa.insert(
        0,
        (
            ["quais são as capitais mais próximas do Brasil?"],
            f"{c1_min} e {c2_min}, a cerca de {min_km} km em linha reta (120 km por rodovia).",
        ),
    )
    sections.append(("DISTANCIAS ENTRE AS CAPITAIS DO BRASIL", dist_qa))

    # 5. Presidentes na História do Brasil (1889–presente)
    pres_qa: list[tuple[list[str], str]] = [
        (["quem foi o primeiro presidente do Brasil?", "primeiro presidente do Brasil"], "Deodoro da Fonseca foi o 1o presidente do Brasil (1889-1891)."),
        (["quem proclamou a República no Brasil?"], "Marechal Deodoro da Fonseca proclamou a República em 15/11/1889."),
        (["quem é o atual presidente do Brasil?", "quem e o presidente atual do Brasil"], "Luiz Inácio Lula da Silva é o atual (39o) presidente, desde 2023."),
        (["quantos presidentes o Brasil já teve?"], "O Brasil teve 39 mandatos presidenciais com 35 pessoas diferentes."),
        (["quem foi o primeiro presidente civil do Brasil?"], "Prudente de Morais, eleito em 1894, foi o 1o presidente civil."),
        (["quem foi a primeira mulher presidente do Brasil?"], "Dilma Rousseff foi a 1a mulher presidente do Brasil (2011-2016)."),
        (["quem construiu Brasília?", "qual presidente inaugurou Brasília?"], "Juscelino Kubitschek construiu e inaugurou Brasília em 1960."),
        (["quem criou a CLT?", "quem criou a Petrobras?"], "Getúlio Vargas criou a CLT (1943) e a Petrobras (1953)."),
        (["quem renunciou à presidência em 1961?"], "Jânio Quadros renunciou em agosto de 1961 após 7 meses no cargo."),
        (["quem foi o primeiro presidente do regime militar?"], "Humberto Castelo Branco foi o 1o presidente do regime militar (1964)."),
        (["quem foi o último presidente do regime militar?"], "João Figueiredo foi o último presidente do regime militar (1979-1985)."),
        (["quem assinou a Lei da Anistia em 1979?"], "João Figueiredo sancionou a Lei da Anistia em agosto de 1979."),
        (["quem foi o presidente da Constituição de 1988?"], "José Sarney era o presidente quando a Constituição de 1988 foi promulgada."),
        (["quem criou o Plano Real?"], "O Plano Real foi lançado no governo Itamar Franco, por FHC, em 1994."),
        (["quais presidentes sofreram impeachment no Brasil?"], "Fernando Collor em 1992 e Dilma Rousseff em 2016 sofreram impeachment."),
        (["o que foi a política do café com leite?"], "Alternância de poder entre elites de São Paulo e Minas Gerais (1898-1930)."),

        (["quem foi Deodoro da Fonseca?", "Deodoro da Fonseca foi qual presidente?"], "1o presidente do Brasil (1889-1891); proclamou a República."),
        (["quem foi Floriano Peixoto?", "Floriano Peixoto foi qual presidente?"], "2o presidente do Brasil (1891-1894), chamado Marechal de Ferro."),
        (["quem foi Prudente de Morais?", "Prudente de Morais foi qual presidente?"], "3o presidente (1894-1898) e o 1o civil eleito por voto direto."),
        (["quem foi Campos Sales?", "Campos Sales foi qual presidente?"], "4o presidente (1898-1902); criou a Política dos Governadores."),
        (["quem foi Rodrigues Alves?", "Rodrigues Alves foi qual presidente?"], "5o presidente (1902-1906); modernizou o Rio e enfrentou a Revolta da Vacina."),
        (["quem foi Afonso Pena?", "Afonso Pena foi qual presidente?"], "6o presidente (1906-1909); morreu no cargo antes do fim do mandato."),
        (["quem foi Nilo Peçanha?", "Nilo Peçanha foi qual presidente?"], "7o presidente (1909-1910); assumiu após Afonso Pena e criou escolas técnicas."),
        (["quem foi Hermes da Fonseca?", "Hermes da Fonseca foi qual presidente?"], "8o presidente (1910-1914); militar que enfrentou a Guerra do Contestado."),
        (["quem foi Venceslau Brás?", "Venceslau Brás foi qual presidente?"], "9o presidente (1914-1918); governou na 1a Guerra e criou o Código Civil."),
        (["quem foi Delfim Moreira?", "Delfim Moreira foi qual presidente?"], "10o presidente (1918-1919); assumiu após a morte de Rodrigues Alves."),
        (["quem foi Epitácio Pessoa?", "Epitácio Pessoa foi qual presidente?"], "11o presidente (1919-1922); governou no centenário da Independência."),
        (["quem foi Artur Bernardes?", "Artur Bernardes foi qual presidente?"], "12o presidente (1922-1926); enfrentou revoltas tenentistas e a Coluna Prestes."),
        (["quem foi Washington Luís?", "Washington Luís foi qual presidente?"], "13o presidente (1926-1930); último da República Velha, deposto em 1930."),
        (["quem foi Júlio Prestes?", "Júlio Prestes tomou posse?"], "Eleito presidente em 1930, mas não assumiu por causa da Revolução de 1930."),
        (["o que foi a Junta Militar de 1930?"], "Junta provisória (Tasso Fragoso, Noronha e Barreto) que passou o poder a Vargas."),
        (["quem foi Getúlio Vargas?", "Getúlio Vargas foi qual presidente?"], "14o (1930-1945) e 17o presidente (1951-1954); liderou a Era Vargas e o Estado Novo."),
        (["quem foi José Linhares?", "José Linhares foi qual presidente?"], "15o presidente (1945-1946); presidente do STF que assumiu após queda de Vargas."),
        (["quem foi Eurico Gaspar Dutra?", "Dutra foi qual presidente?"], "16o presidente (1946-1951); promulgou a Constituição democrática de 1946."),
        (["quem foi Café Filho?", "Café Filho foi qual presidente?"], "18o presidente (1954-1955); vice que assumiu após o suicídio de Vargas."),
        (["quem foi Carlos Luz?", "Carlos Luz foi qual presidente?"], "19o presidente (1955); governou por apenas 3 dias em novembro de 1955."),
        (["quem foi Nereu Ramos?", "Nereu Ramos foi qual presidente?"], "20o presidente (1955-1956); garantiu a posse de Juscelino Kubitschek."),
        (["quem foi Juscelino Kubitschek?", "JK foi qual presidente?"], "21o presidente (1956-1961); do Plano de Metas e construtor de Brasília."),
        (["quem foi Jânio Quadros?", "Jânio Quadros foi qual presidente?"], "22o presidente (1961); eleito com o símbolo da vassoura, renunciou em agosto."),
        (["quem foi Ranieri Mazzilli?", "Ranieri Mazzilli foi qual presidente?"], "23o (1961) e 25o presidente (1964), assumindo interinamente pela Câmara."),
        (["quem foi João Goulart?", "quem foi Jango?", "João Goulart foi qual presidente?"], "24o presidente (1961-1964); defendeu as Reformas de Base e foi deposto em 1964."),
        (["quem foi Castelo Branco?", "Humberto Castelo Branco foi qual presidente?"], "26o presidente (1964-1967) e 1o do Regime Militar."),
        (["quem foi Costa e Silva?", "Artur da Costa e Silva foi qual presidente?"], "27o presidente (1967-1969); editou o AI-5 em dezembro de 1968."),
        (["o que foi a Junta Militar de 1969?"], "Junta de três ministros militares que governou entre agosto e outubro de 1969."),
        (["quem foi Médici?", "Emílio Garrastazu Médici foi qual presidente?"], "28o presidente (1969-1974); período do Milagre Econômico e repressão."),
        (["quem foi Ernesto Geisel?", "Ernesto Geisel foi qual presidente?"], "29o presidente (1974-1979); iniciou a abertura política gradual."),
        (["quem foi João Figueiredo?", "João Figueiredo foi qual presidente?"], "30o presidente (1979-1985); último do Regime Militar, assinou a Anistia."),
        (["quem foi Tancredo Neves?", "Tancredo Neves tomou posse?"], "Eleito em 1985 para encerrar o regime militar, mas morreu antes da posse."),
        (["quem foi José Sarney?", "José Sarney foi qual presidente?"], "31o presidente (1985-1990); 1o civil da Nova República, governou na Carta de 1988."),
        (["quem foi Fernando Collor?", "Fernando Collor foi qual presidente?"], "32o presidente (1990-1992); confiscou a poupança e sofreu impeachment em 1992."),
        (["quem foi Itamar Franco?", "Itamar Franco foi qual presidente?"], "33o presidente (1992-1995); assumiu após Collor e lançou o Plano Real."),
        (["quem foi Fernando Henrique Cardoso?", "quem foi FHC?", "FHC foi qual presidente?"], "34o presidente (1995-2003); sociólogo que consolidou o Plano Real."),
        (["quem é Lula?", "Luiz Inácio Lula da Silva foi qual presidente?"], "35o (2003-2011) e 39o presidente (desde 2023); único eleito 3 vezes."),
        (["quem é Dilma Rousseff?", "Dilma Rousseff foi qual presidente?"], "36a presidente (2011-2016); 1a mulher presidente, sofreu impeachment."),
        (["quem é Michel Temer?", "Michel Temer foi qual presidente?"], "37o presidente (2016-2019); assumiu após o impeachment de Dilma."),
        (["quem é Jair Bolsonaro?", "Jair Bolsonaro foi qual presidente?"], "38o presidente do Brasil (2019-2023), eleito em 2018."),
    ]
    sections.append(("PRESIDENTES NA HISTORIA DO BRASIL (1889-PRESENTE)", pres_qa))

    # Normalize all questions to lowercase ASCII form (preserving readability in answers) and deduplicate
    seen_qa: set[tuple[str, str]] = set()
    cleaned_sections: list[tuple[str, list[tuple[list[str], str]]]] = []
    for sec_name, entries in sections:
        sec_out: list[tuple[list[str], str]] = []
        for qs, ans in entries:
            ans_ascii = to_ascii(ans)
            uniq_qs: list[str] = []
            for q in qs:
                qn = normalize_q(q)
                if qn and (qn, ans_ascii) not in seen_qa:
                    seen_qa.add((qn, ans_ascii))
                    uniq_qs.append(qn)
            if uniq_qs:
                validate_entry(uniq_qs, ans)
                sec_out.append((uniq_qs, ans))
        cleaned_sections.append((sec_name, sec_out))

    return cleaned_sections


def write_dataset(txt_path: pathlib.Path, jsonl_path: pathlib.Path, all_pairs: bool) -> tuple[int, int]:
    sections = build_sections(all_pairs=all_pairs)
    lines = [
        "# Dataset extenso do TINY 2.1 / AMEBA (Brasil: Capitais, Biomas, Distancias e Presidentes)",
        "# Formato: pergunta1 ; pergunta2 | resposta",
        "# Gerado por tools/generate_brazil_dataset.py (100% dentro de MAXQ=72 e CTX=128)",
        "",
    ]
    total_records = 0
    for sec_name, entries in sections:
        lines.append(f"# === {sec_name} ===")
        for qs, ans in entries:
            lines.append(f"{' ; '.join(qs)} | {ans}")
            total_records += 1
        lines.append("")

    txt_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.write_text("\n".join(lines), encoding="utf-8")

    n_pairs, n_jsonl = convert_txt_to_jsonl(txt_path, jsonl_path, group=True, keep_accents=False)
    return n_pairs, n_jsonl


def main() -> int:
    root = pathlib.Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description="Gera dataset extenso do Brasil em data/custom_qa.txt e data/custom_qa.jsonl")
    p.add_argument("--txt", type=pathlib.Path, default=root / "data" / "custom_qa.txt", help="Caminho do .txt")
    p.add_argument("--jsonl", type=pathlib.Path, default=root / "data" / "custom_qa.jsonl", help="Caminho do .jsonl")
    p.add_argument("--all-pairs", action="store_true", help="Incluir todas as 351 combinacoes entre as 27 capitais")
    args = p.parse_args()

    txt_path = args.txt if args.txt.is_absolute() else (root / args.txt)
    jsonl_path = args.jsonl if args.jsonl.is_absolute() else (root / args.jsonl)
    n_pairs, n_jsonl = write_dataset(txt_path, jsonl_path, all_pairs=args.all_pairs)
    print(f"[OK] Gerado {txt_path} e {jsonl_path}: {n_pairs} pares Q&A em {n_jsonl} registros JSONL")
    return 0


if __name__ == "__main__":
    sys.exit(main())
