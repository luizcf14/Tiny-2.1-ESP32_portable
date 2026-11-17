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


def _forest_survival_qa() -> list[tuple[list[str], str]]:
    """Q&As on basic forest survival, orientation, fire, and rescue signaling."""
    return [
        (
            ["qual a regra dos 3?", "regra dos 3 sobrevivencia", "quanto tempo sem agua e comida?"],
            "Regra dos 3: 3 min sem ar, 3h sem abrigo no frio, 3 dias sem água, 3 semanas sem comida.",
        ),
        (
            ["o que fazer primeiro na mata?", "prioridades de sobrevivencia", "sobrevivencia na floresta"],
            "Ordem na mata: acalme-se, garanta segurança e abrigo, água, fogo, sinalização e comida.",
        ),
        (
            ["estou perdido na mata", "o que fazer se me perder?", "metodo pare sobrevivencia"],
            "Use PARE: Pare de andar, Acalme-se, Reflita sobre a rota e Explore marcando o caminho.",
        ),
        (
            ["devo andar ou esperar resgate?", "se perder na mata fico parado?"],
            "Se sabem seu roteiro e há busca, fique visível; só ande se não houver chance de resgate.",
        ),
        (
            ["como nao andar em circulos?", "andar em linha reta na mata"],
            "Alinhe dois pontos fixos à frente (árvores ou morros) e renove o alinhamento ao caminhar.",
        ),
        (
            ["como marcar caminho na mata?", "como sinalizar trilha na selva?"],
            "Marque árvores com tiras visíveis, pedras empilhadas ou galhos na altura dos olhos.",
        ),
        (
            ["como se orientar pelo sol?", "achar norte pelo sol", "onde o sol nasce?"],
            "O Sol nasce perto do Leste e se põe no Oeste; de frente ao nascente, o Norte está à esquerda.",
        ),
        (
            ["orientacao com graveto e sombra", "relogio de sol para direcao"],
            "Finca-se um graveto: marque a ponta da sombra, espere 20 min e marque; a linha vai O a L.",
        ),
        (
            ["achar norte com relogio analogico", "orientacao por relogio no sul"],
            "No Brasil, aponte o 12 ao Sol; a metade do ângulo entre 12 e o ponteiro das horas é Norte.",
        ),
        (
            ["orientacao pelo cruzeiro do sul", "como achar sul pelas estrelas?"],
            "Prolongue 4,5x o braço maior do Cruzeiro do Sul e desça reto ao horizonte para achar o Sul.",
        ),
        (
            ["seguir rio tira da mata?", "seguir corrego quando perdido?"],
            "Rios costumam levar a vilas ou estradas, mas exigem cuidado com cheias e cachoeiras.",
        ),
        (
            ["como fazer fogo na floresta?", "como acender fogueira na mata?"],
            "Reúna isca seca, gravetos finos e lenha média antes de acender; proteja a chama do vento.",
        ),
        (
            ["fogo na chuva ou mata umida", "onde achar lenha seca na mata?"],
            "Pegue galhos mortos ainda presos em árvores e lasque a casca úmida até o miolo seco.",
        ),
        (
            ["o que usar de isca para fogo?", "isca natural para fogueira"],
            "Use raspas finas de madeira (feather stick), capim seco, fibras de cipó ou resina seca.",
        ),
        (
            ["fazer fogo sem fosforo", "fogo sem isqueiro na mata"],
            "Use pederneira, lupa no sol forte, pilha com palha de aço fina ou arco de fricção seco.",
        ),
        (
            ["cuidados ao fazer fogueira", "posso usar pedra de rio no fogo?"],
            "Limpe 1,5 m ao redor até a terra, apague brasas e NUNCA use pedras de rio (explodem!).",
        ),
        (
            ["como sinalizar para resgate?", "pedir socorro na floresta"],
            "Faça X ou SOS gigante em clareira, use espelho ao sol, fumaça de dia e fogo à noite.",
        ),
        (
            ["como fazer fumaca de sinalizacao?", "fumaca branca para resgate"],
            "Com a fogueira acesa, jogue folhas verdes e ramos úmidos na brasa para gerar fumaça densa.",
        ),
        (
            ["codigo sos emergencia", "como sinalizar sos com apito?"],
            "SOS em Morse é 3 curtos, 3 longos e 3 curtos; 3 apitos ou flashes seguidos pedem socorro.",
        ),
        (
            ["por que usar apito na mata?", "apito na sobrevivencia"],
            "O som do apito alcança muito mais longe que a voz e economiza energia sem cansar você.",
        ),
        (
            ["como tratar hipotermia na mata?", "sintomas e acao na hipotermia"],
            "Troque roupa molhada, isole do chão, corte o vento, aqueça o tronco e dê bebida morna doce.",
        ),
        (
            ["o que fazer na insolacao?", "exaustao por calor na mata"],
            "Leve à sombra, afrouxe roupas, resfrie pescoço e axilas com pano úmido e hidrate devagar.",
        ),
        (
            ["como atravessar rio na mata?", "cruzar rio com correnteza"],
            "Solte a cinta da mochila, use cajado como tripé e cruze em diagonal a favor da corrente.",
        ),
        (
            ["o que fazer em areia movedica?", "como sair de atoleiro ou lama?"],
            "Não debata os pés! Jogue o peso para trás, deite para flutuar na lama e role devagar.",
        ),
        (
            ["seguranca com facao na mata", "cuidados ao cortar com faca"],
            "Corte sempre para longe do seu corpo, em base firme, mantendo outras pessoas afastadas.",
        ),
        (
            ["quais nos basicos aprender?", "para que serve no lais de guia?"],
            "Domine lais de guia (alça fixa segura), volta do fiel (poste/árvore) e nó direito.",
        ),
        (
            ["economizar energia na selva", "melhor hora para andar na mata"],
            "Caminhe nas horas mais frescas, evite suar demais, faça pausas curtas e não corra à toa.",
        ),
        (
            ["qual roupa usar na floresta?", "vestimenta segura para mata"],
            "Use calça comprida, manga longa, bota firme, meias secas e chapéu; evite algodão no frio.",
        ),
    ]


def _water_survival_qa() -> list[tuple[list[str], str]]:
    """Q&As on finding, choosing, filtering, and purifying water in nature."""
    return [
        (
            ["qual agua mais segura na mata?", "melhor fonte de agua natural"],
            "Água da chuva captada direto em lona ou folhas limpas é a mais segura na natureza.",
        ),
        (
            ["como coletar orvalho na mata?", "pegar agua do capim de manha"],
            "Ao amanhecer, arraste pano limpo no capim sem veneno e esprema o orvalho num recipiente.",
        ),
        (
            ["agua de rio pode beber direto?", "agua parada ou corrente qual melhor?"],
            "Prefira nascente ou água corrente clara à parada, mas ainda assim ferva ou trate antes.",
        ),
        (
            ["como tirar agua de cipo?", "agua de cipo na floresta"],
            "Corte o cipó alto e depois embaixo só se o líquido for claro, fino e sem cheiro/amargor.",
        ),
        (
            ["onde mais achar agua em planta?", "tem agua no bambu verde?"],
            "Gomos fechados de bambu verde guardam água limpa; coco verde hidrata bem com minerais.",
        ),
        (
            ["posso beber agua de bromelia?", "agua parada em folha e segura?"],
            "Água de bromélias e poças traz larvas e parasitas; só beba após coar e ferver muito bem.",
        ),
        (
            ["qual agua nunca devo beber?", "sinais de agua contaminada"],
            "Evite água parada fétida, oleosa, com espuma, algas densas ou sem vida aquática ao redor.",
        ),
        (
            ["como purificar agua?", "como purificar agua fervendo?", "quanto tempo ferver agua?"],
            "Coe sujeiras num pano e ferva a água em ebulição forte de 1 a 3 min para matar micróbios.",
        ),
        (
            ["como ferver agua sem panela?", "ferver agua em garrafa pet"],
            "Use garrafa PET cheia até a boca (sem ar) sobre brasas suaves ou pedras secas aquecidas.",
        ),
        (
            ["purificar agua com agua sanitaria", "gotas de cloro por litro"],
            "Com água sanitária pura 2-2,5% sem perfume, pingue 2 gotas/L (4 se turva) e espere 30 min.",
        ),
        (
            ["como fazer filtro de agua na mata?", "filtro caseiro com areia"],
            "Faça camadas de pano, carvão de fogueira moído, areia fina e pedrinhas; depois ferva!",
        ),
        (
            ["filtro de areia deixa agua potavel?", "filtro caseiro mata virus?"],
            "Não! Areia e carvão tiram barro e odor, mas não matam vírus; sempre ferva ou clore.",
        ),
        (
            ["purificar agua com sol sodis", "o que e o metodo sodis?"],
            "Encha garrafa PET transparente com água já clara e deixe deitada ao sol forte por 6 horas.",
        ),
        (
            ["como fazer destilador solar?", "agua por transpiracao de folha"],
            "Amarre saco plástico claro em galho ao sol ou cubra buraco úmido com lona e pedra no meio.",
        ),
        (
            ["posso beber agua do mar?", "beber agua salgada faz mal?"],
            "Nunca! O sal da água do mar acelera a desidratação renal; só beba após destilar o vapor.",
        ),
        (
            ["como destilar agua do mar?", "dessalinizar agua na emergencia"],
            "Ferva a água salgada e colete só o vapor condensado escorrendo por tampa inclinada ou tubo.",
        ),
        (
            ["posso beber urina na emergencia?", "beber sangue ou alcool ajuda?"],
            "Jamais beba urina, sangue ou álcool: sobrecarregam os rins e pioram muito a desidratação.",
        ),
        (
            ["agua de qualquer cacto e segura?", "posso beber seiva de cacto?"],
            "Não! Vários cactos têm alcaloides que dão vômito e diarreia, agravando a sede e fraqueza.",
        ),
        (
            ["devo racionar agua ou beber?", "como economizar agua no corpo?"],
            "Racione o suor, não a água! Fique à sombra, respire pelo nariz e beba em pequenos goles.",
        ),
        (
            ["posso comer sem ter agua?", "comer carne seca sem agua faz mal?"],
            "Sem água disponível, evite proteínas e comidas secas, pois digerir consome água do corpo.",
        ),
        (
            ["sinais de desidratacao grave", "como saber se falta agua?"],
            "Urina muito escura ou ausente, boca seca, tontura, dor de cabeça, fraqueza e confusão.",
        ),
        (
            ["como fazer soro caseiro?", "receita soro de reidratacao"],
            "Em 1 litro de água tratada, misture 1 colher de chá rasa de sal e 2 de sopa de açúcar.",
        ),
        (
            ["seiva leitosa de planta pode beber?", "cipo com leite branco faz mal?"],
            "Nunca beba seiva branca, leitosa, amarela ou pegajosa: indica alto risco de veneno!",
        ),
    ]


def _food_survival_qa() -> list[tuple[list[str], str]]:
    """Q&As on safely selecting and preparing food in nature."""
    return [
        (
            ["o que comer na mata?", "como escolher comida na mata?", "regra comida na selva"],
            "Só coma o que identifica com 100% de certeza. Você dura semanas sem comer; na dúvida, evite.",
        ),
        (
            ["posso comer cogumelo selvagem?", "cogumelo na mata e seguro?"],
            "Nunca coma cogumelos desconhecidos: espécies letais parecem comuns e cozinhar não salva.",
        ),
        (
            ["identificar planta venenosa", "quais plantas nunca comer?"],
            "Evite plantas com seiva leitosa, espinhos finos, folhas de 3 em 3 ou vagens desconhecidas.",
        ),
        (
            ["frutas silvestres a evitar", "fruta branca ou amarga na mata"],
            "Evite frutos brancos ou amarelos suspeitos, sabor amargo/ardente e cheiro de amêndoa.",
        ),
        (
            ["se passaro comeu eu posso comer?", "macaco come fruto venenoso?"],
            "Mito perigoso! Aves e macacos digerem toxinas que podem matar um humano rapidamente.",
        ),
        (
            ["mandioca silvestre pode comer?", "perigo da mandioca brava"],
            "Mandioca-brava tem ácido cianídrico letal! Nunca coma raízes desconhecidas cruas ou malfeitas.",
        ),
        (
            ["teste de comestibilidade planta", "como testar planta desconhecida?"],
            "Em último caso, teste 1 parte em jejum: pele, lábio, língua, engula pouco e espere 8h.",
        ),
        (
            ["teste serve para cogumelos?", "quando nao testar planta?"],
            "Nunca teste cogumelos, algas marinhas ou plantas de seiva leitosa ou cheiro forte.",
        ),
        (
            ["se arder na boca no teste?", "sinais ruins no teste de planta"],
            "Se sentir ardor, coceira, dormência, amargor ou enjoo, cuspa na hora, lave e descarte.",
        ),
        (
            ["frutas seguras no brasil", "o que reconhecer facil na mata?"],
            "Coco, goiaba, caju (castanha assada), maracujá, açaí, buriti, pequi e castanha-do-pará.",
        ),
        (
            ["castanha de caju crua faz mal?", "como comer castanha de caju?"],
            "Nunca coma castanha de caju crua (a casca queima)! Asse bem no fogo até soltar o óleo.",
        ),
        (
            ["quais insetos posso comer?", "insetos seguros na sobrevivencia"],
            "Coma grilos, gafanhotos (sem patas, asas e cabeça), tanajuras e cupins sempre bem assados.",
        ),
        (
            ["quais insetos nunca comer?", "posso comer lagarta colorida?"],
            "Evite insetos peludos, com ferrão, cores vivas (alerta de veneno), mau cheiro ou de fezes.",
        ),
        (
            ["larva de coco pode comer?", "bicho de palmeira e comestivel?"],
            "Sim, o gongo/larva de palmeira é rico em energia e proteína, mas sempre asse ou cozinhe bem.",
        ),
        (
            ["todo peixe de rio e comestivel?", "comer peixe e caranguejo na mata"],
            "Peixe de rio saudável pode ser comido bem cozido; cuidado com ferrão de arraia e bagre.",
        ),
        (
            ["posso comer carne ou peixe cru?", "por que cozinhar comida na mata?"],
            "Nunca coma cru! Cozinhar ou assar bem mata vermes, parasitas, bactérias e várias toxinas.",
        ),
        (
            ["posso comer sapo na mata?", "ovos achados na mata sao seguros?"],
            "Evite sapos pela pele tóxica (rãs cozidas sem pele servem); ovos só frescos e bem cozidos.",
        ),
        (
            ["posso comer peixe ja morto?", "sinais de carne estragada"],
            "Não coma bichos achados mortos, nem peixe com guelra cinza, olho opaco fundo ou cheiro azedo.",
        ),
        (
            ["como pescar sem vara na mata?", "armadilha de peixe improvisada"],
            "Monte covo afunilado de taquara na corrente ou anzol de espinho/osso com isca de minhoca.",
        ),
        (
            ["conservar carne sem geladeira", "o que e moquem na mata?"],
            "Corte a carne em tiras magras e defume lentamente numa grelha de madeira (moquém) com fumaça.",
        ),
        (
            ["taboa e planta comestivel?", "raizes de brejo para comer"],
            "Brotos novos, raízes de taboa bem cozidas e palmito cozido dão amido em áreas de brejo.",
        ),
        (
            ["vale a pena cacar perdido?", "gastar energia cacando na mata"],
            "Caçar gasta muita água e energia e gera ferimentos; priorize abrigo, água, pesca e frutos.",
        ),
    ]


def _shelter_survival_qa() -> list[tuple[list[str], str]]:
    """Q&As on choosing safe camp locations and building natural or tarp shelters."""
    return [
        (
            ["quando montar abrigo na mata?", "melhor hora de preparar abrigo"],
            "Comece a montar o abrigo 2 a 3 horas antes de escurecer para trabalhar com segurança e luz.",
        ),
        (
            ["como escolher abrigo?", "como escolher lugar do abrigo?", "onde montar abrigo na mata?"],
            "Busque terreno alto, seco, plano, sem galhos secos acima, longe do vento e a ~60 m da água.",
        ),
        (
            ["posso dormir em leito seco de rio?", "onde nunca montar abrigo?"],
            "Nunca durma em leito seco ou beira baixa de rio: chuvas distantes geram enxurradas à noite.",
        ),
        (
            ["por que olhar acima no abrigo?", "perigo de galhos secos acima"],
            "Sempre cheque acima! Galhos mortos suspensos e coqueiros podem cair com vento ou chuva.",
        ),
        (
            ["locais perigosos para acampar", "posso dormir no topo do morro?"],
            "Evite topos expostos a raios, trilhas de animais, formigueiros, cupinzeiros e colmeias.",
        ),
        (
            ["posso dormir em caverna?", "fazer fogueira dentro de caverna"],
            "Cheque cobras, morcegos e riscos de queda; nunca acenda fogo no fundo por gás tóxico (CO).",
        ),
        (
            ["parte mais importante do abrigo", "como isolar corpo do chao frio?"],
            "O chão rouba calor rápido! Monte cama de isolamento com 20 a 30 cm de folhas secas ou capim.",
        ),
        (
            ["como dormir em chao alagado?", "para que serve jirau na mata?"],
            "Em solo encharcado ou com peçonhentos, construa cama elevada (jirau) ou rede a 60 cm do chão.",
        ),
        (
            ["tamanho ideal do abrigo natural", "abrigo pequeno esquenta mais?"],
            "Faça o abrigo natural justo ao seu corpo: menor volume interno retém muito melhor o calor.",
        ),
        (
            ["como fazer abrigo meia agua?", "o que e abrigo lean to?"],
            "Prenda barra horizontal em duas árvores, apoie varas a 45 graus contra o vento e cubra.",
        ),
        (
            ["como montar abrigo duas aguas?", "como fazer abrigo a frame?"],
            "Apoie uma viga inclinada da cintura ao chão, feche os dois lados com galhos e cubra com folhas.",
        ),
        (
            ["como cobrir teto contra chuva?", "ordem de folhas no telhado"],
            "Empilhe folhas grandes e palha de baixo para cima sobrepostas como telhas, em camada espessa.",
        ),
        (
            ["posicao do abrigo contra vento", "evitar enxurrada no abrigo"],
            "Vire as costas do abrigo para o vento e cave uma valeta rasa em volta para escoar a chuva.",
        ),
        (
            ["como armar abrigo com lona?", "improvisar barraca com poncho"],
            "Estique corda entre duas árvores na altura da cintura, passe a lona e fixe as pontas no chão.",
        ),
        (
            ["distancia da fogueira ao abrigo", "aquecer abrigo com fogo"],
            "Deixe o fogo a 1,5 m da entrada com parede de madeira atrás para refletir calor.",
        ),
        (
            ["como usar manta termica?", "saco de lixo serve para frio?"],
            "Manta térmica retém até 90% do calor; saco plástico grande corta vento deixando rosto livre.",
        ),
        (
            ["por que dormir longe da agua?", "distancia segura do rio na mata"],
            "Ficar 50 a 100 m do rio reduz neblina gelada, mosquitos, animais bebendo à noite e cheias.",
        ),
    ]


def _protection_first_aid_qa() -> list[tuple[list[str], str]]:
    """Q&As on wildlife protection, venomous bites, and field first aid."""
    return [
        (
            ["como evitar picada de cobra?", "prevencao de peconhentos na mata"],
            "Use botas e perneiras, toque a trilha com bastão, não meta mão em ocos e sacuda roupas antes.",
        ),
        (
            ["picada de cobra", "picada de cobra o que fazer?", "fui picado por cobra como agir?"],
            "Calma, deite a vítima, lave com água e sabão, tire anéis, imobilize e leve ao hospital.",
        ),
        (
            ["o que evitar em picada de cobra?", "posso chupar picada de cobra?"],
            "NUNCA corte, chupe, faça torniquete, ponha ervas ou álcool! Só o soro no hospital salva.",
        ),
        (
            ["cobras venenosas do brasil", "principais cobras peconhentas"],
            "Os 4 grupos no Brasil são jararaca, cascavel, surucucu e coral; nunca tente capturar a cobra.",
        ),
        (
            ["picada de escorpiao o que fazer?", "acidente com escorpiao como agir?"],
            "Lave com água/sabão, use compressa morna na dor e vá ao hospital (urgente em criança!).",
        ),
        (
            ["picada de aranha o que fazer?", "picada de aranha armadeira"],
            "Lave o local, use compressa morna para dor sem apertar ou cortar e busque médico rápido.",
        ),
        (
            ["queimadura de taturana o que fazer?", "contato com lagarta lonomia"],
            "Lave com água fria sem esfregar; se houver dor ou sangramento, vá logo ao hospital.",
        ),
        (
            ["o que fazer ao ver uma onca?", "encontrei uma onca devo correr?"],
            "NUNCA corra! Fique ereto, erga os braços para parecer maior, fale firme e recue olhando.",
        ),
        (
            ["como fugir de ataque de abelhas?", "enxame de abelhas o que fazer?"],
            "Corra protegendo rosto para abrigo fechado; não pule na água e raspe o ferrão.",
        ),
        (
            ["encontro com queixada ou javali", "ataque de porco do mato"],
            "Se cruzar com bando de queixadas ou javali agressivo, suba em árvore ou rocha acima de 2 m.",
        ),
        (
            ["evitar jacare ou sucuri no rio", "seguranca ao entrar em rio na mata"],
            "Não nade ao amanhecer ou anoitecer em água turva e não limpe peixes no mesmo local.",
        ),
        (
            ["como tirar carrapato da pele?", "cuidado com picada de carrapato"],
            "Puxe o carrapato bem rente à pele com pinça sem torcer ou queimar; febre após exige médico.",
        ),
        (
            ["queimadura de urtiga o que fazer?", "irritacao por planta urticante"],
            "Lave com água e sabão sem coçar, remova pelos com fita adesiva e aplique compressa fria.",
        ),
        (
            ["como estancar sangramento forte?", "o que fazer em hemorragia?"],
            "Aperte firme a ferida com pano limpo por 10 min; se molhar, ponha mais pano por cima.",
        ),
        (
            ["quando usar um torniquete?", "como fazer torniquete seguro?"],
            "Só em membro com sangue jorrando ou amputação que não estanca; aperte acima e anote a hora.",
        ),
        (
            ["como limpar ferida na natureza?", "evitar infeccao em corte na mata"],
            "Lave bem com água fervida ou limpa e sabão neutro, retire sujeira e cubra com gaze limpa.",
        ),
        (
            ["o que fazer em osso quebrado?", "como imobilizar fratura na mata?"],
            "Não alinhe o osso! Imobilize na mesma posição prendendo acima e abaixo da fratura.",
        ),
        (
            ["torci o tornozelo na trilha", "como cuidar de entorse longe?"],
            "Só tire a bota se tiver apoio firme; enfaixe sem cortar circulação, eleve o pé e resfrie.",
        ),
        (
            ["primeiros socorros queimadura", "o que passar em queimadura?"],
            "Resfrie em água corrente fria por 15 min e cubra com pano úmido; nunca use pasta ou manteiga.",
        ),
        (
            ["devo furar bolha de queimadura?", "furar bolha no pe faz mal?"],
            "Não estoure bolhas de queimadura nem de atrito: a pele intacta impede entrada de infecções.",
        ),
        (
            ["alguem engasgou o que fazer?", "como fazer manobra de heimlich?"],
            "Se não fala nem tosse, dê 5 tapas nas costas e faça compressões rápidas acima do umbigo.",
        ),
        (
            ["como fazer massagem cardiaca?", "pessoa sem respirar rcp basica"],
            "Ligue 192/193 e comprima forte o centro do peito (5 a 6 cm) sem parar a 100 a 120 por min.",
        ),
        (
            ["pessoa desmaiou respirando", "posicao lateral de seguranca"],
            "Se respira normalmente, vire-a de lado (posição lateral de segurança) para manter vias livres.",
        ),
        (
            ["como tratar diarreia na mata?", "diarreia grave na sobrevivencia"],
            "Tome água fervida com soro caseiro em goles constantes; evite gorduras e descanse.",
        ),
    ]


def _floods_landslides_qa() -> list[tuple[list[str], str]]:
    """Q&As on floods, flash floods, landslides, and post-flood health safety."""
    return [
        (
            ["o que fazer em enchente?", "enchente em casa o que fazer?", "acoes imediatas na enchente"],
            "Desligue disjuntor de luz, gás e água, pegue kit com documentos em plástico e vá a ponto alto.",
        ),
        (
            ["posso subir no sotao na enchente?", "onde ir se agua invadir a casa?"],
            "Suba a andares altos ou telhado acessível; nunca fique em sótão fechado sem saída.",
        ),
        (
            ["posso atravessar rua alagada?", "enxurrada rasa derruba adulto?"],
            "Nunca atravesse enxurrada! 15 cm de água corrente já derrubam um adulto e escondem bueiros.",
        ),
        (
            ["posso passar alagamento de carro?", "altura de agua arrasta um carro?"],
            "Nunca dirija em via alagada! 30 cm ja fazem o carro boiar e 50 cm de enxurrada arrastam.",
        ),
        (
            ["carro afundando o que fazer?", "carro travou na enchente"],
            "Saia para lugar alto; se a água subir no carro, solte o cinto e abra o vidro antes de travar.",
        ),
        (
            ["como quebrar vidro de carro?", "porta do carro travou na agua"],
            "Bata nos cantos do vidro lateral (nunca para-brisa) com a haste metálica do encosto.",
        ),
        (
            ["risco eletrico na enchente", "fio caido em rua alagada"],
            "Fique bem longe de postes e fios caídos na água e jamais toque aparelhos com o corpo molhado.",
        ),
        (
            ["o que e cabeca d agua em rio?", "sinais de cabeca d agua cachoeira"],
            "Onda súbita por chuva na serra: se o rio sujar, subir rápido ou trazer folhas, saia já!",
        ),
        (
            ["sinais de deslizamento de terra", "como saber se morro vai cair?"],
            "Rachaduras, portas emperrando, postes/árvores inclinados e água barrenta brotando do chão.",
        ),
        (
            ["risco de deslizamento o que fazer?", "barreira caindo o que faco?"],
            "Saia na hora para local plano e firme e ligue para Defesa Civil (199) ou Bombeiros (193).",
        ),
        (
            ["como evitar leptospirose?", "protecao na agua de enchente"],
            "Evite pisar em lama ou água suja; use botas e luvas impermeáveis ou sacos duplos fechados.",
        ),
        (
            ["sintomas de leptospirose", "peguei agua de enchente sintomas"],
            "Febre alta, dor de cabeça, dor forte nas panturrilhas e olhos vermelhos exigem médico.",
        ),
        (
            ["como limpar casa pos enchente?", "desinfetar lama de enchente"],
            "Retire a lama usando botas e luvas e lave superfícies com 200 ml de água sanitária em 20 L.",
        ),
        (
            ["comida que molhou na enchente", "latas da enchente podem comer?"],
            "Descarte comida aberta, papelão e remédio molhado; salve apenas latas intactas bem lavadas.",
        ),
        (
            ["quando religar luz apos enchente?", "voltar para casa pos alagamento"],
            "Só entre após checar estrutura e vazamento de gás; não religue energia com fiação úmida.",
        ),
        (
            ["agua da torneira apos enchente", "agua de casa depois da enchente"],
            "Ferva ou pingue cloro na água de beber até confirmar limpeza da caixa d'água e rede pública.",
        ),
        (
            ["bichos em casa apos enchente", "cobras apos alagamento"],
            "Cobras, escorpiões e aranhas se escondem em móveis e entulhos; inspecione tudo com um cabo.",
        ),
    ]


def _nuclear_fallout_qa() -> list[tuple[list[str], str]]:
    """Q&As on nuclear detonation response, shielding, decontamination, and fallout."""
    return [
        (
            ["explosao nuclear", "explosao nuclear o que fazer?", "regra ataque nuclear"],
            "Siga 'Entre, Fique dentro e Sintonize': proteja-se da onda e entre rápido em prédio denso.",
        ),
        (
            ["o que fazer ao ver clarao nuclear?", "posso olhar cogumelo nuclear?"],
            "NUNCA olhe o clarão pois cega! Jogue-se de bruços longe de janelas cobrindo a cabeça.",
        ),
        (
            ["posicao onda de choque nuclear", "como deitar em explosao nuclear?"],
            "Deite de bruços, pés apontados ao clarão, proteja a nuca, tape ouvidos e abra leve a boca.",
        ),
        (
            ["quanto tempo antes do fallout?", "quando cai poeira radioativa?"],
            "Você tem cerca de 10 a 15 minutos após o clarão para entrar em abrigo antes do fallout cair.",
        ),
        (
            ["melhor abrigo explosao nuclear", "onde se proteger da radiacao?"],
            "Subsolo, garagem subterrânea ou centro de prédio de concreto sem janelas e longe do telhado.",
        ),
        (
            ["onde nao ficar no fallout?", "carro protege de explosao nuclear?"],
            "Evite carros, casas de madeira ou vidro, último andar ou ficar colado às paredes externas.",
        ),
        (
            ["vedar abrigo contra fallout", "desligar ar no ataque nuclear?"],
            "Feche portas e janelas, desligue ar-condicionado e exaustores de ar externo e vede frestas.",
        ),
        (
            ["fallout", "o que e fallout radioativo?", "perigo da poeira nuclear"],
            "Fallout é poeira e cinza contaminada que cai após a explosão e emite radiação gama e beta.",
        ),
        (
            ["descontaminacao de radiacao", "como limpar fallout do corpo?"],
            "Tire a roupa externa em saco fechado (remove 90% da poeira) e lave pele e cabelo com sabão.",
        ),
        (
            ["por que nao usar condicionador?", "condicionador no fallout nuclear"],
            "NUNCA use condicionador após fallout! Ele gruda radiação nos fios; use só água e sabão.",
        ),
        (
            ["banho sem agua apos fallout", "descontaminar sem chuveiro"],
            "Lave suave sem ferir a pele; se não houver chuveiro, limpe com pano úmido e assoe o nariz.",
        ),
        (
            ["quanto tempo ficar no abrigo?", "quando sair apos explosao nuclear?"],
            "Fique abrigado 24 a 48 horas e so saia quando autoridades ou rádio confirmarem seguro.",
        ),
        (
            ["regra 7 por 10 radiacao", "quanto cai a radiacao do fallout?"],
            "A cada 7x de tempo a radiação cai 10x: após 7h vai a 10%, e após 49h cai para 1%.",
        ),
        (
            ["ferver agua tira radiacao?", "agua segura apos explosao nuclear"],
            "Ferver NÃO tira radiação! Beba só água em garrafas ou caixas lacradas limpas por fora.",
        ),
        (
            ["o que comer apos fallout?", "posso comer horta pos nuclear?"],
            "Coma só latas e pacotes fechados limpos por fora; evite horta, leite ou água expostos.",
        ),
        (
            ["para que serve iodeto de potassio?", "posso tomar iodo de farmacia?"],
            "Iodeto de potássio protege a tireoide só com ordem médica; JAMAIS beba iodo tópico!",
        ),
        (
            ["celular funciona apos explosao?", "informacao apos ataque nuclear"],
            "O EMP pode derrubar luz e internet; tenha um rádio AM/FM a pilhas ou manivela no kit.",
        ),
        (
            ["posso sair para ajudar no fallout?", "receber alguem contaminado"],
            "Não saia durante o fallout; quem entrar deve deixar toda roupa externa e sapatos fora.",
        ),
    ]


def _urban_catastrophe_kit_qa() -> list[tuple[list[str], str]]:
    """Q&As on emergency kits, fires, lightning, earthquakes, blackouts, and Brazil numbers."""
    return [
        (
            ["telefones de emergencia brasil", "numero bombeiros samu defesa civil"],
            "Anote: 193 Bombeiros, 192 SAMU, 190 Polícia Militar, 199 Defesa Civil e 191 PRF.",
        ),
        (
            ["kit de emergencia", "mochila de emergencia 72h", "o que colocar no kit emergencia?"],
            "Água, comida pronta, lanterna, rádio a pilhas, apito, remédios, curativos, faca e cópias.",
        ),
        (
            ["incendio em predio o que fazer?", "posso usar elevador no incendio?"],
            "Nunca use elevador! Desça pelas escadas rastejando sob a fumaça com pano úmido no nariz.",
        ),
        (
            ["abrir porta em incendio", "macaneta quente no fogo o que faco?"],
            "Teste a porta com dorso da mão; se quente, não abra, vede frestas e sinalize na janela.",
        ),
        (
            ["roupa pegou fogo o que fazer?", "apagar fogo no proprio corpo"],
            "Não corra! Pare, deite no chão e role protegendo o rosto com as mãos até abafar a chama.",
        ),
        (
            ["oleo pegou fogo na panela", "agua em panela de oleo em chamas"],
            "NUNCA jogue água! Desligue o fogo e abafe a panela com tampa metálica ou pano bem úmido.",
        ),
        (
            ["vazamento de gas o que fazer?", "cheiro de gas posso ligar luz?"],
            "Não acenda luz, fósforo ou celular! Feche o registro do gás, abra portas e janelas e saia.",
        ),
        (
            ["como fugir de incendio florestal?", "fogo na mata para onde correr?"],
            "Não fuja morro acima! Mova-se para os lados contra o vento ou para área já queimada.",
        ),
        (
            ["tempestade de raios o que fazer?", "onde se proteger de raios?"],
            "Entre em prédio ou carro fechado; nunca fique sob árvore isolada, postes, cercas ou água.",
        ),
        (
            ["sem abrigo na chuva de raios", "posicao de seguranca raios"],
            "Agache na ponta dos pés com os pés bem juntos e cabeça baixa; nunca deite esticado no chão.",
        ),
        (
            ["o que fazer em terremoto?", "predio tremendo como se proteger?"],
            "Abaixe-se, cubra cabeça e nuca sob mesa firme longe de vidros e segure até parar o tremor.",
        ),
        (
            ["preso em escombros o que fazer?", "pedir socorro em desabamento"],
            "Cubra boca/nariz contra poeira, não use isqueiro e bata 3 vezes em canos ou paredes em pausa.",
        ),
        (
            ["apagao prolongado o que fazer?", "posso usar gerador dentro de casa?"],
            "Use lanterna, feche a geladeira e NUNCA ligue gerador ou carvão em ambiente fechado!",
        ),
        (
            ["pane no carro na rodovia", "onde esperar socorro na estrada?"],
            "Ligue pisca-alerta, ponha triângulo a 30+ m e espere fora do carro atrás da mureta.",
        ),
        (
            ["dinheiro no kit de emergencia", "guardar documentos na catastrofe"],
            "Sem luz ou sinal, Pix e cartão param! Leve dinheiro trocado e documentos em saco vedado.",
        ),
    ]


def _critical_first_aid_qa() -> list[tuple[list[str], str]]:
    """Q&As on critical first aid: drowning, strangulation, heart attack, seizures, and wounds."""
    return [
        (
            ["como salvar alguem se afogando?", "socorro em afogamento na agua"],
            "Não entre nadando sem boia! Estenda vara ou jogue garrafas PET fechadas e corda; ligue 193.",
        ),
        (
            ["primeiros socorros afogamento", "afogado sem respirar o que fazer?"],
            "Fora d'água, se não respira: faça 5 ventilações iniciais, depois 30 compressões e 2 sopros.",
        ),
        (
            ["como tirar agua do pulmao afogado?", "o que evitar em afogamento?"],
            "NUNCA aperte a barriga nem vire de cabeça para baixo! Se respira, deite de lado e aqueça.",
        ),
        (
            ["afogado voltou a respirar e agora?", "perigo horas apos afogamento"],
            "Leve ao hospital mesmo se acordado: água aspirada pode inflamar o pulmão horas depois.",
        ),
        (
            ["socorro em estrangulamento", "vitima de enforcamento o que faco?"],
            "Solte logo o pescoço amparando corpo e coluna; se não respirar, inicie RCP e ligue 192.",
        ),
        (
            ["estrangulamento sem sintomas?", "risco apos aperto no pescoco"],
            "Leve urgente ao hospital mesmo falando bem: inchaço interno pode fechar o ar horas depois!",
        ),
        (
            ["quais os sintomas de infarto?", "sinais de ataque cardiaco"],
            "Dor ou aperto no peito irradiando ao braço esquerdo, costas ou queixo, suor frio e náusea.",
        ),
        (
            ["ataque cardiaco o que fazer?", "primeiros socorros em infarto"],
            "Chame 192 (SAMU) já, deixe sentado em repouso absoluto, solte roupas e não o deixe dirigir.",
        ),
        (
            ["posso dar aspirina no infarto?", "aas ajuda em ataque cardiaco?"],
            "Se lúcido, sem alergia/sangramento, mastigar 150-300 mg de AAS com aval do 192 ajuda.",
        ),
        (
            ["diferenca infarto e parada?", "infarto faz massagem cardiaca?"],
            "No infarto a pessoa está acordada com dor (repouso); só faça RCP se desmaiar e não respirar!",
        ),
        (
            ["convulsao o que fazer?", "primeiros socorros em convulsao", "crise epileptica como ajudar?"],
            "Deite no chão, proteja a cabeça com pano macio, solte a gola, conte o tempo e vire de lado.",
        ),
        (
            ["segurar lingua na convulsao?", "o que evitar na convulsao?"],
            "NUNCA ponha nada na boca (ninguém engole a língua!), não segure à força nem dê água.",
        ),
        (
            ["quando convulsao e emergencia?", "convulsao demorando o que fazer?"],
            "Chame 192 se passar de 5 min, repetir sem acordar, for a 1a vez, em grávida ou na água.",
        ),
        (
            ["primeiros socorros em ferimentos", "corte ou ferimento o que fazer?"],
            "Lave mãos, estanque o sangue com pano limpo, lave corte leve em água corrente e cubra seco.",
        ),
        (
            ["objeto cravado no ferimento", "faca ou vidro preso no ferimento"],
            "NUNCA puxe o objeto cravado! Fixe com panos em volta para conter o sangue e vá ao hospital.",
        ),
        (
            ["ferimento furado no peito", "corte soprando ar no torax"],
            "Cubra o furo no peito com plástico preso em 3 lados (1 solto como válvula) e chame 192 já.",
        ),
        (
            ["ferimento na barriga orgao exposto", "corte profundo no abdomen"],
            "Não empurre órgãos para dentro! Cubra com pano limpo úmido, dobre joelhos e chame 192.",
        ),
        (
            ["dedo ou membro amputado?", "como guardar parte amputada?"],
            "Estanque o coto; ponha a parte em pano úmido em saco fechado no gelo sem contato direto.",
        ),
        (
            ["mordida de cao ou prego no pe", "ferimento sujo risco de tetano"],
            "Lave 10 min com água e sabão, não feche o corte e busque vacina contra tétano ou raiva.",
        ),
        (
            ["sinais de ferida infeccionada", "como saber se o corte inflamou?"],
            "Vermelhidão crescendo, calor, inchaço forte, pus amarelado ou febre exigem médico urgente.",
        ),
        (
            ["sangramento no nariz?", "nariz sangrando inclino pra tras?"],
            "Sente inclinado para a FRENTE (nunca para trás!) e aperte a ponta do nariz por 10 min.",
        ),
    ]


def _emergency_power_qa() -> list[tuple[list[str], str]]:
    """Q&As on improvised batteries and small generators to charge mobile devices."""
    return [
        (
            ["como fabricar uma bateria basica?", "como fazer pilha caseira?"],
            "Use 2 metais diferentes (cobre + e zinco ou alumínio -) separados por pano com água e sal.",
        ),
        (
            ["bateria caseira para ligar celular", "quantas pilhas caseiras dao 5v?"],
            "Cada célula dá ~0,9V; ligue 6 grandes em série (+ no -) para somar ~5V ao dispositivo.",
        ),
        (
            ["bateria ar-aluminio celular", "bateria de lata e carvao?"],
            "Ar-alumínio: 6 latas lixadas (-) com carvão moído em pano úmido de salmoura (+) dão ~5V.",
        ),
        (
            ["metais para pilha caseira?", "liquido de bateria improvisada"],
            "Polo +: cobre ou grafite/carvão; Polo -: zinco ou alumínio; líquido: água com sal/vinagre.",
        ),
        (
            ["carregar celular com pilhas aa", "ligar celular com pilha comum"],
            "Ligue 3 pilhas AA em série (4,5V) ou 4 pilhas (6V) com diodo 1N4007 reduzindo para ~5,3V.",
        ),
        (
            ["carregar celular bateria de carro", "posso ligar 12v direto no celular?"],
            "NUNCA ligue 12V direto no USB! Use carregador veicular 12V-5V ou módulo buck / LM7805.",
        ),
        (
            ["quais fios do cabo usb sao 5v?", "cores dos fios usb para carregar"],
            "No cabo USB: fio Vermelho é +5V e Preto é Terra (-); isole verde/branco e não passe 5,5V.",
        ),
        (
            ["como construir gerador celular?", "gerador pequeno para celular", "como fazer dinamo para celular?"],
            "Gire motor DC de ímã permanente (12-24V) com manivela, bike ou água via regulador 5V.",
        ),
        (
            ["qual motor serve de gerador?", "motor de sucata vira gerador?"],
            "Use motor DC de furadeira, impressora, esteira ou limpador de vidro; girar gera tensão DC.",
        ),
        (
            ["gerador do zero com ima?", "gerar energia com ima e bobina"],
            "Gire ímãs fortes rente a bobinas de fio de cobre esmaltado e retifique com ponte de diodos.",
        ),
        (
            ["nao queimar celular no gerador?", "regular 5v no dinamo caseiro"],
            "A manivela oscila! Passe a saída num carregador veicular USB (8-24V->5V) ou LM7805.",
        ),
        (
            ["por que por diodo no gerador?", "evitar descarga no dinamo manual"],
            "Ponha 1 diodo (1N4007/Schottky) em série para a bateria não descarregar no motor parado.",
        ),
        (
            ["celular corta carga no dinamo", "melhor jeito de carregar no dinamo"],
            "Carregue antes um powerbank no gerador manual e depois o celular nele contra oscilações.",
        ),
        (
            ["como fazer bateria durar dias?", "economizar celular na catastrofe"],
            "Ative modo avião e economia, zere o brilho e só ligue sinal em horários combinados.",
        ),
    ]


def build_survival_sections() -> list[tuple[str, list[tuple[list[str], str]]]]:
    """Return ordered survival and catastrophe Q&A sections."""
    return [
        ("SOBREVIVENCIA BASICA NA FLORESTA E ORIENTACAO", _forest_survival_qa()),
        ("AGUA NA NATUREZA: COMO ESCOLHER, CAPTAR E PURIFICAR", _water_survival_qa()),
        ("COMIDA NA NATUREZA: COMO ESCOLHER O QUE COMER", _food_survival_qa()),
        ("COMO ESCOLHER E MONTAR UM ABRIGO NA NATUREZA", _shelter_survival_qa()),
        ("PROTECAO NA MATA, ANIMAIS PECONHENTOS E PRIMEIROS SOCORROS", _protection_first_aid_qa()),
        ("PRIMEIROS SOCORROS: AFOGAMENTO, ESTRANGULAMENTO, INFARTO, CONVULSAO E FERIMENTOS", _critical_first_aid_qa()),
        ("ENERGIA DE EMERGENCIA: BATERIA BASICA E GERADOR PARA CELULAR", _emergency_power_qa()),
        ("ENCHENTES, ALAGAMENTOS E DESLIZAMENTOS", _floods_landslides_qa()),
        ("EXPLOSAO NUCLEAR E FALLOUT RADIOATIVO", _nuclear_fallout_qa()),
        ("OUTRAS CATASTROFES URBANAS E KIT DE EMERGENCIA 72H", _urban_catastrophe_kit_qa()),
    ]


def build_sections(all_pairs: bool = False) -> list[tuple[str, list[tuple[list[str], str]]]]:
    sections: list[tuple[str, list[tuple[list[str], str]]]] = []

    # 1. Conversação Cotidiana, Identidade AMEBA, Hardware ESP32 e Personalizados
    persona = [
        (["oi", "oii", "oi!", "ola", "olaa", "ola!", "opa", "e ai", "e ai?", "eae", "fala", "salve", "alo", "hi", "hello"], "Olá! Sou o AMEBA. Como posso ajudar?"),
        (["tudo bem?", "tudo bem", "oi tudo bem?", "ola tudo bem?", "tudo bom?", "tudo bom", "tudo certo?", "tudo joia?", "td bem?", "beleza?", "blz?"], "Tudo ótimo por aqui no ESP32! E com você?"),
        (
            [
                "o que é o projeto povos da floresta?",
                "o que é o projeto conexão povos da floresta?",
                "o que é conexão povos da floresta?",
            ],
            "Iniciativa humana de conectividade significativa a povos isolados na Amazônia.",
        ),
        (["como vai?", "como vai", "como você está?", "como você esta?", "como vc ta?", "como anda?"], "Tudo ótimo por aqui no ESP32! E com você?"),
        (["tudo bem também", "estou bem", "tudo ótimo", "bem obrigado", "to bem", "tudo certo"], "Que bom! Pode mandar sua pergunta."),
        (["bom dia", "bom dia!", "bom dia ameba"], "Bom dia! Sou o AMEBA no ESP32. Em que posso ajudar?"),
        (["boa tarde", "boa tarde!", "boa tarde ameba"], "Boa tarde! Sou o AMEBA no ESP32. Pode perguntar!"),
        (["boa noite", "boa noite!", "boa noite ameba"], "Boa noite! Estou ligado aqui no ESP32. O que deseja saber?"),
        (["tchau", "até mais", "até logo", "falou", "adeus", "fui", "bye"], "Até mais! Quando precisar, estarei aqui no ESP32."),
        (["obrigado", "obrigada", "valeu", "muito obrigado", "vlw"], "De nada! Estou pronto para a próxima pergunta."),
        (["legal", "muito bom", "top", "show", "bacana", "maneiro"], "Valeu! Pode mandar a próxima pergunta."),
        (["kkk", "kk", "haha", "hehe", "rs"], "Haha! Gostou? Pergunta mais alguma coisa."),
        (["teste", "testando", "test", "123", "alou"], "Teste recebido! O AMEBA está online no ESP32."),
        (["quem é você?", "quem é você", "quem é vc?", "fale sobre você"], "Eu sou o AMEBA rodando em um ESP32. Quem é você?"),
        (["qual seu hardware?", "onde você está rodando?", "onde você mora?"], "Estou rodando direto em um ESP32 sem PSRAM. Qual seu Hardware?"),
        (["qual é o seu nome?", "qual seu nome?", "qual seu nome"], "Meu nome é AMEBA. Quem é você?"),
        (["ameba?", "ameba"], "Sim e daí? Quem é você?!"),
        (["como pode me ajudar?", "como você pode me ajudar?"], "Se nem você sabe, tá difícil."),
        (
            ["o que você sabe fazer?", "sobre o que posso perguntar?", "o que você conhece?"],
            "Sei sobre sobrevivência na mata, catástrofes, estados, capitais, biomas e presidentes do Brasil.",
        ),
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
        (["meu nome é alessandro", "alessandro", "ale"], "Bem vindo ao meu Reino Alessandro"),
        (["meu nome é igor", "igor"], "Bem vindo ao meu Reino Igor"),
        (["meu nome é luize", "luize"], "Bem vindo ao meu Reino Luize"),
        (["meu nome é luiz", "luiz"], "Bem vindo ao meu Reino Luiz"),
        (["meu nome é luan", "luan"], "Bem vindo ao meu Reino Luan"),
        (["meu nome é tasso", "tasso"], "Bem vindo ao meu Reino Tasso"),
    ]
    sections.append(("CONVERSACAO COTIDIANA, IDENTIDADE AMEBA E PERSONALIZADOS", persona))

    # 1B. Perguntas Fora do Escopo / Desconhecidas (evita alucinar texto aleatório)
    unknown_msg = "Não sei responder isso; sou o AMEBA offline com 367 mil parâmetros."
    unknown_hw_msg = "Não sei responder isso; sou o AMEBA offline com 367 mil parâmetros. Qual seu hardware?"
    unknown_who_msg = "Não sei responder isso; sou o AMEBA offline com 367 mil parâmetros. Quem é você?"
    unknown_qa = [
        (["asdf", "qwerty", "blablabla", "??", "xyz", "abc", "nada"], unknown_hw_msg),
        (
            [
                "meu hardware é aasdf",
                "meu hardware é xxx",
                "meu hardware é qwerty",
                "meu hardware é abc",
                "meu hardware é nada",
                "meu hardware é um celular",
                "meu hardware é um pc",
            ],
            "Ual isso parece complexo.",
        ),
        (["qual a cotação do dólar hoje?", "quanto está o dólar?", "qual a cotação do bitcoin?", "quanto está o bitcoin?"], unknown_who_msg),
        (["vai chover amanhã?", "qual a previsão do tempo?", "qual a temperatura hoje?", "que horas são agora?"], unknown_who_msg),
        (["quem ganhou o jogo ontem?", "qual o resultado da mega sena?", "quem vai ganhar a copa?", "qual a senha do wifi?"], unknown_who_msg),
        (["onde eu moro?", "qual o meu nome?", "quantos anos eu tenho?", "qual meu filme favorito?", "qual sua comida favorita?"], unknown_who_msg),
        (["faz um café pra mim", "qual a melhor linguagem de programação?", "me dá o número do pix", "qual o sentido da vida?"], unknown_who_msg),
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
    sections.extend(build_survival_sections())

    # Normalize all questions to lowercase ASCII form (preserving readability in answers) and deduplicate
    seen_qa: set[tuple[str, str]] = set()
    seen_q_to_ans: dict[str, str] = {}
    cleaned_sections: list[tuple[str, list[tuple[list[str], str]]]] = []
    for sec_name, entries in sections:
        sec_out: list[tuple[list[str], str]] = []
        for qs, ans in entries:
            ans_ascii = to_ascii(ans)
            uniq_qs: list[str] = []
            for q in qs:
                qn = normalize_q(q)
                if not qn:
                    continue
                prev_ans = seen_q_to_ans.get(qn)
                if prev_ans is not None and prev_ans != ans_ascii:
                    raise ValueError(
                        f"Conflicting answers for '{qn}': '{prev_ans}' vs '{ans_ascii}'"
                    )
                if (qn, ans_ascii) not in seen_qa:
                    seen_qa.add((qn, ans_ascii))
                    seen_q_to_ans[qn] = ans_ascii
                    uniq_qs.append(qn)
            if uniq_qs:
                validate_entry(uniq_qs, ans)
                sec_out.append((uniq_qs, ans))
        cleaned_sections.append((sec_name, sec_out))

    return cleaned_sections


def write_dataset(txt_path: pathlib.Path, jsonl_path: pathlib.Path, all_pairs: bool) -> tuple[int, int]:
    sections = build_sections(all_pairs=all_pairs)
    lines = [
        "# Dataset extenso do TINY 2.1 / AMEBA (Brasil + Canivete Suico de Sobrevivencia e Catastrofes)",
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
