#include "tiny_session.h"

#include <stdio.h>
#include <string.h>

static TinySession web_sessions[TINY_MAX_SESSIONS];
static TinySession serial_session;

typedef struct {
    const char *uf;
    const char *state;
    const char *capital;
} UfEntry;

static const UfEntry UF_TABLE[27] = {
    {"ac", "acre", "rio branco"},
    {"al", "alagoas", "maceio"},
    {"ap", "amapa", "macapa"},
    {"am", "amazonas", "manaus"},
    {"ba", "bahia", "salvador"},
    {"ce", "ceara", "fortaleza"},
    {"df", "distrito federal", "brasilia"},
    {"es", "espirito santo", "vitoria"},
    {"go", "goias", "goiania"},
    {"ma", "maranhao", "sao luis"},
    {"mt", "mato grosso", "cuiaba"},
    {"ms", "mato grosso do sul", "campo grande"},
    {"mg", "minas gerais", "belo horizonte"},
    {"pa", "para", "belem"},
    {"pb", "paraiba", "joao pessoa"},
    {"pr", "parana", "curitiba"},
    {"pe", "pernambuco", "recife"},
    {"pi", "piaui", "teresina"},
    {"rj", "rio de janeiro", "rio de janeiro"},
    {"rn", "rio grande do norte", "natal"},
    {"rs", "rio grande do sul", "porto alegre"},
    {"ro", "rondonia", "porto velho"},
    {"rr", "roraima", "boa vista"},
    {"sc", "santa catarina", "florianopolis"},
    {"sp", "sao paulo", "sao paulo"},
    {"se", "sergipe", "aracaju"},
    {"to", "tocantins", "palmas"},
};

typedef struct {
    const char *topic;
    const char *k1;
    const char *k2;
    const char *k3;
    const char *canonical_q;
} FollowupRule;

static const FollowupRule FOLLOWUP_RULES[] = {
    /* Convulsao */
    {"convulsao", "evitar", "nunca", "lingua", "segurar lingua na convulsao?"},
    {"convulsao", "192", "emergencia", "demorar", "quando convulsao e emergencia?"},
    {"convulsao", "tempo", "durar", NULL, "quando convulsao e emergencia?"},
    {"convulsao", "fazer", "agir", "ajudar", "convulsao o que fazer?"},
    /* Afogamento */
    {"afogamento", "evitar", "nunca", "pulmao", "o que evitar em afogamento?"},
    {"afogamento", "voltou", "hospital", "horas", "afogado voltou a respirar e agora?"},
    {"afogamento", "respirar", "rcp", "massagem", "primeiros socorros afogamento"},
    {"afogamento", "salvar", "resgatar", NULL, "como salvar alguem se afogando?"},
    {"afogamento", "fazer", "agir", NULL, "primeiros socorros afogamento"},
    /* Estrangulamento */
    {"estrangulamento", "hospital", "depois", "bem", "estrangulamento sem sintomas?"},
    {"estrangulamento", "glote", "inchaco", "falar", "estrangulamento sem sintomas?"},
    {"estrangulamento", "fazer", "socorro", "agir", "socorro em estrangulamento"},
    /* Infarto */
    {"infarto", "sintoma", "sinal", "dor", "quais os sintomas de infarto?"},
    {"infarto", "aspirina", "aas", "remedio", "posso dar aspirina no infarto?"},
    {"infarto", "parada", "diferenca", "rcp", "diferenca infarto e parada?"},
    {"infarto", "fazer", "socorro", "agir", "ataque cardiaco o que fazer?"},
    /* Ferimentos */
    {"ferimento", "cravado", "faca", "vidro", "objeto cravado no ferimento"},
    {"ferimento", "puxar", "tirar", NULL, "objeto cravado no ferimento"},
    {"ferimento", "peito", "torax", NULL, "ferimento furado no peito"},
    {"ferimento", "barriga", "abdomen", "orgao", "ferimento na barriga orgao exposto"},
    {"ferimento", "amputad", "dedo", "gelo", "dedo ou membro amputado?"},
    {"ferimento", "tetano", "mordida", "prego", "mordida de cao ou prego no pe"},
    {"ferimento", "infeccao", "inflam", "pus", "sinais de ferida infeccionada"},
    {"ferimento", "nariz", NULL, NULL, "sangramento no nariz?"},
    {"ferimento", "estancar", "sangr", NULL, "como estancar sangramento forte?"},
    {"ferimento", "torniquete", NULL, NULL, "quando usar um torniquete?"},
    {"ferimento", "limpar", "lavar", NULL, "como limpar ferida na natureza?"},
    {"ferimento", "fazer", "agir", NULL, "primeiros socorros em ferimentos"},
    /* Cobra / Escorpiao */
    {"cobra", "evitar", "nunca", "chupar", "o que evitar em picada de cobra?"},
    {"cobra", "cortar", "torniquete", NULL, "o que evitar em picada de cobra?"},
    {"cobra", "especie", "quais", "venenosa", "cobras venenosas do brasil"},
    {"cobra", "prevenir", "nao ser picado", NULL, "como evitar picada de cobra?"},
    {"cobra", "fazer", "agir", NULL, "picada de cobra o que fazer?"},
    {"escorpiao", "fazer", "agir", "dor", "picada de escorpiao o que fazer?"},
    /* Enchentes e deslizamentos */
    {"enchente", "sotao", "forro", "subir", "posso subir no sotao na enchente?"},
    {"enchente", "a pe", "atravessar", "andar", "posso atravessar rua alagada?"},
    {"enchente", "vidro", "quebrar", NULL, "como quebrar vidro de carro?"},
    {"enchente", "afund", "travou", NULL, "carro afundando o que fazer?"},
    {"enchente", "carro", "dirigir", NULL, "posso passar alagamento de carro?"},
    {"enchente", "religar", "disjuntor", NULL, "quando religar luz apos enchente?"},
    {"enchente", "fio", "poste", "eletric", "risco eletrico na enchente"},
    {"enchente", "limpar", "desinfetar", "sanitaria", "como limpar casa pos enchente?"},
    {"enchente", "comida", "alimento", "lata", "comida que molhou na enchente"},
    {"enchente", "sintoma", "panturrilha", "febre", "sintomas de leptospirose"},
    {"enchente", "leptospirose", "evitar", "doenca", "como evitar leptospirose?"},
    {"enchente", "bicho", "cobra", "escorpiao", "bichos em casa apos enchente"},
    {"enchente", "agua", "torneira", NULL, "agua da torneira apos enchente"},
    {"enchente", "fazer", "agir", NULL, "o que fazer em enchente?"},
    {"deslizamento", "fazer", "agir", "199", "risco de deslizamento o que fazer?"},
    {"deslizamento", "sinal", "sinais", "saber", "sinais de deslizamento de terra"},
    /* Nuclear e Fallout */
    {"nuclear", "clarao", "olhar", NULL, "o que fazer ao ver clarao nuclear?"},
    {"nuclear", "onda", "deitar", "posicao", "posicao onda de choque nuclear"},
    {"nuclear", "quanto tempo cair", "quando cai", "minutos", "quanto tempo antes do fallout?"},
    {"nuclear", "quanto tempo", "sair", "horas", "quanto tempo ficar no abrigo?"},
    {"nuclear", "nao ficar", "evitar", "carro", "onde nao ficar no fallout?"},
    {"nuclear", "abrigo", "esconder", "onde ficar", "melhor abrigo explosao nuclear"},
    {"nuclear", "vedar", "ar", "janela", "vedar abrigo contra fallout"},
    {"nuclear", "condicionador", "cabelo", NULL, "por que nao usar condicionador?"},
    {"nuclear", "sem agua", "chuveiro", NULL, "banho sem agua apos fallout"},
    {"nuclear", "descontamina", "limpar", "roupa", "descontaminacao de radiacao"},
    {"nuclear", "7", "regra", "diminui", "regra 7 por 10 radiacao"},
    {"nuclear", "agua", "ferver", "beber", "ferver agua tira radiacao?"},
    {"nuclear", "comida", "comer", "fruta", "o que comer apos fallout?"},
    {"nuclear", "iodo", "iodeto", "tireoide", "para que serve iodeto de potassio?"},
    {"nuclear", "celular", "radio", "emp", "celular funciona apos explosao?"},
    {"nuclear", "fazer", "agir", NULL, "explosao nuclear o que fazer?"},
    /* Bateria e Gerador */
    {"bateria", "5v", "quantas", "celular", "quantas pilhas caseiras dao 5v?"},
    {"bateria", "lata", "carvao", "aluminio", "bateria ar-aluminio celular"},
    {"bateria", "metais", "metal", "liquido", "metais para pilha caseira?"},
    {"bateria", "aa", "pilha comum", NULL, "carregar celular com pilhas aa"},
    {"bateria", "12v", "carro", "moto", "carregar celular bateria de carro"},
    {"bateria", "usb", "fio", "fios", "quais fios do cabo usb sao 5v?"},
    {"bateria", "fazer", "fabricar", NULL, "como fabricar uma bateria basica?"},
    {"gerador", "motor", "sucata", NULL, "qual motor serve de gerador?"},
    {"gerador", "ima", "bobina", "zero", "gerador do zero com ima?"},
    {"gerador", "queimar", "regular", "5v", "nao queimar celular no gerador?"},
    {"gerador", "diodo", "descarregar", NULL, "por que por diodo no gerador?"},
    {"gerador", "powerbank", "corta", "oscil", "celular corta carga no dinamo"},
    {"gerador", "economi", "durar", NULL, "como fazer bateria durar dias?"},
    {"gerador", "fazer", "construir", NULL, "como construir gerador celular?"},
    /* Agua / Comida / Abrigo / Fogo / Incendio */
    {"agua", "sem panela", "pet", NULL, "como ferver agua sem panela?"},
    {"agua", "ferver", "purificar", "quanto tempo", "como purificar agua?"},
    {"agua", "cloro", "sanitaria", "gota", "purificar agua com agua sanitaria"},
    {"agua", "filtro", "areia", "carvao", "como fazer filtro de agua na mata?"},
    {"agua", "sol", "sodis", NULL, "purificar agua com sol sodis"},
    {"agua", "mar", "salgada", "destilar", "como destilar agua do mar?"},
    {"agua", "urina", "alcool", NULL, "posso beber urina na emergencia?"},
    {"agua", "soro", "receita", NULL, "como fazer soro caseiro?"},
    {"agua", "evitar", "nunca", "impropria", "qual agua nunca devo beber?"},
    {"comida", "cogumelo", NULL, NULL, "posso comer cogumelo selvagem?"},
    {"comida", "evitar", "nunca", "venenos", "identificar planta venenosa"},
    {"comida", "teste", "testar", NULL, "teste de comestibilidade planta"},
    {"comida", "inseto", "grilo", "larva", "quais insetos posso comer?"},
    {"comida", "peixe", NULL, NULL, "todo peixe de rio e comestivel?"},
    {"comida", "fruta", "segur", NULL, "frutas seguras no brasil"},
    {"abrigo", "nunca", "evitar", "onde nao", "onde nunca montar abrigo?"},
    {"abrigo", "chao", "frio", "isolar", "como isolar corpo do chao frio?"},
    {"abrigo", "meia agua", "lean", NULL, "como fazer abrigo meia agua?"},
    {"abrigo", "duas aguas", "a-frame", "a frame", "como montar abrigo duas aguas?"},
    {"abrigo", "fogo", "fogueira", "distancia", "distancia da fogueira ao abrigo"},
    {"abrigo", "lona", "poncho", "barraca", "como armar abrigo com lona?"},
    {"abrigo", "escolher", "onde", "lugar", "como escolher abrigo?"},
    {"fogo", "chuva", "umida", "molhad", "fogo na chuva ou mata umida"},
    {"fogo", "isca", "acender", NULL, "o que usar de isca para fogo?"},
    {"fogo", "sem fosforo", "isqueiro", NULL, "fazer fogo sem fosforo"},
    {"fogo", "cuidado", "pedra", NULL, "cuidados ao fazer fogueira"},
    {"fogo", "fazer", NULL, NULL, "como fazer fogo na floresta?"},
    {"incendio", "porta", "macaneta", NULL, "abrir porta em incendio"},
    {"incendio", "roupa", "corpo", NULL, "roupa pegou fogo o que fazer?"},
    {"incendio", "panela", "oleo", NULL, "oleo pegou fogo na panela"},
    {"incendio", "florest", "mata", NULL, "como fugir de incendio florestal?"},
    {"incendio", "predio", "elevador", NULL, "incendio em predio o que fazer?"},
    {NULL, NULL, NULL, NULL, NULL}
};

static void safe_copy(char *dst, size_t cap, const char *src)
{
    if (!dst || cap == 0) return;
    if (!src) {
        dst[0] = '\0';
        return;
    }
    snprintf(dst, cap, "%s", src);
}

static bool has_sub(const char *hay, const char *needle)
{
    return hay && needle && strstr(hay, needle) != NULL;
}

static bool starts_with(const char *s, const char *prefix)
{
    if (!s || !prefix) return false;
    return strncmp(s, prefix, strlen(prefix)) == 0;
}

static bool ends_with(const char *s, const char *suffix)
{
    if (!s || !suffix) return false;
    size_t ls = strlen(s);
    size_t lsf = strlen(suffix);
    if (lsf > ls) return false;
    return strcmp(s + (ls - lsf), suffix) == 0;
}

static char to_lower_ascii(char c)
{
    if (c >= 'A' && c <= 'Z') return (char)(c + ('a' - 'A'));
    return c;
}

static bool equals_ci(const char *a, const char *b)
{
    if (!a || !b) return false;
    while (*a && *b) {
        if (to_lower_ascii(*a) != to_lower_ascii(*b)) return false;
        a++;
        b++;
    }
    return *a == '\0' && *b == '\0';
}

static bool ends_with_ci(const char *s, const char *suffix)
{
    if (!s || !suffix) return false;
    size_t ls = strlen(s);
    size_t lsf = strlen(suffix);
    if (lsf > ls) return false;
    const char *p = s + (ls - lsf);
    for (size_t i = 0; i < lsf; i++) {
        if (to_lower_ascii(p[i]) != to_lower_ascii(suffix[i])) return false;
    }
    return true;
}

static bool is_alpha_ascii(char c)
{
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}

static bool contains_word(const char *hay, const char *word)
{
    if (!hay || !word || !word[0]) return false;
    size_t lw = strlen(word);
    const char *p = hay;
    while ((p = strstr(p, word)) != NULL) {
        bool left_ok = (p == hay) || !is_alpha_ascii(p[-1]);
        bool right_ok = !is_alpha_ascii(p[lw]);
        if (left_ok && right_ok) return true;
        p += lw;
    }
    return false;
}

static void title_case_clean(char *dst, size_t cap, const char *src)
{
    if (!dst || cap == 0) return;
    dst[0] = '\0';
    if (!src) return;

    while (*src == ' ') src++;
    size_t n = 0;
    bool new_word = true;
    for (; *src && n + 1 < cap; src++) {
        char c = *src;
        if (c == '.' || c == '!' || c == '?') break;
        if (c == ' ') {
            if (n > 0 && dst[n - 1] != ' ') {
                dst[n++] = ' ';
                new_word = true;
            }
            continue;
        }
        if (c >= 'a' && c <= 'z' && new_word) {
            c = (char)(c - ('a' - 'A'));
        }
        dst[n++] = c;
        new_word = false;
    }
    while (n > 0 && dst[n - 1] == ' ') n--;
    dst[n] = '\0';
}

static void extract_clean_phrase(char *dst, size_t cap, const char *src)
{
    if (!dst || cap == 0) return;
    dst[0] = '\0';
    if (!src) return;
    while (*src == ' ') src++;
    size_t n = 0;
    for (; *src && n + 1 < cap; src++) {
        char c = *src;
        if (c == '.' || c == '!' || c == '?') break;
        dst[n++] = c;
    }
    while (n > 0 && dst[n - 1] == ' ') n--;
    dst[n] = '\0';
}

static int count_words(const char *s)
{
    int w = 0;
    bool in_w = false;
    for (; *s; s++) {
        if (*s == ' ') {
            in_w = false;
        } else if (!in_w) {
            in_w = true;
            w++;
        }
    }
    return w;
}

static int find_uf_mention(const char *q)
{
    /* Pass 1: full state name or capital (prefer longer names like Mato Grosso do Sul) */
    int best_idx = -1;
    size_t best_len = 0;
    for (int i = 0; i < 27; i++) {
        size_t sl = strlen(UF_TABLE[i].state);
        bool state_hit = false;
        if (strcmp(UF_TABLE[i].uf, "pa") == 0) {
            /* Disambiguate Para (state) from the Portuguese preposition "para" */
            state_hit = strcmp(q, "para") == 0 || has_sub(q, "do para") ||
                        has_sub(q, "no para") || has_sub(q, "o para") ||
                        has_sub(q, "de para");
        } else {
            state_hit = contains_word(q, UF_TABLE[i].state);
        }
        if (sl > best_len && state_hit) {
            best_idx = i;
            best_len = sl;
        }
        size_t cl = strlen(UF_TABLE[i].capital);
        if (cl > best_len && contains_word(q, UF_TABLE[i].capital)) {
            best_idx = i;
            best_len = cl;
        }
    }
    if (best_idx >= 0) return best_idx;

    /* Pass 2: explicit 2-letter UF code preceded by de/do/da/em/capital/regiao */
    for (int i = 0; i < 27; i++) {
        char p1[16], p2[16];
        snprintf(p1, sizeof(p1), "de %s", UF_TABLE[i].uf);
        snprintf(p2, sizeof(p2), " %s", UF_TABLE[i].uf);
        if (contains_word(q, p1)) return i;
        if (ends_with(q, p2)) return i;
    }
    return -1;
}

static int find_uf_from_topic(const char *topic)
{
    if (!starts_with(topic, "uf:")) return -1;
    const char *uf = topic + 3;
    for (int i = 0; i < 27; i++) {
        if (strcmp(UF_TABLE[i].uf, uf) == 0) return i;
    }
    return -1;
}

static const char *detect_biome(const char *q)
{
    if (has_sub(q, "mata atlantica")) return "bioma:mata atlantica";
    if (contains_word(q, "amazonia")) return "bioma:amazonia";
    if (contains_word(q, "cerrado")) return "bioma:cerrado";
    if (contains_word(q, "caatinga")) return "bioma:caatinga";
    if (contains_word(q, "pampa")) return "bioma:pampa";
    if (contains_word(q, "pantanal")) return "bioma:pantanal";
    return NULL;
}

static const char *detect_topic(const char *q)
{
    if (!q || !q[0]) return NULL;

    if (has_sub(q, "convuls") || has_sub(q, "epilep")) return "convulsao";
    if (has_sub(q, "afoga")) return "afogamento";
    if (has_sub(q, "estrangul") || has_sub(q, "enforc") || has_sub(q, "pescoco")) return "estrangulamento";
    if (has_sub(q, "infart") || has_sub(q, "ataque cardiaco") || has_sub(q, "aspirina") || has_sub(q, "aas")) return "infarto";
    if (has_sub(q, "feriment") || has_sub(q, "corte") || has_sub(q, "cravado") ||
        has_sub(q, "amputad") || has_sub(q, "sangramento") || has_sub(q, "hemorragia") ||
        has_sub(q, "torniquete") || has_sub(q, "ferida")) {
        return "ferimento";
    }
    if (has_sub(q, "cobra") || has_sub(q, "jararaca") || has_sub(q, "cascavel") || has_sub(q, "surucucu")) return "cobra";
    if (has_sub(q, "escorpiao")) return "escorpiao";
    if (has_sub(q, "enchente") || has_sub(q, "alagament") || has_sub(q, "enxurrada") || has_sub(q, "leptospirose")) return "enchente";
    if (has_sub(q, "deslizamento") || has_sub(q, "barreira")) return "deslizamento";
    if (has_sub(q, "nuclear") || has_sub(q, "fallout") || has_sub(q, "radiacao") ||
        has_sub(q, "radioativ") || has_sub(q, "iodeto") || has_sub(q, "clarao")) {
        return "nuclear";
    }
    if (has_sub(q, "gerador") || has_sub(q, "dinamo")) return "gerador";
    if (has_sub(q, "bateria") || has_sub(q, "pilha") || has_sub(q, "cabo usb")) return "bateria";
    if (has_sub(q, "incendio") || has_sub(q, "pegou fogo") || has_sub(q, "panela de oleo")) return "incendio";
    if (has_sub(q, "fogo") || has_sub(q, "fogueira")) return "fogo";
    if (has_sub(q, "abrigo") || has_sub(q, "acampar") || has_sub(q, "lean to") || has_sub(q, "a-frame")) return "abrigo";
    if (has_sub(q, "agua") || has_sub(q, "orvalho") || has_sub(q, "cipo") || has_sub(q, "sodis") || has_sub(q, "soro caseiro")) return "agua";
    if (has_sub(q, "comer") || has_sub(q, "comida") || has_sub(q, "cogumelo") ||
        has_sub(q, "planta venenosa") || has_sub(q, "inseto") || has_sub(q, "peixe") || has_sub(q, "frut")) {
        return "comida";
    }

    const char *biome = detect_biome(q);
    if (biome) return biome;

    int uf_idx = find_uf_mention(q);
    if (uf_idx >= 0) {
        static char uf_buf[12];
        snprintf(uf_buf, sizeof(uf_buf), "uf:%s", UF_TABLE[uf_idx].uf);
        return uf_buf;
    }

    return NULL;
}

static bool looks_like_question_or_command(const char *q)
{
    if (!q || !q[0]) return true;
    if (q[0] == '/') return true;
    if (strchr(q, '?') != NULL) return true;
    if (detect_topic(q) != NULL) return true;

    static const char *const NON_NAME_WORDS[] = {
        "oi", "oii", "oi!", "ola", "ola!", "opa", "e ai", "eae", "fala", "salve", "alo",
        "bom dia", "boa tarde", "boa noite", "tchau", "ate mais", "ate logo", "obrigado",
        "obrigada", "valeu", "vlw", "legal", "sim", "nao", "tudo", "nada", "teste",
        "qual", "quais", "quem", "como", "onde", "quando", "quanto", "o que", "por que",
        "posso", "devo", "tem", "fale", "conta", "ameba", "ajuda", "help", "socorro",
        "presidente", "capital", "bioma", "distancia", "regra", "kit", "cls", "ok",
        NULL
    };
    for (int i = 0; NON_NAME_WORDS[i]; i++) {
        if (strcmp(q, NON_NAME_WORDS[i]) == 0 || starts_with(q, NON_NAME_WORDS[i])) {
            size_t lp = strlen(NON_NAME_WORDS[i]);
            if (q[lp] == '\0' || q[lp] == ' ' || q[lp] == '!') {
                return true;
            }
        }
    }
    return false;
}

static bool match_known_kingdom_name(const char *q, char *proper_name_out, size_t cap)
{
    struct NameAlias { const char *alias; const char *proper; };
    static const NameAlias ALIASES[] = {
        {"alessandro", "Alessandro"},
        {"ale", "Alessandro"},
        {"igor", "Igor"},
        {"luize", "Luize"},
        {"luiz", "Luiz"},
        {"luan", "Luan"},
        {"tasso", "Tasso"},
    };
    for (size_t i = 0; i < sizeof(ALIASES) / sizeof(ALIASES[0]); i++) {
        if (strcmp(q, ALIASES[i].alias) == 0) {
            safe_copy(proper_name_out, cap, ALIASES[i].proper);
            return true;
        }
    }
    return false;
}

static bool is_placeholder_hw(const char *hw)
{
    static const char *const PLACEHOLDERS[] = {"aasdf", "xx", "qwerty", "abc", "nada", NULL};
    for (int i = 0; PLACEHOLDERS[i]; i++) {
        if (strcmp(hw, PLACEHOLDERS[i]) == 0) return true;
    }
    return false;
}

static void extract_name_after_prefix(const char *q, const char *prefix, char *out, size_t cap)
{
    const char *rest = q + strlen(prefix);
    title_case_clean(out, cap, rest);
}

static bool try_rewrite_biome(const char *topic, const char *q, char *out, size_t cap)
{
    const char *biome = topic + 6; /* skip "bioma:" */
    bool is_fem = (strcmp(biome, "amazonia") == 0 || strcmp(biome, "mata atlantica") == 0 || strcmp(biome, "caatinga") == 0);
    const char *prep = is_fem ? "da" : "do";

    if (has_sub(q, "animais") || has_sub(q, "fauna") || has_sub(q, "bichos")) {
        snprintf(out, cap, "animais %s %s", prep, biome);
        return true;
    }
    if (has_sub(q, "plantas") || has_sub(q, "flora") || has_sub(q, "arvore")) {
        if (strcmp(biome, "pampa") == 0) {
            snprintf(out, cap, "qual a vegetacao do pampa?");
        } else if (strcmp(biome, "pantanal") == 0) {
            snprintf(out, cap, "qual o clima do pantanal?");
        } else {
            snprintf(out, cap, "plantas %s %s", prep, biome);
        }
        return true;
    }
    if (has_sub(q, "clima") || has_sub(q, "vegetacao") || has_sub(q, "tempo")) {
        if (strcmp(biome, "mata atlantica") == 0) {
            snprintf(out, cap, "qual o estado atual da mata atlantica?");
        } else if (strcmp(biome, "amazonia") == 0 || strcmp(biome, "pantanal") == 0) {
            snprintf(out, cap, "qual o clima %s %s?", prep, biome);
        } else {
            snprintf(out, cap, "qual a vegetacao %s %s?", prep, biome);
        }
        return true;
    }
    if (has_sub(q, "onde") || has_sub(q, "local") || has_sub(q, "fica") || has_sub(q, "regiao")) {
        if (strcmp(biome, "amazonia") == 0 || strcmp(biome, "pampa") == 0) {
            snprintf(out, cap, "onde fica %s %s no brasil?", is_fem ? "a" : "o", biome);
        } else {
            snprintf(out, cap, "onde fica %s %s?", is_fem ? "a" : "o", biome);
        }
        return true;
    }
    if (has_sub(q, "o que e") || has_sub(q, "caracteristica")) {
        snprintf(out, cap, "o que e %s %s?", is_fem ? "a" : "o", biome);
        return true;
    }
    return false;
}

static bool try_rewrite_uf(int uf_idx, const char *q, char *out, size_t cap)
{
    if (uf_idx < 0 || uf_idx >= 27) return false;
    if (has_sub(q, "regiao")) {
        snprintf(out, cap, "regiao de %s", UF_TABLE[uf_idx].uf);
        return true;
    }
    if (has_sub(q, "capital")) {
        snprintf(out, cap, "qual a capital de %s?", UF_TABLE[uf_idx].uf);
        return true;
    }
    if (has_sub(q, "distancia") || has_sub(q, "km") || has_sub(q, "longe")) {
        int other = find_uf_mention(q);
        if (other >= 0 && other != uf_idx) {
            int a = uf_idx < other ? uf_idx : other;
            int b = uf_idx < other ? other : uf_idx;
            snprintf(out, cap, "qual a distancia entre %s e %s?", UF_TABLE[a].capital, UF_TABLE[b].capital);
            return true;
        }
    }
    return false;
}

static bool try_rewrite_followup(const char *topic, const char *q, char *out, size_t cap)
{
    if (!topic || !topic[0] || !q || strlen(q) > 42u) return false;

    if (starts_with(topic, "bioma:")) {
        return try_rewrite_biome(topic, q, out, cap);
    }
    if (starts_with(topic, "uf:")) {
        return try_rewrite_uf(find_uf_from_topic(topic), q, out, cap);
    }

    for (size_t i = 0; FOLLOWUP_RULES[i].topic != NULL; i++) {
        const FollowupRule *r = &FOLLOWUP_RULES[i];
        if (strcmp(topic, r->topic) != 0) continue;
        if ((r->k1 && has_sub(q, r->k1)) ||
            (r->k2 && has_sub(q, r->k2)) ||
            (r->k3 && has_sub(q, r->k3))) {
            safe_copy(out, cap, r->canonical_q);
            return true;
        }
    }
    return false;
}

void tiny_sessions_init(void)
{
    memset(web_sessions, 0, sizeof(web_sessions));
    memset(&serial_session, 0, sizeof(serial_session));
    serial_session.active = true;
    safe_copy(serial_session.sid, sizeof(serial_session.sid), "serial");
}

void tiny_sanitize_sid(const char *raw_sid, const char *fallback_ip, char *sid_out)
{
    if (!sid_out) return;
    size_t n = 0;
    if (raw_sid) {
        for (size_t i = 0; raw_sid[i] && n < 15; i++) {
            char c = raw_sid[i];
            bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                      (c >= '0' && c <= '9') || c == '_' || c == '-' ||
                      c == '.' || c == ':';
            if (ok) sid_out[n++] = c;
        }
    }
    sid_out[n] = '\0';
    if (n > 0) return;

    if (fallback_ip && fallback_ip[0]) {
        safe_copy(sid_out, 16, fallback_ip);
        return;
    }
    safe_copy(sid_out, 16, "anon");
}

TinySession *tiny_session_get(const char *sid, uint32_t now_ms)
{
    if (sid && strcmp(sid, "serial") == 0) {
        serial_session.active = true;
        serial_session.last_seen_ms = now_ms;
        return &serial_session;
    }

    char clean_sid[16];
    tiny_sanitize_sid(sid, NULL, clean_sid);

    int free_idx = -1;
    int lru_idx = 0;
    uint32_t oldest_ms = 0;
    bool first_lru = true;

    for (int i = 0; i < TINY_MAX_SESSIONS; i++) {
        if (web_sessions[i].active &&
            (uint32_t)(now_ms - web_sessions[i].last_seen_ms) > TINY_SESSION_TTL_MS) {
            memset(&web_sessions[i], 0, sizeof(web_sessions[i]));
        }
        if (web_sessions[i].active) {
            if (strcmp(web_sessions[i].sid, clean_sid) == 0) {
                web_sessions[i].last_seen_ms = now_ms;
                return &web_sessions[i];
            }
            if (first_lru || (int32_t)(web_sessions[i].last_seen_ms - oldest_ms) < 0) {
                oldest_ms = web_sessions[i].last_seen_ms;
                lru_idx = i;
                first_lru = false;
            }
        } else if (free_idx < 0) {
            free_idx = i;
        }
    }

    int slot = (free_idx >= 0) ? free_idx : lru_idx;
    memset(&web_sessions[slot], 0, sizeof(web_sessions[slot]));
    web_sessions[slot].active = true;
    safe_copy(web_sessions[slot].sid, sizeof(web_sessions[slot].sid), clean_sid);
    web_sessions[slot].last_seen_ms = now_ms;
    return &web_sessions[slot];
}

void tiny_session_reset(TinySession *sess, uint32_t now_ms)
{
    if (!sess) return;
    char saved_sid[16];
    safe_copy(saved_sid, sizeof(saved_sid), sess->sid);
    memset(sess, 0, sizeof(*sess));
    sess->active = true;
    safe_copy(sess->sid, sizeof(sess->sid), saved_sid);
    sess->last_seen_ms = now_ms;
}

int tiny_sessions_active_count(uint32_t now_ms)
{
    int count = 0;
    for (int i = 0; i < TINY_MAX_SESSIONS; i++) {
        if (web_sessions[i].active &&
            (uint32_t)(now_ms - web_sessions[i].last_seen_ms) <= TINY_SESSION_TTL_MS) {
            count++;
        }
    }
    if (serial_session.turns > 0) count++;
    return count;
}

static bool is_reset_command(const char *q)
{
    if (!q) return false;
    return strcmp(q, "/reset") == 0 || strcmp(q, "/limpar") == 0 ||
           strcmp(q, "limpar memoria") == 0 || strcmp(q, "resetar memoria") == 0 ||
           strcmp(q, "esqueca tudo") == 0;
}

static void apply_user_name(TinySession *sess, const char *name, char *out, size_t cap)
{
    safe_copy(sess->user_name, sizeof(sess->user_name), name);
    sess->pending_ask = TINY_ASK_NONE;
    snprintf(out, cap, "Bem vindo ao meu Reino %s", sess->user_name);
}

static void apply_user_hw(TinySession *sess, const char *hw_val, char *out, size_t cap)
{
    safe_copy(sess->user_hw, sizeof(sess->user_hw), hw_val);
    sess->pending_ask = TINY_ASK_NONE;
    if (is_placeholder_hw(hw_val)) {
        safe_copy(out, cap, "Ual isso parece complexo.");
    } else {
        snprintf(out, cap, "Ual, %s parece complexo!", sess->user_hw);
    }
}

bool tiny_session_prepare(TinySession *sess,
                          const char *raw_line,
                          const char *clean_q,
                          char *model_q_out,
                          char *direct_ans_out,
                          bool *was_rewritten)
{
    (void)raw_line;
    if (was_rewritten) *was_rewritten = false;
    if (direct_ans_out) direct_ans_out[0] = '\0';
    safe_copy(model_q_out, TINY_MAXQ + 1, clean_q);
    if (!sess || !clean_q || !clean_q[0]) return false;

    /* 1. Reset session command */
    if (is_reset_command(clean_q)) {
        tiny_session_reset(sess, sess->last_seen_ms);
        snprintf(direct_ans_out, TINY_CTX + 1,
                 "Memoria desta sessao (%s) limpa! Podemos recomecar.", sess->sid);
        return true;
    }

    /* 2. Inspect session context command */
    if (strcmp(clean_q, "/memoria") == 0 || strcmp(clean_q, "/contexto") == 0 ||
        strcmp(clean_q, "qual o contexto?") == 0 || strcmp(clean_q, "sobre o que falamos?") == 0) {
        snprintf(direct_ans_out, TINY_CTX + 1,
                 "Sessao %s | Nome:%s | HW:%s | Tema:%s | %ut",
                 sess->sid,
                 sess->user_name[0] ? sess->user_name : "?",
                 sess->user_hw[0] ? sess->user_hw : "?",
                 sess->last_topic[0] ? sess->last_topic : "nenhum",
                 (unsigned)sess->turns);
        return true;
    }

    if (strcmp(clean_q, "qual foi minha ultima pergunta?") == 0 ||
        strcmp(clean_q, "o que eu perguntei antes?") == 0) {
        if (sess->last_q[0]) {
            snprintf(direct_ans_out, TINY_CTX + 1,
                     "Sua ultima pergunta foi: \"%s\".", sess->last_q);
        } else {
            safe_copy(direct_ans_out, TINY_CTX + 1, "Ainda nao houve pergunta anterior nesta sessao.");
        }
        return true;
    }

    /* 3. Recall queries (user name / hardware) */
    if (strcmp(clean_q, "qual o meu nome?") == 0 || strcmp(clean_q, "qual meu nome?") == 0 ||
        strcmp(clean_q, "qual e o meu nome?") == 0 || strcmp(clean_q, "quem sou eu?") == 0 ||
        strcmp(clean_q, "voce lembra meu nome?") == 0 || strcmp(clean_q, "lembra meu nome?") == 0 ||
        strcmp(clean_q, "sabe meu nome?") == 0) {
        if (sess->user_name[0]) {
            if (sess->user_hw[0]) {
                snprintf(direct_ans_out, TINY_CTX + 1,
                         "Voce e %s (usando %s)! Bem-vindo ao meu Reino.",
                         sess->user_name, sess->user_hw);
            } else {
                snprintf(direct_ans_out, TINY_CTX + 1,
                         "Voce e %s! Bem-vindo ao meu Reino.", sess->user_name);
            }
        } else {
            safe_copy(direct_ans_out, TINY_CTX + 1,
                      "Ainda nao sei seu nome nesta sessao. Quem e voce?");
        }
        return true;
    }

    if (strcmp(clean_q, "qual o meu hardware?") == 0 || strcmp(clean_q, "qual meu hardware?") == 0 ||
        strcmp(clean_q, "qual aparelho eu uso?") == 0) {
        if (sess->user_hw[0]) {
            snprintf(direct_ans_out, TINY_CTX + 1,
                     "Nesta sessao voce esta usando: %s.", sess->user_hw);
        } else {
            safe_copy(direct_ans_out, TINY_CTX + 1, "Ainda nao sei! Qual seu hardware?");
        }
        return true;
    }

    /* 4. Explicit user name declaration (works for any name) */
    if (strchr(clean_q, '?') == NULL) {
        static const char *const NAME_PREFIXES[] = {
            "meu nome e ", "eu me chamo ", "me chamo ", "pode me chamar de ",
            "eu sou o ", "eu sou a ", "sou o ", "sou a ", NULL
        };
        for (int i = 0; NAME_PREFIXES[i]; i++) {
            if (starts_with(clean_q, NAME_PREFIXES[i])) {
                char parsed[24];
                extract_name_after_prefix(clean_q, NAME_PREFIXES[i], parsed, sizeof(parsed));
                if (parsed[0] && !equals_ci(parsed, "ameba")) {
                    apply_user_name(sess, parsed, direct_ans_out, TINY_CTX + 1);
                    return true;
                }
            }
        }

        char known_name[24];
        if (match_known_kingdom_name(clean_q, known_name, sizeof(known_name))) {
            apply_user_name(sess, known_name, direct_ans_out, TINY_CTX + 1);
            return true;
        }
    }

    /* 5. Explicit hardware declaration */
    if (strchr(clean_q, '?') == NULL) {
        static const char *const HW_PREFIXES[] = {
            "meu hardware e ", "eu uso um ", "eu uso uma ", "uso um ",
            "uso uma ", "estou em um ", "estou num ", "estou no ", NULL
        };
        for (int i = 0; HW_PREFIXES[i]; i++) {
            if (starts_with(clean_q, HW_PREFIXES[i])) {
                char hw_val[28];
                extract_clean_phrase(hw_val, sizeof(hw_val), clean_q + strlen(HW_PREFIXES[i]));
                if (hw_val[0]) {
                    apply_user_hw(sess, hw_val, direct_ans_out, TINY_CTX + 1);
                    return true;
                }
            }
        }
    }

    /* 6. Reply to AMEBA's previous slot question ("Quem e voce?" / "Qual seu hardware?") */
    if (sess->pending_ask != TINY_ASK_NONE && !looks_like_question_or_command(clean_q)) {
        size_t len = strlen(clean_q);
        int words = count_words(clean_q);
        if (sess->pending_ask == TINY_ASK_WHO_ARE_YOU && len >= 2u && len <= 22u && words >= 1 && words <= 3) {
            char name_buf[24];
            title_case_clean(name_buf, sizeof(name_buf), clean_q);
            if (name_buf[0]) {
                apply_user_name(sess, name_buf, direct_ans_out, TINY_CTX + 1);
                return true;
            }
        } else if (sess->pending_ask == TINY_ASK_WHAT_HARDWARE && len >= 2u && len <= 26u && words >= 1 && words <= 4) {
            char hw_buf[28];
            extract_clean_phrase(hw_buf, sizeof(hw_buf), clean_q);
            if (hw_buf[0]) {
                apply_user_hw(sess, hw_buf, direct_ans_out, TINY_CTX + 1);
                return true;
            }
        }
    }

    /* 7. Personalized greetings once user name is known in this session */
    if (sess->user_name[0]) {
        if (strcmp(clean_q, "oi") == 0 || strcmp(clean_q, "oii") == 0 ||
            strcmp(clean_q, "oi!") == 0 || strcmp(clean_q, "ola") == 0 ||
            strcmp(clean_q, "ola!") == 0 || strcmp(clean_q, "opa") == 0 ||
            strcmp(clean_q, "e ai") == 0 || strcmp(clean_q, "e ai?") == 0 ||
            strcmp(clean_q, "eae") == 0 || strcmp(clean_q, "salve") == 0) {
            snprintf(direct_ans_out, TINY_CTX + 1,
                     "Ola, %s! Sou o AMEBA. Como posso ajudar?", sess->user_name);
            return true;
        }
        if (strcmp(clean_q, "bom dia") == 0 || strcmp(clean_q, "bom dia!") == 0) {
            snprintf(direct_ans_out, TINY_CTX + 1,
                     "Bom dia, %s! Em que posso ajudar?", sess->user_name);
            return true;
        }
        if (strcmp(clean_q, "boa tarde") == 0 || strcmp(clean_q, "boa tarde!") == 0) {
            snprintf(direct_ans_out, TINY_CTX + 1,
                     "Boa tarde, %s! Pode perguntar!", sess->user_name);
            return true;
        }
        if (strcmp(clean_q, "boa noite") == 0 || strcmp(clean_q, "boa noite!") == 0) {
            snprintf(direct_ans_out, TINY_CTX + 1,
                     "Boa noite, %s! Estou aqui no ESP32.", sess->user_name);
            return true;
        }
        if (strcmp(clean_q, "tchau") == 0 || strcmp(clean_q, "ate mais") == 0 ||
            strcmp(clean_q, "ate logo") == 0 || strcmp(clean_q, "falou") == 0) {
            snprintf(direct_ans_out, TINY_CTX + 1,
                     "Ate mais, %s! Quando precisar, estarei aqui.", sess->user_name);
            return true;
        }
    }

    /* 8. Known alias normalization + follow-up rewriting using per-session last_topic */
    if (strcmp(clean_q, "bateria caseira para ligar celular") == 0) {
        safe_copy(model_q_out, TINY_MAXQ + 1, "quantas pilhas caseiras dao 5v?");
        if (was_rewritten) *was_rewritten = true;
        return false;
    }

    const char *explicit_topic = detect_topic(clean_q);
    if (sess->last_topic[0] &&
        (explicit_topic == NULL || (starts_with(sess->last_topic, "uf:") && has_sub(clean_q, "distancia")))) {
        char rewritten_q[TINY_MAXQ + 1];
        if (try_rewrite_followup(sess->last_topic, clean_q, rewritten_q, sizeof(rewritten_q))) {
            safe_copy(model_q_out, TINY_MAXQ + 1, rewritten_q);
            if (was_rewritten) *was_rewritten = true;
            return false;
        }
    }

    return false;
}

void tiny_session_commit(TinySession *sess,
                         const char *clean_q,
                         const char *effective_q,
                         const char *ans,
                         uint32_t now_ms)
{
    if (!sess) return;
    sess->last_seen_ms = now_ms;
    if (is_reset_command(clean_q)) {
        return;
    }

    sess->turns++;
    if (clean_q) {
        safe_copy(sess->last_q, sizeof(sess->last_q), clean_q);
    }
    safe_copy(sess->last_ans, sizeof(sess->last_ans), ans);

    if (ans && (ends_with_ci(ans, "quem e voce?") || ends_with_ci(ans, "quem e voce?!"))) {
        sess->pending_ask = TINY_ASK_WHO_ARE_YOU;
    } else if (ans && ends_with_ci(ans, "qual seu hardware?")) {
        sess->pending_ask = TINY_ASK_WHAT_HARDWARE;
    } else {
        sess->pending_ask = TINY_ASK_NONE;
    }

    const char *t = detect_topic(effective_q);
    if (!t) t = detect_topic(clean_q);
    if (t) {
        safe_copy(sess->last_topic, sizeof(sess->last_topic), t);
    }
}
