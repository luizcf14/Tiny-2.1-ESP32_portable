#include "tiny_rag.h"

#include <ctype.h>
#include <stdio.h>
#include <string.h>

static const uint8_t *g_rag_blob = NULL;
static uint16_t g_rag_docs = 0;

typedef struct {
    const char *kw;
    const char *snippet;
} FallbackFicha;

static const FallbackFicha FALLBACK_FICHAS[] = {
    {"convuls", "[RAG] Proteja a cabeca, vire de lado, nunca segure a lingua. Ligue 192 se passar de 5 min."},
    {"afog",    "[RAG] Retire sem entrar na correnteza. Sem respiracao inicie compressoes e ligue 192."},
    {"oleo",    "[RAG] Nunca jogue agua no oleo quente. Desligue o fogo e abafe com tampa metalica."},
    {"agua",    "[RAG] Filtre agua turva e ferva por 1 min ou use 2 gotas de agua sanitaria por litro (30 min)."},
    {NULL, NULL}
};

static void ascii_lower(const char *src, char *dst, size_t cap) {
    size_t n = 0;
    for (size_t i = 0; src && src[i] && n + 1 < cap; i++) {
        unsigned char c = (unsigned char)src[i];
        if (c == 0xC3u && src[i + 1]) {
            unsigned char n2 = (unsigned char)src[i + 1];
            i++;
            if (n2 >= 0x80u && n2 <= 0x85u) c = 'a';
            else if (n2 >= 0xA0u && n2 <= 0xA5u) c = 'a';
            else if (n2 == 0x87u || n2 == 0xA7u) c = 'c';
            else if (n2 >= 0x88u && n2 <= 0x8Bu) c = 'e';
            else if (n2 >= 0xA8u && n2 <= 0xABu) c = 'e';
            else if (n2 >= 0x92u && n2 <= 0x96u) c = 'o';
            else if (n2 >= 0xB2u && n2 <= 0xB6u) c = 'o';
            else if (n2 >= 0x99u && n2 <= 0x9Cu) c = 'u';
            else if (n2 >= 0xB9u && n2 <= 0xBCu) c = 'u';
            else c = ' ';
        } else if (c >= 'A' && c <= 'Z') {
            c = (unsigned char)(c + ('a' - 'A'));
        } else if (!isalnum(c)) {
            c = ' ';
        }
        dst[n++] = (char)c;
    }
    dst[n] = '\0';
}

bool tiny_rag_init_from_blob(const uint8_t *blob, size_t blob_size) {
    if (!blob || blob_size < sizeof(Rag1Header)) return false;
    Rag1Header hdr;
    memcpy(&hdr, blob, sizeof(Rag1Header));
    if (memcmp(hdr.magic, "RAG1", 4) != 0) return false;
    size_t min_bytes = sizeof(Rag1Header) + (size_t)hdr.num_docs * sizeof(Rag1Entry);
    if (blob_size < min_bytes) return false;
    g_rag_blob = blob;
    g_rag_docs = hdr.num_docs;
    return true;
}

int tiny_rag_search(const char *query_utf8, char *snippet_out, size_t cap) {
    if (!snippet_out || cap == 0) return 0;
    snippet_out[0] = '\0';
    if (!query_utf8) return 0;

    char qnorm[160];
    ascii_lower(query_utf8, qnorm, sizeof(qnorm));

    if (g_rag_blob && g_rag_docs > 0) {
        size_t table_off = sizeof(Rag1Header);
        size_t pool_off = table_off + (size_t)g_rag_docs * sizeof(Rag1Entry);
        int best_idx = -1;
        int best_score = 0;

        for (uint16_t i = 0; i < g_rag_docs; i++) {
            Rag1Entry ent;
            memcpy(&ent, g_rag_blob + table_off + (size_t)i * sizeof(Rag1Entry), sizeof(Rag1Entry));
            char kw[160];
            size_t kl = ent.kw_len < sizeof(kw) - 1u ? ent.kw_len : sizeof(kw) - 1u;
            memcpy(kw, g_rag_blob + pool_off + ent.kw_off, kl);
            kw[kl] = '\0';

            int score = 0;
            char qcopy[160];
            snprintf(qcopy, sizeof(qcopy), "%s", qnorm);
            char *p = qcopy;
            while (*p) {
                while (*p == ' ') p++;
                if (!*p) break;
                char *end = p;
                while (*end && *end != ' ') end++;
                if (*end) *end++ = '\0';
                if (strlen(p) >= 3u && strstr(kw, p) != NULL) score += 3;
                p = end;
            }
            if (score > best_score) {
                best_score = score;
                best_idx = (int)i;
            }
        }

        if (best_idx >= 0 && best_score >= 3) {
            Rag1Entry best;
            memcpy(&best, g_rag_blob + table_off + (size_t)best_idx * sizeof(Rag1Entry), sizeof(Rag1Entry));
            char body[240];
            size_t bl = best.body_len < sizeof(body) - 1u ? best.body_len : sizeof(body) - 1u;
            memcpy(body, g_rag_blob + pool_off + best.body_off, bl);
            body[bl] = '\0';
            snprintf(snippet_out, cap, "[RAG] %s", body);
            return best_score;
        }
        return 0;
    }

    for (int i = 0; FALLBACK_FICHAS[i].kw; i++) {
        if (strstr(qnorm, FALLBACK_FICHAS[i].kw) != NULL) {
            snprintf(snippet_out, cap, "%s", FALLBACK_FICHAS[i].snippet);
            return 3;
        }
    }
    return 0;
}
