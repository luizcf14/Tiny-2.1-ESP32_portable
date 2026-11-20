#ifndef TINY_TOKENIZER_H
#define TINY_TOKENIZER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TINY_TOK_PAD       0
#define TINY_TOK_BOS       1
#define TINY_TOK_EOS       2
#define TINY_TOK_USER      3
#define TINY_TOK_CTX       4
#define TINY_TOK_ACT       5
#define TINY_TOK_ASSISTANT 6
#define TINY_TOK_RESERVED  16
#define TINY_TOK_MAX_SEQ   128

static inline bool tiny_is_word_byte(uint8_t b) {
    return (b >= 'a' && b <= 'z') || (b >= 'A' && b <= 'Z') || b == '_' || b >= 0x80u;
}

/* Busca binária em TINY_BPE_MERGES (ordenado por (left, right)) */
static int tiny_find_merge(const TinyBpeMerge *merges, int num_merges, uint16_t left, uint16_t right,
                           uint16_t *merged_out, uint16_t *rank_out) {
    int lo = 0;
    int hi = num_merges - 1;
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        uint16_t ml = merges[mid].left;
        uint16_t mr = merges[mid].right;
        if (ml == left && mr == right) {
            *merged_out = merges[mid].merged;
            *rank_out = merges[mid].rank;
            return 1;
        }
        if (ml < left || (ml == left && mr < right)) {
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }
    return 0;
}

static int tiny_match_special(const uint8_t *raw, size_t rem, uint16_t *tok_out, size_t *len_out) {
    static const char *const TAGS[7] = {
        "<|pad|>", "<|bos|>", "<|eos|>", "<|user|>", "<|ctx|>", "<|act|>", "<|assistant|>"
    };
    if (rem < 7 || raw[0] != '<' || raw[1] != '|') return 0;
    for (int i = 0; i < 7; i++) {
        size_t tl = 0;
        while (TAGS[i][tl]) tl++;
        if (rem >= tl) {
            bool eq = true;
            for (size_t k = 0; k < tl; k++) {
                if (raw[k] != (uint8_t)TAGS[i][k]) { eq = false; break; }
            }
            if (eq) {
                *tok_out = (uint16_t)i;
                *len_out = tl;
                return 1;
            }
        }
    }
    return 0;
}

/* Codifica UTF-8 em IDs BPE sem usar malloc() */
static size_t tiny_bpe_encode(const char *utf8_text,
                              uint16_t *out_tokens,
                              size_t max_tokens,
                              const TinyBpeMerge *merges,
                              int num_merges) {
    if (!utf8_text || !out_tokens || max_tokens == 0) return 0;
    const uint8_t *raw = (const uint8_t *)utf8_text;
    size_t n = 0;
    while (raw[n]) n++;

    size_t out_n = 0;
    size_t i = 0;
    uint16_t chunk[TINY_TOK_MAX_SEQ];

    while (i < n && out_n < max_tokens) {
        uint16_t stag = 0;
        size_t slen = 0;
        if (tiny_match_special(raw + i, n - i, &stag, &slen)) {
            out_tokens[out_n++] = stag;
            i += slen;
            continue;
        }

        size_t j = i;
        if (raw[i] == 0x20u && i + 1 < n && tiny_is_word_byte(raw[i + 1])) {
            j = i + 2;
            while (j < n && tiny_is_word_byte(raw[j])) j++;
        } else if (tiny_is_word_byte(raw[i])) {
            j = i + 1;
            while (j < n && tiny_is_word_byte(raw[j])) j++;
        } else {
            j = i + 1;
        }

        size_t clen = j - i;
        if (clen > TINY_TOK_MAX_SEQ) clen = TINY_TOK_MAX_SEQ;
        for (size_t k = 0; k < clen; k++) {
            chunk[k] = (uint16_t)(raw[i + k] + TINY_TOK_RESERVED);
        }

        while (clen >= 2) {
            uint16_t best_rank = 0xFFFFu;
            uint16_t best_l = 0, best_r = 0, best_m = 0;
            bool found = false;
            for (size_t k = 0; k + 1 < clen; k++) {
                uint16_t m = 0, rk = 0;
                if (tiny_find_merge(merges, num_merges, chunk[k], chunk[k + 1], &m, &rk)) {
                    if (!found || rk < best_rank) {
                        best_rank = rk;
                        best_l = chunk[k];
                        best_r = chunk[k + 1];
                        best_m = m;
                        found = true;
                    }
                }
            }
            if (!found) break;

            size_t w = 0;
            for (size_t k = 0; k < clen;) {
                if (k + 1 < clen && chunk[k] == best_l && chunk[k + 1] == best_r) {
                    chunk[w++] = best_m;
                    k += 2;
                } else {
                    chunk[w++] = chunk[k++];
                }
            }
            clen = w;
        }

        for (size_t k = 0; k < clen && out_n < max_tokens; k++) {
            out_tokens[out_n++] = chunk[k];
        }
        i = j;
    }
    return out_n;
}

/* Buffer seguro para montar codepoints UTF-8 completos durante streaming SSE */
typedef struct {
    uint8_t buf[4];
    uint8_t len;
    uint8_t expected;
} TinyUtf8Stream;

static inline void tiny_utf8_init(TinyUtf8Stream *st) {
    st->len = 0;
    st->expected = 0;
}

static size_t tiny_utf8_feed(TinyUtf8Stream *st, uint8_t b, char *out_utf8) {
    if (st->len == 0) {
        if (b < 0x80u) {
            out_utf8[0] = (char)b;
            out_utf8[1] = '\0';
            return 1;
        }
        if ((b & 0xE0u) == 0xC0u) st->expected = 2;
        else if ((b & 0xF0u) == 0xE0u) st->expected = 3;
        else if ((b & 0xF8u) == 0xF0u) st->expected = 4;
        else return 0;
        st->buf[0] = b;
        st->len = 1;
        return 0;
    }
    st->buf[st->len++] = b;
    if (st->len >= st->expected) {
        size_t n = st->len;
        for (size_t k = 0; k < n; k++) out_utf8[k] = (char)st->buf[k];
        out_utf8[n] = '\0';
        st->len = 0;
        st->expected = 0;
        return n;
    }
    return 0;
}

#ifdef __cplusplus
}
#endif

#endif /* TINY_TOKENIZER_H */
