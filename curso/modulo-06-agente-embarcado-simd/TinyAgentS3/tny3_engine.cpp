#include "tny3_engine.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifdef ARDUINO
#include <Arduino.h>
#include <esp_heap_caps.h>
#include <esp_timer.h>
#endif

typedef struct {
    const uint8_t *data;
    int rows;
    int cols;
    uint8_t quant_mode;
} Tny3Table;

static Tny3Info g_info;
static bool g_ready = false;
static Tny3MatvecMode g_matvec_mode = TNY3_MATVEC_INT8_MAC4;

static Tny3Table g_emb;
static Tny3Table g_wq[TNY3_MAX_LAYERS], g_wk[TNY3_MAX_LAYERS], g_wv[TNY3_MAX_LAYERS], g_wo[TNY3_MAX_LAYERS];
static Tny3Table g_wgate[TNY3_MAX_LAYERS], g_wup[TNY3_MAX_LAYERS], g_wdown[TNY3_MAX_LAYERS];
static const uint8_t *g_g1[TNY3_MAX_LAYERS], *g_g2[TNY3_MAX_LAYERS], *g_gf;

/* KV Cache GQA (reduzido em 4x frente a MHA) */
static int8_t *g_kc[TNY3_MAX_LAYERS], *g_vc[TNY3_MAX_LAYERS];
static float *g_ks[TNY3_MAX_LAYERS], *g_vs[TNY3_MAX_LAYERS];

/* Buffers de trabalho alinhados em 16 bytes para instruções vetoriais PIE 128-bit */
static float  __attribute__((aligned(16))) bx[TNY3_MAX_D];
static float  __attribute__((aligned(16))) bh[TNY3_MAX_D];
static float  __attribute__((aligned(16))) bq[TNY3_MAX_D];
static float  __attribute__((aligned(16))) bk[TNY3_MAX_D];
static float  __attribute__((aligned(16))) bv[TNY3_MAX_D];
static float  __attribute__((aligned(16))) ba[TNY3_MAX_D];
static float  __attribute__((aligned(16))) bt[TNY3_MAX_D];
static float  __attribute__((aligned(16))) bgate[TNY3_MAX_HID];
static float  __attribute__((aligned(16))) bup[TNY3_MAX_HID];
static float  __attribute__((aligned(16))) bs[TNY3_MAX_CTX];
static int8_t __attribute__((aligned(16))) bxq[TNY3_MAX_HID];

static float *g_rope_cos = NULL;
static float *g_rope_sin = NULL;

static inline float read_f32(const uint8_t *p) {
    float v;
    if ((((uintptr_t)p) & 3u) == 0u) {
#ifdef __GNUC__
        memcpy(&v, __builtin_assume_aligned(p, 4), sizeof(float));
#else
        memcpy(&v, p, sizeof(float));
#endif
    } else {
        memcpy(&v, p, sizeof(float));
    }
    return v;
}

static size_t table_byte_size(int rows, int cols, uint8_t qmode) {
    if (qmode == TNY3_QUANT_INT8) {
        size_t padded = (size_t)(((cols + 3) / 4) * 4);
        return (size_t)rows * (4u + padded);
    }
    size_t n_blocks = (size_t)((cols + 31) / 32);
    return (size_t)rows * n_blocks * 20u;
}

static bool map_table(const uint8_t *blob, size_t blob_size, size_t *offset,
                      Tny3Table *t, int rows, int cols, uint8_t qmode, const char **err_msg) {
    size_t bytes = table_byte_size(rows, cols, qmode);
    if (*offset + bytes > blob_size) {
        if (err_msg) *err_msg = "TNY3 truncado ao mapear tabela";
        return false;
    }
    t->data = blob + *offset;
    t->rows = rows;
    t->cols = cols;
    t->quant_mode = qmode;
    *offset += bytes;
    return true;
}

static const uint8_t *map_gains(const uint8_t *blob, size_t blob_size, size_t *offset,
                                int d, const char **err_msg) {
    size_t bytes = (size_t)d * sizeof(float);
    if (*offset + bytes > blob_size) {
        if (err_msg) *err_msg = "TNY3 truncado ao mapear ganhos RMSNorm";
        return NULL;
    }
    const uint8_t *ptr = blob + *offset;
    *offset += bytes;
    return ptr;
}

static void *alloc_kv_mem(size_t bytes, bool *used_psram) {
#ifdef ARDUINO
    if (ESP.getFreeHeap() < 220000u && psramFound()) {
        *used_psram = true;
        return ps_malloc(bytes);
    }
#endif
    *used_psram = false;
    return malloc(bytes);
}

void tny3_free(void) {
    for (int l = 0; l < TNY3_MAX_LAYERS; l++) {
        free(g_kc[l]); g_kc[l] = NULL;
        free(g_vc[l]); g_vc[l] = NULL;
        free(g_ks[l]); g_ks[l] = NULL;
        free(g_vs[l]); g_vs[l] = NULL;
    }
    free(g_rope_cos); g_rope_cos = NULL;
    free(g_rope_sin); g_rope_sin = NULL;
    g_ready = false;
}

bool tny3_init(const uint8_t *blob, size_t blob_size, const char **err_msg) {
    if (err_msg) *err_msg = NULL;
    tny3_free();

    if (!blob || blob_size < sizeof(Tny3Header)) {
        if (err_msg) *err_msg = "Blob TNY3 nulo ou menor que 32 bytes";
        return false;
    }
    memcpy(&g_info.hdr, blob, sizeof(Tny3Header));
    const Tny3Header *h = &g_info.hdr;

    if (memcmp(h->magic, "TNY3", 4) != 0 || h->version != 1u) {
        if (err_msg) *err_msg = "Header TNY3 inválido";
        return false;
    }
    if (h->d < 16u || h->d > TNY3_MAX_D || (h->d % 16u) != 0u ||
        h->layers < 1u || h->layers > TNY3_MAX_LAYERS ||
        h->q_heads < 1u || h->kv_heads < 1u || (h->q_heads % h->kv_heads) != 0u ||
        (h->d % h->q_heads) != 0u || h->hid > TNY3_MAX_HID ||
        h->ctx_len > TNY3_MAX_CTX || h->vocab_size > TNY3_MAX_VOCAB) {
        if (err_msg) *err_msg = "Dimensões TNY3 fora dos limites do motor";
        return false;
    }

    g_info.head_dim = (int)(h->d / h->q_heads);
    g_info.kv_dim = (int)(h->kv_heads * (uint16_t)g_info.head_dim);
    g_info.groups = (int)(h->q_heads / h->kv_heads);

    size_t off = sizeof(Tny3Header);
    if (!map_table(blob, blob_size, &off, &g_emb, (int)h->vocab_size, (int)h->d, TNY3_QUANT_INT8, err_msg)) {
        tny3_free();
        return false;
    }

    for (int l = 0; l < (int)h->layers; l++) {
        g_g1[l] = map_gains(blob, blob_size, &off, (int)h->d, err_msg);
        if (!g_g1[l] ||
            !map_table(blob, blob_size, &off, &g_wq[l], (int)h->d, (int)h->d, h->quant_mode, err_msg) ||
            !map_table(blob, blob_size, &off, &g_wk[l], g_info.kv_dim, (int)h->d, h->quant_mode, err_msg) ||
            !map_table(blob, blob_size, &off, &g_wv[l], g_info.kv_dim, (int)h->d, h->quant_mode, err_msg) ||
            !map_table(blob, blob_size, &off, &g_wo[l], (int)h->d, (int)h->d, h->quant_mode, err_msg)) {
            tny3_free();
            return false;
        }
        g_g2[l] = map_gains(blob, blob_size, &off, (int)h->d, err_msg);
        if (!g_g2[l] ||
            !map_table(blob, blob_size, &off, &g_wgate[l], (int)h->hid, (int)h->d, h->quant_mode, err_msg) ||
            !map_table(blob, blob_size, &off, &g_wup[l], (int)h->hid, (int)h->d, h->quant_mode, err_msg) ||
            !map_table(blob, blob_size, &off, &g_wdown[l], (int)h->d, (int)h->hid, h->quant_mode, err_msg)) {
            tny3_free();
            return false;
        }
    }
    g_gf = map_gains(blob, blob_size, &off, (int)h->d, err_msg);
    if (!g_gf || off != blob_size) {
        if (err_msg) *err_msg = "Tamanho do blob TNY3 inconsistente";
        tny3_free();
        return false;
    }

    int half_hd = g_info.head_dim / 2;
    size_t rope_elems = (size_t)h->ctx_len * (size_t)half_hd;
    g_rope_cos = (float *)malloc(rope_elems * sizeof(float));
    g_rope_sin = (float *)malloc(rope_elems * sizeof(float));
    if (!g_rope_cos || !g_rope_sin) {
        if (err_msg) *err_msg = "Sem memória para cache RoPE";
        tny3_free();
        return false;
    }
    for (int p = 0; p < (int)h->ctx_len; p++) {
        for (int i = 0; i < half_hd; i++) {
            float freq = 1.0f / powf(h->rope_theta, (float)(2 * i) / (float)g_info.head_dim);
            float ang = (float)p * freq;
            g_rope_cos[p * half_hd + i] = cosf(ang);
            g_rope_sin[p * half_hd + i] = sinf(ang);
        }
    }

    bool any_psram = false;
    for (int l = 0; l < (int)h->layers; l++) {
        bool ps = false;
        g_kc[l] = (int8_t *)alloc_kv_mem((size_t)h->ctx_len * (size_t)g_info.kv_dim, &ps);
        any_psram = any_psram || ps;
        g_vc[l] = (int8_t *)alloc_kv_mem((size_t)h->ctx_len * (size_t)g_info.kv_dim, &ps);
        g_ks[l] = (float *)alloc_kv_mem((size_t)h->ctx_len * sizeof(float), &ps);
        g_vs[l] = (float *)alloc_kv_mem((size_t)h->ctx_len * sizeof(float), &ps);
        if (!g_kc[l] || !g_vc[l] || !g_ks[l] || !g_vs[l]) {
            if (err_msg) *err_msg = "Heap insuficiente para KV Cache GQA";
            tny3_free();
            return false;
        }
    }

    g_info.using_psram = any_psram;
    g_info.kv_ram_bytes = (uint32_t)h->layers * (uint32_t)h->ctx_len * 2u *
                          ((uint32_t)g_info.kv_dim + (uint32_t)sizeof(float));
    tny3_reset_kv_cache();
    g_ready = true;
    return true;
}

const Tny3Info *tny3_get_info(void) { return g_ready ? &g_info : NULL; }
void tny3_set_matvec_mode(Tny3MatvecMode mode) { g_matvec_mode = mode; }
Tny3MatvecMode tny3_get_matvec_mode(void) { return g_matvec_mode; }

void tny3_reset_kv_cache(void) {
    if (!g_ready) return;
    for (int l = 0; l < (int)g_info.hdr.layers; l++) {
        memset(g_kc[l], 0, (size_t)g_info.hdr.ctx_len * (size_t)g_info.kv_dim);
        memset(g_vc[l], 0, (size_t)g_info.hdr.ctx_len * (size_t)g_info.kv_dim);
        memset(g_ks[l], 0, (size_t)g_info.hdr.ctx_len * sizeof(float));
        memset(g_vs[l], 0, (size_t)g_info.hdr.ctx_len * sizeof(float));
    }
}

static float quantize_vec_int8(const float *x, int cols, int8_t *xq) {
    float big = 0.0f;
    for (int i = 0; i < cols; i++) {
        float v = x[i] < 0.0f ? -x[i] : x[i];
        if (v > big) big = v;
    }
    if (big < 1e-20f) big = 1e-20f;
    const float k = 127.0f / big;
    for (int i = 0; i < cols; i++) {
        float v = x[i] * k;
        xq[i] = (int8_t)(v < 0.0f ? v - 0.5f : v + 0.5f);
    }
    return big / 127.0f;
}

static inline int32_t dotprod_s8_mac4(const int8_t *w, const int8_t *xq, int cols) {
    int32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    int c = 0;
    for (; c + 3 < cols; c += 4) {
        a0 += (int32_t)w[c]     * (int32_t)xq[c];
        a1 += (int32_t)w[c + 1] * (int32_t)xq[c + 1];
        a2 += (int32_t)w[c + 2] * (int32_t)xq[c + 2];
        a3 += (int32_t)w[c + 3] * (int32_t)xq[c + 3];
    }
    int32_t acc = (a0 + a1) + (a2 + a3);
    for (; c < cols; c++) acc += (int32_t)w[c] * (int32_t)xq[c];
    return acc;
}

/* Kernel SIMD 128-bit PIE do ESP32-S3 (16 MACs int8 por ciclo quando TNY3_USE_PIE_ASM ativo) */
static inline int32_t dotprod_s8_pie_128(const int8_t *w, const int8_t *xq, int cols) {
#if defined(CONFIG_IDF_TARGET_ESP32S3) && defined(TNY3_USE_PIE_ASM)
    int32_t acc = 0;
    int chunks16 = cols >> 4;
    if (chunks16 > 0 && ((((uintptr_t)w) & 15u) == 0u) && ((((uintptr_t)xq) & 15u) == 0u)) {
        const int8_t *pw = w;
        const int8_t *px = xq;
        __asm__ __volatile__(
            "wur.accx_0 %2\n"
            "wur.accx_1 %2\n"
            "loopgtz %3, 1f\n"
            "  ee.vld.128.ip q0, %0, 16\n"
            "  ee.vld.128.ip q1, %1, 16\n"
            "  ee.vmulas.s8.accx q0, q1\n"
            "1:\n"
            "rur.accx_0 %2\n"
            : "+r"(pw), "+r"(px), "+r"(acc)
            : "r"(chunks16)
        );
        return acc;
    }
#endif
    return dotprod_s8_mac4(w, xq, cols);
}

static void matvec_tny3(const Tny3Table *t, const float *x, float *y, Tny3MatvecMode mode) {
    const int rows = t->rows;
    const int cols = t->cols;

    if (t->quant_mode == TNY3_QUANT_Q4_0) {
        const int n_blocks = (cols + 31) / 32;
        const size_t stride = (size_t)n_blocks * 20u;
        const uint8_t *row_ptr = t->data;
        for (int r = 0; r < rows; r++) {
            float total = 0.0f;
            const uint8_t *blk = row_ptr;
            for (int b = 0; b < n_blocks; b++) {
                const float sc = read_f32(blk);
                const uint8_t *nib = blk + 4;
                int base = b * 32;
                int limit = cols - base;
                if (limit > 32) limit = 32;
                float bs_sum = 0.0f;
                for (int i = 0; i < limit; i += 2) {
                    uint8_t byte = nib[i >> 1];
                    int8_t q0 = (int8_t)(byte & 0x0Fu) - 8;
                    bs_sum += (float)q0 * x[base + i];
                    if (i + 1 < limit) {
                        int8_t q1 = (int8_t)((byte >> 4) & 0x0Fu) - 8;
                        bs_sum += (float)q1 * x[base + i + 1];
                    }
                }
                total += bs_sum * sc;
                blk += 20;
            }
            y[r] = total;
            row_ptr += stride;
        }
        return;
    }

    const size_t padded_cols = (size_t)(((cols + 3) / 4) * 4);
    const size_t stride = 4u + padded_cols;
    const uint8_t *row_ptr = t->data;

    if (mode == TNY3_MATVEC_FP32_SCALAR) {
        for (int r = 0; r < rows; r++) {
            const float sc = read_f32(row_ptr);
            const int8_t *w = (const int8_t *)(row_ptr + 4);
            float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
            int c = 0;
            for (; c + 3 < cols; c += 4) {
                s0 += (float)w[c]     * x[c];
                s1 += (float)w[c + 1] * x[c + 1];
                s2 += (float)w[c + 2] * x[c + 2];
                s3 += (float)w[c + 3] * x[c + 3];
            }
            float s = (s0 + s1) + (s2 + s3);
            for (; c < cols; c++) s += (float)w[c] * x[c];
            y[r] = s * sc;
            row_ptr += stride;
        }
        return;
    }

    const float x_scale = quantize_vec_int8(x, cols, bxq);
    for (int r = 0; r < rows; r++) {
        const float w_scale = read_f32(row_ptr);
        const int8_t *w = (const int8_t *)(row_ptr + 4);
        int32_t acc = (mode == TNY3_MATVEC_S3_PIE_SIMD)
                          ? dotprod_s8_pie_128(w, bxq, cols)
                          : dotprod_s8_mac4(w, bxq, cols);
        y[r] = (float)acc * (w_scale * x_scale);
        row_ptr += stride;
    }
}

static void rms_norm(const float *x, const uint8_t *g, int d, float *y) {
    float s = 0.0f;
    for (int i = 0; i < d; i++) s += x[i] * x[i];
    const float inv = 1.0f / sqrtf(s / (float)d + 1e-5f);
    for (int i = 0; i < d; i++) {
        y[i] = x[i] * inv * read_f32(g + (size_t)i * 4u);
    }
}

static void apply_rope_vec(float *vec, int n_heads, int hd, int pos) {
    int half = hd / 2;
    const float *c = g_rope_cos + (size_t)pos * (size_t)half;
    const float *s = g_rope_sin + (size_t)pos * (size_t)half;
    for (int h = 0; h < n_heads; h++) {
        float *vh = vec + h * hd;
        for (int i = 0; i < half; i++) {
            float x0 = vh[2 * i];
            float x1 = vh[2 * i + 1];
            vh[2 * i]     = x0 * c[i] - x1 * s[i];
            vh[2 * i + 1] = x0 * s[i] + x1 * c[i];
        }
    }
}

static inline float safe_exp(float x) {
    return (x < -60.0f) ? 0.0f : expf(x);
}

int tny3_step_greedy(int tok, int p, float *out_logits) {
    if (!g_ready || p < 0 || p >= (int)g_info.hdr.ctx_len) return 0;
    const int d = (int)g_info.hdr.d;
    const int hd = g_info.head_dim;
    const int kv_dim = g_info.kv_dim;
    const int q_heads = (int)g_info.hdr.q_heads;
    const int groups = g_info.groups;
    const float att_div = sqrtf((float)hd);

    const size_t emb_stride = 4u + (size_t)(((d + 3) / 4) * 4);
    const uint8_t *erow = g_emb.data + (size_t)tok * emb_stride;
    const float es = read_f32(erow);
    const int8_t *ew = (const int8_t *)(erow + 4);
    for (int i = 0; i < d; i++) bx[i] = (float)ew[i] * es;

    for (int l = 0; l < (int)g_info.hdr.layers; l++) {
        rms_norm(bx, g_g1[l], d, bh);
        matvec_tny3(&g_wq[l], bh, bq, g_matvec_mode);
        matvec_tny3(&g_wk[l], bh, bk, g_matvec_mode);
        matvec_tny3(&g_wv[l], bh, bv, g_matvec_mode);

        apply_rope_vec(bq, q_heads, hd, p);
        apply_rope_vec(bk, (int)g_info.hdr.kv_heads, hd, p);
        g_ks[l][p] = quantize_vec_int8(bk, kv_dim, g_kc[l] + (size_t)p * (size_t)kv_dim);
        g_vs[l][p] = quantize_vec_int8(bv, kv_dim, g_vc[l] + (size_t)p * (size_t)kv_dim);

        for (int qh = 0; qh < q_heads; qh++) {
            const int qoff = qh * hd;
            const int koff = (qh / groups) * hd;
            float big = -1e30f;
            for (int j = 0; j <= p; j++) {
                const int8_t *kp = g_kc[l] + (size_t)j * (size_t)kv_dim + koff;
                float s = 0.0f;
                for (int i = 0; i < hd; i++) s += bq[qoff + i] * (float)kp[i];
                s = (s * g_ks[l][j]) / att_div;
                bs[j] = s;
                if (s > big) big = s;
            }
            float sum = 0.0f;
            for (int j = 0; j <= p; j++) {
                bs[j] = safe_exp(bs[j] - big);
                sum += bs[j];
            }
            for (int i = 0; i < hd; i++) ba[qoff + i] = 0.0f;
            if (sum > 0.0f) {
                for (int j = 0; j <= p; j++) {
                    if (bs[j] == 0.0f) continue;
                    const int8_t *vp = g_vc[l] + (size_t)j * (size_t)kv_dim + koff;
                    const float w = bs[j] * g_vs[l][j] / sum;
                    for (int i = 0; i < hd; i++) ba[qoff + i] += w * (float)vp[i];
                }
            }
        }

        matvec_tny3(&g_wo[l], ba, bt, g_matvec_mode);
        for (int i = 0; i < d; i++) bx[i] += bt[i];

        rms_norm(bx, g_g2[l], d, bh);
        matvec_tny3(&g_wgate[l], bh, bgate, g_matvec_mode);
        matvec_tny3(&g_wup[l], bh, bup, g_matvec_mode);
        for (int i = 0; i < (int)g_info.hdr.hid; i++) {
            float gval = bgate[i];
            float silu = gval / (1.0f + expf(-gval));
            bgate[i] = silu * bup[i];
        }
        matvec_tny3(&g_wdown[l], bgate, bt, g_matvec_mode);
        for (int i = 0; i < d; i++) bx[i] += bt[i];
    }

    rms_norm(bx, g_gf, d, bh);
    matvec_tny3(&g_emb, bh, out_logits, g_matvec_mode);

    int best = 0;
    for (int i = 1; i < (int)g_info.hdr.vocab_size; i++) {
        if (out_logits[i] > out_logits[best]) best = i;
    }
    return best;
}

void tny3_benchmark_matvec(uint32_t *us_fp32, uint32_t *us_mac4, uint32_t *us_pie) {
#ifdef ARDUINO
    if (!g_ready) { *us_fp32 = *us_mac4 = *us_pie = 0; return; }
    for (int i = 0; i < (int)g_info.hdr.d; i++) bh[i] = 0.05f * (float)((i % 9) - 4);
    uint64_t t0 = esp_timer_get_time();
    matvec_tny3(&g_wgate[0], bh, bgate, TNY3_MATVEC_FP32_SCALAR);
    uint64_t t1 = esp_timer_get_time();
    matvec_tny3(&g_wgate[0], bh, bgate, TNY3_MATVEC_INT8_MAC4);
    uint64_t t2 = esp_timer_get_time();
    matvec_tny3(&g_wgate[0], bh, bgate, TNY3_MATVEC_S3_PIE_SIMD);
    uint64_t t3 = esp_timer_get_time();
    *us_fp32 = (uint32_t)(t1 - t0);
    *us_mac4 = (uint32_t)(t2 - t1);
    *us_pie  = (uint32_t)(t3 - t2);
#else
    (void)us_fp32; (void)us_mac4; (void)us_pie;
#endif
}
