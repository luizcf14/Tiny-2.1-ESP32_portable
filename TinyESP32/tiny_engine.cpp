#include "tiny_engine.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifdef ARDUINO
#include <Arduino.h>
#else
#include <time.h>
#endif

typedef struct {
    const uint8_t *data;
    int rows;
    int cols;
} TinyTable;

static TinyModelInfo model_info;
static bool model_ready = false;

static int D, L, H, HD, HID;
static float att_div;

static TinyTable emb, pos;
static TinyTable wq[TINY_MAXL], wk[TINY_MAXL], wv[TINY_MAXL], wo[TINY_MAXL];
static TinyTable f1[TINY_MAXL], f2[TINY_MAXL];
static const uint8_t *g1[TINY_MAXL], *g2[TINY_MAXL], *gf;

/* KV cache allocated on heap in tiny_init() so it does not overflow dram0_0_seg .bss */
static int8_t *kc[TINY_MAXL], *vc[TINY_MAXL];
static float *ks[TINY_MAXL], *vs[TINY_MAXL];

/* Small working buffers (~5.5 KB total) */
static float bx[TINY_MAXD], bh[TINY_MAXD], bq[TINY_MAXD], bt[TINY_MAXD], ba[TINY_MAXD];
static float bm[TINY_MAXHID], bs[TINY_CTX], score[TINY_VOCAB];
static float layer_act[TINY_MAXL];
static float last_conf = 1.0f;

static inline uint16_t read_u16(const uint8_t *p)
{
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

static inline float read_f32(const uint8_t *p)
{
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

static bool map_table(const uint8_t *blob, size_t blob_size, size_t *offset,
                      TinyTable *t, int rows, int cols, const char **err_msg)
{
    size_t stride = 4u + (size_t)cols;
    size_t bytes = (size_t)rows * stride;
    if (*offset + bytes > blob_size) {
        if (err_msg) *err_msg = "TINY.BIN truncated while reading table";
        return false;
    }
    t->data = blob + *offset;
    t->rows = rows;
    t->cols = cols;
    *offset += bytes;
    return true;
}

static const uint8_t *map_gains(const uint8_t *blob, size_t blob_size, size_t *offset,
                                const char **err_msg)
{
    size_t bytes = (size_t)D * sizeof(float);
    if (*offset + bytes > blob_size) {
        if (err_msg) *err_msg = "TINY.BIN truncated while reading gains";
        return NULL;
    }
    const uint8_t *ptr = blob + *offset;
    *offset += bytes;
    return ptr;
}

static void *tiny_alloc_kv(size_t bytes)
{
#ifdef ARDUINO
    if (ESP.getFreeHeap() < 180000u && psramFound()) {
        return ps_malloc(bytes);
    }
#endif
    return malloc(bytes);
}

void tiny_free(void)
{
    for (int l = 0; l < TINY_MAXL; l++) {
        free(kc[l]); kc[l] = NULL;
        free(vc[l]); vc[l] = NULL;
        free(ks[l]); ks[l] = NULL;
        free(vs[l]); vs[l] = NULL;
    }
    model_ready = false;
}

bool tiny_init(const uint8_t *blob, size_t blob_size, const char **err_msg)
{
    if (err_msg) *err_msg = NULL;
    tiny_free();

    if (!blob || blob_size < 12u) {
        if (err_msg) *err_msg = "Model blob is null or too short";
        return false;
    }
    if (memcmp(blob, "TNY2", 4) != 0) {
        if (err_msg) *err_msg = "Invalid model magic (expected TNY2)";
        return false;
    }

    D   = (int)read_u16(blob + 4);
    L   = (int)read_u16(blob + 6);
    H   = (int)read_u16(blob + 8);
    HID = (int)read_u16(blob + 10);

    if (D < 8 || D > TINY_MAXD || L < 1 || L > TINY_MAXL ||
        H < 1 || D % H != 0 || HID < 8 || HID > TINY_MAXHID) {
        if (err_msg) *err_msg = "Unsupported model dimensions";
        return false;
    }

    HD = D / H;
    att_div = sqrtf((float)HD);

    size_t offset = 12u;
    if (!map_table(blob, blob_size, &offset, &emb, TINY_VOCAB, D, err_msg)) {
        tiny_free();
        return false;
    }
    if (!map_table(blob, blob_size, &offset, &pos, TINY_CTX, D, err_msg)) {
        tiny_free();
        return false;
    }

    for (int l = 0; l < L; l++) {
        g1[l] = map_gains(blob, blob_size, &offset, err_msg);
        if (!g1[l] ||
            !map_table(blob, blob_size, &offset, &wq[l], D, D, err_msg) ||
            !map_table(blob, blob_size, &offset, &wk[l], D, D, err_msg) ||
            !map_table(blob, blob_size, &offset, &wv[l], D, D, err_msg) ||
            !map_table(blob, blob_size, &offset, &wo[l], D, D, err_msg)) {
            tiny_free();
            return false;
        }
        g2[l] = map_gains(blob, blob_size, &offset, err_msg);
        if (!g2[l] ||
            !map_table(blob, blob_size, &offset, &f1[l], HID, D, err_msg) ||
            !map_table(blob, blob_size, &offset, &f2[l], D, HID, err_msg)) {
            tiny_free();
            return false;
        }
    }

    gf = map_gains(blob, blob_size, &offset, err_msg);
    if (!gf) {
        tiny_free();
        return false;
    }
    if (offset != blob_size) {
        if (err_msg) *err_msg = "Model blob size mismatch";
        tiny_free();
        return false;
    }

    for (int l = 0; l < L; l++) {
        kc[l] = (int8_t *)tiny_alloc_kv((size_t)TINY_CTX * (size_t)D);
        vc[l] = (int8_t *)tiny_alloc_kv((size_t)TINY_CTX * (size_t)D);
        ks[l] = (float *)tiny_alloc_kv((size_t)TINY_CTX * sizeof(float));
        vs[l] = (float *)tiny_alloc_kv((size_t)TINY_CTX * sizeof(float));
        if (!kc[l] || !vc[l] || !ks[l] || !vs[l]) {
            if (err_msg) *err_msg = "Out of heap memory allocating KV cache";
            tiny_free();
            return false;
        }
    }

    uint32_t kv_bytes = (uint32_t)L * (2u * (uint32_t)TINY_CTX * (uint32_t)D +
                                       2u * (uint32_t)TINY_CTX * (uint32_t)sizeof(float));
    uint32_t buf_bytes = (uint32_t)(sizeof(bx) + sizeof(bh) + sizeof(bq) + sizeof(bt) +
                                    sizeof(ba) + sizeof(bm) + sizeof(bs) + sizeof(score) +
                                    sizeof(layer_act));

    model_info.d = D;
    model_info.layers = L;
    model_info.heads = H;
    model_info.head_dim = HD;
    model_info.hidden = HID;
    model_info.ram_bytes = kv_bytes + buf_bytes;
    model_info.flash_bytes = (uint32_t)blob_size;
    model_ready = true;
    return true;
}

const TinyModelInfo *tiny_get_info(void)
{
    return model_ready ? &model_info : NULL;
}

static void matvec(const TinyTable *t, const float *x, float *y)
{
    const int rows = t->rows;
    const int cols = t->cols;
    const size_t stride = 4u + (size_t)cols;
    const uint8_t *row_ptr = t->data;

    for (int r = 0; r < rows; r++) {
        const float scale = read_f32(row_ptr);
        const int8_t *w = (const int8_t *)(row_ptr + 4);
        float s0 = 0.0f;
        float s1 = 0.0f;
        float s2 = 0.0f;
        float s3 = 0.0f;
        int c = 0;
        for (; c + 3 < cols; c += 4) {
            s0 += (float)w[c]     * x[c];
            s1 += (float)w[c + 1] * x[c + 1];
            s2 += (float)w[c + 2] * x[c + 2];
            s3 += (float)w[c + 3] * x[c + 3];
        }
        float s = (s0 + s1) + (s2 + s3);
        for (; c < cols; c++) {
            s += (float)w[c] * x[c];
        }
        y[r] = s * scale;
        row_ptr += stride;
    }
}

static void norm(const float *x, const uint8_t *g, float *y)
{
    float s = 0.0f;
    for (int i = 0; i < D; i++) {
        s += x[i] * x[i];
    }
    const float inv = 1.0f / sqrtf(s / (float)D + 1e-5f);
    for (int i = 0; i < D; i++) {
        y[i] = x[i] * inv * read_f32(g + (size_t)i * 4u);
    }
}

static inline float safe_exp(float x)
{
    if (x < -60.0f) return 0.0f;
    return expf(x);
}

static float store(int8_t *dst, const float *src)
{
    float big = 0.0f;
    for (int i = 0; i < D; i++) {
        float v = src[i] < 0.0f ? -src[i] : src[i];
        if (v > big) big = v;
    }
    if (big < 1e-20f) big = 1e-20f;
    const float k = 127.0f / big;
    for (int i = 0; i < D; i++) {
        float v = src[i] * k;
        dst[i] = (int8_t)(v < 0.0f ? v - 0.5f : v + 0.5f);
    }
    return big / 127.0f;
}

static void step(int tok, int p, float *out_score)
{
    if (tok < 0 || tok >= TINY_VOCAB) tok = 0;
    if (p < 0 || p >= TINY_CTX) return;

    const size_t emb_stride = 4u + (size_t)D;
    const uint8_t *erow = emb.data + (size_t)tok * emb_stride;
    const uint8_t *prow = pos.data + (size_t)p * emb_stride;
    const float es = read_f32(erow);
    const float ps = read_f32(prow);
    const int8_t *e = (const int8_t *)(erow + 4);
    const int8_t *q = (const int8_t *)(prow + 4);

    for (int i = 0; i < D; i++) {
        bx[i] = (float)e[i] * es + (float)q[i] * ps;
    }

    for (int l = 0; l < L; l++) {
        norm(bx, g1[l], bh);
        matvec(&wk[l], bh, bt);
        ks[l][p] = store(kc[l] + (size_t)p * (size_t)D, bt);
        matvec(&wv[l], bh, bt);
        vs[l][p] = store(vc[l] + (size_t)p * (size_t)D, bt);
        matvec(&wq[l], bh, bq);

        for (int h = 0; h < H; h++) {
            const int hoff = h * HD;
            float big = -1e30f;
            for (int j = 0; j <= p; j++) {
                const int8_t *kp = kc[l] + (size_t)j * (size_t)D + hoff;
                float s = 0.0f;
                for (int i = 0; i < HD; i++) {
                    s += bq[hoff + i] * (float)kp[i];
                }
                s = (s * ks[l][j]) / att_div;
                bs[j] = s;
                if (s > big) big = s;
            }
            float sum = 0.0f;
            for (int j = 0; j <= p; j++) {
                bs[j] = safe_exp(bs[j] - big);
                sum += bs[j];
            }
            for (int i = 0; i < HD; i++) {
                ba[hoff + i] = 0.0f;
            }
            if (sum > 0.0f) {
                for (int j = 0; j <= p; j++) {
                    if (bs[j] == 0.0f) continue;
                    const int8_t *vp = vc[l] + (size_t)j * (size_t)D + hoff;
                    const float w = bs[j] * vs[l][j] / sum;
                    for (int i = 0; i < HD; i++) {
                        ba[hoff + i] += w * (float)vp[i];
                    }
                }
            }
        }

        matvec(&wo[l], ba, bt);
        for (int i = 0; i < D; i++) {
            bx[i] += bt[i];
        }

        norm(bx, g2[l], bh);
        matvec(&f1[l], bh, bm);
        float s = 0.0f;
        for (int i = 0; i < HID; i++) {
            if (bm[i] < 0.0f) {
                bm[i] = 0.0f;
            } else {
                s += bm[i];
            }
        }
        layer_act[l] = s / (float)HID;
        matvec(&f2[l], bm, bt);
        for (int i = 0; i < D; i++) {
            bx[i] += bt[i];
        }
    }

    norm(bx, gf, bh);
    matvec(&emb, bh, out_score);
}

static int chosen(int tok)
{
    float sum = 0.0f;
    for (int i = 0; i < TINY_VOCAB; i++) {
        sum += safe_exp(score[i] - score[tok]);
    }
    last_conf = (sum > 0.0f) ? (1.0f / sum) : 1.0f;
    return tok;
}

static int pick(int creativity)
{
    int best = 0;
    for (int i = 1; i < TINY_VOCAB; i++) {
        if (score[i] > score[best]) best = i;
    }
    if (creativity <= 0) {
        return chosen(best);
    }
    if (creativity > 9) creativity = 9;

    const float t = (float)creativity / 5.0f;
    float sum = 0.0f;
    for (int i = 0; i < TINY_VOCAB; i++) {
        float w = safe_exp((score[i] - score[best]) / t);
        if (w < 0.22f) w = 0.0f;
        bs[i] = w;
        sum += w;
    }
    float r = sum * (float)(rand() % 10000) / 10000.0f;
    for (int i = 0; i < TINY_VOCAB; i++) {
        r -= bs[i];
        if (r < 0.0f && bs[i] > 0.0f) {
            return chosen(i);
        }
    }
    return chosen(best);
}

int tiny_clean_question(const char *line, char *q_out)
{
    static const char latin1_ascii[64] = {
        'A','A','A','A','A','A','A','C','E','E','E','E','I','I','I','I',
        'D','N','O','O','O','O','O','x','O','U','U','U','U','Y', 0 ,'s',
        'a','a','a','a','a','a','a','c','e','e','e','e','i','i','i','i',
        'd','n','o','o','o','o','o','/','o','u','u','u','u','y', 0 ,'y'
    };
    if (!line || !q_out) return 0;
    int n = 0;
    for (int i = 0; line[i] && n < TINY_MAXQ; i++) {
        unsigned char c = (unsigned char)line[i];
        char ch = 0;
        if (c == 0xC3u && (unsigned char)line[i + 1] >= 0x80u && (unsigned char)line[i + 1] <= 0xBFu) {
            ch = latin1_ascii[(unsigned char)line[i + 1] - 0x80u];
            i++;
        } else if (c >= 32u && c <= 126u) {
            ch = (char)c;
        }
        if (!ch) continue;
        if (ch >= 'A' && ch <= 'Z') {
            ch = (char)(ch + ('a' - 'A'));
        }
        if (ch >= 'a' && ch <= 'z' && n >= 2 && q_out[n - 1] == ch && q_out[n - 2] == ch) {
            continue;
        }
        q_out[n++] = ch;
    }
    while (n > 0 && q_out[n - 1] == ' ') {
        n--;
    }
    q_out[n] = '\0';
    return n;
}

int tiny_think(const char *clean_q,
               char *ans_out,
               int creativity,
               uint32_t *elapsed_ms,
               TinyStepFn on_step,
               TinyLetterFn on_letter,
               void *user)
{
    if (!model_ready || !clean_q || !ans_out) {
        if (ans_out) ans_out[0] = '\0';
        if (elapsed_ms) *elapsed_ms = 0;
        return 0;
    }

#ifdef ARDUINO
    const uint32_t t0 = millis();
#else
    const clock_t t0 = clock();
#endif

    int n = (int)strlen(clean_q);
    if (n > TINY_MAXQ) n = TINY_MAXQ;

    int p = 0;
    for (; p < n; p++) {
        int qtok = (int)(unsigned char)clean_q[p] - 30;
        if (qtok < 2 || qtok >= TINY_VOCAB) qtok = 2;
        step(qtok, p, score);
        if (on_step) {
            on_step(p, n, layer_act, L, user);
        }
#ifdef ARDUINO
        yield();
#endif
    }

    step(1, p++, score);
#ifdef ARDUINO
    yield();
#endif

    int letters = 0;
    for (; p < TINY_CTX; p++) {
        const int tok = pick(creativity);
        if (tok <= 1) break;
        const char ch = (char)(tok + 30);
        ans_out[letters++] = ch;
        if (on_letter) {
            on_letter(ch, last_conf, layer_act, L, user);
        }
        step(tok, p, score);
#ifdef ARDUINO
        yield();
#endif
    }
    ans_out[letters] = '\0';

    if (elapsed_ms) {
#ifdef ARDUINO
        *elapsed_ms = (uint32_t)(millis() - t0);
#else
        *elapsed_ms = (uint32_t)(((float)(clock() - t0) * 1000.0f) / (float)CLOCKS_PER_SEC);
#endif
    }
    return letters;
}
