/*
 * Módulo 2 (Lab 3 - Hardware): Profiling em Microssegundos do step() no ESP32-S3-WROOM-1 (16MB)
 * ==============================================================================================
 * Mede quanto tempo gastam cada etapa de um Transformer (D=96, L=4, H=4, HID=256) direto no
 * silício Xtensa LX7 com esp_timer_get_time(), e compara FPU float32 vs emulação double.
 *
 * Arduino IDE -> Tools:
 *   Board: ESP32S3 Dev Module | USB CDC On Boot: Enabled | CPU Freq: 240MHz | Flash Size: 16MB
 */

#include <Arduino.h>
#include <esp_timer.h>
#include <math.h>
#include <pgmspace.h>
#include <stdarg.h>
#include <stdint.h>

#if defined(ARDUINO_USB_CDC_ON_BOOT) && ARDUINO_USB_CDC_ON_BOOT
#define PROF_DUAL_SERIAL 1
#else
#define PROF_DUAL_SERIAL 0
#endif

static const int P_VOCAB = 97;
static const int P_CTX   = 128;
static const int P_D     = 96;
static const int P_L     = 4;
static const int P_H     = 4;
static const int P_HD    = 24;
static const int P_HID   = 256;

/* Pesos sintéticos alinhados em Flash (PROGMEM) para medir custo real de barramento/cache/FPU */
static const int8_t PROGMEM W_SQUARE[P_D * P_D] = { 17, -23, 31, -9, 12, -41, 7, 19 };
static const int8_t PROGMEM W_FFN1[P_HID * P_D] = { -11, 25, -7, 14, -33, 8, 19, -5 };
static const int8_t PROGMEM W_FFN2[P_D * P_HID] = { 9, -18, 22, -6, 15, -27, 4, 13 };
static const int8_t PROGMEM W_EMB[P_VOCAB * P_D] = { 13, -19, 27, -8, 16, -29, 5, 21 };

static float bx[P_D], bh[P_D], bq[P_D], bt[P_D], ba[P_D], bm[P_HID], bs[P_CTX], logits[P_VOCAB];
static int8_t kc[P_L][P_CTX][P_D];
static int8_t vc[P_L][P_CTX][P_D];
static float ks[P_L][P_CTX], vs[P_L][P_CTX];
static float gains[P_D];

static void conPrintf(const char *fmt, ...) {
  char buf[256];
  va_list args;
  va_start(args, fmt);
  int n = vsnprintf(buf, sizeof(buf), fmt, args);
  va_end(args);
  if (n <= 0) return;
  size_t len = (size_t)n < sizeof(buf) ? (size_t)n : sizeof(buf) - 1u;
  Serial.write((const uint8_t *)buf, len);
#if PROF_DUAL_SERIAL
  Serial0.write((const uint8_t *)buf, len);
#endif
}

static void matvec_unroll4_f32(const int8_t *w_base, int rows, int cols, const float *x, float *y) {
  const float scale = 0.005f;
  for (int r = 0; r < rows; r++) {
    const int8_t *w = w_base + (size_t)r * (size_t)cols;
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
    y[r] = s * scale;
  }
}

/* Versão propositalmente lenta para demonstrar por que verify_port.py proíbe double! */
static float matvec_double_naive(const int8_t *w_base, int rows, int cols, const float *x) {
  volatile double sink = 0.0;
  for (int r = 0; r < rows; r++) {
    const int8_t *w = w_base + (size_t)r * (size_t)cols;
    double s = 0.0;
    for (int c = 0; c < cols; c++) {
      s += (double)w[c] * (double)x[c];
    }
    sink += sqrt(fabs(s) + 1.0);
  }
  return (float)sink;
}

static void rms_norm_f32(const float *x, const float *g, float *y) {
  float s = 0.0f;
  for (int i = 0; i < P_D; i++) s += x[i] * x[i];
  const float inv = 1.0f / sqrtf(s / (float)P_D + 1e-5f);
  for (int i = 0; i < P_D; i++) y[i] = x[i] * inv * g[i];
}

static float store_kv_f32(int8_t *dst, const float *src) {
  float big = 0.0f;
  for (int i = 0; i < P_D; i++) {
    float v = src[i] < 0.0f ? -src[i] : src[i];
    if (v > big) big = v;
  }
  if (big < 1e-20f) big = 1e-20f;
  const float k = 127.0f / big;
  for (int i = 0; i < P_D; i++) {
    float v = src[i] * k;
    dst[i] = (int8_t)(v < 0.0f ? v - 0.5f : v + 0.5f);
  }
  return big / 127.0f;
}

static void runProfile(int ctx_pos) {
  uint64_t t_norm = 0, t_qkv_wo = 0, t_att_loop = 0, t_ffn = 0, t_head = 0;

  for (int i = 0; i < P_D; i++) {
    bx[i] = 0.05f * (float)((i % 7) - 3);
    gains[i] = 1.0f;
  }

  /* Preenche KV cache até ctx_pos - 1 */
  for (int l = 0; l < P_L; l++) {
    for (int j = 0; j < ctx_pos; j++) {
      ks[l][j] = 0.01f;
      vs[l][j] = 0.01f;
      for (int i = 0; i < P_D; i++) {
        kc[l][j][i] = (int8_t)((i + j) % 21 - 10);
        vc[l][j][i] = (int8_t)((i - j) % 21 - 10);
      }
    }
  }

  const float att_div = sqrtf((float)P_HD);
  uint64_t t_total_0 = esp_timer_get_time();

  for (int l = 0; l < P_L; l++) {
    uint64_t t0 = esp_timer_get_time();
    rms_norm_f32(bx, gains, bh);
    uint64_t t1 = esp_timer_get_time();
    t_norm += (t1 - t0);

    matvec_unroll4_f32(W_SQUARE, P_D, P_D, bh, bt);
    ks[l][ctx_pos] = store_kv_f32(kc[l][ctx_pos], bt);
    matvec_unroll4_f32(W_SQUARE, P_D, P_D, bh, bt);
    vs[l][ctx_pos] = store_kv_f32(vc[l][ctx_pos], bt);
    matvec_unroll4_f32(W_SQUARE, P_D, P_D, bh, bq);
    uint64_t t2 = esp_timer_get_time();
    t_qkv_wo += (t2 - t1);

    for (int h = 0; h < P_H; h++) {
      const int hoff = h * P_HD;
      float big = -1e30f;
      for (int j = 0; j <= ctx_pos; j++) {
        float s = 0.0f;
        for (int i = 0; i < P_HD; i++) s += bq[hoff + i] * (float)kc[l][j][hoff + i];
        s = (s * ks[l][j]) / att_div;
        bs[j] = s;
        if (s > big) big = s;
      }
      float sum = 0.0f;
      for (int j = 0; j <= ctx_pos; j++) {
        float d = bs[j] - big;
        bs[j] = (d < -60.0f) ? 0.0f : expf(d);
        sum += bs[j];
      }
      for (int i = 0; i < P_HD; i++) ba[hoff + i] = 0.0f;
      if (sum > 0.0f) {
        for (int j = 0; j <= ctx_pos; j++) {
          const float w = bs[j] * vs[l][j] / sum;
          for (int i = 0; i < P_HD; i++) ba[hoff + i] += w * (float)vc[l][j][hoff + i];
        }
      }
    }
    uint64_t t3 = esp_timer_get_time();
    t_att_loop += (t3 - t2);

    matvec_unroll4_f32(W_SQUARE, P_D, P_D, ba, bt);
    for (int i = 0; i < P_D; i++) bx[i] += bt[i];
    uint64_t t4 = esp_timer_get_time();
    t_qkv_wo += (t4 - t3);

    rms_norm_f32(bx, gains, bh);
    uint64_t t5 = esp_timer_get_time();
    t_norm += (t5 - t4);

    matvec_unroll4_f32(W_FFN1, P_HID, P_D, bh, bm);
    for (int i = 0; i < P_HID; i++) if (bm[i] < 0.0f) bm[i] = 0.0f;
    matvec_unroll4_f32(W_FFN2, P_D, P_HID, bm, bt);
    for (int i = 0; i < P_D; i++) bx[i] += bt[i];
    uint64_t t6 = esp_timer_get_time();
    t_ffn += (t6 - t5);
  }

  uint64_t th0 = esp_timer_get_time();
  rms_norm_f32(bx, gains, bh);
  matvec_unroll4_f32(W_EMB, P_VOCAB, P_D, bh, logits);
  uint64_t th1 = esp_timer_get_time();
  t_head = (th1 - th0);

  uint64_t t_total = th1 - t_total_0;
  uint64_t t_all_matvec = t_qkv_wo + t_ffn + t_head;

  conPrintf("\n=== PROFILING DE 1 TOKEN (pos=%d, D=%d, L=%d, HID=%d) ===\n",
            ctx_pos, P_D, P_L, P_HID);
  conPrintf("  matvec Q,K,V,Wo (4x) : %6llu us (%5.1f%%)\n",
            (unsigned long long)t_qkv_wo, 100.0 * (double)t_qkv_wo / (double)t_total);
  conPrintf("  matvec FFN f1+f2 (4x): %6llu us (%5.1f%%)\n",
            (unsigned long long)t_ffn, 100.0 * (double)t_ffn / (double)t_total);
  conPrintf("  Head emb + norm final: %6llu us (%5.1f%%)\n",
            (unsigned long long)t_head, 100.0 * (double)t_head / (double)t_total);
  conPrintf("  Atenção KV + expf()  : %6llu us (%5.1f%%)\n",
            (unsigned long long)t_att_loop, 100.0 * (double)t_att_loop / (double)t_total);
  conPrintf("  RMSNorm (8x)         : %6llu us (%5.1f%%)\n",
            (unsigned long long)t_norm, 100.0 * (double)t_norm / (double)t_total);
  conPrintf("  ---------------------------------------------------\n");
  conPrintf("  TOTAL por token      : %6llu us (%.1f tokens/s teóricos)\n",
            (unsigned long long)t_total, 1000000.0 / (double)t_total);
  conPrintf("  Fração em matvec()   : %.1f%%\n",
            100.0 * (double)t_all_matvec / (double)t_total);

  /* Comparação FPU float32 vs double emulação em uma matriz FFN (256x96) */
  uint64_t tf0 = esp_timer_get_time();
  matvec_unroll4_f32(W_FFN1, P_HID, P_D, bh, bm);
  uint64_t tf1 = esp_timer_get_time();
  volatile float d_res = matvec_double_naive(W_FFN1, P_HID, P_D, bh);
  uint64_t tf2 = esp_timer_get_time();
  (void)d_res;

  conPrintf("\n[FPU float32 vs double software em FFN1 256x96]\n");
  conPrintf("  float32 unroll 4x : %llu us\n", (unsigned long long)(tf1 - tf0));
  conPrintf("  double (emulado)  : %llu us (%.1fx mais lento!)\n",
            (unsigned long long)(tf2 - tf1), (double)(tf2 - tf1) / (double)(tf1 - tf0));
}

void setup() {
  Serial.begin(115200);
#if PROF_DUAL_SERIAL
  Serial.setTxTimeoutMs(0);
  Serial0.begin(115200);
#endif
  uint32_t t0 = millis();
  while (!Serial && (millis() - t0) < 3500u) delay(50);
  delay(400);

  runProfile(32);
  conPrintf("\nPressione Enter no Serial para rodar outro profiling.\n> ");
}

void loop() {
  bool hit = false;
  while (Serial.available()) {
    int c = Serial.read();
    if (c == '\n' || c == '\r') hit = true;
  }
#if PROF_DUAL_SERIAL
  while (Serial0.available()) {
    int c = Serial0.read();
    if (c == '\n' || c == '\r') hit = true;
  }
#endif
  if (hit) {
    runProfile(32);
    conPrintf("\n> ");
  }
  delay(20);
}
