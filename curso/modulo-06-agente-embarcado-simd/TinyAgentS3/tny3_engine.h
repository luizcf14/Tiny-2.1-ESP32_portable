#ifndef TNY3_ENGINE_H
#define TNY3_ENGINE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TNY3_MAX_D       256
#define TNY3_MAX_HID     768
#define TNY3_MAX_LAYERS  8
#define TNY3_MAX_CTX     512
#define TNY3_MAX_VOCAB   4096

#define TNY3_QUANT_INT8  1
#define TNY3_QUANT_Q4_0  2

typedef enum {
    TNY3_MATVEC_FP32_SCALAR = 0,
    TNY3_MATVEC_INT8_MAC4   = 1,
    TNY3_MATVEC_S3_PIE_SIMD = 2
} Tny3MatvecMode;

typedef struct __attribute__((packed)) {
    char     magic[4];
    uint8_t  version;
    uint8_t  quant_mode;
    uint16_t vocab_size;
    uint16_t ctx_len;
    uint16_t d;
    uint16_t layers;
    uint16_t q_heads;
    uint16_t kv_heads;
    uint16_t hid;
    uint16_t flags;
    uint16_t reserved;
    float    rope_theta;
    uint32_t total_bytes;
} Tny3Header;

typedef struct {
    Tny3Header hdr;
    int head_dim;
    int kv_dim;
    int groups;
    uint32_t kv_ram_bytes;
    bool using_psram;
} Tny3Info;

bool tny3_init(const uint8_t *blob, size_t blob_size, const char **err_msg);
void tny3_free(void);
const Tny3Info *tny3_get_info(void);
void tny3_set_matvec_mode(Tny3MatvecMode mode);
Tny3MatvecMode tny3_get_matvec_mode(void);
void tny3_reset_kv_cache(void);

/* Executa um passo autorregressivo (RoPE + GQA + SwiGLU) e preenche out_logits[vocab_size] */
int tny3_step_greedy(int tok, int pos, float *out_logits);

/* Benchmark comparativo dos modos de matvec() em microssegundos */
void tny3_benchmark_matvec(uint32_t *us_fp32, uint32_t *us_mac4, uint32_t *us_pie);

#ifdef __cplusplus
}
#endif

#endif /* TNY3_ENGINE_H */
