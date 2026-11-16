#ifndef TINY_ENGINE_H
#define TINY_ENGINE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TINY_VOCAB  97
#define TINY_CTX    128
#define TINY_MAXD   128
#define TINY_MAXHID 512
#define TINY_MAXL   8
#define TINY_MAXQ   72

typedef struct {
    int d;
    int layers;
    int heads;
    int head_dim;
    int hidden;
    uint32_t ram_bytes;
    uint32_t flash_bytes;
} TinyModelInfo;

typedef void (*TinyStepFn)(int pos, int total, const float *layer_act, int layers, void *user);
typedef void (*TinyLetterFn)(char ch, float confidence, const float *layer_act, int layers, void *user);

bool tiny_init(const uint8_t *blob, size_t blob_size, const char **err_msg);
void tiny_free(void);
const TinyModelInfo *tiny_get_info(void);
int tiny_clean_question(const char *line, char *q_out);
int tiny_think(const char *clean_q,
               char *ans_out,
               int creativity,
               uint32_t *elapsed_ms,
               TinyStepFn on_step,
               TinyLetterFn on_letter,
               void *user);

#ifdef __cplusplus
}
#endif

#endif /* TINY_ENGINE_H */
