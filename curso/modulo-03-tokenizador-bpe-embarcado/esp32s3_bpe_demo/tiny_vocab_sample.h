/*
 * Exemplo compacto de vocabulário BPE pronto para compilação imediata do sketch demo.
 * Para usar o vocabulário completo de 4.096 tokens, gere ../out/tiny_vocab.h com lab03_paridade_bpe.py
 * e substitua o #include no sketch esp32s3_bpe_demo.ino.
 */
#ifndef TINY_VOCAB_SAMPLE_H
#define TINY_VOCAB_SAMPLE_H

#include <stdint.h>
#ifdef ARDUINO
#include <pgmspace.h>
#else
#ifndef PROGMEM
#define PROGMEM
#endif
#endif

typedef struct {
    uint32_t offset;
    uint8_t  len;
} TinyTokenSlice;

typedef struct {
    uint16_t left;
    uint16_t right;
    uint16_t merged;
    uint16_t rank;
} TinyBpeMerge;

/* Merges demonstrativos PT-BR (0..15 especiais, 16..271 bytes ASCII/UTF-8, 272..283 merges) */
#define TINY_SAMPLE_NUM_MERGES 12

/*
 * Ordenado estritamente por (left, right) para busca binária:
 *   (48 ' ', 113 'a') -> 274 (" a") rank 2
 *   (48 ' ', 116 'd') -> 276 (" d") rank 4
 *   (48 ' ', 127 'o') -> 278 (" o") rank 6
 *   (48 ' ', 134 'v') -> 280 (" v") rank 8
 *   (113 'a', 130 'r') -> 273 ("ar") rank 1
 *   (117 'e', 126 'n') -> 272 ("en") rank 0
 *   (130 'r', 113 'a') -> 282 ("ra") rank 10
 *   (274 " a", 129 'q') -> 275 (" aq") rank 3  [exemplo simples]
 *   (276 " d", 113 'a') -> 277 (" da") rank 5
 *   (278 " o", 131 's') -> 279 (" os") rank 7
 *   (280 " v", 272 "en") -> 281 (" ven") rank 9
 *   (195+16=211 0xC3, 163+16=179 0xA3) -> 283 ("ã" UTF-8) rank 11
 */
static const TinyBpeMerge TINY_SAMPLE_MERGES[TINY_SAMPLE_NUM_MERGES] PROGMEM = {
    {48u, 113u, 274u, 2u},
    {48u, 116u, 276u, 4u},
    {48u, 127u, 278u, 6u},
    {48u, 134u, 280u, 8u},
    {113u, 130u, 273u, 1u},
    {117u, 126u, 272u, 0u},
    {130u, 113u, 282u, 10u},
    {211u, 179u, 283u, 11u},
    {274u, 129u, 275u, 3u},
    {276u, 113u, 277u, 5u},
    {278u, 131u, 279u, 7u},
    {280u, 272u, 281u, 9u}
};

static const char *const TINY_SAMPLE_SPECIALS[7] = {
    "<|pad|>", "<|bos|>", "<|eos|>", "<|user|>", "<|ctx|>", "<|act|>", "<|assistant|>"
};

static const char *const TINY_SAMPLE_MERGE_STRS[12] = {
    "en", "ar", " a", " aq", " d", " da", " o", " os", " v", " ven", "ra", "\xC3\xA3"
};

#endif /* TINY_VOCAB_SAMPLE_H */
