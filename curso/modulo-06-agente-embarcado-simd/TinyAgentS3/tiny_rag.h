#ifndef TINY_RAG_H
#define TINY_RAG_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct __attribute__((packed)) {
    char     magic[4];
    uint16_t num_docs;
    uint16_t reserved;
} Rag1Header;

typedef struct __attribute__((packed)) {
    uint32_t title_off;
    uint16_t title_len;
    uint32_t kw_off;
    uint16_t kw_len;
    uint32_t body_off;
    uint16_t body_len;
    uint16_t flags;
} Rag1Entry;

bool tiny_rag_init_from_blob(const uint8_t *blob, size_t blob_size);
int tiny_rag_search(const char *query_utf8, char *snippet_out, size_t cap);

#ifdef __cplusplus
}
#endif

#endif /* TINY_RAG_H */
