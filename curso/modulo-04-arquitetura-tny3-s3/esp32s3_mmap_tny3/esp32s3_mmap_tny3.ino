/*
 * Módulo 4 (Hardware): Zero-Copy na Partição model_bin (8 MB) via esp_partition_mmap()
 * ====================================================================================
 * Demonstra no ESP32-S3-WROOM-1 (16MB) como mapear até 8 MB de pesos direto da Flash
 * sem recompilar o firmware nem gastar SRAM com os pesos, e como dimensionar o KV Cache
 * GQA automaticamente entre N16 (sem PSRAM) e N16R2/N16R8 (com PSRAM).
 */

#include <Arduino.h>
#include <esp_partition.h>
#include <esp_system.h>
#include <pgmspace.h>
#include <stdarg.h>
#include <stdint.h>
#include <string.h>

#if defined(ARDUINO_USB_CDC_ON_BOOT) && ARDUINO_USB_CDC_ON_BOOT
#define MMAP_DUAL_SERIAL 1
#else
#define MMAP_DUAL_SERIAL 0
#endif

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

static_assert(sizeof(Tny3Header) == 32, "Tny3Header deve ter exatamente 32 bytes!");

/* Cabeçalho TNY3 de demonstração caso a partição model_bin ainda esteja virgem (0xFF) */
static const Tny3Header DEMO_TNY3_HDR PROGMEM = {
    {'T', 'N', 'Y', '3'}, 1, 1, 4096, 256, 256, 8, 8, 2, 688, 0, 0, 10000.0f, 6721536u
};

static void conPrintf(const char *fmt, ...) {
  char buf[256];
  va_list args;
  va_start(args, fmt);
  int n = vsnprintf(buf, sizeof(buf), fmt, args);
  va_end(args);
  if (n <= 0) return;
  size_t len = (size_t)n < sizeof(buf) ? (size_t)n : sizeof(buf) - 1u;
  Serial.write((const uint8_t *)buf, len);
#if MMAP_DUAL_SERIAL
  Serial0.write((const uint8_t *)buf, len);
#endif
}

static void inspectTny3Header(const Tny3Header *h, const char *source_label) {
  uint32_t hd = h->d / h->q_heads;
  uint32_t kv_dim = h->kv_heads * hd;
  uint32_t kv_bytes_256 = (uint32_t)h->layers * 256u * 2u * (kv_dim + 4u);
  uint32_t kv_bytes_512 = (uint32_t)h->layers * 512u * 2u * (kv_dim + 4u);

  conPrintf("\n[Cabeçalho TNY3 lido via %s]\n", source_label);
  conPrintf("  Versão=%u | Quant=%s | Vocab=%u | CTX=%u\n",
            (unsigned)h->version, h->quant_mode == 2 ? "Q4_0 (4-bit)" : "int8 (8-bit)",
            (unsigned)h->vocab_size, (unsigned)h->ctx_len);
  conPrintf("  D=%u | L=%u | Q_Heads=%u | KV_Heads=%u (GQA %u:1) | HID=%u\n",
            (unsigned)h->d, (unsigned)h->layers, (unsigned)h->q_heads,
            (unsigned)h->kv_heads, (unsigned)(h->q_heads / h->kv_heads), (unsigned)h->hid);
  conPrintf("  Tamanho do modelo: %lu bytes (%.2f MB)\n",
            (unsigned long)h->total_bytes, (double)h->total_bytes / (1024.0 * 1024.0));
  conPrintf("  KV Cache GQA CTX=256: %lu bytes (~%lu KB)\n",
            (unsigned long)kv_bytes_256, (unsigned long)(kv_bytes_256 / 1024u));

  if (psramFound()) {
    conPrintf("  [PSRAM detectada: %lu KB] Alocando CTX=512 (%lu KB) com ps_malloc()...\n",
              (unsigned long)(ESP.getPsramSize() / 1024u), (unsigned long)(kv_bytes_512 / 1024u));
    void *kv = ps_malloc(kv_bytes_512);
    conPrintf("  ps_malloc -> %s (ptr=0x%08lX)\n", kv ? "OK" : "FALHOU", (unsigned long)(uintptr_t)kv);
    if (kv) free(kv);
  } else {
    conPrintf("  [Modo N16 sem PSRAM] Alocando CTX=256 (%lu KB) na SRAM interna...\n",
              (unsigned long)(kv_bytes_256 / 1024u));
    void *kv = malloc(kv_bytes_256);
    conPrintf("  malloc -> %s (ptr=0x%08lX, heap restante=%lu B)\n",
              kv ? "OK" : "FALHOU", (unsigned long)(uintptr_t)kv, (unsigned long)ESP.getFreeHeap());
    if (kv) free(kv);
  }
}

void setup() {
  Serial.begin(115200);
#if MMAP_DUAL_SERIAL
  Serial.setTxTimeoutMs(0);
  Serial0.begin(115200);
#endif
  uint32_t t0 = millis();
  while (!Serial && (millis() - t0) < 3500u) delay(50);
  delay(400);

  conPrintf("\n=== MÓDULO 4: ZERO-COPY MMAP NA PARTIÇÃO model_bin ===\n");

  const esp_partition_t *part = esp_partition_find_first(
      ESP_PARTITION_TYPE_DATA, (esp_partition_subtype_t)0x99, "model_bin");

  if (!part) {
    conPrintf("[AVISO] Partição 'model_bin' (subtype 0x99) não encontrada.\n");
    conPrintf("        Copie partitions_s3_16mb.csv como partitions.csv na pasta do sketch!\n");
    inspectTny3Header(&DEMO_TNY3_HDR, "PROGMEM Demo (fallback)");
    return;
  }

  conPrintf("[OK] Partição 'model_bin' encontrada: offset=0x%06lX | size=%lu KB\n",
            (unsigned long)part->address, (unsigned long)(part->size / 1024u));

  const void *mapped_ptr = NULL;
  esp_partition_mmap_handle_t mmap_handle;
  esp_err_t err = esp_partition_mmap(
      part, 0, part->size, ESP_PARTITION_MMAP_DATA, &mapped_ptr, &mmap_handle);

  if (err != ESP_OK || !mapped_ptr) {
    conPrintf("[ERRO] esp_partition_mmap falhou: 0x%X\n", (unsigned)err);
    return;
  }

  conPrintf("[MMAP] Ponteiro DROM mapeado em 0x%08lX (0 bytes de RAM consumidos!)\n",
            (unsigned long)(uintptr_t)mapped_ptr);

  const Tny3Header *hdr = (const Tny3Header *)mapped_ptr;
  if (memcmp(hdr->magic, "TNY3", 4) == 0) {
    inspectTny3Header(hdr, "esp_partition_mmap() na Flash");
  } else {
    conPrintf("[INFO] A partição 'model_bin' está vazia (bytes iniciais: %02X %02X %02X %02X).\n",
              ((const uint8_t *)mapped_ptr)[0], ((const uint8_t *)mapped_ptr)[1],
              ((const uint8_t *)mapped_ptr)[2], ((const uint8_t *)mapped_ptr)[3]);
    conPrintf("       Para gravar um modelo sem recompilar:\n");
    conPrintf("       esptool.py --chip esp32s3 write_flash 0x%06lX seu_modelo_tny3.bin\n",
              (unsigned long)part->address);
    inspectTny3Header(&DEMO_TNY3_HDR, "PROGMEM Demo (aguardando upload em model_bin)");
  }

  esp_partition_munmap(mmap_handle);
}

void loop() {
  delay(1000);
}
