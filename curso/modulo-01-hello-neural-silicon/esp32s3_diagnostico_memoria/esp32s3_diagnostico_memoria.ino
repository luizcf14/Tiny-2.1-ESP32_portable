/*
 * Módulo 1 (Lab 2 - Hardware): Diagnóstico de Memória e Zero-Copy para ESP32-S3-WROOM-1 (16MB)
 * ============================================================================================
 * Como usar na Arduino IDE:
 *   - Board:            ESP32S3 Dev Module
 *   - USB CDC On Boot:  Enabled
 *   - USB Mode:         Hardware CDC and JTAG
 *   - Flash Size:       16MB (128Mb)
 *   - Flash Mode:       QIO 80MHz (ou DIO)
 *   - PSRAM:            OPI PSRAM (N16R8) / QSPI PSRAM (N16R2) / Disabled (N16)
 *   - Serial Monitor:   115200 baud
 */

#include <Arduino.h>
#include <esp_heap_caps.h>
#include <esp_partition.h>
#include <esp_system.h>
#include <pgmspace.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdlib.h>

#if defined(ARDUINO_USB_CDC_ON_BOOT) && ARDUINO_USB_CDC_ON_BOOT
#define DIAG_DUAL_SERIAL 1
#else
#define DIAG_DUAL_SERIAL 0
#endif

/* Bloco simulando 4 KB de pesos neurais gravados em Flash (PROGMEM / DROM), alinhado em 4 bytes */
static const uint32_t DEMO_FLASH_WEIGHTS[1024] PROGMEM = {
    0x32594E54u, /* "TNY2" em little-endian */
    0x00040060u, /* D=96, L=4 */
    0x01000004u, /* H=4, HID=256 */
    0x3B800000u  /* Exemplo de scale float32 */
};

static bool cdcWasConnected = false;

static void conPrint(const char *s) {
  Serial.print(s);
#if DIAG_DUAL_SERIAL
  Serial0.print(s);
#endif
}

static void conPrintln(const char *s) {
  Serial.println(s);
#if DIAG_DUAL_SERIAL
  Serial0.println(s);
#endif
}

static void conPrintf(const char *fmt, ...) {
  char buf[256];
  va_list args;
  va_start(args, fmt);
  int n = vsnprintf(buf, sizeof(buf), fmt, args);
  va_end(args);
  if (n <= 0) return;
  size_t len = (size_t)n < sizeof(buf) ? (size_t)n : sizeof(buf) - 1u;
  Serial.write((const uint8_t *)buf, len);
#if DIAG_DUAL_SERIAL
  Serial0.write((const uint8_t *)buf, len);
#endif
}

static void printPartitionTable(void) {
  conPrintln("\n[4] Tabela de Partições da Flash:");
  conPrintln("  Nome         | Tipo | Sub | Offset     | Tamanho");
  conPrintln("  -------------+------+-----+------------+------------");

  esp_partition_iterator_t it =
      esp_partition_find(ESP_PARTITION_TYPE_ANY, ESP_PARTITION_SUBTYPE_ANY, NULL);
  while (it != NULL) {
    const esp_partition_t *p = esp_partition_get(it);
    conPrintf("  %-12s | 0x%02X | 0x%02X| 0x%08lX | %5lu KB\n", p->label,
              (unsigned)p->type, (unsigned)p->subtype,
              (unsigned long)p->address, (unsigned long)(p->size / 1024u));
    it = esp_partition_next(it);
  }
  esp_partition_iterator_release(it);
}

static void runZeroCopyDemo(void) {
  conPrintln("\n[5] Demonstração Zero-Copy (Flash DROM vs SRAM malloc vs PSRAM):");

  uint32_t heapBefore = ESP.getFreeHeap();
  uint32_t magic = pgm_read_dword(&DEMO_FLASH_WEIGHTS[0]);
  uint32_t heapAfterFlash = ESP.getFreeHeap();

  conPrintf("  Flash PROGMEM (4096 B): addr=0x%08lX | magic=0x%08lX | RAM consumida=%lu B\n",
            (unsigned long)(uintptr_t)DEMO_FLASH_WEIGHTS, (unsigned long)magic,
            (unsigned long)(heapBefore - heapAfterFlash));

  /* Aloca 100 KB simulando o KV Cache (kc, vc, ks, vs) de tiny_engine.cpp */
  const size_t kvSimBytes = 102400u;
  uint8_t *sramKv = (uint8_t *)heap_caps_malloc(kvSimBytes, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
  uint32_t heapAfterSram = ESP.getFreeHeap();
  if (sramKv) {
    sramKv[0] = 0x7F;
    conPrintf("  SRAM malloc (100 KB)  : addr=0x%08lX | heap livre apos=%lu B (delta=-%lu B)\n",
              (unsigned long)(uintptr_t)sramKv, (unsigned long)heapAfterSram,
              (unsigned long)(heapAfterFlash - heapAfterSram));
    free(sramKv);
  } else {
    conPrintln("  SRAM malloc (100 KB)  : FALHOU (heap interno insuficiente)");
  }

  if (psramFound()) {
    uint8_t *psramKv = (uint8_t *)ps_malloc(kvSimBytes);
    if (psramKv) {
      psramKv[0] = 0x7F;
      conPrintf("  PSRAM ps_malloc(100KB): addr=0x%08lX | PSRAM livre=%lu B\n",
                (unsigned long)(uintptr_t)psramKv, (unsigned long)ESP.getFreePsram());
      free(psramKv);
    }
  } else {
    conPrintln("  PSRAM ps_malloc       : não ativa (placa N16 ou PSRAM desabilitada em Tools)");
  }
}

static void printFullDiagnostic(void) {
  uint32_t flashBytes = ESP.getFlashChipSize();
  uint32_t freeInternal = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
  uint32_t largestInternal = heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL);

  conPrintln("\n======================================================================");
  conPrintln("  DIAGNÓSTICO DE SILÍCIO — MÓDULO 1 (ESP32-S3-WROOM-1 16MB)");
  conPrintln("======================================================================");

  conPrintf("[1] Chip: %s | %d cores @ %u MHz | Revisão %d\n",
            ESP.getChipModel(), ESP.getChipCores(), ESP.getCpuFreqMHz(),
            ESP.getChipRevision());

  conPrintf("\n[2] Flash SPI: %lu MB (%lu bytes) | Velocidade %lu MHz\n",
            (unsigned long)(flashBytes / (1024u * 1024u)),
            (unsigned long)flashBytes,
            (unsigned long)(ESP.getFlashChipSpeed() / 1000000u));
  if (flashBytes < 16u * 1024u * 1024u) {
    conPrintln("  [AVISO] Flash reportada < 16 MB! Confira Tools -> Flash Size: 16MB (128Mb).");
  } else {
    conPrintln("  [OK] 16 MB disponíveis (pronto para partição TNY3-S3 de 8 MB nos Módulos 4-6).");
  }

  conPrintln("\n[3] Memória Volátil (SRAM Interna + PSRAM):");
  conPrintf("  SRAM Total Heap       : %lu bytes\n", (unsigned long)ESP.getHeapSize());
  conPrintf("  SRAM Interna Livre    : %lu bytes (~%lu KB)\n",
            (unsigned long)freeInternal, (unsigned long)(freeInternal / 1024u));
  conPrintf("  Maior Bloco Contínuo  : %lu bytes (~%lu KB)\n",
            (unsigned long)largestInternal, (unsigned long)(largestInternal / 1024u));
  conPrintf("  Capacidade KV (100 KB): cabem ~%lu instâncias de KV Cache na SRAM interna\n",
            (unsigned long)(freeInternal / 102400u));

  if (psramFound()) {
    conPrintf("  PSRAM Detectada       : %lu MB (%lu bytes) | Livre: %lu bytes\n",
              (unsigned long)(ESP.getPsramSize() / (1024u * 1024u)),
              (unsigned long)ESP.getPsramSize(),
              (unsigned long)ESP.getFreePsram());
  } else {
    conPrintln("  PSRAM Detectada       : Nenhuma (modo N16 ou Tools -> PSRAM Disabled)");
  }

  printPartitionTable();
  runZeroCopyDemo();
  conPrintln("\nDigite qualquer linha ou 'status' e pressione Enter para rodar novamente.");
  conPrint("> ");
}

void setup() {
  Serial.begin(115200);
#if DIAG_DUAL_SERIAL
  Serial.setTxTimeoutMs(0);
  Serial0.begin(115200);
#endif

  uint32_t t0 = millis();
  while (!Serial && (millis() - t0) < 3500u) {
    delay(50);
  }
  delay(400);

  printFullDiagnostic();
#if DIAG_DUAL_SERIAL
  cdcWasConnected = (bool)Serial;
#endif
}

static bool drainTrigger(Stream &st) {
  bool hit = false;
  while (st.available()) {
    int c = st.read();
    if (c == '\n' || c == '\r' || c == 's' || c == 'S') {
      hit = true;
    }
  }
  return hit;
}

void loop() {
#if DIAG_DUAL_SERIAL
  bool cdcNow = (bool)Serial;
  if (cdcNow && !cdcWasConnected) {
    cdcWasConnected = true;
    delay(250);
    printFullDiagnostic();
  } else if (!cdcNow) {
    cdcWasConnected = false;
  }
#endif

  bool trigger = drainTrigger(Serial);
#if DIAG_DUAL_SERIAL
  trigger = drainTrigger(Serial0) || trigger;
#endif

  if (trigger) {
    delay(40);
    drainTrigger(Serial);
#if DIAG_DUAL_SERIAL
    drainTrigger(Serial0);
#endif
    printFullDiagnostic();
  }
  delay(20);
}
