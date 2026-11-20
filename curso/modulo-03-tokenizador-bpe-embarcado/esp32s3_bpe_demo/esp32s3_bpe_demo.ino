/*
 * Módulo 3 (Hardware): Demonstração de Tokenizador BPE UTF-8 no ESP32-S3-WROOM-1 (16MB)
 * =====================================================================================
 * Codifica frases com acentuação UTF-8 sem malloc(), mede a latência em microssegundos
 * e reconstrói codepoints multi-byte em streaming com TinyUtf8Stream.
 */

#include <Arduino.h>
#include <esp_timer.h>
#include <stdarg.h>
#include <stdint.h>

#include "tiny_vocab_sample.h"
#include "tiny_tokenizer.h"

#if defined(ARDUINO_USB_CDC_ON_BOOT) && ARDUINO_USB_CDC_ON_BOOT
#define BPE_DUAL_SERIAL 1
#else
#define BPE_DUAL_SERIAL 0
#endif

static char serialBuf[192];
static size_t serialLen = 0;

static void conPrintf(const char *fmt, ...) {
  char buf[320];
  va_list args;
  va_start(args, fmt);
  int n = vsnprintf(buf, sizeof(buf), fmt, args);
  va_end(args);
  if (n <= 0) return;
  size_t len = (size_t)n < sizeof(buf) ? (size_t)n : sizeof(buf) - 1u;
  Serial.write((const uint8_t *)buf, len);
#if BPE_DUAL_SERIAL
  Serial0.write((const uint8_t *)buf, len);
#endif
}

static void printTokenPiece(uint16_t tid) {
  if (tid < 7u) {
    conPrintf("%s", TINY_SAMPLE_SPECIALS[tid]);
  } else if (tid >= 16u && tid < 272u) {
    uint8_t b = (uint8_t)(tid - 16u);
    if (b >= 32u && b <= 126u) conPrintf("'%c'", (char)b);
    else conPrintf("<0x%02X>", (unsigned)b);
  } else if (tid >= 272u && tid < 272u + TINY_SAMPLE_NUM_MERGES) {
    conPrintf("\"%s\"", TINY_SAMPLE_MERGE_STRS[tid - 272u]);
  } else {
    conPrintf("<id:%u>", (unsigned)tid);
  }
}

static void runDemoPhrase(const char *text) {
  uint16_t toks[128];
  uint64_t t0 = esp_timer_get_time();
  size_t n_tok = tiny_bpe_encode(text, toks, 128, TINY_SAMPLE_MERGES, TINY_SAMPLE_NUM_MERGES);
  uint64_t dt = esp_timer_get_time() - t0;

  conPrintf("\nEntrada UTF-8 (%u bytes): %s\n", (unsigned)strlen(text), text);
  conPrintf("Tokens BPE (%u tokens em %llu us):\n  [", (unsigned)n_tok, (unsigned long long)dt);
  for (size_t i = 0; i < n_tok; i++) {
    conPrintf("%s%u:", (i > 0 ? ", " : ""), (unsigned)toks[i]);
    printTokenPiece(toks[i]);
  }
  conPrintf("]\nDecodificado via TinyUtf8Stream: ");

  TinyUtf8Stream st;
  tiny_utf8_init(&st);
  char utf8_ch[8];
  for (size_t i = 0; i < n_tok; i++) {
    uint16_t tid = toks[i];
    if (tid >= 16u && tid < 272u) {
      if (tiny_utf8_feed(&st, (uint8_t)(tid - 16u), utf8_ch) > 0) conPrintf("%s", utf8_ch);
    } else if (tid >= 272u && tid < 272u + TINY_SAMPLE_NUM_MERGES) {
      const char *p = TINY_SAMPLE_MERGE_STRS[tid - 272u];
      for (; *p; p++) {
        if (tiny_utf8_feed(&st, (uint8_t)*p, utf8_ch) > 0) conPrintf("%s", utf8_ch);
      }
    } else if (tid < 7u) {
      conPrintf("%s", TINY_SAMPLE_SPECIALS[tid]);
    }
  }
  conPrintf("\n");
}

void setup() {
  Serial.begin(115200);
#if BPE_DUAL_SERIAL
  Serial.setTxTimeoutMs(0);
  Serial0.begin(115200);
#endif
  uint32_t t0 = millis();
  while (!Serial && (millis() - t0) < 3500u) delay(50);
  delay(400);

  conPrintf("\n=== MÓDULO 3: DEMO TOKENIZADOR BPE UTF-8 NO ESP32-S3 ===\n");
  runDemoPhrase("<|user|>ligar ventilação da bancada em 29°C<|eos|>");
  runDemoPhrase("ação e atenção com acentuação preservada!");
  conPrintf("\nDigite uma frase em UTF-8 no Serial para tokenizar:\n> ");
}

static void pollInput(Stream &s) {
  while (s.available()) {
    int c = s.read();
    if (c < 0) break;
    if (c == '\n' || c == '\r') {
      if (serialLen > 0) {
        serialBuf[serialLen] = '\0';
        serialLen = 0;
        runDemoPhrase(serialBuf);
        conPrintf("\n> ");
      }
    } else if (serialLen + 1 < sizeof(serialBuf)) {
      serialBuf[serialLen++] = (char)c;
    }
  }
}

void loop() {
  pollInput(Serial);
#if BPE_DUAL_SERIAL
  pollInput(Serial0);
#endif
  delay(10);
}
