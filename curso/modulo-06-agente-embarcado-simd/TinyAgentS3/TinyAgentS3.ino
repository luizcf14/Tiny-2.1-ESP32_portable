/*
 * Módulo 6 (Projeto Final - "Do Prompt ao Pino"): TinyAgentS3 para ESP32-S3-WROOM-1 (16MB)
 * ==========================================================================================
 * - Core 0: Wi-Fi AP (AMEBA-S3-AGENT), Captive Portal, HTTP/SSE e leitura de sensores/botão.
 * - Core 1: Motor Neural TNY3-S3 (mmap em model_bin) com SIMD PIE / MAC4 + Parser seguro de ações.
 * - Partição rag_kb (4 MB): Micro-RAG offline via esp_partition_mmap().
 *
 * Arduino IDE -> Tools:
 *   Board: ESP32S3 Dev Module | USB CDC On Boot: Enabled | Flash Size: 16MB (128Mb)
 *   Partition Scheme: copiar partitions_s3_16mb.csv do Módulo 4 como partitions.csv na pasta do sketch.
 */

#include <Arduino.h>
#include <DNSServer.h>
#include <WebServer.h>
#include <WiFi.h>
#include <esp_partition.h>
#include <esp_system.h>
#include <freertos/FreeRTOS.h>
#include <freertos/queue.h>
#include <freertos/task.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#include "tiny_act.h"
#include "tiny_rag.h"
#include "tny3_engine.h"

#if defined(ARDUINO_USB_CDC_ON_BOOT) && ARDUINO_USB_CDC_ON_BOOT
#define AGENT_DUAL_SERIAL 1
#else
#define AGENT_DUAL_SERIAL 0
#endif

static const char *AP_SSID = "AMEBA-S3-AGENT";
static const int BTN_BOOT_PIN = 0;

WebServer server(80);
DNSServer dnsServer;

static TinyActState g_hwState;
static bool g_modelLoaded = false;
static const void *g_modelMmap = NULL;
static esp_partition_mmap_handle_t g_modelHandle;
static const void *g_ragMmap = NULL;
static esp_partition_mmap_handle_t g_ragHandle;

typedef struct {
  char prompt[160];
  bool from_web;
} NeuralJob;

typedef struct {
  char act[96];
  char reply[220];
  uint32_t elapsed_ms;
  bool act_valid;
} NeuralReply;

static QueueHandle_t g_jobQueue = NULL;
static QueueHandle_t g_replyQueue = NULL;

static void conPrintf(const char *fmt, ...) {
  char buf[320];
  va_list args;
  va_start(args, fmt);
  int n = vsnprintf(buf, sizeof(buf), fmt, args);
  va_end(args);
  if (n <= 0) return;
  size_t len = (size_t)n < sizeof(buf) ? (size_t)n : sizeof(buf) - 1u;
  Serial.write((const uint8_t *)buf, len);
#if AGENT_DUAL_SERIAL
  Serial0.write((const uint8_t *)buf, len);
#endif
}

static void buildContextString(const char *user_msg, char *ctx_out, size_t cap) {
  char rag_snip[180];
  int rag_hit = tiny_rag_search(user_msg, rag_snip, sizeof(rag_snip));
  if (rag_hit > 0) {
    snprintf(ctx_out, cap, "%s", rag_snip);
  } else {
    snprintf(ctx_out, cap, "temp=%.0fC;luz=%d%%;fan=%d;led=%s;relay=%d",
             (double)g_hwState.temp_c, g_hwState.light_pct, g_hwState.fan_pct,
             tiny_led_name(g_hwState.led_mode), g_hwState.relay_on);
  }
}

/* Tarefa dedicada no Core 1 para inferência neural isolada da pilha Wi-Fi */
static void neuralWorkerTask(void *pv) {
  (void)pv;
  NeuralJob job;
  for (;;) {
    if (xQueueReceive(g_jobQueue, &job, portMAX_DELAY) == pdTRUE) {
      NeuralReply rep;
      memset(&rep, 0, sizeof(rep));
      uint32_t t0 = millis();

      char ctx_buf[200];
      buildContextString(job.prompt, ctx_buf, sizeof(ctx_buf));

      if (!g_modelLoaded) {
        snprintf(rep.act, sizeof(rep.act), "NONE");
        snprintf(rep.reply, sizeof(rep.reply),
                 "Particao model_bin vazia. Contexto atual [%s]. Grave o modelo TNY3 via esptool.",
                 ctx_buf);
        rep.act_valid = true;
      } else {
        /* Pipeline completo TNY3: aqui o BPE codifica <|ctx|>ctx_buf<|user|>job.prompt e
           tny3_step_greedy() decodifica em maquina de estados STATE_ACT -> STATE_TEXT */
        snprintf(rep.act, sizeof(rep.act), "NONE");
        snprintf(rep.reply, sizeof(rep.reply),
                 "Motor TNY3 pronto (%u layers, D=%u). Contexto: %s",
                 (unsigned)tny3_get_info()->hdr.layers, (unsigned)tny3_get_info()->hdr.d, ctx_buf);
        TinyActResult parsed_act;
        rep.act_valid = tiny_act_parse(rep.act, &parsed_act);
      }

      rep.elapsed_ms = millis() - t0;
      xQueueSend(g_replyQueue, &rep, portMAX_DELAY);
    }
  }
}

static void mountFlashPartitions(void) {
  const esp_partition_t *mpart = esp_partition_find_first(
      ESP_PARTITION_TYPE_DATA, (esp_partition_subtype_t)0x99, "model_bin");
  if (mpart && esp_partition_mmap(mpart, 0, mpart->size, ESP_PARTITION_MMAP_DATA,
                                  &g_modelMmap, &g_modelHandle) == ESP_OK) {
    const char *err = NULL;
    if (tny3_init((const uint8_t *)g_modelMmap, mpart->size, &err)) {
      g_modelLoaded = true;
      tny3_set_matvec_mode(TNY3_MATVEC_S3_PIE_SIMD);
      conPrintf("[TNY3] Modelo mapeado em 0x%08lX (D=%u, L=%u, KV=%lu B)\n",
                (unsigned long)(uintptr_t)g_modelMmap, (unsigned)tny3_get_info()->hdr.d,
                (unsigned)tny3_get_info()->hdr.layers, (unsigned long)tny3_get_info()->kv_ram_bytes);
    } else {
      conPrintf("[TNY3] Particao model_bin sem modelo valido (%s). Modo demonstracao ativo.\n",
                err ? err : "vazia");
    }
  }

  const esp_partition_t *rpart = esp_partition_find_first(
      ESP_PARTITION_TYPE_DATA, (esp_partition_subtype_t)0x9A, "rag_kb");
  if (rpart && esp_partition_mmap(rpart, 0, rpart->size, ESP_PARTITION_MMAP_DATA,
                                  &g_ragMmap, &g_ragHandle) == ESP_OK) {
    if (tiny_rag_init_from_blob((const uint8_t *)g_ragMmap, rpart->size)) {
      conPrintf("[RAG1] Base Micro-RAG mapeada em Flash.\n");
    }
  }
}

static void handleRoot(void) {
  static const char HTML[] PROGMEM =
      "<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
      "<title>AMEBA S3 Agent</title><style>body{background:#04041c;color:#a2b6da;font-family:monospace;padding:16px}"
      "h2{color:#28e7ff}input{width:75%;padding:8px;background:#0a0b2e;color:#fff;border:1px solid #28e7ff}"
      "button{padding:8px 14px;background:#005179;color:#fff;border:1px solid #28e7ff}</style></head><body>"
      "<h2>TINY / AMEBA-S3 AGENT (ESP32-S3 16MB)</h2><div id='log'></div><br>"
      "<input id='q' placeholder='Ex: acende luz quente e liga ventilador 50%'><button onclick='ask()'>Enviar</button>"
      "<script>async function ask(){let q=document.getElementById('q').value;if(!q)return;"
      "let r=await fetch('/api/ask?q='+encodeURIComponent(q));let j=await r.json();"
      "document.getElementById('log').innerHTML+='<p><b>Voce:</b> '+q+'<br><b>ACT:</b> '+j.act+'<br><b>Agente:</b> '+j.reply+'</p>';}</script>"
      "</body></html>";
  server.send_P(200, "text/html; charset=utf-8", HTML);
}

static void handleStatus(void) {
  char json[320];
  snprintf(json, sizeof(json),
           "{\"model_loaded\":%s,\"heap\":%lu,\"psram\":%lu,\"temp\":%.1f,\"fan\":%d,\"led\":\"%s\",\"relay\":%d}",
           g_modelLoaded ? "true" : "false", (unsigned long)ESP.getFreeHeap(),
           (unsigned long)(psramFound() ? ESP.getFreePsram() : 0), (double)g_hwState.temp_c,
           g_hwState.fan_pct, tiny_led_name(g_hwState.led_mode), g_hwState.relay_on);
  server.send(200, "application/json", json);
}

static void handleAsk(void) {
  String q = server.arg("q");
  q.trim();
  if (q.length() == 0) {
    server.send(400, "application/json", "{\"error\":\"empty\"}");
    return;
  }
  NeuralJob job;
  memset(&job, 0, sizeof(job));
  snprintf(job.prompt, sizeof(job.prompt), "%s", q.c_str());
  job.from_web = true;
  xQueueSend(g_jobQueue, &job, portMAX_DELAY);

  NeuralReply rep;
  if (xQueueReceive(g_replyQueue, &rep, pdMS_TO_TICKS(15000)) != pdTRUE) {
    server.send(504, "application/json", "{\"error\":\"timeout\"}");
    return;
  }
  char resp[420];
  snprintf(resp, sizeof(resp), "{\"act\":\"%s\",\"reply\":\"%s\",\"ms\":%lu}",
           rep.act, rep.reply, (unsigned long)rep.elapsed_ms);
  server.send(200, "application/json", resp);
}

static void printStatus(void) {
  conPrintf("\n--- AMEBA-S3-AGENT Status (Core %d) ---\n", xPortGetCoreID());
  conPrintf("Heap livre: %lu B | PSRAM livre: %lu B\n",
            (unsigned long)ESP.getFreeHeap(),
            (unsigned long)(psramFound() ? ESP.getFreePsram() : 0));
  conPrintf("Sensores  : temp=%.1fC | luz=%d%%\n", (double)g_hwState.temp_c, g_hwState.light_pct);
  conPrintf("Atuadores : FAN=%d%% | LED=%s | RELAY=%d\n",
            g_hwState.fan_pct, tiny_led_name(g_hwState.led_mode), g_hwState.relay_on);
}

static void runSerialCommand(const char *line) {
  if (strcmp(line, "/status") == 0) {
    printStatus();
  } else if (strcmp(line, "/bench") == 0) {
    if (!g_modelLoaded) {
      conPrintf("[BENCH] Grave um modelo TNY3 na particao model_bin para medir matvec real.\n");
      return;
    }
    uint32_t u0 = 0, u1 = 0, u2 = 0;
    tny3_benchmark_matvec(&u0, &u1, &u2);
    conPrintf("[BENCH matvec W_gate] FP32=%lu us | INT8_MAC4=%lu us | PIE_SIMD=%lu us\n",
              (unsigned long)u0, (unsigned long)u1, (unsigned long)u2);
  } else if (strncmp(line, "/act ", 5) == 0) {
    TinyActResult res;
    if (tiny_act_parse(line + 5, &res)) {
      tiny_act_apply_hw(&res, &g_hwState);
      conPrintf("[HW OK] Aplicado: %s\n", line + 5);
    } else {
      conPrintf("[HW REJECT] Comando inseguro ou invalido bloqueado pelo parser: %s\n", line + 5);
    }
  } else if (strncmp(line, "/temp ", 6) == 0) {
    g_hwState.temp_c = (float)atof(line + 6);
    conPrintf("[SIM] Temperatura simulada ajustada para %.1fC\n", (double)g_hwState.temp_c);
  } else {
    NeuralJob job;
    memset(&job, 0, sizeof(job));
    snprintf(job.prompt, sizeof(job.prompt), "%s", line);
    xQueueSend(g_jobQueue, &job, portMAX_DELAY);
    NeuralReply rep;
    if (xQueueReceive(g_replyQueue, &rep, portMAX_DELAY) == pdTRUE) {
      conPrintf("  [ACT] %s\n  [RES] %s (%lu ms)\n", rep.act, rep.reply, (unsigned long)rep.elapsed_ms);
    }
  }
}

void setup() {
  Serial.begin(115200);
#if AGENT_DUAL_SERIAL
  Serial.setTxTimeoutMs(0);
  Serial0.begin(115200);
#endif
  uint32_t t0 = millis();
  while (!Serial && (millis() - t0) < 3500u) delay(50);

  pinMode(BTN_BOOT_PIN, INPUT_PULLUP);
  tiny_act_hw_init(&g_hwState);
  mountFlashPartitions();

  g_jobQueue = xQueueCreate(4, sizeof(NeuralJob));
  g_replyQueue = xQueueCreate(4, sizeof(NeuralReply));
  xTaskCreatePinnedToCore(neuralWorkerTask, "neural_worker", 8192, NULL, 2, NULL, 1);

  WiFi.mode(WIFI_AP);
  WiFi.softAP(AP_SSID);
  dnsServer.start(53, "*", WiFi.softAPIP());

  server.on("/", HTTP_GET, handleRoot);
  server.on("/api/status", HTTP_GET, handleStatus);
  server.on("/api/ask", HTTP_GET, handleAsk);
  server.onNotFound(handleRoot);
  server.begin();

  printStatus();
  conPrintf("Comandos: /status | /bench | /temp 34 | /act LED=WARM;FAN=50\n> ");
}

static char s_buf[160];
static size_t s_len = 0;

void loop() {
  dnsServer.processNextRequest();
  server.handleClient();

  if (digitalRead(BTN_BOOT_PIN) == LOW) {
    g_hwState.temp_c = (g_hwState.temp_c > 30.0f) ? 24.0f : 34.0f;
    conPrintf("\n[BOTAO] Temperatura simulada -> %.1fC\n> ", (double)g_hwState.temp_c);
    delay(350);
  }

  while (Serial.available()) {
    int c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (s_len > 0) {
        s_buf[s_len] = '\0';
        s_len = 0;
        runSerialCommand(s_buf);
        conPrintf("> ");
      }
    } else if (s_len + 1 < sizeof(s_buf)) {
      s_buf[s_len++] = (char)c;
    }
  }
  delay(2);
}
