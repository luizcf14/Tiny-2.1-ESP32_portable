/*
 * TINY 2.1 - Transformer Neural Network ported to ESP32 (Arduino IDE)
 * =====================================================================
 * Configuracao recomendada na Arduino IDE:
 *   - Placa:            ESP32 Dev Module / ESP32S3 Dev Module / ESP32C3 Dev Module
 *   - USB CDC On Boot:  Enabled (em placas com USB nativo: S3, C3, C6, S2)
 *   - Partition Scheme: Huge APP (3MB No OTA/1MB SPIFFS) ou o partitions.csv local (2MB+)
 *   - Flash Mode:       DIO (recomendado se houver falha de boot em S3/C3)
 *   - Upload Speed:     921600 (ou 115200)
 *   - Serial Monitor:   115200 baud (Newline / Both NL & CR / No Line Ending)
 *
 * Interfaces:
 *   1) Serial USB (115200 baud) - pergunte direto ou use /help, /demo, /0../9, /wifi
 *   2) Wi-Fi AP + Web Neon UI  - conecte na rede "TINY-2.1-ESP32" e abra http://192.168.4.1
 */

#include <WiFi.h>
#include <WebServer.h>
#include <Preferences.h>
#include <ESPmDNS.h>
#include <esp_system.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

#include "tiny_engine.h"
#include "tiny_weights.h"
#include "web_ui.h"

#if defined(ARDUINO_USB_CDC_ON_BOOT) && ARDUINO_USB_CDC_ON_BOOT
#define TINY_DUAL_SERIAL 1
#else
#define TINY_DUAL_SERIAL 0
#endif

/* Credenciais opcionais do roteador (deixe "" para configurar pela Web ou Serial) */
static const char *DEFAULT_STA_SSID = "";
static const char *DEFAULT_STA_PASS = "";

/* Rede Access Point propria do ESP32 (sempre ativa para acesso imediato) */
static const char *AP_SSID = "TINY-2.1-ESP32";
static const char *AP_PASS = ""; /* vazio = rede aberta; ou defina >= 8 caracteres */

static const int DEFAULT_CREATIVITY = 5;

WebServer server(80);
Preferences prefs;

static int currentCreativity = DEFAULT_CREATIVITY;
static bool isThinking = false;
static bool wifiStartedOnce = false;
static bool pendingWifiReconnect = false;
static uint32_t wifiReconnectAt = 0;
static String staSsid;
static String staPass;
static bool mdnsStarted = false;

static char serialBuf[128];
static size_t serialLen = 0;
static bool serialBurst = false;
static uint32_t lastRxMs = 0;
static uint32_t lastEolMs = 0;
static int lastEolChar = 0;

static bool userInteracted = false;
static int bootReminders = 0;
static uint32_t lastReminderMs = 0;
#if TINY_DUAL_SERIAL
static bool cdcWasConnected = false;
static bool pendingCdcBanner = false;
static uint32_t cdcBannerAt = 0;
#endif

static const char *DEMO_QUESTIONS[] = {
    "what is the capital of estonia",
    "is a whale a fish",
    "what is 7 times 8",
    "which countries border estonia",
    "spell computer",
    "what is the symbol for gold",
    "how do you say thank you in spanish",
    "what is a group of lions called",
    "what comes after 199",
    "what time is it",
    "tell me a joke",
    "name the continents",
    NULL
};

/* ---- Console helpers (espelha Serial + Serial0 quando CDC On Boot ativo) ---- */

static void conWrite(uint8_t b)
{
    Serial.write(b);
#if TINY_DUAL_SERIAL
    Serial0.write(b);
#endif
}

static void conPrint(const char *s)
{
    Serial.print(s);
#if TINY_DUAL_SERIAL
    Serial0.print(s);
#endif
}

static void conPrintln(const char *s)
{
    Serial.println(s);
#if TINY_DUAL_SERIAL
    Serial0.println(s);
#endif
}

static void conPrintf(const char *fmt, ...)
{
    char buf[384];
    va_list args;
    va_start(args, fmt);
    int n = vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    if (n <= 0) return;
    size_t len = (size_t)n < sizeof(buf) ? (size_t)n : sizeof(buf) - 1u;
    Serial.write((const uint8_t *)buf, len);
#if TINY_DUAL_SERIAL
    Serial0.write((const uint8_t *)buf, len);
#endif
}

static bool conAvailable(void)
{
    if (Serial.available()) return true;
#if TINY_DUAL_SERIAL
    if (Serial0.available()) return true;
#endif
    return false;
}

static void conDrain(void)
{
    while (Serial.available()) Serial.read();
#if TINY_DUAL_SERIAL
    while (Serial0.available()) Serial0.read();
#endif
}

/* ---- JSON helpers ---- */

static void appendJsonEscaped(String &dst, const char *src)
{
    for (; *src; src++) {
        char c = *src;
        if (c == '"' || c == '\\') {
            dst += '\\';
            dst += c;
        } else if ((unsigned char)c >= 32u) {
            dst += c;
        }
    }
}

static void formatLayerArray(char *out, size_t cap, const float *acts, int layers)
{
    size_t pos = 0;
    pos += (size_t)snprintf(out + pos, cap - pos, "[");
    for (int l = 0; l < layers && pos + 16 < cap; l++) {
        pos += (size_t)snprintf(out + pos, cap - pos, "%s%.4f", (l > 0 ? "," : ""), (double)acts[l]);
    }
    snprintf(out + pos, cap - pos, "]");
}

/* ---- Wi-Fi setup ---- */

static void startWifi()
{
    prefs.begin("tiny", false);
    staSsid = prefs.getString("ssid", "");
    staPass = prefs.getString("pass", "");
    prefs.end();

    if (staSsid.length() == 0 && DEFAULT_STA_SSID[0] != '\0') {
        staSsid = DEFAULT_STA_SSID;
        staPass = DEFAULT_STA_PASS;
    }

    const char *apPass = (AP_PASS[0] != '\0') ? AP_PASS : NULL;

    if (wifiStartedOnce) {
        WiFi.disconnect(true);
        delay(100);
    }
    wifiStartedOnce = true;

    if (staSsid.length() > 0) {
        WiFi.mode(WIFI_AP_STA);
        WiFi.softAP(AP_SSID, apPass);
        conPrintf("Conectando ao Wi-Fi \"%s\"", staSsid.c_str());
        WiFi.begin(staSsid.c_str(), staPass.c_str());
        uint32_t t0 = millis();
        while (WiFi.status() != WL_CONNECTED && (millis() - t0) < 8000u) {
            delay(250);
            conPrint(".");
        }
        conPrintln("");
        if (WiFi.status() == WL_CONNECTED) {
            conPrintf("Wi-Fi conectado! IP: http://%s\n", WiFi.localIP().toString().c_str());
            if (!mdnsStarted && MDNS.begin("tiny")) {
                MDNS.addService("http", "tcp", 80);
                mdnsStarted = true;
                conPrintln("mDNS ativo: http://tiny.local");
            }
        } else {
            conPrintln("Nao conectou ao roteador em 8s; seguindo com Access Point.");
        }
    } else {
        WiFi.mode(WIFI_AP);
        WiFi.softAP(AP_SSID, apPass);
    }

    conPrintf("Access Point \"%s\" -> http://%s\n",
              AP_SSID, WiFi.softAPIP().toString().c_str());
}

/* ---- SSE streaming callbacks ---- */

static void sseOnStep(int pos, int total, const float *layer_act, int layers, void *user)
{
    if (pos != 0 && (pos % 2) != 0 && pos != total - 1) return;
    Client *client = (Client *)user;
    if (!client || !client->connected()) return;

    char acts[144];
    char pkt[240];
    formatLayerArray(acts, sizeof(acts), layer_act, layers);
    int n = snprintf(pkt, sizeof(pkt),
                     "data: {\"t\":\"s\",\"i\":%d,\"n\":%d,\"a\":%s}\n\n",
                     pos, total, acts);
    if (n > 0) {
        client->write((const uint8_t *)pkt, (size_t)n);
    }
}

static void sseOnLetter(char ch, float confidence, const float *layer_act, int layers, void *user)
{
    Client *client = (Client *)user;
    if (!client || !client->connected()) return;

    char chEsc[4];
    if (ch == '"' || ch == '\\') {
        chEsc[0] = '\\';
        chEsc[1] = ch;
        chEsc[2] = '\0';
    } else {
        chEsc[0] = ch;
        chEsc[1] = '\0';
    }

    char acts[144];
    char pkt[256];
    formatLayerArray(acts, sizeof(acts), layer_act, layers);
    int n = snprintf(pkt, sizeof(pkt),
                     "data: {\"t\":\"l\",\"c\":\"%s\",\"f\":%.4f,\"a\":%s}\n\n",
                     chEsc, (double)confidence, acts);
    if (n > 0) {
        client->write((const uint8_t *)pkt, (size_t)n);
    }
}

/* ---- HTTP Route Handlers ---- */

static void handleRoot()
{
    server.sendHeader("Content-Encoding", "gzip");
    server.sendHeader("Cache-Control", "no-cache");
    server.send_P(200, "text/html; charset=utf-8",
                  (const char *)INDEX_HTML_GZ, INDEX_HTML_GZ_LEN);
}

static void handleStatus()
{
    const TinyModelInfo *info = tiny_get_info();
    String json;
    json.reserve(320);

    bool staOk = (WiFi.status() == WL_CONNECTED);
    json += "{";
    json += "\"d\":" + String(info ? info->d : 0);
    json += ",\"layers\":" + String(info ? info->layers : 0);
    json += ",\"heads\":" + String(info ? info->heads : 0);
    json += ",\"hidden\":" + String(info ? info->hidden : 0);
    json += ",\"ram_bytes\":" + String(info ? info->ram_bytes : 0);
    json += ",\"flash_bytes\":" + String(info ? info->flash_bytes : 0);
    json += ",\"heap\":" + String(ESP.getFreeHeap());
    json += ",\"creativity\":" + String(currentCreativity);
    json += ",\"ap_ssid\":\""; appendJsonEscaped(json, AP_SSID); json += "\"";
    json += ",\"ap_ip\":\"" + WiFi.softAPIP().toString() + "\"";
    json += ",\"sta_connected\":" + String(staOk ? "true" : "false");
    json += ",\"sta_ssid\":\""; appendJsonEscaped(json, staSsid.c_str()); json += "\"";
    json += ",\"sta_ip\":\"" + (staOk ? WiFi.localIP().toString() : String("")) + "\"";
    json += "}";

    server.sendHeader("Cache-Control", "no-cache");
    server.send(200, "application/json", json);
}

static void handleAsk()
{
    if (isThinking) {
        server.send(409, "application/json", "{\"error\":\"TINY is busy\"}");
        return;
    }

    String rawQ = server.arg("q");
    int creat = currentCreativity;
    if (server.hasArg("c")) {
        int c = server.arg("c").toInt();
        if (c >= 0 && c <= 9) creat = c;
    }

    char cleanQ[TINY_MAXQ + 1];
    if (tiny_clean_question(rawQ.c_str(), cleanQ) == 0) {
        server.send(400, "application/json", "{\"error\":\"Empty question\"}");
        return;
    }

    isThinking = true;
    auto client = server.client();
    client.setNoDelay(true);

    client.print("HTTP/1.1 200 OK\r\n"
                 "Content-Type: text/event-stream; charset=utf-8\r\n"
                 "X-Content-Type-Options: nosniff\r\n"
                 "Cache-Control: no-cache\r\n"
                 "Connection: close\r\n"
                 "Access-Control-Allow-Origin: *\r\n\r\n");

    char ans[TINY_CTX + 1];
    uint32_t elapsed = 0;

    int letters = tiny_think(cleanQ, ans, creat, &elapsed,sseOnStep, sseOnLetter, (void *)&client);

    if (client.connected()) {
        char donePkt[128];
        int n = snprintf(donePkt, sizeof(donePkt),
                         "data: {\"t\":\"d\",\"ltr\":%d,\"ms\":%lu,\"heap\":%u}\n\n",
                         letters, (unsigned long)elapsed, (unsigned)ESP.getFreeHeap());
        if (n > 0) {
            client.write((const uint8_t *)donePkt, (size_t)n);
        }
        client.flush();
    }
    client.stop();

    float lps = elapsed > 0 ? ((float)letters * 1000.0f / (float)elapsed) : 0.0f;
    conPrintf("\n[WEB] > %s\n  %s\n  [%d letters in %lu ms | %.1f ltr/s | heap %u B]\n> ",
              cleanQ, ans, letters, (unsigned long)elapsed, (double)lps,
              (unsigned)ESP.getFreeHeap());

    isThinking = false;
}

static void handleWifiSave()
{
    String newSsid = server.arg("ssid");
    String newPass = server.arg("pass");
    newSsid.trim();

    prefs.begin("tiny", false);
    if (newSsid.length() == 0) {
        prefs.remove("ssid");
        prefs.remove("pass");
    } else {
        prefs.putString("ssid", newSsid);
        prefs.putString("pass", newPass);
    }
    prefs.end();

    String msg = (newSsid.length() == 0)
        ? "Credenciais removidas. Mantendo apenas Access Point."
        : ("Credenciais salvas! Conectando em " + newSsid + "...");

    String resp = "{\"ok\":true,\"msg\":\"";
    appendJsonEscaped(resp, msg.c_str());
    resp += "\"}";
    server.send(200, "application/json", resp);

    pendingWifiReconnect = true;
    wifiReconnectAt = millis() + 350u;
}

/* ---- Serial Terminal ---- */

static void serialOnLetter(char ch, float confidence, const float *layer_act, int layers, void *user)
{
    (void)confidence;
    (void)layer_act;
    (void)layers;
    (void)user;
    conWrite((uint8_t)ch);
}

static void printStatusSerial()
{
    const TinyModelInfo *info = tiny_get_info();
    conPrintln("\n--- TINY 2.1 ESP32 Status ---");
    if (info) {
        conPrintf("Modelo:       %d layers, %d wide, %d heads, HID %d\n",
                  info->layers, info->d, info->heads, info->hidden);
        conPrintf("Flash pesos:  %lu bytes (Zero-Copy PROGMEM/DROM)\n",
                  (unsigned long)info->flash_bytes);
        conPrintf("RAM motor:    %lu bytes (KV Cache + buffers)\n",
                  (unsigned long)info->ram_bytes);
    }
    conPrintf("Heap livre:   %u bytes\n", (unsigned)ESP.getFreeHeap());
    conPrintf("Creativity:   %d (0 = exato .. 9 = criativo)\n", currentCreativity);
    conPrintf("Access Point: %s -> http://%s\n",
              AP_SSID, WiFi.softAPIP().toString().c_str());
    if (WiFi.status() == WL_CONNECTED) {
        conPrintf("Wi-Fi STA:    %s -> http://%s (http://tiny.local)\n",
                  staSsid.c_str(), WiFi.localIP().toString().c_str());
    } else {
        conPrintln("Wi-Fi STA:    Desconectado");
    }
    conPrintln("-----------------------------");
}

static void printHelpSerial()
{
    conPrintln("\nComandos disponiveis no Serial:");
    conPrintln("  <pergunta>          Pergunta ao TINY (ex: what is the capital of peru)");
    conPrintln("  /0 .. /9            Ajusta creativity (0 = deterministico, 5 = padrao)");
    conPrintln("  /demo               Executa perguntas de demonstracao");
    conPrintln("  /status             Mostra memoria RAM/Flash e IPs Wi-Fi");
    conPrintln("  /wifi <ssid> <pwd>  Salva e conecta no roteador Wi-Fi");
    conPrintln("  /wifi clear         Apaga credenciais do roteador");
    conPrintln("  cls                 Limpa a tela ANSI");
    conPrintln("  /help               Mostra esta ajuda");
}

static void askSerial(const char *cleanQ)
{
    char ans[TINY_CTX + 1];
    uint32_t elapsed = 0;
    isThinking = true;
    conPrint("  ");
    int letters = tiny_think(cleanQ, ans, currentCreativity, &elapsed,
                             NULL, serialOnLetter, NULL);
    isThinking = false;
    if (letters == 0) {
        conPrint(".");
    }
    float lps = elapsed > 0 ? ((float)letters * 1000.0f / (float)elapsed) : 0.0f;
    conPrintf("\n   [%d letters in %lu ms | %.1f ltr/s]\n",
              letters, (unsigned long)elapsed, (double)lps);
}

static void runSerialDemo()
{
    conPrintln("\n[DEMO] Pressione qualquer tecla no Serial para parar...");
    for (int i = 0; DEMO_QUESTIONS[i]; i++) {
        if (conAvailable()) {
            conDrain();
            conPrintln("[DEMO] Interrompido.");
            return;
        }
        conPrintf("\n> %s\n", DEMO_QUESTIONS[i]);
        askSerial(DEMO_QUESTIONS[i]);
        server.handleClient();
        uint32_t t0 = millis();
        while (millis() - t0 < 800u) {
            server.handleClient();
            if (conAvailable()) {
                conDrain();
                conPrintln("[DEMO] Interrompido.");
                return;
            }
            delay(20);
        }
    }
    conPrintln("[DEMO] Concluido.");
}

static void handleSerialLine(const char *line)
{
    if (strncasecmp(line, "/wifi ", 6) == 0) {
        const char *rest = line + 6;
        while (*rest == ' ') rest++;
        if (strcasecmp(rest, "clear") == 0) {
            prefs.begin("tiny", false);
            prefs.remove("ssid");
            prefs.remove("pass");
            prefs.end();
            conPrintln("Credenciais Wi-Fi apagadas.");
            startWifi();
            return;
        }
        const char *sp = strchr(rest, ' ');
        String ssid, pass;
        if (sp) {
            ssid = String(rest).substring(0, (unsigned int)(sp - rest));
            pass = String(sp + 1);
            pass.trim();
        } else {
            ssid = String(rest);
        }
        ssid.trim();
        if (ssid.length() == 0) {
            conPrintln("Uso: /wifi <ssid> <senha>  ou  /wifi clear");
            return;
        }
        prefs.begin("tiny", false);
        prefs.putString("ssid", ssid);
        prefs.putString("pass", pass);
        prefs.end();
        startWifi();
        return;
    }

    char q[TINY_MAXQ + 1];
    if (tiny_clean_question(line, q) == 0) {
        return;
    }

    if (strcasecmp(q, "help") == 0 || strcasecmp(q, "/help") == 0) {
        printHelpSerial();
    } else if (strcasecmp(q, "cls") == 0) {
        conPrint("\033[2J\033[H");
    } else if (strcasecmp(q, "/status") == 0 || strcasecmp(q, "/mem") == 0) {
        printStatusSerial();
    } else if (strcasecmp(q, "/demo") == 0) {
        runSerialDemo();
    } else if (q[0] == '/' && q[1] >= '0' && q[1] <= '9' && q[2] == '\0') {
        currentCreativity = q[1] - '0';
        conPrintf("Creativity = %d\n", currentCreativity);
    } else if (strncasecmp(q, "/creativity ", 12) == 0 && q[12] >= '0' && q[12] <= '9') {
        currentCreativity = q[12] - '0';
        conPrintf("Creativity = %d\n", currentCreativity);
    } else {
        askSerial(q);
    }
}

static void pollStream(Stream &st)
{
    while (st.available()) {
        int ch = st.read();
        if (ch < 0) break;
        if (ch == '\r' || ch == '\n') {
            userInteracted = true;
            serialBurst = false;
            if (serialLen > 0) {
                serialBuf[serialLen] = '\0';
                serialLen = 0;
                conPrintln("");
                handleSerialLine(serialBuf);
                conPrint("\n> ");
            } else if (!(ch == '\n' && lastEolChar == '\r' && (uint32_t)(millis() - lastEolMs) < 50u)) {
                conPrint("\n[TINY 2.1 pronto] > ");
            }
            lastEolChar = ch;
            lastEolMs = millis();
        } else if (ch == '\b' || ch == 127) {
            userInteracted = true;
            if (serialLen > 0) {
                serialLen--;
                conPrint("\b \b");
            }
        } else if (ch >= 32 && ch <= 126 && serialLen + 1 < sizeof(serialBuf)) {
            userInteracted = true;
            if (st.available() > 0) {
                serialBurst = true;
            }
            serialBuf[serialLen++] = (char)ch;
            conWrite((uint8_t)ch);
            lastRxMs = millis();
        }
    }
}

static void pollSerial()
{
    pollStream(Serial);
#if TINY_DUAL_SERIAL
    pollStream(Serial0);
#endif
    if (serialLen > 0 && serialBurst && (uint32_t)(millis() - lastRxMs) >= 250u) {
        serialBuf[serialLen] = '\0';
        serialLen = 0;
        serialBurst = false;
        conPrintln("");
        handleSerialLine(serialBuf);
        conPrint("\n> ");
    }
}

/* ---- Arduino Lifecycle ---- */

void setup()
{
    Serial.begin(115200);
#if TINY_DUAL_SERIAL
    Serial.setTxTimeoutMs(0);
    Serial0.begin(115200);
#endif

    uint32_t tWait = millis();
    while (!Serial && (millis() - tWait) < 3500u) {
        delay(50);
    }
    delay(500);

    conPrintln("\n==========================================");
    conPrintln("  TINY 2.1 - Transformer Neural Interface");
    conPrintln("==========================================");

    srand((unsigned int)esp_random());

    const char *err = NULL;
    if (!tiny_init(TINY_BIN_DATA, TINY_BIN_SIZE, &err)) {
        conPrintf("ERRO ao iniciar TINY: %s\n", err ? err : "desconhecido");
        while (true) delay(1000);
    }

    startWifi();

    server.on("/", HTTP_GET, handleRoot);
    server.on("/api/status", HTTP_GET, handleStatus);
    server.on("/api/ask", HTTP_GET, handleAsk);
    server.on("/api/wifi", HTTP_POST, handleWifiSave);
    server.begin();

    printStatusSerial();
    printHelpSerial();
    conPrint("\n> ");

    lastReminderMs = millis();
#if TINY_DUAL_SERIAL
    cdcWasConnected = (bool)Serial;
#endif
}

void loop()
{
    server.handleClient();
    pollSerial();

#if TINY_DUAL_SERIAL
    bool cdcNow = (bool)Serial;
    if (cdcNow && !cdcWasConnected) {
        cdcWasConnected = true;
        pendingCdcBanner = true;
        cdcBannerAt = millis() + 250u;
    } else if (!cdcNow) {
        cdcWasConnected = false;
        pendingCdcBanner = false;
    }
    if (pendingCdcBanner && (int32_t)(millis() - cdcBannerAt) >= 0) {
        pendingCdcBanner = false;
        bootReminders = 2;
        printStatusSerial();
        printHelpSerial();
        conPrint("\n> ");
    }
#endif

    if (!userInteracted && bootReminders < 2 && (uint32_t)(millis() - lastReminderMs) >= 4000u) {
        lastReminderMs = millis();
        bootReminders++;
        conPrintf("\n[TINY 2.1 ONLINE | Wi-Fi AP: %s -> http://%s | Digite uma pergunta ou /help]\n> ",
                  AP_SSID, WiFi.softAPIP().toString().c_str());
    }

    if (pendingWifiReconnect && (int32_t)(millis() - wifiReconnectAt) >= 0) {
        pendingWifiReconnect = false;
        startWifi();
    }
    delay(1);
}
