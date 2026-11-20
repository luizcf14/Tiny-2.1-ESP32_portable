#include "tiny_act.h"

#include <ctype.h>
#include <stdlib.h>
#include <string.h>

#ifdef ARDUINO
#include <Arduino.h>
#ifndef TINY_LED_RGB_PIN
#ifdef RGB_BUILTIN
#define TINY_LED_RGB_PIN RGB_BUILTIN
#else
#define TINY_LED_RGB_PIN 48
#endif
#endif
#define TINY_FAN_PWM_PIN 4
#define TINY_RELAY_PIN   5
#endif

static bool parse_strict_int(const char *s, int *out) {
    if (!s || !s[0]) return false;
    for (int i = 0; s[i]; i++) {
        if (!isdigit((unsigned char)s[i])) return false;
    }
    long v = strtol(s, NULL, 10);
    if (v < 0 || v > 100000) return false;
    *out = (int)v;
    return true;
}

bool tiny_act_parse(const char *act_str, TinyActResult *out) {
    if (!out) return false;
    memset(out, 0, sizeof(*out));
    if (!act_str) return false;

    char buf[96];
    size_t n = 0;
    while (act_str[n] && n + 1 < sizeof(buf)) {
        buf[n] = act_str[n];
        n++;
    }
    buf[n] = '\0';

    char *start = buf;
    while (*start == ' ') start++;
    while (n > 0 && buf[n - 1] == ' ') buf[--n] = '\0';
    if (start[0] == '\0') return false;

    if (strcmp(start, "NONE") == 0) {
        out->valid = true;
        out->is_none = true;
        return true;
    }

    char *tok = start;
    while (tok && *tok) {
        char *semi = strchr(tok, ';');
        if (semi) *semi = '\0';

        char *eq = strchr(tok, '=');
        if (!eq || eq == tok || eq[1] == '\0') {
            memset(out, 0, sizeof(*out));
            return false;
        }
        *eq = '\0';
        const char *key = tok;
        const char *val = eq + 1;

        if (strcmp(key, "FAN") == 0) {
            int pct = -1;
            if (!parse_strict_int(val, &pct) || pct < 0 || pct > 100) {
                memset(out, 0, sizeof(*out));
                return false;
            }
            out->has_fan = true;
            out->fan_pct = pct;
        } else if (strcmp(key, "RELAY") == 0) {
            int st = -1;
            if (!parse_strict_int(val, &st) || (st != 0 && st != 1)) {
                memset(out, 0, sizeof(*out));
                return false;
            }
            out->has_relay = true;
            out->relay_on = st;
        } else if (strcmp(key, "LED") == 0) {
            out->has_led = true;
            if (strcmp(val, "OFF") == 0) out->led_mode = TINY_LED_OFF;
            else if (strcmp(val, "WARM") == 0) out->led_mode = TINY_LED_WARM;
            else if (strcmp(val, "COOL") == 0) out->led_mode = TINY_LED_COOL;
            else if (strcmp(val, "ALERT") == 0) out->led_mode = TINY_LED_ALERT;
            else {
                memset(out, 0, sizeof(*out));
                return false;
            }
        } else {
            /* Qualquer chave fora da whitelist é rejeitada com segurança */
            memset(out, 0, sizeof(*out));
            return false;
        }

        tok = semi ? (semi + 1) : NULL;
    }

    out->valid = true;
    return true;
}

const char *tiny_led_name(TinyLedMode m) {
    switch (m) {
        case TINY_LED_WARM:  return "warm";
        case TINY_LED_COOL:  return "cool";
        case TINY_LED_ALERT: return "alert";
        default:             return "off";
    }
}

void tiny_act_hw_init(TinyActState *state) {
    if (state) {
        state->fan_pct = 0;
        state->led_mode = TINY_LED_OFF;
        state->relay_on = 0;
        state->temp_c = 25.0f;
        state->light_pct = 45;
    }
#ifdef ARDUINO
    pinMode(TINY_FAN_PWM_PIN, OUTPUT);
    pinMode(TINY_RELAY_PIN, OUTPUT);
    analogWrite(TINY_FAN_PWM_PIN, 0);
    digitalWrite(TINY_RELAY_PIN, LOW);
#ifdef RGB_BUILTIN
    neopixelWrite(TINY_LED_RGB_PIN, 0, 0, 0);
#endif
#endif
}

void tiny_act_apply_hw(const TinyActResult *res, TinyActState *state) {
    if (!res || !res->valid || res->is_none || !state) return;

    if (res->has_fan) {
        state->fan_pct = res->fan_pct;
#ifdef ARDUINO
        analogWrite(TINY_FAN_PWM_PIN, (res->fan_pct * 255) / 100);
#endif
    }
    if (res->has_relay) {
        state->relay_on = res->relay_on;
#ifdef ARDUINO
        digitalWrite(TINY_RELAY_PIN, res->relay_on ? HIGH : LOW);
#endif
    }
    if (res->has_led) {
        state->led_mode = res->led_mode;
#ifdef ARDUINO
        uint8_t r = 0, g = 0, b = 0;
        if (res->led_mode == TINY_LED_WARM) { r = 64; g = 34; b = 6; }
        else if (res->led_mode == TINY_LED_COOL) { r = 40; g = 55; b = 64; }
        else if (res->led_mode == TINY_LED_ALERT) { r = 80; g = 0; b = 0; }
        neopixelWrite(TINY_LED_RGB_PIN, r, g, b);
#endif
    }
}
