#ifndef TINY_ACT_H
#define TINY_ACT_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    TINY_LED_OFF = 0,
    TINY_LED_WARM = 1,
    TINY_LED_COOL = 2,
    TINY_LED_ALERT = 3
} TinyLedMode;

typedef struct {
    int fan_pct;          /* 0..100 */
    TinyLedMode led_mode; /* OFF, WARM, COOL, ALERT */
    int relay_on;         /* 0 ou 1 */
    float temp_c;         /* leitura de sensor ou simulada */
    int light_pct;        /* 0..100 */
} TinyActState;

typedef struct {
    bool valid;
    bool is_none;
    bool has_fan;
    int fan_pct;
    bool has_led;
    TinyLedMode led_mode;
    bool has_relay;
    int relay_on;
} TinyActResult;

void tiny_act_hw_init(TinyActState *state);
bool tiny_act_parse(const char *act_str, TinyActResult *out);
void tiny_act_apply_hw(const TinyActResult *res, TinyActState *state);
const char *tiny_led_name(TinyLedMode m);

#ifdef __cplusplus
}
#endif

#endif /* TINY_ACT_H */
