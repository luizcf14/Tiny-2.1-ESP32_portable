#ifndef TINY_SESSION_H
#define TINY_SESSION_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "tiny_engine.h"

#ifdef __cplusplus
extern "C" {
#endif

#define TINY_MAX_SESSIONS   12
#define TINY_SESSION_TTL_MS 1800000UL

typedef enum {
    TINY_ASK_NONE = 0,
    TINY_ASK_WHO_ARE_YOU = 1,
    TINY_ASK_WHAT_HARDWARE = 2
} TinyPendingAsk;

typedef struct {
    bool active;
    char sid[16];
    char user_name[24];
    char user_hw[28];
    char last_topic[28];
    char last_q[TINY_MAXQ + 1];
    char last_ans[96];
    uint8_t pending_ask;
    uint16_t turns;
    uint32_t last_seen_ms;
} TinySession;

void tiny_sessions_init(void);
void tiny_sanitize_sid(const char *raw_sid, const char *fallback_ip, char *sid_out);
TinySession *tiny_session_get(const char *sid, uint32_t now_ms);
void tiny_session_reset(TinySession *sess, uint32_t now_ms);
int tiny_sessions_active_count(uint32_t now_ms);

/*
 * Prepare a turn using per-session context memory.
 * Returns true if handled directly in direct_ans_out without invoking tiny_think().
 * Returns false if tiny_think() should run on model_q_out (which may be rewritten).
 */
bool tiny_session_prepare(TinySession *sess,
                          const char *raw_line,
                          const char *clean_q,
                          char *model_q_out,
                          char *direct_ans_out,
                          bool *was_rewritten);

void tiny_session_commit(TinySession *sess,
                         const char *clean_q,
                         const char *effective_q,
                         const char *ans,
                         uint32_t now_ms);

#ifdef __cplusplus
}
#endif

#endif /* TINY_SESSION_H */
