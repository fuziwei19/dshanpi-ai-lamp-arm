/**
 * @file app_feishu.c
 * @brief 飞书 IM 频道 <-> 云端 AI 双向桥（实现）
 *
 * 组件来源：https://github.com/tuya/TClaw/tree/master/IM
 *   - im/channels/feishu_bot.c  WebSocket(长连接) + Protobuf 收，REST 发
 *   - im/bus/message_bus.c      入站/出站队列
 *   - im/proxy/http_proxy.c     HTTP CONNECT / SOCKS5（可选）
 *   - im/certs/*                TLS 证书（open.feishu.cn 走内置 CA 包，不依赖 iot-dns）
 *
 * 线程模型（3 条线程）
 * -------------------
 *   AI 组件线程   : 事件回调 -> app_feishu_on_ai_event()  ← 只 memcpy + 放信号量
 *   feishu_in     : message_bus_pop_inbound -> ai_agent_send_text()（并负责重试启动）
 *   feishu_out    : 等信号量 / 轮询回合状态 -> feishu_send_message()  ← 真正的阻塞 HTTP 在这里
 *   feishu_bot 内部: feishu_ws_task（WS 长连接，IM 组件自带）
 *
 * ★ 「这一轮说完了没有」怎么判定
 * -----------------------------
 * 本版 TuyaOpen 里 `AI_USER_EVT_END` **只声明、没有任何地方 notify**（已全仓 grep 确认），
 * 所以不能拿它当回合结束。改用两个可靠信号：
 *   1) 文字流闭合：收到 AI_USER_EVT_TEXT_STREAM_STOP 且此后没再开新流
 *      （一轮里可能因为 MCP 工具调用出现多段文字，我们整轮累积）
 *   2) 会话状态回落：ai_mode_get_state() == AI_MODE_STATE_IDLE
 *      （即语音那条链路用的 LISTEN/THINK/SPEAK/IDLE 状态机，SPEAK->IDLE 精确对应播报结束）
 * 两者同时满足才 flush + 回发飞书；另外兜一个 45s 硬超时，避免万一卡住就一直不回。
 *
 * ★ 为什么出站要单独一条线程
 *   事件回调跑在 AI 组件的收流线程上，feishu_send_message() 是一次 HTTPS 请求
 *   （几百 ms ~ 2s）。直接在回调里发，会把整条 AI 流水线卡住（TTS 断续、情绪灯延迟）。
 *
 * @copyright Copyright (c) 2026
 */

#include "app_feishu.h"

#include <stdio.h>
#include <string.h>

#include "tal_api.h"

#include "ai_agent.h"        /* ai_agent_send_text() */
#include "ai_manage_mode.h"  /* ai_mode_get_state() / AI_MODE_STATE_IDLE */
#include "tuya_ai_agent.h"   /* tuya_ai_agent_is_ready() */
#include "app_vision.h"      /* app_vision_maybe_capture()：拍照意图 */
#include "cJSON.h"

#include "im_api.h"          /* message_bus_* / feishu_* / http_proxy_* */

/***********************************************************
************************macro define************************
***********************************************************/
#define APP_FEISHU_CONF           "/etc/feishu.json"  /* 运行时凭证（可选） */
#define APP_FEISHU_REPLY_MAX      (4096)              /* 单轮回复文本上限 */
#define APP_FEISHU_POLL_MS        (300)               /* 出站线程轮询步长 */
#define APP_FEISHU_START_RETRY_MS (5000)              /* 启动失败首次重试间隔 */
#define APP_FEISHU_START_RETRY_MAX_MS (30000)         /* 重试间隔上限（退避）*/
#define APP_FEISHU_IDLE_WAIT_MS   (60000)             /* 等 AI 空闲的最长时间 */
#define APP_FEISHU_IDLE_TICK_MS   (200)               /* 注入前空闲检测步长 */
#define APP_FEISHU_TURN_TIMEOUT_MS (45000)            /* 回合兜底超时 */
#define APP_FEISHU_QUIET_MS       (800)               /* 文字流闭合后的静默窗 */
#define APP_FEISHU_THRD_STACK     (6144)
#define APP_FEISHU_THRD_PRIO      (4)                 /* 与 app_local_ctl 一致 */

/***********************************************************
***********************variable define**********************
***********************************************************/
static BOOL_T          s_inited  = FALSE;
static volatile BOOL_T s_running = FALSE;
static volatile BOOL_T s_started = FALSE;        /* feishu_bot_start() 是否成功过 */
static BOOL_T          s_ever_tried = FALSE;     /* 是否尝试过启动（配合 s_last_try_ms 限流） */
static uint32_t        s_last_try_ms = 0;
static uint32_t        s_retry_ms    = APP_FEISHU_START_RETRY_MS; /* 当前重试间隔（失败则翻倍） */

static MUTEX_HANDLE    s_lock    = NULL;

/* 最近一次收到消息的会话（open_id 或 oc_xxx），回复目标 */
static char            s_chat_id[96] = {0};

/* ---- 一轮飞书对话的状态 ---- */
static BOOL_T          s_pending         = FALSE;  /* 这轮要回发飞书 */
static BOOL_T          s_busy_seen       = FALSE;  /* 见过 AI 真的开始干活（收到过文字） */
static BOOL_T          s_stream_open     = FALSE;  /* TEXT_STREAM 是否还开着 */
static uint32_t        s_stream_stop_ms  = 0;      /* 文字流闭合时刻（0=本轮还没收到过 STOP） */
static uint32_t        s_inject_ms       = 0;      /* 本轮注入时刻 */
static uint32_t        s_reply_len       = 0;
static BOOL_T          s_reply_truncated = FALSE;
static char            s_reply_buf[APP_FEISHU_REPLY_MAX] = {0};

/* 出站单槽信箱（AI 一轮只有一条最终回复，不需要队列） */
static SEM_HANDLE      s_out_sem  = NULL;
static char            s_out_chat[96] = {0};
static char           *s_out_text = NULL;

/* 出站待发照片槽（一轮最多一张；与文本分开，照片先发、AI 的描述随后到） */
static uint8_t        *s_out_photo      = NULL;
static uint32_t        s_out_photo_len  = 0;
static char            s_out_photo_chat[96] = {0};
static BOOL_T          s_photo_queued   = FALSE;   /* 本轮是否已经收下过一张（去重） */

static THREAD_HANDLE   s_in_thrd  = NULL;
static THREAD_HANDLE   s_out_thrd = NULL;

/***********************************************************
***********************function define**********************
***********************************************************/

/* -------- 运行时配置 /etc/feishu.json -------- */

/**
 * {"app_id":"cli_xxx","app_secret":"yyy","allow_from":"ou_aaa,ou_bbb"}
 *
 * 只覆盖出现过的字段；文件不存在就沿用 im_secrets.h / KV 的值。
 * 必须在 feishu_bot_init() **之前** 调用：set_* 会写 KV，init 再从 KV 读回来。
 */
static void __load_conf(void)
{
    FILE *fp = fopen(APP_FEISHU_CONF, "rb");
    if (NULL == fp) {
        PR_NOTICE("feishu: %s not found (fallback: im_secrets.h / KV)", APP_FEISHU_CONF);
        return;
    }

    char   buf[2048];
    size_t n = fread(buf, 1, sizeof(buf) - 1, fp);
    fclose(fp);
    buf[n] = '\0';

    cJSON *root = cJSON_Parse(buf);
    if (NULL == root) {
        PR_ERR("feishu: %s is not valid JSON, ignored", APP_FEISHU_CONF);
        return;
    }

    cJSON *app_id     = cJSON_GetObjectItem(root, "app_id");
    cJSON *app_secret = cJSON_GetObjectItem(root, "app_secret");
    cJSON *allow_from = cJSON_GetObjectItem(root, "allow_from");

    if (cJSON_IsString(app_id) && NULL != app_id->valuestring && '\0' != app_id->valuestring[0]) {
        (void)feishu_set_app_id(app_id->valuestring);
        PR_NOTICE("feishu: app_id loaded from %s", APP_FEISHU_CONF);
    }
    if (cJSON_IsString(app_secret) && NULL != app_secret->valuestring && '\0' != app_secret->valuestring[0]) {
        (void)feishu_set_app_secret(app_secret->valuestring);
    }
    if (cJSON_IsString(allow_from) && NULL != allow_from->valuestring) {
        (void)feishu_set_allow_from(allow_from->valuestring);
        PR_NOTICE("feishu: allow_from = \"%s\"", allow_from->valuestring);
    }

    cJSON_Delete(root);
}

/* -------- 出站：单槽信箱 -------- */

/* 调用者必须持有 s_lock */
static BOOL_T __queue_out_locked(const char *chat_id, const char *text)
{
    if (NULL == text || '\0' == text[0]) {
        return FALSE;
    }

    char *copy = tal_malloc(strlen(text) + 1);
    if (NULL == copy) {
        PR_ERR("feishu: OOM while copying reply");
        return FALSE;
    }
    strcpy(copy, text);

    if (NULL != s_out_text) {
        tal_free(s_out_text); /* 丢旧的：同一轮只会有一条最终回复 */
    }
    s_out_text = copy;
    snprintf(s_out_chat, sizeof(s_out_chat), "%s", (NULL != chat_id && '\0' != chat_id[0]) ? chat_id : s_chat_id);
    return TRUE;
}

/** 收下一张待发照片（非阻塞：只是一次 malloc + memcpy）。调用者必须持有 s_lock。 */
static BOOL_T __queue_photo_locked(const char *chat_id, const uint8_t *jpeg, uint32_t len)
{
    if (NULL == jpeg || 0 == len) {
        return FALSE;
    }

    uint8_t *copy = tal_malloc(len);
    if (NULL == copy) {
        PR_ERR("feishu: OOM while copying photo (%u bytes)", (unsigned)len);
        return FALSE;
    }
    memcpy(copy, jpeg, len);

    if (NULL != s_out_photo) {
        tal_free(s_out_photo); /* 丢旧的：一轮只发第一张 */
    }
    s_out_photo     = copy;
    s_out_photo_len = len;
    snprintf(s_out_photo_chat, sizeof(s_out_photo_chat), "%s",
             (NULL != chat_id && '\0' != chat_id[0]) ? chat_id : s_chat_id);
    return TRUE;
}

/** 取走待发照片（取走即清空）；没有则返回 NULL 且 *len=0。调用者负责 tal_free。 */
static uint8_t *__take_photo_locked(char *chat, size_t chat_size, uint32_t *len)
{
    uint8_t *jpeg = s_out_photo;

    *len            = s_out_photo_len;
    s_out_photo     = NULL;
    s_out_photo_len = 0;
    snprintf(chat, chat_size, "%s", s_out_photo_chat);
    return jpeg;
}

/** 收尾本轮：把累积文本投进信箱（非阻塞）。调用者必须持有 s_lock。 */
static BOOL_T __flush_locked(const char *reason)
{
    BOOL_T posted = FALSE;

    if (s_pending) {
        if (s_reply_len > 0) {
            posted = __queue_out_locked(s_chat_id, s_reply_buf);
        } else {
            PR_WARN("feishu: turn finished (%s) but no text was produced", reason);
        }
        PR_NOTICE("feishu: turn finished (%s), reply=%u bytes: \"%.160s\"%s", reason, (unsigned)s_reply_len,
                  s_reply_buf, s_reply_truncated ? " ...[TRUNCATED]" : "");
    }

    s_pending         = FALSE;
    s_busy_seen       = FALSE;
    s_stream_open     = FALSE;
    s_stream_stop_ms  = 0;
    s_reply_len       = 0;
    s_reply_buf[0]    = '\0';
    s_reply_truncated = FALSE;
    s_photo_queued    = FALSE;

    return posted;
}

/* -------- 入站：等 AI 空闲 -> 喂给云端 -------- */

static BOOL_T __wait_idle(uint32_t max_ms)
{
    uint32_t start = tal_system_get_millisecond();

    for (;;) {
        if (!s_running) {
            return FALSE;
        }
        /* 两个条件：AI client 已连上云端，且当前没有在听/在想/在说 */
        if (tuya_ai_agent_is_ready() && (AI_MODE_STATE_IDLE == ai_mode_get_state())) {
            return TRUE;
        }
        if ((uint32_t)(tal_system_get_millisecond() - start) >= max_ms) {
            return FALSE;
        }
        tal_system_sleep(APP_FEISHU_IDLE_TICK_MS);
    }
}

/**
 * 启动飞书 WS（要拿 tenant token + WS 端点，依赖公网）。
 * 启动时刻网络通常还没就绪，所以由入站线程反复重试；内部限流 + 失败退避，
 * 避免凭证填错时每 5s 就朝 open.feishu.cn 打一次 HTTPS。
 */
static void __try_start(void)
{
    if (s_started || !feishu_is_configured()) {
        return;
    }

    uint32_t now = tal_system_get_millisecond();
    if (s_ever_tried && ((uint32_t)(now - s_last_try_ms) < s_retry_ms)) {
        return;
    }
    s_ever_tried  = TRUE;
    s_last_try_ms = now;

    OPERATE_RET rt = feishu_bot_start();
    if (OPRT_OK == rt) {
        s_started  = TRUE;
        s_retry_ms = APP_FEISHU_START_RETRY_MS;
        PR_NOTICE("feishu: bot started (app_id=%s)", feishu_get_app_id());
    } else {
        /* 退避：5s -> 10s -> 20s -> 30s（封顶）。凭证错/网络不通时不至于一直刷。 */
        if (s_retry_ms < APP_FEISHU_START_RETRY_MAX_MS) {
            s_retry_ms = (s_retry_ms * 2u > APP_FEISHU_START_RETRY_MAX_MS) ? APP_FEISHU_START_RETRY_MAX_MS
                                                                          : s_retry_ms * 2u;
        }
        PR_WARN("feishu: start failed rt=%d (next retry in %dms)", rt, s_retry_ms);
    }
}

static void __in_task(void *arg)
{
    (void)arg;

    PR_NOTICE("feishu: inbound bridge started");

    while (s_running) {
        /* 始终尝试拉起 WS（未配置时内部直接返回；失败时限流重试）。
           注意：**不要**因为没起来就不消费总线，否则消息会永远卡在队列里。 */
        __try_start();

        im_msg_t msg = {0};
        if (OPRT_OK != message_bus_pop_inbound(&msg, APP_FEISHU_POLL_MS)) {
            continue;
        }
        if (NULL == msg.content) {
            continue;
        }

        if (!s_started) {
            PR_WARN("feishu: inbound while bot not started (configured=%d), reply will be dropped",
                    (int)feishu_is_configured());
        }

        PR_NOTICE("feishu: inbound chat=%s len=%u text=\"%.80s\"", msg.chat_id, (unsigned)strlen(msg.content),
                  msg.content);

        /* 开一轮：记回复目标、清空上一轮残留 */
        tal_mutex_lock(s_lock);
        snprintf(s_chat_id, sizeof(s_chat_id), "%s", msg.chat_id);
        s_pending         = TRUE;
        s_busy_seen       = FALSE;
        s_stream_open     = FALSE;
        s_stream_stop_ms  = 0;
        s_reply_len       = 0;
        s_reply_buf[0]    = '\0';
        s_reply_truncated = FALSE;
        s_inject_ms       = 0;
        s_photo_queued    = FALSE;
        tal_mutex_unlock(s_lock);

        /* 拍照意图（「拍张照 / 看看前面」等，关键词表复用 app_vision）：
           交给 app_vision 去拍 —— 它拍完会走 app_feishu_on_photo() 把这帧交回来，
           同时自己也把「图 + 引导语」送给云端多模态 AI，于是用户会先收到照片、
           再收到 AI 的描述。命中时**不要**再 ai_agent_send_text()，否则白喂一遍。 */
        if (app_vision_maybe_capture(msg.content, (int)strlen(msg.content))) {
            PR_NOTICE("feishu: photo intent -> vision capture requested (chat=%s)", msg.chat_id);
            tal_mutex_lock(s_lock);
            s_inject_ms = tal_system_get_millisecond(); /* 超时兜底仍按此刻起算 */
            tal_mutex_unlock(s_lock);
            im_free(msg.content);
            continue;
        }

        OPERATE_RET rt   = OPRT_COM_ERROR;
        BOOL_T      busy = FALSE;

        if (__wait_idle(APP_FEISHU_IDLE_WAIT_MS)) {
            rt = ai_agent_send_text(msg.content);
            PR_NOTICE("feishu: ai_agent_send_text -> rt=%d", rt);
            if (OPRT_OK == rt) {
                /* 超时计时从「真正喂进去」开始算，否则等空闲的时间会把兜底超时提前耗尽 */
                tal_mutex_lock(s_lock);
                s_inject_ms = tal_system_get_millisecond();
                tal_mutex_unlock(s_lock);
            }
        } else {
            busy = TRUE;
            PR_WARN("feishu: ai busy for %dms, drop this message", APP_FEISHU_IDLE_WAIT_MS);
        }

        if (OPRT_OK != rt) {
            /* 喂不进去 / 等不到空闲：撤掉这轮，并给飞书一个明确交代 */
            tal_mutex_lock(s_lock);
            s_pending = FALSE;
            tal_mutex_unlock(s_lock);

            app_feishu_send_text(msg.chat_id, busy ? "（设备正忙，请稍后再发一次）"
                                                   : "（设备端 AI 暂时不可用，请稍后再试）");
        }

        im_free(msg.content);
    }

    PR_NOTICE("feishu: inbound bridge exit");
}

/* -------- 出站线程 -------- */

/* 这些事件在飞书那轮里太吵，不打印 */
static BOOL_T __event_is_noisy(int type)
{
    return (type == AI_USER_EVT_TTS_DATA) || (type == AI_USER_EVT_TTS_PRE) || (type == AI_USER_EVT_MIC_DATA);
}

/** 取出信箱里的待发文本（取走即清空）；没有则返回 NULL。调用者负责 tal_free。 */
static char *__take_out_locked(char *chat, size_t chat_size)
{
    char *text = s_out_text;
    s_out_text = NULL;
    snprintf(chat, chat_size, "%s", s_out_chat);
    return text;
}

/** 真正的阻塞 HTTPS 发送 —— 只被出站线程调用 */
static void __deliver(const char *chat, const char *text)
{
    if ('\0' == chat[0]) {
        PR_WARN("feishu: no chat_id, reply dropped: \"%.160s\"", text);
    } else if (!s_started) {
        PR_WARN("feishu: bot not started, reply dropped: \"%.160s\"", text);
    } else {
        OPERATE_RET rt = feishu_send_message(chat, text);
        PR_NOTICE("feishu: sent %u bytes to %s rt=%d: \"%.160s\"", (unsigned)strlen(text), chat, rt, text);
    }
}

/** 真正的阻塞图片上传 + 发送 —— 只被出站线程调用 */
static void __deliver_photo(const char *chat, const uint8_t *jpeg, uint32_t len)
{
    if ('\0' == chat[0]) {
        PR_WARN("feishu: no chat_id, photo dropped (%u bytes)", (unsigned)len);
    } else if (!s_started) {
        PR_WARN("feishu: bot not started, photo dropped (%u bytes)", (unsigned)len);
    } else {
        OPERATE_RET rt = feishu_send_image(chat, jpeg, len);
        PR_NOTICE("feishu: sent photo %u bytes to %s rt=%d", (unsigned)len, chat, rt);
    }
}

static void __out_task(void *arg)
{
    (void)arg;

    PR_NOTICE("feishu: outbound sender started");

    while (s_running) {
        /* 1) 等信箱有东西（顺便当成 300ms 轮询节拍） */
        (void)tal_semaphore_wait(s_out_sem, APP_FEISHU_POLL_MS);

        /* 2) 回合收尾判定：文字流闭合 + 静默窗 + 会话状态回落 IDLE；兜底超时 */
        tal_mutex_lock(s_lock);
        if (s_pending) {
            uint32_t now      = tal_system_get_millisecond();
            BOOL_T   timedout = (0 != s_inject_ms) &&
                                ((uint32_t)(now - s_inject_ms) >= APP_FEISHU_TURN_TIMEOUT_MS);
            /* 静默窗：文字流闭合后还要安静 N ms，避开 MCP 多段文字之间的短暂 IDLE */
            BOOL_T   quiet    = (0 == s_stream_stop_ms) ||
                                ((uint32_t)(now - s_stream_stop_ms) >= APP_FEISHU_QUIET_MS);

            if (s_busy_seen && !s_stream_open && quiet && (AI_MODE_STATE_IDLE == ai_mode_get_state())) {
                (void)__flush_locked("stream closed + ai idle");
            } else if (timedout) {
                (void)__flush_locked("timeout");
            }
        }
        tal_mutex_unlock(s_lock);

        /* 3) 待发照片优先：照片比 AI 的文字描述先到，先发照片观感更好 */
        uint8_t *jpeg                        = NULL;
        uint32_t jlen                        = 0;
        char     pchat[sizeof(s_out_photo_chat)] = {0};

        tal_mutex_lock(s_lock);
        jpeg = __take_photo_locked(pchat, sizeof(pchat), &jlen);
        tal_mutex_unlock(s_lock);

        if (NULL != jpeg) {
            __deliver_photo(pchat, jpeg, jlen);
            tal_free(jpeg);
        }

        /* 4) 把信箱里的文本发出去（无论是回合收尾还是主动 send_text 投递的） */
        char *text = NULL;
        char  chat[sizeof(s_out_chat)] = {0};

        tal_mutex_lock(s_lock);
        text = __take_out_locked(chat, sizeof(chat));
        tal_mutex_unlock(s_lock);

        if (NULL != text) {
            __deliver(chat, text);
            tal_free(text);
        }
    }

    PR_NOTICE("feishu: outbound sender exit");
}

/***********************************************************
***********************public api***************************
***********************************************************/

OPERATE_RET app_feishu_init(void)
{
    OPERATE_RET rt = OPRT_OK; /* TUYA_CALL_ERR_RETURN 要求作用域内有 rt */

    if (s_inited) {
        return OPRT_OK; /* 幂等 */
    }

    TUYA_CALL_ERR_RETURN(tal_mutex_create_init(&s_lock));
    TUYA_CALL_ERR_RETURN(tal_semaphore_create_init(&s_out_sem, 0, 8));

    /* 运行时凭证优先（改文件 + 重启即可，不用重新编译固件） */
    __load_conf();

    TUYA_CALL_ERR_LOG(message_bus_init());
    TUYA_CALL_ERR_LOG(http_proxy_init());

    rt = feishu_bot_init();
    if (OPRT_OK != rt) {
        PR_ERR("feishu: feishu_bot_init failed rt=%d", rt);
        return rt;
    }

    s_running = TRUE;

    THREAD_CFG_T cfg = {0};
    cfg.thrdname   = "feishu_in";
    cfg.priority   = APP_FEISHU_THRD_PRIO;
    cfg.stackDepth = APP_FEISHU_THRD_STACK;
    if (OPRT_OK != tal_thread_create_and_start(&s_in_thrd, NULL, NULL, __in_task, NULL, &cfg)) {
        PR_ERR("feishu: create inbound thread failed");
        s_running = FALSE;
        return OPRT_COM_ERROR;
    }

    memset(&cfg, 0, sizeof(cfg));
    cfg.thrdname   = "feishu_out";
    cfg.priority   = APP_FEISHU_THRD_PRIO;
    cfg.stackDepth = APP_FEISHU_THRD_STACK;
    if (OPRT_OK != tal_thread_create_and_start(&s_out_thrd, NULL, NULL, __out_task, NULL, &cfg)) {
        PR_ERR("feishu: create outbound thread failed");
        s_running = FALSE;
        return OPRT_COM_ERROR;
    }

    s_inited = TRUE;

    PR_NOTICE("feishu: bridge ready (configured=%d, app_id=%s, conf=%s)", (int)feishu_is_configured(),
              feishu_is_configured() ? feishu_get_app_id() : "(none)", APP_FEISHU_CONF);
    return OPRT_OK;
}

void app_feishu_on_ai_event(AI_NOTIFY_EVENT_T *event)
{
    if (NULL == event) {
        return;
    }

    /* 不是飞书触发的那一轮（比如语音唤醒）-> 完全不插手，绝不漏到飞书 */
    if (!s_pending) {
        return;
    }

    if (!__event_is_noisy((int)event->type)) {
        PR_DEBUG("feishu: ai event type=%d (state=%d)", (int)event->type, (int)ai_mode_get_state());
    }

    switch (event->type) {
    case AI_USER_EVT_TEXT_STREAM_START:
    case AI_USER_EVT_TEXT_STREAM_DATA: {
        /* 注：**不在 START 时清空缓冲** —— 带 MCP 工具调用的一轮会分成多段文字，
           整轮累积才是完整答复（STRART/DATA 都带内容）。 */
        AI_NOTIFY_TEXT_T *t = (AI_NOTIFY_TEXT_T *)event->data;

        tal_mutex_lock(s_lock);
        if (s_pending) {
            s_busy_seen   = TRUE;
            s_stream_open = (AI_USER_EVT_TEXT_STREAM_START == event->type) ? TRUE : s_stream_open;

            if (NULL != t && NULL != t->data && t->datalen > 0) {
                if ((uint32_t)(s_reply_len + t->datalen) < APP_FEISHU_REPLY_MAX) {
                    memcpy(s_reply_buf + s_reply_len, t->data, t->datalen);
                    s_reply_len += t->datalen;
                    s_reply_buf[s_reply_len] = '\0';
                } else if (!s_reply_truncated) {
                    s_reply_truncated = TRUE;
                    PR_WARN("feishu: reply exceeds %d bytes, truncated", APP_FEISHU_REPLY_MAX);
                }
            }
        }
        tal_mutex_unlock(s_lock);
    } break;

    case AI_USER_EVT_TEXT_STREAM_STOP: {
        /* 文字流闭合：真正的 flush 时机交给出站线程（要等会话状态回 IDLE + 静默窗） */
        tal_mutex_lock(s_lock);
        if (s_pending) {
            s_busy_seen      = TRUE;
            s_stream_open    = FALSE;
            s_stream_stop_ms = tal_system_get_millisecond();
        }
        tal_mutex_unlock(s_lock);
    } break;

    case AI_USER_EVT_TEXT_STREAM_ABORT:
    case AI_USER_EVT_CHAT_BREAK: {
        /* 用户打断：这一轮作废，不回发 */
        tal_mutex_lock(s_lock);
        s_pending        = FALSE;
        s_busy_seen      = FALSE;
        s_stream_open    = FALSE;
        s_stream_stop_ms = 0;
        s_reply_len      = 0;
        s_reply_buf[0]   = '\0';
        tal_mutex_unlock(s_lock);
        PR_NOTICE("feishu: turn aborted by user interrupt, nothing sent");
    } break;

    default:
        break;
    }
}

BOOL_T app_feishu_is_started(void)
{
    return s_started;
}

BOOL_T app_feishu_is_configured(void)
{
    return feishu_is_configured();
}

const char *app_feishu_chat_id(void)
{
    return s_chat_id;
}

const char *app_feishu_app_id(void)
{
    return feishu_get_app_id();
}

OPERATE_RET app_feishu_send_text(const char *chat_id, const char *text)
{
    if (NULL == s_lock) {
        return OPRT_RESOURCE_NOT_READY;
    }

    tal_mutex_lock(s_lock);
    BOOL_T ok = __queue_out_locked(chat_id, text);
    tal_mutex_unlock(s_lock);

    if (!ok) {
        return OPRT_INVALID_PARM;
    }

    (void)tal_semaphore_post(s_out_sem);
    return OPRT_OK;
}

void app_feishu_on_photo(const uint8_t *jpeg, uint32_t len)
{
    if (NULL == s_lock || NULL == jpeg || 0 == len) {
        return;
    }

    tal_mutex_lock(s_lock);

    /* 只有当下一轮飞书对话正在进行时才收这张图；否则是语音/本机命令拍的，直接丢 */
    if (!s_pending || s_photo_queued) {
        tal_mutex_unlock(s_lock);
        PR_NOTICE("feishu: photo not for feishu (pending=%d, queued=%d), ignored", (int)s_pending, (int)s_photo_queued);
        return;
    }

    if (__queue_photo_locked(s_chat_id, jpeg, len)) {
        s_photo_queued = TRUE;
    }
    tal_mutex_unlock(s_lock);

    (void)tal_semaphore_post(s_out_sem);
    PR_NOTICE("feishu: photo queued (%u bytes)", (unsigned)len);
}

OPERATE_RET app_feishu_send_image(const char *chat_id, const uint8_t *jpeg, uint32_t len)
{
    if (NULL == s_lock) {
        return OPRT_RESOURCE_NOT_READY;
    }

    tal_mutex_lock(s_lock);
    BOOL_T ok = __queue_photo_locked(chat_id, jpeg, len);
    tal_mutex_unlock(s_lock);

    if (!ok) {
        return OPRT_INVALID_PARM;
    }

    (void)tal_semaphore_post(s_out_sem);
    return OPRT_OK;
}

OPERATE_RET app_feishu_test_inject(const char *chat_id, const char *text)
{
    if (!s_inited) {
        return OPRT_RESOURCE_NOT_READY;
    }
    if (NULL == text || '\0' == text[0]) {
        return OPRT_INVALID_PARM;
    }

    im_msg_t in = {0};
    snprintf(in.channel, sizeof(in.channel), "%s", IM_CHAN_FEISHU);
    snprintf(in.chat_id, sizeof(in.chat_id), "%s",
             (NULL != chat_id && '\0' != chat_id[0]) ? chat_id : "ou_test_fake");

    in.content = im_strdup(text);
    if (NULL == in.content) {
        return OPRT_COM_ERROR;
    }

    OPERATE_RET rt = message_bus_push_inbound(&in);
    if (OPRT_OK != rt) {
        im_free(in.content);
    }
    PR_NOTICE("feishu: test inject chat=%s rt=%d", in.chat_id, rt);
    return rt;
}

OPERATE_RET app_feishu_clear_credentials(void)
{
    /* 写空串：feishu_bot_init() 读到空值时会忽略，等价于「没配」。
       注意必须在 init 之后调用（init 里的 set_* 是给下次启动用的）。 */
    OPERATE_RET r1 = feishu_set_app_id("");
    OPERATE_RET r2 = feishu_set_app_secret("");
    PR_NOTICE("feishu: credentials cleared (app_id rt=%d, app_secret rt=%d)", r1, r2);
    return (OPRT_OK == r1 && OPRT_OK == r2) ? OPRT_OK : OPRT_COM_ERROR;
}
