/**
 * @file app_emo_lamp.c
 * @brief 情绪灯效 + 空闲流水灯（独立线程 + 信号量 debounce，绝不阻塞对话主流程）
 *
 * 一、情绪灯效
 * ------------
 * TuyaOpen 的 skill_emotion.c 收到云端情绪技能后会
 *     ai_user_event_notify(AI_USER_EVT_EMOTION, &emo)
 * emo.name 取值来自 cAI_AGENT_EMO[]（21 个），本文件各配一个颜色。
 *
 * 二、空闲流水灯
 * ------------
 * chat_bot 运行且「空闲」时，灯环跑一个旋转流水灯（默认 chase）。
 * 「空闲」的判据直接用会话状态机 AI_MODE_STATE_IDLE（ai_manage_mode.h）：
 *   IDLE -> 空闲；LISTEN/UPLOAD/THINK/SPEAK -> 对话中，空闲灯必须熄灭。
 * 实测 SPEAK -> IDLE 精确发生在 TTS 播完那一刻，比「超时猜测」可靠得多。
 *
 * 三、尊重人工灯效（关键）
 * ------------------------
 * 如果灯已经是「被人为打开」的（on=true 且亮度>0，且不是本模块点的），
 * 则：不显示情绪色、也不点亮空闲流水灯 —— 完全不动人家的灯。
 *
 * 怎么区分「人点的」和「我们点的」？
 * ws2812d 每次真正改动灯效都会把 state.version 自增。本模块每次成功写灯后
 * 立即把应答里的 version 记进 sg_ver_owned。因此：
 *   state.on && state.bright>0:
 *       version == sg_ver_owned  -> 我们自己点的（空闲灯/情绪灯）
 *       否则                      -> 人工/其它程序点的  -> 让路
 * 一旦让路就不会再把灯抢回来；人工关闭灯之后本模块才恢复控制。
 *
 * 手动关灯的行为
 * --------------
 * 如果空闲流水灯正亮着时被人手动关掉（`{"cmd":"off"}` 或 ws2812d 被外部重置），
 * 本模块会「尊重」这个动作：停止自动点亮空闲灯（sg_idle_suppressed），
 * 直到下一次对话（收到情绪事件）才恢复自动 —— 这样用户想安静地关灯时
 * 不会 0.5 秒后又被点亮。想立刻恢复可重启 app。
 *
 * 灯控通道
 * --------
 * 板载 24 颗 WS2812 由常驻守护 ws2812d 驱动，UNIX socket /run/ws2812d.sock，
 * JSON 行协议：
 *   亮  {"cmd":"set","on":true,"color":"#FF2020","brightness":90,"effect":"static"}
 *   查  {"cmd":"state"}   -> {"ok":true,"state":{on,color,brightness,effect,count,version,color_hex}}
 *   关  {"cmd":"off"}
 * effect 可选：static / breathing / rainbow / chase / blink
 *
 * 配置 /etc/emo-lamp.json（全部可省）：
 *   {
 *     "enabled": true,          // 情绪灯效总开关
 *     "hold_ms": 2000,          // 情绪色显示时长
 *     "respect_manual": true,   // 人工点亮的灯不覆盖
 *     "idle_enabled": true,     // 空闲流水灯
 *     "idle_after_ms": 1200,    // 进入 IDLE 后延迟多久点亮
 *     "idle_effect": "chase",   // static/breathing/rainbow/chase/blink
 *     "idle_color": "#00C0FF",
 *     "idle_brightness": 35
 *   }
 *
 * @copyright Copyright (c) 2026
 */

#include "app_emo_lamp.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/un.h>
#include <unistd.h>

#include "tal_api.h"
#include "cJSON.h"
#include "ai_manage_mode.h"

/***********************************************************
************************macro define************************
***********************************************************/
#define APP_EMO_LED_SOCK        "/run/ws2812d.sock"
#define APP_EMO_CONF            "/etc/emo-lamp.json"

#define APP_EMO_NAME_MAX        (32)
#define APP_EMO_RESP_MAX        (640)
#define APP_EMO_CMD_MAX         (224)

#define APP_EMO_HOLD_MS_DEF     (2000)   /* 情绪色显示时长 */
#define APP_EMO_HOLD_MS_MIN     (300)
#define APP_EMO_HOLD_MS_MAX     (10000)

/* 空闲流水灯 */
#define APP_EMO_IDLE_AFTER_DEF  (1200)   /* 进入 IDLE 后延迟多久点亮 */
#define APP_EMO_IDLE_AFTER_MAX  (60000)
#define APP_EMO_IDLE_BRIGHT_DEF (35)
#define APP_EMO_IDLE_BRIGHT_MIN (1)
#define APP_EMO_IDLE_COLOR_DEF  "#00C0FF"
#define APP_EMO_IDLE_EFFECT_DEF "chase"

#define APP_EMO_THRD_STACK      (4096)
#define APP_EMO_THRD_PRIO       (4)
#define APP_EMO_SEM_MAX         (8)      /* 计数信号量上限，多来几次也不会 post 失败 */
#define APP_EMO_IO_TMO_MS       (400)    /* socket 收发超时，防止灯守护卡住把线程拖死 */
#define APP_EMO_TICK_MS         (500)    /* 线程空转/空闲检测节拍 */

/***********************************************************
***********************typedef define***********************
***********************************************************/
typedef struct {
    const char *name;      /* 云端情绪名（大写） */
    const char *hex;       /* 灯色 */
    int         bright;    /* 亮度 0-100 */
    const char *cn;        /* 中文注释（仅日志可读性） */
} EMO_COLOR_T;

/* ws2812d 的 state 快照 */
typedef struct {
    BOOL_T on;
    int    bright;
    int    version;        /* 每次改动自增；用来认领「是不是我们点的」 */
    char   hex[16];
    char   effect[24];
} LED_STATE_T;

/* 灯当前归谁管 */
typedef enum {
    OWN_FREE = 0,   /* 灯是灭的 */
    OWN_SELF,       /* 本模块点的（空闲灯 / 情绪灯） */
    OWN_USER,       /* 人工或其它程序点的 -> 让路 */
} LAMP_OWN_E;

/***********************************************************
***********************variable define**********************
***********************************************************/
/* 情绪 -> 颜色。顺序无关，查找用线性扫描（21 项，足够快） */
static const EMO_COLOR_T sg_emo_map[] = {
    { "ANGRY",      "#FF2020", 90, "愤怒" },
    { "SHOCKED",    "#B040FF", 85, "震惊" },
    { "SURPRISE",   "#B040FF", 85, "惊讶" },
    { "FEARFUL",    "#7030C0", 80, "害怕" },
    { "SAD",        "#2060FF", 80, "悲伤" },
    { "HAPPY",      "#FFC020", 85, "开心" },
    { "LAUGHING",   "#FFC020", 85, "大笑" },
    { "FUNNY",      "#FFC020", 85, "逗趣" },
    { "DELICIOUS",  "#FFA020", 85, "馋" },
    { "LOVING",     "#FF5FA0", 85, "喜欢" },
    { "KISSY",      "#FF5FA0", 85, "亲昵" },
    { "WINK",       "#FF5FA0", 80, "眨眼" },
    { "EMBARRASSED","#FF7040", 85, "害羞" },
    { "CONFIDENT",  "#00C0FF", 80, "自信" },
    { "COOL",       "#00C0FF", 80, "酷" },
    { "THINKING",   "#00C0A0", 75, "思考" },
    { "CONFUSED",   "#00C0A0", 75, "困惑" },
    { "RELAXED",    "#4060C0", 70, "放松" },
    { "SLEEP",      "#4060C0", 55, "困" },
    { "SILLY",      "#A0E000", 80, "调皮" },
    { "NEUTRAL",    "#FFE0B0", 60, "中性" },
};
#define EMO_MAP_CNT (sizeof(sg_emo_map) / sizeof(sg_emo_map[0]))

/* ws2812d 支持的动态效果 */
static const char *sg_effect_names[] = { "static", "breathing", "rainbow", "chase", "blink" };
#define EFFECT_CNT (sizeof(sg_effect_names) / sizeof(sg_effect_names[0]))

static SEM_HANDLE    sg_sem      = NULL;
static MUTEX_HANDLE  sg_lock     = NULL;
static THREAD_HANDLE sg_thrd     = NULL;
static volatile BOOL_T sg_running = FALSE;
static volatile BOOL_T sg_enabled = TRUE;
static int            sg_hold_ms  = APP_EMO_HOLD_MS_DEF;

static char           sg_pending[APP_EMO_NAME_MAX] = {0};   /* 待处理情绪名 */
static char           sg_last_color[16]            = {0};

/* ---- 灯的所有权 / 空闲流水灯 ---- */
static int            sg_ver_owned  = -1;      /* 本模块最后一次写灯后拿到的 version */
static BOOL_T         sg_idle_lit   = FALSE;   /* 空闲流水灯是否由本模块点亮 */
static BOOL_T         sg_idle_suppressed = FALSE; /* 灯被人手动关掉 -> 暂停自动点亮 */
static SYS_TIME_T     sg_idle_since = 0;       /* 进入 IDLE 的时刻（0=尚未记录） */

static BOOL_T         sg_respect_manual = TRUE; /* 人工点亮的灯不覆盖 */
static BOOL_T         sg_idle_enabled   = TRUE;
static int            sg_idle_after_ms  = APP_EMO_IDLE_AFTER_DEF;
static int            sg_idle_bright    = APP_EMO_IDLE_BRIGHT_DEF;
static char           sg_idle_color[16]  = APP_EMO_IDLE_COLOR_DEF;
static char           sg_idle_effect[24] = APP_EMO_IDLE_EFFECT_DEF;

/***********************************************************
***********************function define**********************
***********************************************************/
/* -------------------- 小工具 -------------------- */

/* 情绪名归一化：去空白 + 转大写（云端一般是 "ANGRY" 这种） */
static void __norm_name(const char *in, char *out, size_t cap)
{
    size_t i = 0;

    if (NULL == in || cap < 2) {
        if (cap > 0) { out[0] = '\0'; }
        return;
    }
    while (*in == ' ' || *in == '\t' || *in == '\r' || *in == '\n') { in++; }
    for (; *in != '\0' && i + 1 < cap; in++) {
        char c = *in;
        if (c == ' ' || c == '\t' || c == '\r' || c == '\n') { continue; }
        if (c >= 'a' && c <= 'z') { c = (char)(c - 'a' + 'A'); }
        out[i++] = c;
    }
    out[i] = '\0';
}

static const EMO_COLOR_T *__lookup(const char *name)
{
    size_t i;

    for (i = 0; i < EMO_MAP_CNT; i++) {
        if (0 == strcmp(sg_emo_map[i].name, name)) {
            return &sg_emo_map[i];
        }
    }
    /* 未知情绪 -> 暖白（NEUTRAL） */
    return &sg_emo_map[EMO_MAP_CNT - 1];
}

static uint32_t __elapsed_ms(SYS_TIME_T since)
{
    return (uint32_t)(tal_system_get_millisecond() - since);
}

static BOOL_T __effect_valid(const char *e)
{
    size_t i;

    if (NULL == e) { return FALSE; }
    for (i = 0; i < EFFECT_CNT; i++) {
        if (0 == strcmp(sg_effect_names[i], e)) { return TRUE; }
    }
    return FALSE;
}

/* -------------------- 与 ws2812d 通信 -------------------- */

/* 发一行 JSON，收一行 JSON。0=成功 */
static int __led_send(const char *request, char *resp, size_t resp_cap)
{
    int                fd;
    struct sockaddr_un addr;
    struct timeval     tv;
    size_t             used = 0;
    size_t             off  = 0;
    size_t             want;
    ssize_t            n;

    if (NULL == request) {
        return -1;
    }
    if (resp != NULL && resp_cap > 0) {
        resp[0] = '\0';
    }

    fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) {
        return -1;
    }

    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    snprintf(addr.sun_path, sizeof(addr.sun_path), "%s", APP_EMO_LED_SOCK);

    tv.tv_sec  = 0;
    tv.tv_usec = APP_EMO_IO_TMO_MS * 1000;
    (void)setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
    (void)setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    if (0 != connect(fd, (struct sockaddr *)&addr, sizeof(addr))) {
        (void)close(fd);
        return -1;
    }

    want = strlen(request);
    while (off < want) {
        n = send(fd, request + off, want - off, 0);
        if (n > 0) { off += (size_t)n; continue; }
        if (n < 0 && EINTR == errno) { continue; }
        (void)close(fd);
        return -1;
    }
    (void)send(fd, "\n", 1, 0);

    if (resp != NULL && resp_cap > 1) {
        while (used + 1 < resp_cap) {
            n = recv(fd, resp + used, resp_cap - 1 - used, 0);
            if (n <= 0) {
                break;
            }
            used += (size_t)n;
            resp[used] = '\0';
            if (NULL != memchr(resp, '\n', used)) {
                break;
            }
        }
    }

    (void)close(fd);
    return (off == want) ? 0 : -1;
}

/* 解析 {"ok":true,"state":{...}}；成功返回 TRUE */
static BOOL_T __parse_state(const char *resp, LED_STATE_T *st)
{
    cJSON *root  = NULL;
    cJSON *state = NULL;
    cJSON *j;
    BOOL_T ok    = FALSE;

    if (NULL == resp || NULL == st) {
        return FALSE;
    }
    memset(st, 0, sizeof(*st));
    st->version = -1;

    root = cJSON_Parse(resp);
    if (NULL == root) {
        return FALSE;
    }
    state = cJSON_GetObjectItem(root, "state");
    if (cJSON_IsObject(state)) {
        j = cJSON_GetObjectItem(state, "on");
        if (cJSON_IsBool(j)) { st->on = cJSON_IsTrue(j); }

        j = cJSON_GetObjectItem(state, "brightness");
        if (cJSON_IsNumber(j)) { st->bright = j->valueint; }

        j = cJSON_GetObjectItem(state, "version");
        if (cJSON_IsNumber(j)) { st->version = j->valueint; }

        j = cJSON_GetObjectItem(state, "color_hex");
        if (cJSON_IsString(j) && NULL != j->valuestring) {
            snprintf(st->hex, sizeof(st->hex), "%s", j->valuestring);
        }

        j = cJSON_GetObjectItem(state, "effect");
        if (cJSON_IsString(j) && NULL != j->valuestring) {
            snprintf(st->effect, sizeof(st->effect), "%s", j->valuestring);
        }
        ok = TRUE;
    }
    cJSON_Delete(root);
    return ok;
}

/* 读一次灯效状态 */
static BOOL_T __led_query(LED_STATE_T *st)
{
    char resp[APP_EMO_RESP_MAX];

    if (0 != __led_send("{\"cmd\":\"state\"}", resp, sizeof(resp))) {
        return FALSE;
    }
    return __parse_state(resp, st);
}

/* 写灯效，并把应答里的 version 记成「我们写的」 */
static BOOL_T __led_apply(const char *req, LED_STATE_T *st)
{
    char resp[APP_EMO_RESP_MAX];

    if (0 != __led_send(req, resp, sizeof(resp))) {
        return FALSE;
    }
    if (!__parse_state(resp, st)) {
        return FALSE;
    }
    if (st->version >= 0) {
        sg_ver_owned = st->version;
    }
    return TRUE;
}

/* 灯当前归谁管 */
static LAMP_OWN_E __who_owns(const LED_STATE_T *st)
{
    if (!st->on || st->bright <= 0) {
        return OWN_FREE;
    }
    if (sg_ver_owned >= 0 && st->version == sg_ver_owned) {
        return OWN_SELF;
    }
    return OWN_USER;
}

/* -------------------- 灯效动作（都在独立线程里跑） -------------------- */

/* 点亮 / 维持空闲流水灯。
 * 每个 tick 都按灯的真实状态决策（不能只信本地标记，否则灯被外部关掉后会一直卡住）：
 *   OWN_SELF  -> 还在我们手里，保持
 *   OWN_USER  -> 被人工点亮，让路（也不点亮空闲灯）
 *   OWN_FREE  -> 灯是灭的：
 *                  之前是我们点着的  -> 说明被人手动关掉了，尊重，不再自动点亮
 *                  其它情况          -> 正常点亮空闲灯
 */
static void __idle_on(void)
{
    char        req[APP_EMO_CMD_MAX];
    LED_STATE_T st;

    if (!sg_idle_enabled || sg_idle_suppressed) {
        return;
    }
    if (!__led_query(&st)) {
        return;
    }

    switch (__who_owns(&st)) {
        case OWN_SELF:
            sg_idle_lit = TRUE;
            return;
        case OWN_USER:
            /* 灯被人点着 -> 不抢，也不点亮 */
            sg_idle_lit = FALSE;
            return;
        default:
            break;   /* OWN_FREE */
    }

    if (sg_idle_lit) {
        /* 我们点着的灯被人关掉了 -> 尊重，熄灭空闲灯且不再自动点亮 */
        sg_idle_lit       = FALSE;
        sg_idle_suppressed = TRUE;
        PR_NOTICE("emo_lamp: idle lamp was turned off externally, stay off "
                  "(will resume after next conversation)");
        return;
    }

    snprintf(req, sizeof(req),
             "{\"cmd\":\"set\",\"on\":true,\"color\":\"%s\",\"brightness\":%d,\"effect\":\"%s\"}",
             sg_idle_color, sg_idle_bright, sg_idle_effect);

    if (__led_apply(req, &st)) {
        sg_idle_lit = TRUE;
        PR_DEBUG("emo_lamp: idle '%s' on (%s bright=%d)",
                 sg_idle_effect, sg_idle_color, sg_idle_bright);
    }
}

/* 熄灭空闲流水灯（只关自己点的那盏） */
static void __idle_off(void)
{
    LED_STATE_T st;

    if (!sg_idle_lit) {
        return;
    }
    sg_idle_lit = FALSE;

    if (!__led_query(&st)) {
        return;
    }
    if (OWN_SELF == __who_owns(&st)) {
        (void)__led_apply("{\"cmd\":\"off\"}", &st);
        PR_DEBUG("emo_lamp: idle off");
    }
}

/* 显示一个情绪色。返回 TRUE = 真的显示了；FALSE = 被人工灯效挡住 */
static BOOL_T __emo_show(const char *name)
{
    const EMO_COLOR_T *e = __lookup(name);
    char               req[APP_EMO_CMD_MAX];
    LED_STATE_T        st;

    sg_idle_since = 0;   /* 有情绪 = 不空闲，空闲计时从头来 */

    /* 有对话了 -> 解除「被人手动关灯」的暂停，空闲灯效恢复自动 */
    if (sg_idle_suppressed) {
        sg_idle_suppressed = FALSE;
        PR_DEBUG("emo_lamp: idle lamp effect resumed (new conversation)");
    }

    if (sg_respect_manual) {
        if (!__led_query(&st)) {
            return FALSE;
        }
        if (OWN_USER == __who_owns(&st)) {
            PR_NOTICE("emo_lamp: skip %s(%s) -- lamp is manually ON, respect user",
                      name, e->cn);
            return FALSE;
        }
    }

    /* 覆盖前先摘掉「空闲灯」标记：马上要变成情绪色了 */
    sg_idle_lit = FALSE;

    snprintf(req, sizeof(req),
             "{\"cmd\":\"set\",\"on\":true,\"color\":\"%s\",\"brightness\":%d,\"effect\":\"static\"}",
             e->hex, e->bright);

    if (__led_apply(req, &st)) {
        snprintf(sg_last_color, sizeof(sg_last_color), "%s", e->hex);
        PR_NOTICE("emo_lamp: %s(%s) -> %s bright=%d", name, e->cn, e->hex, e->bright);
        return TRUE;
    }
    PR_WARN("emo_lamp: set failed (%s)", name);
    return FALSE;
}

/* 情绪显示结束 -> 关灯（之后由空闲检测决定是否点亮流水灯） */
static void __emo_end(void)
{
    LED_STATE_T st;

    if (__led_query(&st) && OWN_SELF == __who_owns(&st)) {
        (void)__led_apply("{\"cmd\":\"off\"}", &st);
    }
    sg_last_color[0] = '\0';
    PR_DEBUG("emo_lamp: emo end");
}

/* 空闲检测（每个 tick 调一次）：
 *   对话中        -> 熄灭空闲灯
 *   空闲且无人占用 -> 延迟 idle_after_ms 后点亮流水灯
 */
static void __idle_check(void)
{
    if (!sg_running) {
        return;
    }

    if (!sg_enabled || !sg_idle_enabled) {
        __idle_off();
        return;
    }

    /* 「空闲」= 会话状态机在 IDLE（SPEAK/LISTEN/THINK/UPLOAD 都算忙） */
    if (AI_MODE_STATE_IDLE != ai_mode_get_state()) {
        sg_idle_since = 0;
        __idle_off();
        return;
    }

    if (0 == sg_idle_since) {
        sg_idle_since = tal_system_get_millisecond();
        return;   /* 刚空闲下来，先等一下再点亮 */
    }
    if (__elapsed_ms(sg_idle_since) < (uint32_t)sg_idle_after_ms) {
        return;
    }

    __idle_on();
}

/* 从共享状态取走待处理情绪名；有则返回 TRUE */
static BOOL_T __take_pending(char *out, size_t cap)
{
    BOOL_T has = FALSE;

    if (NULL != sg_lock) {
        tal_mutex_lock(sg_lock);
    }
    if (sg_pending[0] != '\0') {
        snprintf(out, cap, "%s", sg_pending);
        sg_pending[0] = '\0';
        has = TRUE;
    }
    if (NULL != sg_lock) {
        tal_mutex_unlock(sg_lock);
    }
    return has;
}

/* -------------------- 工作线程 -------------------- */

static void __emo_task(void *arg)
{
    char name[APP_EMO_NAME_MAX];

    (void)arg;

    while (sg_running) {
        /* 等新情绪（带超时，超时那一拍顺便做空闲灯检测） */
        if (OPRT_OK != tal_semaphore_wait(sg_sem, APP_EMO_TICK_MS)) {
            __idle_check();
            continue;
        }
        if (!sg_running) {
            break;
        }
        if (!sg_enabled) {
            /* 关掉了：把积压的事件丢掉 */
            (void)__take_pending(name, sizeof(name));
            continue;
        }
        if (!__take_pending(name, sizeof(name))) {
            continue;
        }

        if (!__emo_show(name)) {
            continue;   /* 人工灯效占用 / 写灯失败 -> 本轮不显示 */
        }

        /* 显示 hold_ms；期间每来一个新情绪就立刻刷新并重新计时（debounce） */
        for (;;) {
            if (OPRT_OK != tal_semaphore_wait(sg_sem, (uint32_t)sg_hold_ms)) {
                break;   /* 超时 -> 显示结束 */
            }
            if (!sg_running || !sg_enabled) {
                break;
            }
            if (!__take_pending(name, sizeof(name))) {
                continue; /* 空信号量，继续等剩余时间 */
            }
            if (!__emo_show(name)) {
                break;    /* 中途被人工灯效接管 -> 收工 */
            }
        }

        if (!sg_running) {
            break;
        }
        __emo_end();

        /* 情绪结束后如果已经空闲，让空闲延迟重新计时 */
        if (AI_MODE_STATE_IDLE == ai_mode_get_state()) {
            sg_idle_since = 0;
        }
    }

    PR_NOTICE("emo_lamp: task exit");
}

/* -------------------- 对外接口 -------------------- */

static void __load_conf(void)
{
    FILE  *fp;
    long   sz;
    char  *buf;
    cJSON *root;
    cJSON *j;

    fp = fopen(APP_EMO_CONF, "r");
    if (NULL == fp) {
        return;
    }
    if (0 != fseek(fp, 0, SEEK_END)) { fclose(fp); return; }
    sz = ftell(fp);
    if (sz <= 0 || sz > 4096) { fclose(fp); return; }
    rewind(fp);

    buf = (char *)tal_malloc((size_t)sz + 1);
    if (NULL == buf) { fclose(fp); return; }

    if (fread(buf, 1, (size_t)sz, fp) != (size_t)sz) {
        tal_free(buf);
        fclose(fp);
        return;
    }
    buf[sz] = '\0';
    fclose(fp);

    root = cJSON_Parse(buf);
    tal_free(buf);
    if (NULL == root) {
        return;
    }

    j = cJSON_GetObjectItem(root, "enabled");
    if (cJSON_IsBool(j)) {
        sg_enabled = cJSON_IsTrue(j);
    }
    j = cJSON_GetObjectItem(root, "respect_manual");
    if (cJSON_IsBool(j)) {
        sg_respect_manual = cJSON_IsTrue(j);
    }
    j = cJSON_GetObjectItem(root, "hold_ms");
    if (cJSON_IsNumber(j)) {
        int v = j->valueint;
        if (v < APP_EMO_HOLD_MS_MIN) { v = APP_EMO_HOLD_MS_MIN; }
        if (v > APP_EMO_HOLD_MS_MAX) { v = APP_EMO_HOLD_MS_MAX; }
        sg_hold_ms = v;
    }

    /* ---- 空闲流水灯 ---- */
    j = cJSON_GetObjectItem(root, "idle_enabled");
    if (cJSON_IsBool(j)) {
        sg_idle_enabled = cJSON_IsTrue(j);
    }
    j = cJSON_GetObjectItem(root, "idle_after_ms");
    if (cJSON_IsNumber(j)) {
        int v = j->valueint;
        if (v < 0) { v = 0; }
        if (v > APP_EMO_IDLE_AFTER_MAX) { v = APP_EMO_IDLE_AFTER_MAX; }
        sg_idle_after_ms = v;
    }
    j = cJSON_GetObjectItem(root, "idle_brightness");
    if (cJSON_IsNumber(j)) {
        int v = j->valueint;
        if (v < APP_EMO_IDLE_BRIGHT_MIN) { v = APP_EMO_IDLE_BRIGHT_MIN; }
        if (v > 100) { v = 100; }
        sg_idle_bright = v;
    }
    j = cJSON_GetObjectItem(root, "idle_color");
    if (cJSON_IsString(j) && NULL != j->valuestring && j->valuestring[0] != '\0') {
        snprintf(sg_idle_color, sizeof(sg_idle_color), "%.*s", 15, j->valuestring);
    }
    j = cJSON_GetObjectItem(root, "idle_effect");
    if (cJSON_IsString(j) && NULL != j->valuestring) {
        char eff[24];
        snprintf(eff, sizeof(eff), "%.*s", 23, j->valuestring);
        if (__effect_valid(eff)) {
            snprintf(sg_idle_effect, sizeof(sg_idle_effect), "%s", eff);
        } else {
            PR_WARN("emo_lamp: unknown idle_effect '%s', fallback '%s'",
                    eff, APP_EMO_IDLE_EFFECT_DEF);
        }
    }

    cJSON_Delete(root);
    PR_NOTICE("emo_lamp: conf (emo=%d hold=%dms respect=%d | idle=%d after=%dms %s %s b=%d)",
              (int)sg_enabled, sg_hold_ms, (int)sg_respect_manual,
              (int)sg_idle_enabled, sg_idle_after_ms,
              sg_idle_effect, sg_idle_color, sg_idle_bright);
}

void app_emo_lamp_notify(const char *emotion_name)
{
    char name[APP_EMO_NAME_MAX];

    if (!sg_running || !sg_enabled || NULL == emotion_name) {
        return;
    }

    __norm_name(emotion_name, name, sizeof(name));
    if ('\0' == name[0]) {
        return;
    }

    /* 只写全局缓冲 + 放信号量：微秒级，不碰 socket，不阻塞事件回调 */
    if (NULL != sg_lock) {
        tal_mutex_lock(sg_lock);
    }
    snprintf(sg_pending, sizeof(sg_pending), "%s", name);
    if (NULL != sg_lock) {
        tal_mutex_unlock(sg_lock);
    }
    (void)tal_semaphore_post(sg_sem);
}

void app_emo_lamp_set_enabled(BOOL_T on)
{
    sg_enabled = on;
    PR_NOTICE("emo_lamp: enabled=%d", (int)on);
}

BOOL_T app_emo_lamp_is_enabled(void)
{
    return sg_enabled;
}

const char *app_emo_lamp_last_color(void)
{
    return sg_last_color;
}

OPERATE_RET app_emo_lamp_init(void)
{
    THREAD_CFG_T cfg;

    if (sg_running) {
        return OPRT_OK; /* 幂等 */
    }

    __load_conf();

    /* 认领上一次进程遗留的灯效：灯正亮着 + 效果就是我们的空闲效果
       -> 认为是我们自己点的（重启残留），接管；否则视为人工灯效，不碰。 */
    if (sg_idle_enabled) {
        LED_STATE_T st;
        if (__led_query(&st) && st.on && st.bright > 0 &&
            0 == strcmp(st.effect, sg_idle_effect)) {
            sg_ver_owned = st.version;
            sg_idle_lit  = TRUE;
            PR_NOTICE("emo_lamp: adopted leftover idle lamp ('%s' v=%d)",
                      st.effect, st.version);
        }
    }

    if (OPRT_OK != tal_mutex_create_init(&sg_lock)) {
        PR_ERR("emo_lamp: create mutex failed");
        return OPRT_COM_ERROR;
    }
    if (OPRT_OK != tal_semaphore_create_init(&sg_sem, 0, APP_EMO_SEM_MAX)) {
        PR_ERR("emo_lamp: create semaphore failed");
        return OPRT_COM_ERROR;
    }

    sg_running = TRUE;

    memset(&cfg, 0, sizeof(cfg));
    cfg.thrdname   = "app_emo_lamp";
    cfg.priority   = APP_EMO_THRD_PRIO;
    cfg.stackDepth = APP_EMO_THRD_STACK;
    cfg.psram_mode = 0;

    if (OPRT_OK != tal_thread_create_and_start(&sg_thrd, NULL, NULL, __emo_task, NULL, &cfg)) {
        PR_ERR("emo_lamp: create thread failed");
        sg_running = FALSE;
        return OPRT_COM_ERROR;
    }

    PR_NOTICE("emo_lamp: ready (hold=%dms idle=%d after=%dms)",
              sg_hold_ms, (int)sg_idle_enabled, sg_idle_after_ms);
    return OPRT_OK;
}
