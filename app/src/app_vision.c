/**
 * @file app_vision.c
 * @brief AI 图片理解实现：拍照 -> 图+文本一起送给多模态 agent
 *
 * 上传协议（与官方 ai_picture_input_recognize() 同一套路）
 * ----------------------------------------------------
 *   tuya_ai_input_start(TRUE);
 *   tuya_ai_image_input(ts, jpeg, len, len);     // 图片
 *   tuya_ai_text_input(prompt, plen, plen);      // 引导语
 *   tuya_ai_input_stop();
 * 官方实现用的是固定 prompt RECOGNIZE_IMAGE_PROMPT（"请解释上传图片的内容…"）。
 * 这里换成更口语化的引导语，并且在有关键词上下文时把用户原话一起带上，
 * 让云端知道"用户到底在问什么"。
 *
 * 线程模型
 * --------
 * 单个常驻工作线程 + 计数信号量：
 *   调用方(request/maybe_capture) 只写缓冲 + post 信号量，绝不阻塞；
 *   线程里做耗时的拍照与网络上传，一次只处理一张，天然串行、不会堆积。
 *
 * @copyright Copyright (c) 2026
 */

#include "app_vision.h"

#include <errno.h>
#include <stdio.h>
#include <string.h>

#include "tal_api.h"
#include "tuya_ai_agent.h"
#include "cJSON.h"

#include "app_camera.h"
#include "app_feishu.h"   /* app_feishu_on_photo()：把这一帧也回传给飞书 */

/***********************************************************
************************macro define************************
***********************************************************/
/* 配置文件（可选）。字段：
 *   "auto"   : true/false   关键词自动拍照开关（默认 true）
 *   "prompt" : "字符串"      固定引导语。设置后每次上传都用它，
 *                            用来和云端 Agent 工作流的选择器对齐
 *                            （平台侧工作流要求 prompt 文本与选择器一致）。
 *   例：{"auto":true,"prompt":"image_recognition"}
 */
#define APP_VISION_CONF           "/etc/ai-vision.json"

#define APP_VISION_TEXT_MAX       (256)     /* 用户原话缓冲 */
#define APP_VISION_PROMPT_MAX     (448)
#define APP_VISION_THRD_STACK     (8192)    /* 线程里有拍照 + 网络调用，栈给足 */
#define APP_VISION_THRD_PRIO      (4)
#define APP_VISION_TICK_MS        (500)
#define APP_VISION_MIN_GAP_MS     (1200)    /* 两次拍照最小间隔，防连点 */
#define APP_VISION_TEXT_TAIL      (180)     /* 拼进 prompt 的用户原话最多取多少字节 */

/* 无上下文时的默认引导语 */
#define APP_VISION_PROMPT_DEF \
    "请你结合下面这张设备摄像头刚刚拍到的实时画面，用一到两句自然口语描述你看到了什么。"

/* 关键词：命中即自动拍照。中英文都放进来（小写后做子串匹配） */
static const char *sg_keywords[] = {
    "看一下", "看看", "瞧瞧", "瞅瞅", "看一眼",
    "前面有什么", "前面是什么", "前面有啥", "前面有什", "前面是什",
    "那是什么", "这是什么", "这是啥", "那是啥",
    "你能看到", "你能看见", "看见什么", "看到什么", "你看到",
    "拍张照", "拍个照", "拍照", "拍一张", "拍下来", "拍个",
    "摄像头", "镜头", "画面里", "识别一下", "认一下", "形容一下", "描述一下",
    "look", "see what", "what's in front", "what is in front",
    "in front of you", "take a photo", "take a picture",
    "camera", "snapshot", "describe what you see", "what can you see",
};
#define APP_VISION_KW_CNT (sizeof(sg_keywords) / sizeof(sg_keywords[0]))

/***********************************************************
***********************variable define**********************
***********************************************************/
static SEM_HANDLE    sg_sem      = NULL;
static MUTEX_HANDLE  sg_lock     = NULL;
static THREAD_HANDLE sg_thrd     = NULL;
static volatile BOOL_T sg_running = FALSE;
static volatile BOOL_T sg_auto    = TRUE;   /* 关键词自动拍照开关 */
static volatile BOOL_T sg_busy    = FALSE;  /* 有任务在处理中 */

static char     sg_text[APP_VISION_TEXT_MAX] = {0};
static uint64_t sg_last_ms                   = 0;
static char     sg_prompt[APP_VISION_PROMPT_MAX] = {0};  /* 非空 = 固定引导语 */
/* 非空 = **仅本次生效**的完整引导语（app_vision_request_prompt 写入）。
   优先级最高：一次性的 > 配置里的固定引导语 > 由用户原话自动拼的那句。
   场景：挥手打招呼时外部想精确指定 AI 说什么（「打个招呼并介绍你自己」），
   而自动拼接的模板只会说「用一到两句自然口语回答他」，讲不出自我介绍。 */
static char     sg_once_prompt[APP_VISION_PROMPT_MAX] = {0};

/***********************************************************
***********************function define**********************
***********************************************************/

/* 全小写拷贝（只动 ASCII，中文原样保留），便于中英关键词统一匹配 */
static void __to_lower(const char *src, size_t src_len, char *dst, size_t cap)
{
    size_t i;

    for (i = 0; i + 1 < cap && i < src_len; i++) {
        char c = src[i];
        if (c >= 'A' && c <= 'Z') {
            c = (char)(c - 'A' + 'a');
        }
        dst[i] = c;
    }
    dst[i] = '\0';
}

/* 关键词命中判断 */
static BOOL_T __hit_keyword(const char *text, size_t len)
{
    char   low[APP_VISION_TEXT_MAX];
    size_t i;
    size_t n = (len < sizeof(low) - 1) ? len : sizeof(low) - 1;

    if (n == 0) {
        return FALSE;
    }
    __to_lower(text, n, low, sizeof(low));

    for (i = 0; i < APP_VISION_KW_CNT; i++) {
        if (NULL != strstr(low, sg_keywords[i])) {
            PR_NOTICE("vision: keyword hit \"%s\"", sg_keywords[i]);
            return TRUE;
        }
    }
    return FALSE;
}

/* 真正上传（在工作线程里跑） */
static void __send_image(uint8_t *jpeg, uint32_t len, const char *user_text,
                         const char *once_prompt)
{
    char        prompt[APP_VISION_PROMPT_MAX];
    uint64_t    ts;
    OPERATE_RET rt;

    if (NULL == jpeg || 0 == len) {
        return;
    }

    /* 飞书 IM：同一张 JPEG 顺便回传给飞书用户。
       由桥接层自己判断是不是飞书轮次（语音轮次会被自动忽略），
       所以这里无条件回调，不需要知道调用方是谁。 */
    app_feishu_on_photo(jpeg, len);

    if (!tuya_ai_agent_is_ready()) {
        PR_WARN("vision: ai agent not ready, drop photo (%u bytes)", (unsigned)len);
        return;
    }

    if (NULL != once_prompt && once_prompt[0] != '\0') {
        /* 本次请求的专属引导语：直接照用，不再套任何模板 */
        snprintf(prompt, sizeof(prompt), "%s", once_prompt);
    } else if (sg_prompt[0] != '\0') {
        /* 配置了固定引导语：直接用，用于对齐云端工作流选择器 */
        snprintf(prompt, sizeof(prompt), "%s", sg_prompt);
    } else if (NULL != user_text && user_text[0] != '\0') {
        snprintf(prompt, sizeof(prompt),
                 "用户刚刚说：\"%.*s\"。请你结合这张设备摄像头刚拍到的实时画面，"
                 "用一到两句自然口语回答他。",
                 APP_VISION_TEXT_TAIL, user_text);
    } else {
        snprintf(prompt, sizeof(prompt), "%s", APP_VISION_PROMPT_DEF);
    }

    ts = tal_system_get_millisecond();

    (void)tuya_ai_input_start(TRUE);
    rt = tuya_ai_image_input(ts, jpeg, len, len);
    if (OPRT_OK == rt) {
        (void)tuya_ai_text_input((uint8_t *)prompt, strlen(prompt), strlen(prompt));
        PR_NOTICE("vision: photo sent to ai agent (%u bytes)", (unsigned)len);
    } else {
        PR_ERR("vision: image input failed, rt=%d", rt);
    }
    (void)tuya_ai_input_stop();
}

static void __vision_task(void *arg)
{
    char text[APP_VISION_TEXT_MAX];
    char once[APP_VISION_PROMPT_MAX];

    (void)arg;

    while (sg_running) {
        uint8_t  *jpeg = NULL;
        uint32_t  jlen = 0;

        if (OPRT_OK != tal_semaphore_wait(sg_sem, APP_VISION_TICK_MS)) {
            continue;
        }
        if (!sg_running) {
            break;
        }

        /* 取走本次请求的用户原话 + 一次性引导语（一起拿，避免拍照期间被覆盖） */
        if (NULL != sg_lock) {
            tal_mutex_lock(sg_lock);
        }
        snprintf(text, sizeof(text), "%s", sg_text);
        sg_text[0] = '\0';
        snprintf(once, sizeof(once), "%s", sg_once_prompt);
        sg_once_prompt[0] = '\0';
        if (NULL != sg_lock) {
            tal_mutex_unlock(sg_lock);
        }

        if (OPRT_OK != app_camera_snapshot(&jpeg, &jlen)) {
            PR_ERR("vision: snapshot failed: %s", app_camera_last_error());
            sg_busy = FALSE;
            continue;
        }

        __send_image(jpeg, jlen, text, once);
        app_camera_free(jpeg);

        sg_busy = FALSE;
    }

    PR_NOTICE("vision: task exit");
}

/* -------------------- 对外接口 -------------------- */

BOOL_T app_vision_request(const char *user_text)
{
    uint64_t now;

    if (!sg_running) {
        return FALSE;
    }
    if (sg_busy) {
        PR_WARN("vision: busy, request dropped");
        return FALSE;
    }

    now = tal_system_get_millisecond();
    if (sg_last_ms != 0 && (now - sg_last_ms) < APP_VISION_MIN_GAP_MS) {
        PR_NOTICE("vision: throttled (%.0fms since last)", (double)(now - sg_last_ms));
        return FALSE;
    }
    sg_last_ms = now;

    if (NULL != sg_lock) {
        tal_mutex_lock(sg_lock);
    }
    if (NULL != user_text) {
        snprintf(sg_text, sizeof(sg_text), "%.*s", APP_VISION_TEXT_MAX - 1, user_text);
    } else {
        sg_text[0] = '\0';
    }
    if (NULL != sg_lock) {
        tal_mutex_unlock(sg_lock);
    }

    sg_busy = TRUE;
    (void)tal_semaphore_post(sg_sem);

    PR_NOTICE("vision: capture requested (text=\"%s\")", (NULL != user_text) ? user_text : "");
    return TRUE;
}

/**
 * 请求拍照，并**完整指定**送给云端的引导语（不套任何模板）。
 *
 * 与 app_vision_request() 的区别：后者会包一层
 * 「用户刚刚说：\"xxx\"。请你结合这张实时画面，用一到两句自然口语回答他。」
 * 这个模板适合「回答问题」，但不适合「让它主动自我介绍」——
 * 那种场景需要调用方自己把话说全。
 */
BOOL_T app_vision_request_prompt(const char *prompt)
{
    if (NULL == prompt || '\0' == prompt[0]) {
        return FALSE;
    }
    if (!sg_running || sg_busy) {
        return FALSE;
    }

    /* 先写一次性引导语，再走公共受理流程（节流 / 排队逻辑全部复用） */
    if (NULL != sg_lock) {
        tal_mutex_lock(sg_lock);
    }
    snprintf(sg_once_prompt, sizeof(sg_once_prompt), "%.*s", APP_VISION_PROMPT_MAX - 1, prompt);
    if (NULL != sg_lock) {
        tal_mutex_unlock(sg_lock);
    }

    if (!app_vision_request(NULL)) {
        /* 被节流 / 正忙：撤掉刚写的一次性引导语，别污染下一次请求 */
        if (NULL != sg_lock) {
            tal_mutex_lock(sg_lock);
        }
        sg_once_prompt[0] = '\0';
        if (NULL != sg_lock) {
            tal_mutex_unlock(sg_lock);
        }
        PR_WARN("vision: prompt request rejected (busy/throttled)");
        return FALSE;
    }

    PR_NOTICE("vision: capture requested with custom prompt (%u bytes)", (unsigned)strlen(prompt));
    return TRUE;
}

BOOL_T app_vision_maybe_capture(const char *text, int len)
{
    if (!sg_running || !sg_auto || NULL == text || len <= 0) {
        return FALSE;
    }
    if (!__hit_keyword(text, (size_t)len)) {
        return FALSE;
    }

    /* 关键词命中的文本可能没有 '\0' 结尾，做个安全拷贝再传 */
    {
        char tmp[APP_VISION_TEXT_MAX];
        size_t n = ((size_t)len < sizeof(tmp) - 1) ? (size_t)len : (sizeof(tmp) - 1);
        memcpy(tmp, text, n);
        tmp[n] = '\0';
        return app_vision_request(tmp);
    }
}

void app_vision_set_auto_enabled(BOOL_T on)
{
    sg_auto = on;
    PR_NOTICE("vision: auto capture enabled=%d", (int)on);
}

/* 读 /etc/ai-vision.json（不存在则用默认值，不是错误） */
static void __load_conf(void)
{
    FILE  *fp;
    char   buf[1024];
    size_t n;
    cJSON *root;
    cJSON *j;

    fp = fopen(APP_VISION_CONF, "r");
    if (NULL == fp) {
        return; /* 正常情况：没配置就用默认 */
    }
    n = fread(buf, 1, sizeof(buf) - 1, fp);
    fclose(fp);
    buf[n] = '\0';

    root = cJSON_Parse(buf);
    if (NULL == root) {
        PR_WARN("vision: %s parse failed", APP_VISION_CONF);
        return;
    }

    j = cJSON_GetObjectItem(root, "auto");
    if (cJSON_IsBool(j)) {
        sg_auto = cJSON_IsTrue(j);
    }

    j = cJSON_GetObjectItem(root, "prompt");
    if (cJSON_IsString(j) && NULL != j->valuestring && j->valuestring[0] != '\0') {
        snprintf(sg_prompt, sizeof(sg_prompt), "%s", j->valuestring);
        PR_NOTICE("vision: fixed prompt = \"%s\"", sg_prompt);
    }

    cJSON_Delete(root);
}

OPERATE_RET app_vision_init(void)
{
    THREAD_CFG_T cfg;

    if (sg_running) {
        return OPRT_OK; /* 幂等 */
    }

    /* 摄像头模块（读 /etc/ai-camera.dev）由本模块一并初始化，
       app 层不必单独调 app_camera_init() */
    (void)app_camera_init();

    /* 可选配置：/etc/ai-vision.json */
    __load_conf();

    if (OPRT_OK != tal_mutex_create_init(&sg_lock)) {
        PR_ERR("vision: create mutex failed");
        return OPRT_COM_ERROR;
    }
    if (OPRT_OK != tal_semaphore_create_init(&sg_sem, 0, 4)) {
        PR_ERR("vision: create semaphore failed");
        return OPRT_COM_ERROR;
    }

    sg_running = TRUE;

    memset(&cfg, 0, sizeof(cfg));
    cfg.thrdname   = "app_vision";
    cfg.priority   = APP_VISION_THRD_PRIO;
    cfg.stackDepth = APP_VISION_THRD_STACK;
    cfg.psram_mode = 0;

    if (OPRT_OK != tal_thread_create_and_start(&sg_thrd, NULL, NULL, __vision_task, NULL, &cfg)) {
        PR_ERR("vision: create thread failed");
        sg_running = FALSE;
        return OPRT_COM_ERROR;
    }

    PR_NOTICE("vision: ready (camera=%s, auto keyword=%d)",
              app_camera_device(), (int)sg_auto);
    return OPRT_OK;
}
