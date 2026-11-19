/**
 * @file app_mcp_tools.c
 * @brief 应用侧端侧 MCP 工具：WS2812 灯环控制 + 机械臂人脸跟随 / 迎宾
 *
 * 移植来源：小智 AI 那套 xiaozhi_mcp_led.py（当时是 WebSocket 桥接到 xiaozhi.me 云），
 *          现在改成 TuyaOpen 的**端侧** MCP：工具直接在设备上跑，由涂鸦 AI 通过
 *          MCP(JSON-RPC) 调用，不再需要外部 MCP 桥接服务。
 *
 * 对应关系（小智工具 -> 端侧工具）：
 *   set_light_power / set_light_color / set_light_brightness / set_light_effect /
 *   control_light      -> device_light_control（参数可选，只传要改的）+ device_light_power_set
 *   get_light_state    -> device_light_state_get
 *   start/stop/get_face_follow_status -> device_face_follow_start/stop/status_get
 *   start/stop/get_greeting_status    -> device_greeting_start/stop/status_get
 *
 * 干活的方式：
 *   灯环    : 走常驻守护 ws2812d 的 UNIX socket（/run/ws2812d.sock，JSON 行协议）
 *   跟随/迎宾: 先写 /etc/face-follow.json | /etc/vision-greet.json，再用 systemctl
 *             拉起 face-follow.service | vision-greet.service。
 *             这两个单元的 KillSignal=SIGINT，停止时脚本会平滑回收纳位 + 释放力矩。
 *
 * !! 执行环境注意 !!：ai_mcp_server 是 tal_workq_schedule(WORKQ_SYSTEM, ...) 调回调，
 *   即跑在系统工作队列线程上（单线程、栈约 6KB）：
 *     1) 只用 static / 堆缓冲，不在栈上放大数组；
 *     2) 不做长时间阻塞（start 后等 4s，stop 最多轮询等 6s）。
 */
#ifdef APP_MCP_SELFTEST
/* 独立自测模式：不引入 TuyaOpen 头文件，只用 libc，
 *   板子上直接：gcc -DAPP_MCP_SELFTEST app_mcp_tools.c -o /tmp/app_mcp_selftest
 */
#else
#include "app_mcp_tools.h"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <stdbool.h>
#include <ctype.h>
#include <errno.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>

#if defined(ENABLE_COMP_AI_MCP) && (ENABLE_COMP_AI_MCP == 1)
#include "tal_api.h"
#include "ai_mcp_server.h"
#include "app_vision.h"
#include "app_camera.h"
#include "cJSON.h"
#endif

/***********************************************************
************************macro define************************
***********************************************************/
/* ---- 灯环 ---- */
#define APP_LED_SOCK            "/run/ws2812d.sock"

/* ---- 机械臂：人脸跟随 / 迎宾 ---- */
#define APP_FOLLOW_UNIT         "face-follow"
#define APP_GREET_UNIT          "vision-greet"
#define APP_FOLLOW_CFG          "/etc/face-follow.json"
#define APP_GREET_CFG           "/etc/vision-greet.json"
#define APP_FOLLOW_LOG          "/var/log/xiaozhi-linux/face_follow.log"
#define APP_GREET_LOG           "/var/log/xiaozhi-linux/vision_greet.log"

/* ---- 迎宾基准位（机械臂当前姿态）---- */
#define APP_POSE_CFG            "/etc/greet-pose.json"
#define APP_POSE_SCRIPT         "/root/arm_pose_save.py"
#define APP_POSE_PY             "/root/miniconda3/envs/lerobot/bin/python"
#define APP_POSE_LOG            "/tmp/greet_pose_save.log"

/* ---- 机械臂一次性小动作（点头 / 摇摇头）---- */
#define APP_ARM_DO_SCRIPT       "/root/arm_do.py"
#define APP_ARM_LOG             "/tmp/arm_do.log"

#define APP_SYSTEMCTL           "/usr/bin/systemctl"
#define APP_TIMEOUT_BIN         "/usr/bin/timeout"

/* ---- 缓冲（都是 static，别放栈上）---- */
#define APP_LED_RESP_MAX        (1024)
#define APP_CMD_OUT_MAX         (768)
#define APP_LOG_BUF_MAX         (3072)
#define APP_LOG_INCLUDE_CHARS   (500)   /* 回给大模型的日志尾巴最多这么多字符 */
#define APP_FOLLOW_BAND_DEF     (25)    /* 允许带默认 25% */
#define APP_CAM_ID_DEF          (12)    /* 摄像头节点号（与跟随/迎宾脚本一致） */

/***********************************************************
***********************底层工具层（纯 libc）*****************
***********************************************************/
/* 这一层不依赖 TuyaOpen / MCP，方便在板子上直接
 *   gcc -DAPP_MCP_SELFTEST app_mcp_tools.c -o /tmp/app_mcp_selftest
 * 单跑自检（验证 socket 协议、systemctl、日志读取这些真正干活的部分）。 */

static void app_str_trim(char *s)
{
    size_t len;

    if (s == NULL) {
        return;
    }
    len = strlen(s);
    while (len > 0 && (s[len - 1] == '\n' || s[len - 1] == '\r' ||
                       s[len - 1] == ' '  || s[len - 1] == '\t')) {
        s[--len] = '\0';
    }
}

/* 只保留字符串末尾 max_chars 个字符（用于日志尾巴） */
static void app_keep_tail(char *s, size_t max_chars)
{
    size_t len;

    if (s == NULL) {
        return;
    }
    len = strlen(s);
    if (len <= max_chars) {
        return;
    }
    memmove(s, s + (len - max_chars), max_chars + 1);
}

/* 跑一条命令并收集 stdout（cmd 必须是代码里写死的字面量，绝不拼接用户输入） */
static int app_sh(const char *cmd, char *out, size_t out_len)
{
    FILE *fp;
    int   rc;

    if (out != NULL && out_len > 0) {
        out[0] = '\0';
    }
    fp = popen(cmd, "r");
    if (fp == NULL) {
        return -1;
    }
    if (out != NULL && out_len > 1) {
        size_t n = fread(out, 1, out_len - 1, fp);
        out[n] = '\0';
    } else {
        char sink[128];
        while (fread(sink, 1, sizeof(sink), fp) > 0) {
        }
    }
    rc = pclose(fp);
    if (rc == -1) {
        return -1;
    }
    return WIFEXITED(rc) ? WEXITSTATUS(rc) : -1;
}

/* systemctl is-active <unit>：把状态字符串写进 out（active/inactive/failed/...） */
static int app_unit_state(const char *unit, char *out, size_t out_len)
{
    char cmd[160];

    snprintf(cmd, sizeof(cmd), APP_TIMEOUT_BIN " 5 " APP_SYSTEMCTL " is-active %s", unit);
    int rc = app_sh(cmd, out, out_len);
    app_str_trim(out);
    return rc;
}

/* 服务算不算"在跑"（activating/deactivating 也算，别让用户重复点） */
static int app_unit_running(const char *unit)
{
    static char state[48];

    if (app_unit_state(unit, state, sizeof(state)) != 0 && state[0] == '\0') {
        return 0;
    }
    if (strcmp(state, "active") == 0 || strcmp(state, "activating") == 0 ||
        strcmp(state, "deactivating") == 0 || strcmp(state, "reloading") == 0) {
        return 1;
    }
    return 0;
}

static int app_unit_pid(const char *unit)
{
    char cmd[160];
    char out[64];

    snprintf(cmd, sizeof(cmd), APP_TIMEOUT_BIN " 5 " APP_SYSTEMCTL " show -p MainPID --value %s", unit);
    if (app_sh(cmd, out, sizeof(out)) != 0) {
        return 0;
    }
    app_str_trim(out);
    return atoi(out);
}

/* 读取文件末尾 back_bytes 字节（日志/配置都能用） */
static int app_file_tail(const char *path, long back_bytes, char *out, size_t out_len)
{
    FILE *fp;
    long  size;
    size_t n;

    if (out != NULL && out_len > 0) {
        out[0] = '\0';
    }
    fp = fopen(path, "rb");
    if (fp == NULL) {
        return -1;
    }
    if (fseek(fp, 0, SEEK_END) != 0) {
        fclose(fp);
        return -1;
    }
    size = ftell(fp);
    if (size < 0) {
        fclose(fp);
        return -1;
    }
    if (size > back_bytes) {
        if (fseek(fp, size - back_bytes, SEEK_SET) != 0) {
            fclose(fp);
            return -1;
        }
    } else {
        rewind(fp);   /* 文件比 back_bytes 还小：必须回到开头，否则 fread 读到 EOF 为空 */
    }
    if (out != NULL && out_len > 1) {
        n = fread(out, 1, out_len - 1, fp);
        out[n] = '\0';
    }
    fclose(fp);
    return 0;
}

/* 文件最后修改时间（秒）。文件不存在返回 0 —— 用来判断后台任务有没有产出新结果 */
static long app_file_mtime(const char *path)
{
    struct stat st;

    if (path == NULL || stat(path, &st) != 0) {
        return 0;
    }
    return (long)st.st_mtime;
}

static int app_log_has(const char *path, long back_bytes, const char *needle)
{
    static char buf[2048];

    if (app_file_tail(path, back_bytes, buf, sizeof(buf)) != 0) {
        return 0;
    }
    return (strstr(buf, needle) != NULL) ? 1 : 0;
}

/* 原子写文件（先写 .tmp 再 rename），避免启动器读到半截 JSON */
static int app_write_file(const char *path, const char *data)
{
    char  tmp[160];
    FILE *fp;
    size_t len;

    if (path == NULL || data == NULL) {
        return -1;
    }
    snprintf(tmp, sizeof(tmp), "%s.tmp", path);
    len = strlen(data);

    fp = fopen(tmp, "wb");
    if (fp == NULL) {
        return -1;
    }
    if (fwrite(data, 1, len, fp) != len) {
        fclose(fp);
        unlink(tmp);
        return -1;
    }
    fflush(fp);
    fclose(fp);
    chmod(tmp, 0644);
    if (rename(tmp, path) != 0) {
        unlink(tmp);
        return -1;
    }
    return 0;
}

/* 给 ws2812d 发一行 JSON，读回一行 JSON。0=成功 */
static int app_led_send(const char *request, char *resp, size_t resp_len,
                        char *err, size_t err_len)
{
    int                fd;
    struct sockaddr_un addr;
    struct timeval     tv;
    size_t             used = 0;
    ssize_t            n;

    if (resp != NULL && resp_len > 0) {
        resp[0] = '\0';
    }
    if (err != NULL && err_len > 0) {
        err[0] = '\0';
    }
    if (request == NULL || resp == NULL || resp_len < 2) {
        if (err) {
            snprintf(err, err_len, "参数错误");
        }
        return -1;
    }

    fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) {
        snprintf(err, err_len, "创建 socket 失败：%s", strerror(errno));
        return -1;
    }

    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, APP_LED_SOCK, sizeof(addr.sun_path) - 1);

    tv.tv_sec  = 2;
    tv.tv_usec = 0;
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

    if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) != 0) {
        snprintf(err, err_len,
                 "连接灯环守护 %s 失败（%s），请确认 ws2812d 服务在运行",
                 APP_LED_SOCK, strerror(errno));
        close(fd);
        return -1;
    }
    if (send(fd, request, strlen(request), 0) < 0 || send(fd, "\n", 1, 0) < 0) {
        snprintf(err, err_len, "发送灯环指令失败：%s", strerror(errno));
        close(fd);
        return -1;
    }

    while (used + 1 < resp_len) {
        n = recv(fd, resp + used, resp_len - 1 - used, 0);
        if (n <= 0) {
            break;
        }
        used += (size_t)n;
        resp[used] = '\0';
        if (memchr(resp, '\n', used) != NULL) {
            break;
        }
    }
    close(fd);

    if (used == 0) {
        snprintf(err, err_len, "灯环守护没有回复（超时）");
        return -1;
    }
    app_str_trim(resp);
    return 0;
}

/* 颜色：#RRGGBB / #RGB 或颜色名（红/蓝/暖白/rainbow...），过滤掉会破坏 JSON 的字符 */
static int app_color_ok(const char *s)
{
    size_t i, len;

    if (s == NULL || s[0] == '\0') {
        return 0;
    }
    if (s[0] == '#') {
        len = strlen(s + 1);
        if (len != 3 && len != 6) {
            return 0;
        }
        for (i = 1; s[i] != '\0'; i++) {
            if (!isxdigit((unsigned char)s[i])) {
                return 0;
            }
        }
        return 1;
    }
    len = strlen(s);
    if (len > 24) {
        return 0;
    }
    for (i = 0; i < len; i++) {
        unsigned char c = (unsigned char)s[i];
        if (c < 0x20 || c == '"' || c == '\\') {
            return 0;
        }
    }
    return 1;
}

/* 光效别名归一（与 ws2812d.py 的 EFFECT_ALIAS 保持一致） */
static int app_effect_norm(const char *in, char *out, size_t out_len)
{
    static const struct {
        const char *alias;
        const char *value;
    } map[] = {
        {"static", "static"}, {"常亮", "static"}, {"静态", "static"}, {"solid", "static"},
        {"breathing", "breathing"}, {"呼吸", "breathing"}, {"呼吸灯", "breathing"}, {"breath", "breathing"},
        {"rainbow", "rainbow"}, {"彩虹", "rainbow"}, {"流光", "rainbow"},
        {"chase", "chase"}, {"流水", "chase"}, {"跑马灯", "chase"}, {"追逐", "chase"},
        {"blink", "blink"}, {"闪烁", "blink"}, {"闪", "blink"},
    };
    size_t i;

    if (in == NULL) {
        return 0;
    }
    for (i = 0; i < sizeof(map) / sizeof(map[0]); i++) {
        if (strcasecmp(in, map[i].alias) == 0) {
            snprintf(out, out_len, "%s", map[i].value);
            return 1;
        }
    }
    return 0;
}

#ifdef APP_MCP_SELFTEST
/* ---- 独立自检入口（只在 -DAPP_MCP_SELFTEST 时编译）---- */
int main(int argc, char **argv)
{
    static char resp[APP_LED_RESP_MAX];
    static char err[256];
    const char *req = (argc > 1) ? argv[1] : "{\"cmd\":\"state\"}";

    printf("== app_led_send(%s)\n", req);
    if (app_led_send(req, resp, sizeof(resp), err, sizeof(err)) != 0) {
        printf("   FAIL: %s\n", err);
    } else {
        printf("   OK  : %s\n", resp);
    }

    printf("== unit running: %s=%d  %s=%d\n",
           APP_FOLLOW_UNIT, app_unit_running(APP_FOLLOW_UNIT),
           APP_GREET_UNIT, app_unit_running(APP_GREET_UNIT));
    printf("== face-follow MainPID = %d\n", app_unit_pid(APP_FOLLOW_UNIT));
    printf("== follow cfg = %s",
           (app_file_tail(APP_FOLLOW_CFG, 256, resp, sizeof(resp)) == 0) ? resp : "(none)\n");
    printf("== greet log tail (last 300B) = %s",
           (app_file_tail(APP_GREET_LOG, 300, resp, sizeof(resp)) == 0) ? resp : "(none)\n");

    printf("== color/effect 校验: '红'=%d '#FF8800'=%d '坏\"色'=%d '呼吸'=%d 'xx'=%d\n",
           app_color_ok("红"), app_color_ok("#FF8800"), app_color_ok("坏\"色"),
           app_effect_norm("呼吸", resp, sizeof(resp)), app_effect_norm("xx", resp, sizeof(resp)));

    printf("== write /tmp/ selftest = %d\n",
           app_write_file("/tmp/app_mcp_selftest.json", "{\"ok\":1}"));
    {
        static char t[64] = "  abcdefghij  ";
        app_keep_tail(t, 5);
        printf("== keep_tail='%s'  log_has(ok)=%d  log_has(nope)=%d\n",
               t, app_log_has("/tmp/app_mcp_selftest.json", 256, "ok"),
               app_log_has("/tmp/app_mcp_selftest.json", 256, "nope"));
    }
    return 0;
}
#endif /* APP_MCP_SELFTEST */

/***********************************************************
******************** MCP 工具层 ****************************
***********************************************************/
#if defined(ENABLE_COMP_AI_MCP) && (ENABLE_COMP_AI_MCP == 1)

/* ---- 参数取值 ---- */
static const MCP_PROPERTY_T *__prop(const MCP_PROPERTY_LIST_T *list, const char *name)
{
    int i;

    if (list == NULL || name == NULL) {
        return NULL;
    }
    for (i = 0; i < list->count; i++) {
        const MCP_PROPERTY_T *p = list->properties[i];
        if (p != NULL && p->name != NULL && strcmp(p->name, name) == 0) {
            return p;
        }
    }
    return NULL;
}

/* 没声明默认值的属性：只有调用方真的传了值，has_default 才会变 true */
static bool __given(const MCP_PROPERTY_T *p)
{
    return (p != NULL) && (p->has_default == true);
}

static OPERATE_RET __reply_set(MCP_RETURN_VALUE_T *ret_val, bool success,
                               const char *msg, cJSON *extra)
{
    cJSON *json = cJSON_CreateObject();

    if (json == NULL) {
        if (extra != NULL) {
            cJSON_Delete(extra);
        }
        return OPRT_MALLOC_FAILED;
    }
    cJSON_AddBoolToObject(json, "success", success);
    cJSON_AddStringToObject(json, "message", (msg != NULL) ? msg : "");
    if (extra != NULL) {
        cJSON_AddItemToObject(json, "state", extra);
    }
    ai_mcp_return_value_set_json(ret_val, json);
    return OPRT_OK;
}

static OPERATE_RET __reply_ok(MCP_RETURN_VALUE_T *ret_val, const char *msg, cJSON *extra)
{
    return __reply_set(ret_val, true, msg, extra);
}

static OPERATE_RET __reply_fail(MCP_RETURN_VALUE_T *ret_val, const char *msg)
{
    return __reply_set(ret_val, false, msg, NULL);
}

/* ---------------------------------------------------------------
 * 灯环：把一条指令发给 ws2812d，并把结果翻译成给大模型看的中文
 * --------------------------------------------------------------- */
static OPERATE_RET __light_do(const char *request, MCP_RETURN_VALUE_T *ret_val, const char *ok_msg)
{
    static char resp[APP_LED_RESP_MAX];
    static char err[256];
    static char msg[256];
    cJSON *root, *ok, *state;

    if (app_led_send(request, resp, sizeof(resp), err, sizeof(err)) != 0) {
        PR_ERR("app_mcp: led cmd failed, req=%s err=%s", request, err);
        return __reply_fail(ret_val, err);
    }

    root = cJSON_Parse(resp);
    if (root == NULL) {
        snprintf(msg, sizeof(msg), "灯环守护返回的内容无法解析：%.200s", resp);
        return __reply_fail(ret_val, msg);
    }

    ok = cJSON_GetObjectItem(root, "ok");
    if (!cJSON_IsTrue(ok)) {
        cJSON *e = cJSON_GetObjectItem(root, "error");
        snprintf(msg, sizeof(msg), "灯环拒绝了这条指令：%.200s",
                 (e != NULL && cJSON_IsString(e)) ? e->valuestring : resp);
        cJSON_Delete(root);
        return __reply_fail(ret_val, msg);
    }

    state = cJSON_DetachItemFromObject(root, "state");
    cJSON_Delete(root);

    if (state != NULL) {
        cJSON *on   = cJSON_GetObjectItem(state, "on");
        cJSON *hex  = cJSON_GetObjectItem(state, "color_hex");
        cJSON *br   = cJSON_GetObjectItem(state, "brightness");
        cJSON *eff  = cJSON_GetObjectItem(state, "effect");
        snprintf(msg, sizeof(msg), "%s：现在%s，颜色 %s，亮度 %d%%，光效 %s",
                 ok_msg,
                 (on != NULL && cJSON_IsTrue(on)) ? "已开灯" : "已关灯",
                 (hex != NULL && cJSON_IsString(hex)) ? hex->valuestring : "未知",
                 (br != NULL && cJSON_IsNumber(br)) ? br->valueint : 0,
                 (eff != NULL && cJSON_IsString(eff)) ? eff->valuestring : "未知");
    } else {
        snprintf(msg, sizeof(msg), "%s", ok_msg);
    }

    PR_INFO("app_mcp: led ok, req=%s", request);
    return __reply_ok(ret_val, msg, state);
}

static OPERATE_RET __light_power_set(const MCP_PROPERTY_LIST_T *properties,
                                     MCP_RETURN_VALUE_T *ret_val, void *user_data)
{
    const MCP_PROPERTY_T *p_on = __prop(properties, "on");
    char req[64];

    (void)user_data;
    if (!__given(p_on) || p_on->default_val.type != MCP_PROPERTY_TYPE_BOOLEAN) {
        return __reply_fail(ret_val, "缺少参数 on：true=开灯，false=关灯");
    }
    snprintf(req, sizeof(req), "{\"cmd\":\"%s\"}", p_on->default_val.bool_val ? "on" : "off");
    return __light_do(req, ret_val, p_on->default_val.bool_val ? "灯环已打开" : "灯环已关闭");
}

static OPERATE_RET __light_control(const MCP_PROPERTY_LIST_T *properties,
                                   MCP_RETURN_VALUE_T *ret_val, void *user_data)
{
    const MCP_PROPERTY_T *p_on     = __prop(properties, "on");
    const MCP_PROPERTY_T *p_color  = __prop(properties, "color");
    const MCP_PROPERTY_T *p_bright = __prop(properties, "brightness");
    const MCP_PROPERTY_T *p_effect = __prop(properties, "effect");
    static char req[APP_CMD_OUT_MAX];
    static char effect[24];
    static char msg[256];
    int  n = 0;

    (void)user_data;
    if (!__given(p_on) && !__given(p_color) && !__given(p_bright) && !__given(p_effect)) {
        return __reply_fail(ret_val, "没有收到要修改的参数：请至少给出 on / color / brightness / effect 之一");
    }

    n += snprintf(req + n, sizeof(req) - n, "{\"cmd\":\"set\"");

    if (__given(p_on)) {
        n += snprintf(req + n, sizeof(req) - n, ",\"on\":%s",
                      p_on->default_val.bool_val ? "true" : "false");
    }
    if (__given(p_color)) {
        const char *c = p_color->default_val.str_val;
        if (!app_color_ok(c)) {
            return __reply_fail(ret_val,
                                "颜色格式不支持：请用 #RRGGBB（如 #FF8800），"
                                "或 红/绿/蓝/黄/紫/青/橙/粉/白/暖白/冷白/彩虹");
        }
        n += snprintf(req + n, sizeof(req) - n, ",\"color\":\"%s\"", c);
    }
    if (__given(p_bright)) {
        int v = p_bright->default_val.int_val;
        if (v < 0)   { v = 0; }
        if (v > 100) { v = 100; }
        n += snprintf(req + n, sizeof(req) - n, ",\"brightness\":%d", v);
    }
    if (__given(p_effect)) {
        if (!app_effect_norm(p_effect->default_val.str_val, effect, sizeof(effect))) {
            return __reply_fail(ret_val,
                                "不支持的光效：static(常亮) / breathing(呼吸) / "
                                "rainbow(彩虹) / chase(流水) / blink(闪烁)");
        }
        n += snprintf(req + n, sizeof(req) - n, ",\"effect\":\"%s\"", effect);
    }
    n += snprintf(req + n, sizeof(req) - n, "}");

    if (n >= (int)sizeof(req)) {
        return __reply_fail(ret_val, "参数太长，无法组装指令");
    }
    snprintf(msg, sizeof(msg), "灯环设置已生效");
    return __light_do(req, ret_val, msg);
}

static OPERATE_RET __light_state_get(const MCP_PROPERTY_LIST_T *properties,
                                     MCP_RETURN_VALUE_T *ret_val, void *user_data)
{
    (void)properties;
    (void)user_data;
    return __light_do("{\"cmd\":\"state\"}", ret_val, "灯环当前状态");
}

/* ---------------------------------------------------------------
 * 机械臂：人脸跟随 / 迎宾（都是 systemd 单元，启动前写参数文件）
 * --------------------------------------------------------------- */
static void __unit_report_fail(char *msg, size_t msg_len, const char *what,
                               const char *unit, int rc, const char *out,
                               const char *log_path)
{
    static char tail[APP_LOG_BUF_MAX];

    if (app_file_tail(log_path, 4096, tail, sizeof(tail)) == 0) {
        app_keep_tail(tail, APP_LOG_INCLUDE_CHARS);
    } else {
        tail[0] = '\0';
    }
    snprintf(msg, msg_len,
             "%s启动失败（服务 %s 没起来，systemctl rc=%d%s%.200s）。最近日志：%.500s",
             what, unit, rc, (out != NULL && out[0] != '\0') ? "：" : "",
             (out != NULL) ? out : "", tail);
}

static OPERATE_RET __face_follow_start(const MCP_PROPERTY_LIST_T *properties,
                                       MCP_RETURN_VALUE_T *ret_val, void *user_data)
{
    const MCP_PROPERTY_T *p_aim  = __prop(properties, "aim");
    const MCP_PROPERTY_T *p_band = __prop(properties, "band_percent");
    const MCP_PROPERTY_T *p_dry  = __prop(properties, "dry_run");
    static char cfg[192];
    static char out[APP_CMD_OUT_MAX];
    static char msg[APP_LOG_INCLUDE_CHARS + 320];
    const char *aim = "head";
    int  band = APP_FOLLOW_BAND_DEF;
    bool dry  = false;
    int  rc;

    (void)user_data;

    if (app_unit_running(APP_FOLLOW_UNIT)) {
        return __reply_ok(ret_val, "人脸跟随已经在运行了，机械臂正在跟着画面里的人转，不用重复启动。", NULL);
    }
    if (app_unit_running(APP_GREET_UNIT)) {
        return __reply_fail(ret_val,
                            "迎宾模式正在运行、机械臂被它占用。请先停止迎宾"
                            "（调用 device_greeting_stop），再启动人脸跟随。");
    }

    if (__given(p_aim) && p_aim->default_val.str_val != NULL) {
        const char *s = p_aim->default_val.str_val;
        if (strcasecmp(s, "body") == 0 || strcmp(s, "身体") == 0 || strcmp(s, "人体") == 0) {
            aim = "body";
        }
    }
    if (__given(p_band)) {
        band = p_band->default_val.int_val;
        if (band < 5)  { band = 5; }
        if (band > 60) { band = 60; }
    }
    if (__given(p_dry)) {
        dry = p_dry->default_val.bool_val;
    }

    snprintf(cfg, sizeof(cfg), "{\"aim\":\"%s\",\"band\":%.2f,\"cam_id\":%d,\"dry_run\":%s}",
             aim, (double)band / 100.0, APP_CAM_ID_DEF, dry ? "true" : "false");
    if (app_write_file(APP_FOLLOW_CFG, cfg) != 0) {
        snprintf(msg, sizeof(msg), "写入跟随参数 %s 失败，无法启动人脸跟随。", APP_FOLLOW_CFG);
        return __reply_fail(ret_val, msg);
    }
    PR_INFO("app_mcp: %s start, cfg=%s", APP_FOLLOW_UNIT, cfg);

    rc = app_sh(APP_TIMEOUT_BIN " 60 " APP_SYSTEMCTL " start " APP_FOLLOW_UNIT " 2>&1",
                out, sizeof(out));
    tal_system_sleep(4000);   /* 等 precheck + 解释器 + 模型加载 */

    if (!app_unit_running(APP_FOLLOW_UNIT)) {
        __unit_report_fail(msg, sizeof(msg), "人脸跟随", APP_FOLLOW_UNIT, rc, out, APP_FOLLOW_LOG);
        PR_ERR("app_mcp: %s not running after start", APP_FOLLOW_UNIT);
        return __reply_fail(ret_val, msg);
    }
    if (app_log_has(APP_FOLLOW_LOG, 4096, "Traceback")) {
        static char tail[APP_LOG_BUF_MAX];
        if (app_file_tail(APP_FOLLOW_LOG, 4096, tail, sizeof(tail)) == 0) {
            app_keep_tail(tail, APP_LOG_INCLUDE_CHARS);
        } else {
            tail[0] = '\0';
        }
        snprintf(msg, sizeof(msg),
                 "人脸跟随进程起来了但立刻报错，请检查硬件（机械臂串口 /dev/ttyACM0、摄像头）。最近日志：%.500s",
                 tail);
        return __reply_fail(ret_val, msg);
    }

    snprintf(msg, sizeof(msg),
             "人脸跟随已启动（对准%s，允许带 %d%%%s）。%s"
             "想让它停下就说\"停止跟随\"。",
             (strcmp(aim, "body") == 0) ? "人体框中心" : "人脸",
             band, dry ? "，演示模式：机械臂不会动" : "",
             dry ? "这是只跑视觉的演示模式。"
                 : "机械臂大约 10~25 秒加载模型后开始跟着动，请注意安全，"
                   "不要把手放在机械臂活动范围内。");
    return __reply_ok(ret_val, msg, NULL);
}

static OPERATE_RET __face_follow_stop(const MCP_PROPERTY_LIST_T *properties,
                                      MCP_RETURN_VALUE_T *ret_val, void *user_data)
{
    static char out[APP_CMD_OUT_MAX];
    static char tail[APP_LOG_BUF_MAX];
    int i;

    (void)properties;
    (void)user_data;

    if (!app_unit_running(APP_FOLLOW_UNIT)) {
        return __reply_ok(ret_val, "人脸跟随当前没有在运行，机械臂处于待机状态。", NULL);
    }

    app_sh(APP_TIMEOUT_BIN " 60 " APP_SYSTEMCTL " --no-block stop " APP_FOLLOW_UNIT " 2>&1",
           out, sizeof(out));

    /* 最多等 6 秒：单元收到 SIGINT 后脚本会平滑回收纳位 + 释放力矩 */
    for (i = 0; i < 12; i++) {
        tal_system_sleep(500);
        if (!app_unit_running(APP_FOLLOW_UNIT)) {
            break;
        }
    }

    if (app_file_tail(APP_FOLLOW_LOG, 4096, tail, sizeof(tail)) != 0) {
        tail[0] = '\0';
    }

    if (app_unit_running(APP_FOLLOW_UNIT)) {
        return __reply_ok(ret_val,
                          "已发出停止命令，服务还在退出中（机械臂正在平滑归位）。"
                          "稍后用状态查询确认即可。", NULL);
    }
    if (strstr(tail, "力矩已释放") != NULL) {
        return __reply_ok(ret_val,
                          "已停止人脸跟随，机械臂已平滑回到收纳位并释放力矩"
                          "（现在可以用手自由摆动）。", NULL);
    }
    return __reply_ok(ret_val,
                      "已停止人脸跟随。没读到\"力矩已释放\"日志——如果刚才是演示模式"
                      "(dry_run) 则属正常，否则请确认机械臂是否已回到收纳位。", NULL);
}

static OPERATE_RET __face_follow_status_get(const MCP_PROPERTY_LIST_T *properties,
                                            MCP_RETURN_VALUE_T *ret_val, void *user_data)
{
    static char cfg[256];
    static char msg[512];

    (void)properties;
    (void)user_data;

    if (app_file_tail(APP_FOLLOW_CFG, 192, cfg, sizeof(cfg)) != 0) {
        cfg[0] = '\0';
    }

    if (app_unit_running(APP_FOLLOW_UNIT)) {
        snprintf(msg, sizeof(msg),
                 "人脸跟随正在运行中（进程号 %d）。当前参数：%s"
                 "想让它停下就说\"停止跟随\"。",
                 app_unit_pid(APP_FOLLOW_UNIT), cfg);
        return __reply_ok(ret_val, msg, NULL);
    }
    return __reply_ok(ret_val,
                      "人脸跟随当前没有在运行，机械臂处于待机状态。"
                      "想启动就说\"启动人脸跟随\"。", NULL);
}

static OPERATE_RET __greeting_start(const MCP_PROPERTY_LIST_T *properties,
                                    MCP_RETURN_VALUE_T *ret_val, void *user_data)
{
    const MCP_PROPERTY_T *p_dry = __prop(properties, "dry_run");
    static char raw[768];
    static char cfg[768];
    static char out[APP_CMD_OUT_MAX];
    static char msg[APP_LOG_INCLUDE_CHARS + 320];
    cJSON *root = NULL;
    char  *printed = NULL;
    bool   dry = false;
    int    rc;

    (void)user_data;

    if (app_unit_running(APP_GREET_UNIT)) {
        return __reply_ok(ret_val, "迎宾模式已经在运行了，不用重复启动。", NULL);
    }
    if (__given(p_dry)) {
        dry = p_dry->default_val.bool_val;
    }

    /* vision-greet.service 里写了 Conflicts=face-follow.service：
       启动迎宾时 systemd 会先把人脸跟随优雅停掉（那边会先回收纳位） */

    /* 只改我们管的 cam_id / dry_run，保留文件里其它键
       （greet_say / pan_amp / pan_cycles / absent_sec ...），
       否则每次开启迎宾都会把用户手写的参数抹掉。 */
    if (app_file_tail(APP_GREET_CFG, (long)sizeof(raw) - 1, raw, sizeof(raw)) != 0) {
        raw[0] = '\0';
    }
    root = (raw[0] != '\0') ? cJSON_Parse(raw) : NULL;
    if (root == NULL || !cJSON_IsObject(root)) {
        if (root != NULL) {
            cJSON_Delete(root);
        }
        root = cJSON_CreateObject();
    }
    if (root == NULL) {
        return __reply_fail(ret_val, "内存不足，无法写入迎宾参数。");
    }
    cJSON_DeleteItemFromObject(root, "cam_id");
    cJSON_AddNumberToObject(root, "cam_id", APP_CAM_ID_DEF);
    cJSON_DeleteItemFromObject(root, "dry_run");
    cJSON_AddBoolToObject(root, "dry_run", dry ? 1 : 0);

    printed = cJSON_PrintUnformatted(root);
    cJSON_Delete(root);
    if (printed == NULL) {
        return __reply_fail(ret_val, "序列化迎宾参数失败。");
    }
    snprintf(cfg, sizeof(cfg), "%.500s", printed);
    cJSON_free(printed);

    if (app_write_file(APP_GREET_CFG, cfg) != 0) {
        snprintf(msg, sizeof(msg), "写入迎宾参数 %s 失败，无法启动迎宾模式。", APP_GREET_CFG);
        return __reply_fail(ret_val, msg);
    }
    PR_INFO("app_mcp: %s start, cfg=%s", APP_GREET_UNIT, cfg);

    rc = app_sh(APP_TIMEOUT_BIN " 60 " APP_SYSTEMCTL " start " APP_GREET_UNIT " 2>&1",
                out, sizeof(out));
    tal_system_sleep(4000);

    if (!app_unit_running(APP_GREET_UNIT)) {
        __unit_report_fail(msg, sizeof(msg), "迎宾模式", APP_GREET_UNIT, rc, out, APP_GREET_LOG);
        PR_ERR("app_mcp: %s not running after start", APP_GREET_UNIT);
        return __reply_fail(ret_val, msg);
    }
    if (app_log_has(APP_GREET_LOG, 4096, "Traceback")) {
        static char tail[APP_LOG_BUF_MAX];
        if (app_file_tail(APP_GREET_LOG, 4096, tail, sizeof(tail)) == 0) {
            app_keep_tail(tail, APP_LOG_INCLUDE_CHARS);
        } else {
            tail[0] = '\0';
        }
        snprintf(msg, sizeof(msg),
                 "迎宾模式进程起来了但立刻报错，请检查硬件（机械臂串口 /dev/ttyACM0、摄像头）。最近日志：%.500s",
                 tail);
        return __reply_fail(ret_val, msg);
    }

    snprintf(msg, sizeof(msg),
             "迎宾模式已启动%s。有人靠近时机械臂会在基准位上左右小幅旋转，"
             "并让 AI 主动打招呼；人离开后挥手告别并进入省电待机。"
             "（如果之前开着人脸跟随，它已被自动停止）",
             dry ? "（演示模式：机械臂不会动，只跑视觉和灯效）" : "");
    return __reply_ok(ret_val, msg, NULL);
}

static OPERATE_RET __greeting_stop(const MCP_PROPERTY_LIST_T *properties,
                                   MCP_RETURN_VALUE_T *ret_val, void *user_data)
{
    static char out[APP_CMD_OUT_MAX];
    static char tail[APP_LOG_BUF_MAX];
    int i;

    (void)properties;
    (void)user_data;

    if (!app_unit_running(APP_GREET_UNIT)) {
        return __reply_ok(ret_val, "迎宾模式当前没有在运行。", NULL);
    }

    app_sh(APP_TIMEOUT_BIN " 60 " APP_SYSTEMCTL " --no-block stop " APP_GREET_UNIT " 2>&1",
           out, sizeof(out));

    for (i = 0; i < 12; i++) {
        tal_system_sleep(500);
        if (!app_unit_running(APP_GREET_UNIT)) {
            break;
        }
    }

    if (app_file_tail(APP_GREET_LOG, 4096, tail, sizeof(tail)) != 0) {
        tail[0] = '\0';
    }

    if (app_unit_running(APP_GREET_UNIT)) {
        return __reply_ok(ret_val,
                          "已发出停止命令，服务还在退出中（机械臂正在归位）。"
                          "稍后用状态查询确认即可。", NULL);
    }
    if (strstr(tail, "力矩已释放") != NULL) {
        return __reply_ok(ret_val, "已停止迎宾模式，机械臂已回到收纳位并释放力矩。", NULL);
    }
    return __reply_ok(ret_val,
                      "已停止迎宾模式。没读到\"力矩已释放\"日志——如果刚才是演示模式"
                      "(dry_run) 则属正常，否则请确认机械臂是否已回到收纳位。", NULL);
}

static OPERATE_RET __greeting_status_get(const MCP_PROPERTY_LIST_T *properties,
                                         MCP_RETURN_VALUE_T *ret_val, void *user_data)
{
    static char cfg[128];
    static char msg[384];

    (void)properties;
    (void)user_data;

    if (app_file_tail(APP_GREET_CFG, 96, cfg, sizeof(cfg)) != 0) {
        cfg[0] = '\0';
    }

    if (app_unit_running(APP_GREET_UNIT)) {
        snprintf(msg, sizeof(msg),
                 "迎宾模式正在运行中（进程号 %d）。当前参数：%s",
                 app_unit_pid(APP_GREET_UNIT), cfg);
        return __reply_ok(ret_val, msg, NULL);
    }
    return __reply_ok(ret_val, "迎宾模式当前没有在运行。想开启就说\"开启迎宾模式\"。", NULL);
}

/* ---------------------------------------------------------------
 * 迎宾基准位：把「机械臂此刻的真实姿态」记下来
 *   之后迎宾启动时先转到这个位置，有人靠近就在这个位置上左右小幅旋转
 * --------------------------------------------------------------- */
static OPERATE_RET __greeting_pose_save(const MCP_PROPERTY_LIST_T *properties,
                                        MCP_RETURN_VALUE_T *ret_val, void *user_data)
{
    static char cmd[512];
    static char out[APP_CMD_OUT_MAX];
    static char raw[768];
    static char msg[768];
    cJSON *root = NULL;
    cJSON *bp   = NULL;
    long   t0, t1;
    int    i;

    (void)properties;
    (void)user_data;

    if (app_unit_running(APP_FOLLOW_UNIT) || app_unit_running(APP_GREET_UNIT)) {
        return __reply_fail(ret_val,
                            "机械臂现在被人脸跟随或迎宾模式占用着，串口读不了姿态。"
                            "请先停止它们（说\"停止跟随\"或\"停止迎宾\"），再让我记姿势。");
    }

    t0 = app_file_mtime(APP_POSE_CFG);

    /* 后台跑：读一次姿态要 10~20 秒（起 lerobot 解释器 + 连串口），
       不能在这里同步等，否则会把系统工作队列线程卡住。 */
    snprintf(cmd, sizeof(cmd),
             "nohup " APP_POSE_PY " " APP_POSE_SCRIPT " --out " APP_POSE_CFG
             " >" APP_POSE_LOG " 2>&1 &");
    app_sh(cmd, out, sizeof(out));

    t1 = t0;
    for (i = 0; i < 8; i++) {
        tal_system_sleep(1000);
        t1 = app_file_mtime(APP_POSE_CFG);
        if (t1 != 0 && t1 != t0) {
            break;
        }
    }

    if (t1 == 0 || t1 == t0) {
        /* 还没落盘：要么还在读，要么已经失败 —— 看日志再决定怎么回 */
        if (app_log_has(APP_POSE_LOG, 2048, "Traceback") ||
            app_log_has(APP_POSE_LOG, 2048, "[error]")) {
            static char tail[1024];
            if (app_file_tail(APP_POSE_LOG, 2048, tail, sizeof(tail)) == 0) {
                app_keep_tail(tail, APP_LOG_INCLUDE_CHARS);
            } else {
                tail[0] = '\0';
            }
            snprintf(msg, sizeof(msg), "读取机械臂姿态失败了。最近日志：%.500s", tail);
            return __reply_fail(ret_val, msg);
        }
        return __reply_ok(ret_val,
                          "正在读取机械臂此刻的姿态并保存为迎宾基准位（大约 10~20 秒）。"
                          "稍后再问一句\"迎宾基准位是什么\"就能看到结果。", NULL);
    }

    if (app_file_tail(APP_POSE_CFG, (long)sizeof(raw) - 1, raw, sizeof(raw)) != 0) {
        raw[0] = '\0';
    }
    root = cJSON_Parse(raw);
    bp   = (root != NULL) ? cJSON_GetObjectItem(root, "base_pose") : NULL;
    if (bp != NULL && cJSON_IsObject(bp)) {
        cJSON *pan  = cJSON_GetObjectItem(bp, "shoulder_pan.pos");
        cJSON *lift = cJSON_GetObjectItem(bp, "shoulder_lift.pos");
        cJSON *elb  = cJSON_GetObjectItem(bp, "elbow_flex.pos");
        cJSON *wfl  = cJSON_GetObjectItem(bp, "wrist_flex.pos");

        snprintf(msg, sizeof(msg),
                 "记好了！机械臂此刻的姿态已经成为迎宾基准位："
                 "水平 %.1f°、大臂 %.1f°、肘 %.1f°、腕 %.1f°。"
                 "之后开启迎宾模式时，机械臂会先转到这个位置；"
                 "有人靠近就在这个位置上左右小幅旋转，并让我主动打招呼。",
                 (pan != NULL) ? pan->valuedouble : 0.0,
                 (lift != NULL) ? lift->valuedouble : 0.0,
                 (elb != NULL) ? elb->valuedouble : 0.0,
                 (wfl != NULL) ? wfl->valuedouble : 0.0);
        cJSON_Delete(root);
        PR_NOTICE("app_mcp: greeting base pose saved");
        return __reply_ok(ret_val, msg, NULL);
    }
    if (root != NULL) {
        cJSON_Delete(root);
    }
    return __reply_ok(ret_val,
                      "姿态文件已经更新，但内容读不出来，建议重新说一次\"记住现在的姿势\"。", NULL);
}

static OPERATE_RET __greeting_pose_get(const MCP_PROPERTY_LIST_T *properties,
                                       MCP_RETURN_VALUE_T *ret_val, void *user_data)
{
    static char raw[768];
    static char msg[512];
    cJSON *root = NULL;
    cJSON *bp   = NULL;

    (void)properties;
    (void)user_data;

    if (app_file_tail(APP_POSE_CFG, (long)sizeof(raw) - 1, raw, sizeof(raw)) != 0) {
        raw[0] = '\0';
    }
    if (raw[0] == '\0') {
        return __reply_ok(ret_val,
                          "还没有保存过迎宾基准位，迎宾模式会用默认的 HOME 姿态。"
                          "想让机械臂用现在的姿势迎宾，就说\"记住现在的姿势\"。", NULL);
    }

    root = cJSON_Parse(raw);
    bp   = (root != NULL) ? cJSON_GetObjectItem(root, "base_pose") : NULL;
    if (bp == NULL || !cJSON_IsObject(bp)) {
        if (root != NULL) {
            cJSON_Delete(root);
        }
        return __reply_ok(ret_val,
                          "迎宾基准位文件在，但里面没有 base_pose，建议重新说一次"
                          "\"记住现在的姿势\"保存一遍。", NULL);
    }

    {
        cJSON *pan  = cJSON_GetObjectItem(bp, "shoulder_pan.pos");
        cJSON *lift = cJSON_GetObjectItem(bp, "shoulder_lift.pos");
        cJSON *elb  = cJSON_GetObjectItem(bp, "elbow_flex.pos");
        cJSON *wfl  = cJSON_GetObjectItem(bp, "wrist_flex.pos");
        cJSON *sa   = cJSON_GetObjectItem(root, "saved_at");

        snprintf(msg, sizeof(msg),
                 "迎宾基准位：水平 %.1f°、大臂 %.1f°、肘 %.1f°、腕 %.1f°；保存于 %s。"
                 "开启迎宾后机械臂会先转到这个位置。",
                 (pan != NULL) ? pan->valuedouble : 0.0,
                 (lift != NULL) ? lift->valuedouble : 0.0,
                 (elb != NULL) ? elb->valuedouble : 0.0,
                 (wfl != NULL) ? wfl->valuedouble : 0.0,
                 (sa != NULL && cJSON_IsString(sa)) ? sa->valuestring : "未知");
    }
    cJSON_Delete(root);
    return __reply_ok(ret_val, msg, NULL);
}

/* ---------------------------------------------------------------
 * 摄像头：拍照，并把照片作为 MCP 的 image content **直接返回**给云端 LLM。
 *
 * 为什么不是"拍完另外用 tuya_ai_image_input 上传"
 * ------------------------------------------------
 * 实测把 JPEG 用 tuya_ai_image_input 传上云后，云端 LLM 的上下文里并没有这张图
 * （AI 会回答"我看不到你说的摄像头画面"）。真正能让模型"看到"的通道是
 * MCP 工具返回值本身：MCP 协议支持 content 里带
 *     {"type":"image","mimeType":"image/jpeg","data":"<base64>"}
 * TuyaOpen 的 ai_mcp_return_value_set_image() 就是干这个的（自动 base64）。
 * 所以这里在回调里同步拍一张，然后 set_image 交给框架组装响应。
 *
 * 回调在 tal_workq_schedule 线程上跑：实测抓一帧 <1s，与已有 greeting_start
 * （等 4s）同一量级，可以接受。  MARK: APP_MCP_PHOTO_IMAGE_CONTENT
 * --------------------------------------------------------------- */
static OPERATE_RET __take_photo(const MCP_PROPERTY_LIST_T *properties,
                                MCP_RETURN_VALUE_T *ret_val, void *user_data)
{
    uint8_t    *jpeg = NULL;
    uint32_t    jlen = 0;
    OPERATE_RET rt   = OPRT_OK;

    (void)properties;
    (void)user_data;

    if (OPRT_OK != app_camera_snapshot(&jpeg, &jlen)) {
        static char err[256];
        snprintf(err, sizeof(err), "拍照失败：%s", app_camera_last_error());
        return __reply_fail(ret_val, err);
    }

    rt = ai_mcp_return_value_set_image(ret_val, "image/jpeg", jpeg, jlen);
    app_camera_free(jpeg);

    if (OPRT_OK != rt) {
        return __reply_fail(ret_val, "照片编码失败，请稍后再试一次");
    }

    PR_NOTICE("app_mcp: take_photo -> %u bytes jpeg returned as MCP image content",
              (unsigned)jlen);
    return OPRT_OK;
}

/* ---------------------------------------------------------------
 * 机械臂一次性小动作：点头 / 摇摇头
 *
 *   真正干活的是 /root/arm_do.py，它自己选路径：
 *     ① 迎宾在跑 -> 走 /run/vision-greet.sock，让迎宾顺手做一个
 *        （不停服务、不重启、也不跟它抢 /dev/ttyACM0）
 *     ② 没在跑   -> 自己连机械臂：转到基准位 -> 做动作 -> 收回纳位 -> 松力矩
 *        若此时是人脸跟随占着串口，会先停跟随、做完再拉起来
 *
 *   这里用 nohup 后台拉起并立刻回话：连 lerobot + 连串口要 2~4 秒、动作本身 3~5 秒，
 *   不能在系统工作队列线程（栈约 6KB）上同步等。
 * --------------------------------------------------------------- */
static OPERATE_RET __arm_do_async(const char *action, const char *ok_msg,
                                  MCP_RETURN_VALUE_T *ret_val)
{
    static char cmd[384];
    static char sink[64];

    snprintf(cmd, sizeof(cmd),
             "nohup " APP_POSE_PY " " APP_ARM_DO_SCRIPT " %s >" APP_ARM_LOG " 2>&1 &",
             action);
    app_sh(cmd, sink, sizeof(sink));
    PR_NOTICE("app_mcp: arm action dispatched: %s", action);
    return __reply_ok(ret_val, ok_msg, NULL);
}

static OPERATE_RET __arm_nod(const MCP_PROPERTY_LIST_T *properties,
                             MCP_RETURN_VALUE_T *ret_val, void *user_data)
{
    (void)properties;
    (void)user_data;
    return __arm_do_async(
        "nod",
        "好，我点个头。（机械臂先转到迎宾基准位再做，做完自己收回；迎宾在跑时 1~2 秒，没在跑要等十几秒）",
        ret_val);
}

static OPERATE_RET __arm_shake_head(const MCP_PROPERTY_LIST_T *properties,
                                    MCP_RETURN_VALUE_T *ret_val, void *user_data)
{
    (void)properties;
    (void)user_data;
    return __arm_do_async(
        "shake_head",
        "好，我摇摇头。（机械臂先转到迎宾基准位再做，做完自己收回；迎宾在跑时 1~2 秒，没在跑要等十几秒）",
        ret_val);
}

/***********************************************************
********************工具注册*******************************
***********************************************************/
static OPERATE_RET __app_mcp_tools_register(void)
{
    OPERATE_RET rt = OPRT_OK;

    /* ---------------- 灯环（24 颗 WS2812，由 ws2812d 守护驱动） ---------------- */
    TUYA_CALL_ERR_GOTO(AI_MCP_TOOL_ADD(
        "device_light_power_set",
        "打开或关闭【本设备自带的 WS2812 灯环】（机身内 24 颗灯珠，属于本机硬件）。\n"
        "重要：它控制的是设备自身的灯环，不是用户家里的智能灯，"
        "请直接调用本工具，不要使用智能家居/云端设备控制去查找灯设备。\n"
        "用户说\"开灯 / 关灯 / 把灯打开 / 灯关掉 / 打开你的灯环\"时调用。\n"
        "参数：\n"
        "- on (bool)：true=开灯，false=关灯。",
        __light_power_set,
        NULL,
        MCP_PROP_BOOL("on", "true=开灯，false=关灯。")
    ), err);

    TUYA_CALL_ERR_GOTO(AI_MCP_TOOL_ADD(
        "device_light_control",
        "设置本设备自带灯环的颜色 / 亮度 / 光效 / 开关，只传需要改的参数即可。\n"
        "这是本机灯环，不是智能家居灯，请直接调用本工具，不要查找云端设备列表。\n"
        "用户说\"把灯调成蓝色\"\"亮一点\"\"调暗一点\"\"来个呼吸灯\"\"彩虹灯\"时调用。\n"
        "参数（都可选）：\n"
        "- on (bool)：开关\n"
        "- color (string)：颜色，#RRGGBB 或颜色名（红/绿/蓝/黄/紫/青/橙/粉/白/暖白/冷白/彩虹）\n"
        "- brightness (int 0-100)：亮度\n"
        "- effect (string)：static 常亮 / breathing 呼吸 / rainbow 彩虹 / chase 流水 / blink 闪烁",
        __light_control,
        NULL,
        MCP_PROP_BOOL("on", "true=开灯，false=关灯。"),
        MCP_PROP_STR("color", "颜色：#RRGGBB 或 红/绿/蓝/黄/紫/青/橙/粉/白/暖白/冷白/彩虹。"),
        MCP_PROP_INT_RANGE("brightness", "亮度百分比 0-100。", 0, 100),
        MCP_PROP_STR("effect", "光效：static 常亮 / breathing 呼吸 / rainbow 彩虹 / chase 流水 / blink 闪烁。")
    ), err);

    TUYA_CALL_ERR_GOTO(AI_MCP_TOOL_ADD(
        "device_light_state_get",
        "查询本设备自带灯环当前的开关、颜色、亮度和光效（本机硬件，非智能家居灯）。\n"
        "用户问\"灯开着吗 / 灯是什么颜色\"时调用。",
        __light_state_get,
        NULL
    ), err);

    /* ---------------- 机械臂：人脸跟随 ---------------- */
    TUYA_CALL_ERR_GOTO(AI_MCP_TOOL_ADD(
        "device_face_follow_start",
        "启动机械臂的人脸跟随：摄像头看到人后，机械臂自动转过去对准。\n"
        "用户说\"启动人脸跟随 / 开始跟随我 / 让机械臂跟着我 / 打开追踪 / 机械臂转过来对着我\"时调用。\n"
        "重要：调用成功后机械臂会立即运动，回复里务必提醒用户注意周围安全、"
        "不要把手放在机械臂活动范围内。\n"
        "参数（都可选）：\n"
        "- aim (string)：head=对准人脸（默认），body=对准人体框中心\n"
        "- band_percent (int 5-60)：允许带，越大越稳越不爱动，默认 25\n"
        "- dry_run (bool)：true=只跑视觉不驱动机械臂（演示/自检），默认 false",
        __face_follow_start,
        NULL,
        MCP_PROP_STR("aim", "head=对准人脸（默认），body=对准人体框中心。"),
        MCP_PROP_INT_RANGE("band_percent", "允许带 5-60，越大越稳，默认 25。", 5, 60),
        MCP_PROP_BOOL("dry_run", "true=只跑视觉不驱动机械臂（演示/自检），默认 false。")
    ), err);

    TUYA_CALL_ERR_GOTO(AI_MCP_TOOL_ADD(
        "device_face_follow_stop",
        "停止机械臂的人脸跟随，机械臂会平滑回到收纳位置并释放力矩"
        "（之后可以用手自由摆动）。\n"
        "用户说\"停止跟随 / 别跟着我了 / 关掉追踪 / 机械臂回原位\"时调用。",
        __face_follow_stop,
        NULL
    ), err);

    TUYA_CALL_ERR_GOTO(AI_MCP_TOOL_ADD(
        "device_face_follow_status_get",
        "查询人脸跟随当前是否在运行、跑的是什么参数。\n"
        "用户问\"机械臂在跟着我吗 / 追踪开着吗\"时调用。",
        __face_follow_status_get,
        NULL
    ), err);

    /* ---------------- 机械臂：迎宾模式 ---------------- */
    TUYA_CALL_ERR_GOTO(AI_MCP_TOOL_ADD(
        "device_greeting_start",
        "启动迎宾模式：有人靠近时机械臂在【基准位】上左右小幅旋转，并让 AI 主动打招呼；"
        "人离开后挥手告别并进入省电待机。\n"
        "用户说\"开启迎宾 / 欢迎模式 / 打招呼模式 / 有人来打招呼\"时调用。\n"
        "注意：迎宾与人脸跟随互斥，启动迎宾会自动停止人脸跟随。\n"
        "参数：\n"
        "- dry_run (bool)：true=不驱动机械臂，只跑视觉与灯效（演示/自检），默认 false",
        __greeting_start,
        NULL,
        MCP_PROP_BOOL("dry_run", "true=不驱动机械臂，只跑视觉与灯效，默认 false。")
    ), err);

    TUYA_CALL_ERR_GOTO(AI_MCP_TOOL_ADD(
        "device_greeting_stop",
        "停止迎宾模式，机械臂回到收纳位并释放力矩。\n"
        "用户说\"关闭迎宾 / 停止迎宾模式\"时调用。",
        __greeting_stop,
        NULL
    ), err);

    TUYA_CALL_ERR_GOTO(AI_MCP_TOOL_ADD(
        "device_greeting_status_get",
        "查询迎宾模式当前是否在运行。\n"
        "用户问\"迎宾模式开着吗\"时调用。",
        __greeting_status_get,
        NULL
    ), err);

    TUYA_CALL_ERR_GOTO(AI_MCP_TOOL_ADD(
        "device_greeting_pose_save",
        "让机械臂记住【它此时此刻的姿态】，把这个姿态保存成迎宾模式的基准位。\n"
        "之后开启迎宾模式时，机械臂会先转到这个位置，有人靠近就在这个位置上左右小幅旋转。\n"
        "用户说\"记住现在的姿势 / 把现在的位置记下来 / 以当前姿势迎宾 / "
        "用这个姿势当基准\"时调用。\n"
        "注意：需要机械臂空闲（人脸跟随与迎宾都没有在跑），否则会读不到串口。\n"
        "执行需要约 10~20 秒，调用后请告诉用户稍等，再问一次\"迎宾基准位是什么\"即可确认。",
        __greeting_pose_save,
        NULL
    ), err);

    TUYA_CALL_ERR_GOTO(AI_MCP_TOOL_ADD(
        "device_greeting_pose_get",
        "查询迎宾模式当前使用的基准位姿态（水平 / 大臂 / 肘 / 腕角度、保存时间）。\n"
        "用户问\"迎宾基准位是什么 / 机械臂迎宾时站在什么姿态 / 记的姿势是什么\"时调用。",
        __greeting_pose_get,
        NULL
    ), err);

    /* ---------------- 摄像头：图片理解 ---------------- */
    TUYA_CALL_ERR_GOTO(AI_MCP_TOOL_ADD(
        "device_take_photo",
        "用本设备自带的摄像头拍一张照片。照片会作为本工具的返回内容【直接给你看到】，"
        "你能真实看到设备正前方 / 周围画面的内容。\n"
        "用户说\"你看看前面有什么\"\"你能看到什么\"\"看一下这是什么\"\"帮我拍张照\""
        "\"镜头里是什么\"\"形容一下你看到的\"时调用。\n"
        "调用后请根据返回的图片内容，用自然口语回答用户；同一次提问只调用一次。",
        __take_photo,
        NULL
    ), err);


    /* ---------------- 机械臂：一次性小动作（点头 / 摇摇头） ---------------- */
    TUYA_CALL_ERR_GOTO(AI_MCP_TOOL_ADD(
        "device_arm_nod",
        "让机械臂【点头】：灯头上下点两下，表示「对 / 好 / 同意 / 知道了」。\n"
        "用户说「点点头」「点头」「你同意吗」「好不好」「知道了吗」时调用。\n"
        "机械臂会先转到迎宾基准位，做完自己收回收纳位。迎宾服务在跑时 1~2 秒就做完；\n"
        "没在跑时要现连机械臂，约 15~20 秒，请告诉用户稍等一下。\n"
        "如果此刻正在人脸跟随，会先短暂停一下、做完自动恢复。",
        __arm_nod,
        NULL
    ), err);

    TUYA_CALL_ERR_GOTO(AI_MCP_TOOL_ADD(
        "device_arm_shake_head",
        "让机械臂【摇摇头】：灯头左右拧两下，表示「不对 / 不行 / 不知道」。\n"
        "用户说「摇摇头」「摇头」「不对吧」「你不知道吗」时调用。\n"
        "机械臂会先转到迎宾基准位，做完自己收回收纳位。迎宾服务在跑时 1~2 秒就做完；\n"
        "没在跑时要现连机械臂，约 15~20 秒，请告诉用户稍等一下。\n"
        "如果此刻正在人脸跟随，会先短暂停一下、做完自动恢复。",
        __arm_shake_head,
        NULL
    ), err);

    PR_NOTICE("app_mcp: tools registered (light x3, face_follow x3, greeting x3, pose x2, photo x1, arm x2)");
    return OPRT_OK;

err:
    PR_ERR("app_mcp: register tools failed, rt=%d", rt);
    return rt;
}

static OPERATE_RET __app_mcp_on_mqtt_connected(void *data)
{
    (void)data;
    return __app_mcp_tools_register();
}

OPERATE_RET app_mcp_tools_init(void)
{
    /* 必须晚于 ai_mcp_init()：订阅顺序 = 回调顺序，等在 MQTT_CONNECTED
       时 ai_mcp 先把 MCP server 建好，我们才注册得进去 */
    return tal_event_subscribe(EVENT_MQTT_CONNECTED, "app_mcp_tools",
                               __app_mcp_on_mqtt_connected, SUBSCRIBE_TYPE_ONETIME);
}

#else /* !ENABLE_COMP_AI_MCP */

/* MCP 组件没开时留个空实现，方便 app_chat_bot.c 无条件调用 */
#ifndef APP_MCP_SELFTEST
OPERATE_RET app_mcp_tools_init(void)
{
    return OPRT_OK;
}
#endif

#endif /* ENABLE_COMP_AI_MCP */
