/**
 * @file app_local_ctl.c
 * @brief 本机控制通道：把一个 UNIX socket 暴露给板上的本地进程（Python），
 *        用一段文本「戳」一下 AI（ai_agent_send_text），让 AI 主动说话。
 *
 * 为什么需要它
 * ------------
 * `ai_agent_send_text()` 是**进程内**的 C 接口，内部走
 *     tuya_ai_input_start(true) -> tuya_ai_text_input() -> tuya_ai_input_stop()
 * 而板上的「视觉迎宾 / 人脸跟随」是**独立进程**（Python），没法直接调它。
 * 所以这里在 app 里开一个本地 socket 当「嘴」：脚本往 socket 写一行文本即可。
 * 这样迎宾脚本检测到有人靠近时，就能让 AI 主动打招呼（而不是只让机械臂动）。
 *
 * 协议（UTF-8，每行一条，以 \n 结尾）
 * ---------------------------------
 *   {"cmd":"say","text":"打个招呼"}   -> 把 text 交给 AI，AI 会 TTS 说出来
 *   {"cmd":"status"}                  -> 查询 AI agent 是否已就绪
 *   {"cmd":"photo","text":"可选问题"}  -> 拍一张照（可带问题）送给多模态 AI 识别
 *   <其它任意文本>                     -> 等价于 {"cmd":"say","text":"<整行>"}（方便调试）
 * 回复（同样一行 JSON）：
 *   成功  {"ok":true,"cmd":"say","rt":0}
 *   失败  {"ok":false,"cmd":"say","rt":-3,"error":"ai agent not ready"}
 *
 * 调试（板子上）
 * -------------
 *   printf '打个招呼\n' | python3 -c "import socket,sys;s=socket.socket(socket.AF_UNIX);\
 *   s.connect('/run/tuya_chatbot.sock');s.sendall(sys.stdin.buffer.read());print(s.recv(256))"
 *   或直接跑 /root/notify_ai.py 打个招呼
 *
 * 线程说明
 * --------
 * 独立线程阻塞在 accept/read 上；客户端 socket 设了收发超时，避免半开连接把线程挂死。
 */

#include "app_local_ctl.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/un.h>
#include <unistd.h>

#include "tal_api.h"

#include "ai_agent.h"
#include "tuya_ai_agent.h"
#include "app_vision.h"
#include "app_feishu.h"
#include "cJSON.h"

/***********************************************************
************************macro define************************
***********************************************************/
#define APP_LOCAL_CTL_SOCK       "/run/tuya_chatbot.sock"
#define APP_LOCAL_CTL_BACKLOG    (4)
#define APP_LOCAL_CTL_LINE_MAX   (1024)     /* 单行最大长度（含 '\0'） */
#define APP_LOCAL_CTL_TEXT_MAX   (768)      /* 送给 AI 的文本上限 */
#define APP_LOCAL_CTL_RX_TMO_S   (5)        /* 客户端读超时（秒） */
#define APP_LOCAL_CTL_TX_TMO_S   (3)        /* 客户端写超时（秒） */
#define APP_LOCAL_CTL_THRD_STACK (4096)
#define APP_LOCAL_CTL_THRD_PRIO  (4)

/***********************************************************
***********************variable define**********************
***********************************************************/
static int           sg_listen_fd = -1;
static THREAD_HANDLE sg_thrd      = NULL;
static volatile BOOL_T sg_running = FALSE;

/***********************************************************
***********************function define**********************
***********************************************************/
/* 去掉首尾空白，返回去空白后的起始指针（原地修改） */
static char *__trim(char *s)
{
    char *end = NULL;

    if (NULL == s) {
        return NULL;
    }
    while (*s == ' ' || *s == '\t' || *s == '\r' || *s == '\n') {
        s++;
    }
    end = s + strlen(s);
    while (end > s && (end[-1] == ' ' || end[-1] == '\t' || end[-1] == '\r' || end[-1] == '\n')) {
        *--end = '\0';
    }
    return s;
}

static void __reply(int fd, const char *json)
{
    size_t len = strlen(json);
    size_t off = 0;

    while (off < len) {
        ssize_t n = write(fd, json + off, len - off);
        if (n > 0) {
            off += (size_t)n;
            continue;
        }
        if (n < 0 && errno == EINTR) {
            continue;
        }
        break; /* 对端关了 / 超时，放弃 */
    }
}

/* 处理一行请求：识出 cmd/text -> 调 ai_agent_send_text -> 回一行 JSON */
static void __handle_line(int fd, char *line)
{
    static char reply[APP_LOCAL_CTL_LINE_MAX];
    char        text[APP_LOCAL_CTL_TEXT_MAX] = {0};
    char        cmd[32]                      = "say";
    char        chat[96]                     = {0};   /* 可选：指定飞书 chat_id / open_id */
    char        prompt[512]                  = {0};   /* 可选：photo 命令的完整引导语 */
    char       *s                            = __trim(line);
    cJSON      *root                         = NULL;
    BOOL_T      is_say                       = FALSE;

    if (NULL == s || '\0' == *s) {
        return;
    }

    root = cJSON_Parse(s);
    if (NULL != root && cJSON_IsObject(root)) {
        cJSON *jcmd = cJSON_GetObjectItem(root, "cmd");
        cJSON *jtxt = cJSON_GetObjectItem(root, "text");
        if (NULL != jcmd && cJSON_IsString(jcmd)) {
            snprintf(cmd, sizeof(cmd), "%s", jcmd->valuestring);
        }
        if (NULL != jtxt && cJSON_IsString(jtxt)) {
            snprintf(text, sizeof(text), "%.*s", (int)(sizeof(text) - 1), jtxt->valuestring);
            is_say = TRUE;
        }
        /* 可选目标：{"chat":"ou_xxx"} / {"chat_id":"oc_xxx"}，用于把消息发到指定会话 */
        {
            cJSON *jchat = cJSON_GetObjectItem(root, "chat");
            if (NULL == jchat || !cJSON_IsString(jchat)) {
                jchat = cJSON_GetObjectItem(root, "chat_id");
            }
            if (NULL != jchat && cJSON_IsString(jchat)) {
                snprintf(chat, sizeof(chat), "%.*s", (int)(sizeof(chat) - 1), jchat->valuestring);
            }
        }
        /* 可选完整引导语：{"cmd":"photo","prompt":"..."} */
        {
            cJSON *jp = cJSON_GetObjectItem(root, "prompt");
            if (NULL != jp && cJSON_IsString(jp)) {
                snprintf(prompt, sizeof(prompt), "%.*s", (int)(sizeof(prompt) - 1), jp->valuestring);
            }
        }
    } else {
        /* 不是 JSON：整行当成要说的文本（方便 shell 里直接 echo） */
        snprintf(text, sizeof(text), "%.*s", (int)(sizeof(text) - 1), s);
        snprintf(cmd, sizeof(cmd), "say");
        is_say = TRUE;
    }

    if (NULL != root) {
        cJSON_Delete(root);
    }

    if (0 == strcmp(cmd, "status")) {
        BOOL_T ready = tuya_ai_agent_is_ready();
        snprintf(reply, sizeof(reply), "{\"ok\":true,\"cmd\":\"status\",\"ready\":%s}\n",
                 ready ? "true" : "false");
        __reply(fd, reply);
        return;
    }

    /* 让本地进程也能请求「拍一张照给 AI 看」。
       与端侧 MCP 工具 device_take_photo 走同一条路径（app_vision 的工作线程），
       好处是不用对着设备说话也能从命令行自测整条图片理解链路；
       顺带也让板上的 Python 脚本能主动请求「看一眼」。 */
    if (0 == strcmp(cmd, "photo")) {
        /* 给了 prompt 就原样送给云端（适合「主动自我介绍」这类需要说全的场景），
           否则沿用旧的自动拼接模板（适合「回答提问」）。 */
        BOOL_T ok = ('\0' != prompt[0])
                        ? app_vision_request_prompt(prompt)
                        : app_vision_request(('\0' != text[0]) ? text : NULL);
        PR_NOTICE("local ctl: photo -> %s (%s)", ok ? "accepted" : "rejected",
                  ('\0' != prompt[0]) ? "custom prompt" : "auto template");
        snprintf(reply, sizeof(reply), "{\"ok\":%s,\"cmd\":\"photo\"}\n", ok ? "true" : "false");
        __reply(fd, reply);
        return;
    }

    /* 飞书 IM 频道诊断（板子上可以直接调）：
       {"cmd":"feishu"}                  -> 查配置 / 连接状态
       {"cmd":"feishu_send","text":".."} -> 给最近一次飞书会话发一条文本（测出站）
       {"cmd":"feishu_inject","text":".."} -> 灌一条假入站消息，测「总线 -> AI -> 回合判定」
                                             （不需要飞书凭证；出站会因未启动而 drop） */
    if (0 == strcmp(cmd, "feishu")) {
        snprintf(reply, sizeof(reply),
                 "{\"ok\":true,\"cmd\":\"feishu\",\"configured\":%s,\"started\":%s,"
                 "\"app_id\":\"%s\",\"chat_id\":\"%s\"}\n",
                 app_feishu_is_configured() ? "true" : "false",
                 app_feishu_is_started() ? "true" : "false",
                 app_feishu_app_id(), app_feishu_chat_id());
        __reply(fd, reply);
        return;
    }

    if (0 == strcmp(cmd, "feishu_inject")) {
        /* chat 为空时走「最近一次会话」，否则发到指定的 open_id / chat_id */
        OPERATE_RET frt = app_feishu_test_inject(('\0' != chat[0]) ? chat : NULL, text);
        snprintf(reply, sizeof(reply), "{\"ok\":%s,\"cmd\":\"feishu_inject\",\"rt\":%d,\"chat\":\"%s\"}\n",
                 (OPRT_OK == frt) ? "true" : "false", frt, chat);
        __reply(fd, reply);
        return;
    }

    if (0 == strcmp(cmd, "feishu_clear")) {
        OPERATE_RET frt = app_feishu_clear_credentials();
        snprintf(reply, sizeof(reply),
                 "{\"ok\":%s,\"cmd\":\"feishu_clear\",\"rt\":%d,\"hint\":\"restart to take effect\"}\n",
                 (OPRT_OK == frt) ? "true" : "false", frt);
        __reply(fd, reply);
        return;
    }

    if (0 == strcmp(cmd, "feishu_send")) {
        OPERATE_RET frt = app_feishu_send_text(NULL, text);
        snprintf(reply, sizeof(reply), "{\"ok\":%s,\"cmd\":\"feishu_send\",\"rt\":%d}\n",
                 (OPRT_OK == frt) ? "true" : "false", frt);
        __reply(fd, reply);
        return;
    }

    if (!is_say || '\0' == text[0]) {
        snprintf(reply, sizeof(reply),
                 "{\"ok\":false,\"cmd\":\"%s\",\"error\":\"need text\"}\n", cmd);
        __reply(fd, reply);
        return;
    }

    if (!tuya_ai_agent_is_ready()) {
        /* AI client 还没连上云端（MQTT 刚起 / 正在重连）：让调用方过一会重试 */
        PR_WARN("local ctl: ai agent not ready, drop text");
        snprintf(reply, sizeof(reply),
                 "{\"ok\":false,\"cmd\":\"say\",\"error\":\"ai agent not ready\"}\n");
        __reply(fd, reply);
        return;
    }

    OPERATE_RET rt = ai_agent_send_text(text);
    PR_NOTICE("local ctl: say \"%s\" -> rt=%d", text, rt);

    snprintf(reply, sizeof(reply), "{\"ok\":%s,\"cmd\":\"say\",\"rt\":%d}\n",
             (OPRT_OK == rt) ? "true" : "false", rt);
    __reply(fd, reply);
}

/* 逐字节读到 '\n'；返回读到的字节数，0 = 对端正常关闭且无待处理数据，-1 = 出错 */
static int __read_line(int fd, char *buf, size_t cap)
{
    size_t used = 0;

    while (used + 1 < cap) {
        char    ch = 0;
        ssize_t n  = read(fd, &ch, 1);

        if (0 == n) {
            return (0 == used) ? 0 : (int)used;
        }
        if (n < 0) {
            if (EINTR == errno) {
                continue;
            }
            if (EAGAIN == errno || EWOULDBLOCK == errno) {
                break; /* 超时：把已收到的当一行处理 */
            }
            return -1;
        }
        if ('\n' == ch) {
            break;
        }
        if ('\r' == ch) {
            continue;
        }
        buf[used++] = ch;
    }
    buf[used] = '\0';
    return (int)used;
}

static void __serve_client(int fd)
{
    char      line[APP_LOCAL_CTL_LINE_MAX];
    struct timeval tv;

    tv.tv_sec  = APP_LOCAL_CTL_RX_TMO_S;
    tv.tv_usec = 0;
    (void)setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    tv.tv_sec  = APP_LOCAL_CTL_TX_TMO_S;
    tv.tv_usec = 0;
    (void)setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

    for (;;) {
        int n = __read_line(fd, line, sizeof(line));
        if (n <= 0) {
            break;
        }
        __handle_line(fd, line);
    }
}

static void __ctl_task(void *arg)
{
    (void)arg;

    while (sg_running) {
        int cfd = accept(sg_listen_fd, NULL, NULL);
        if (cfd < 0) {
            if (EINTR == errno) {
                continue;
            }
            if (!sg_running) {
                break;
            }
            tal_system_sleep(50);
            continue;
        }
        __serve_client(cfd);
        (void)close(cfd);
    }
    PR_NOTICE("local ctl: task exit");
}

OPERATE_RET app_local_ctl_init(void)
{
    struct sockaddr_un addr;
    THREAD_CFG_T       cfg = {0};

    if (sg_running) {
        return OPRT_OK; /* 幂等 */
    }

    /* 上一次异常退出可能留下 stale socket 文件 */
    (void)unlink(APP_LOCAL_CTL_SOCK);

    sg_listen_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (sg_listen_fd < 0) {
        PR_ERR("local ctl: socket() failed, errno=%d", errno);
        return OPRT_COM_ERROR;
    }

    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    snprintf(addr.sun_path, sizeof(addr.sun_path), "%s", APP_LOCAL_CTL_SOCK);

    if (bind(sg_listen_fd, (struct sockaddr *)&addr, sizeof(addr)) != 0) {
        PR_ERR("local ctl: bind(%s) failed, errno=%d", APP_LOCAL_CTL_SOCK, errno);
        (void)close(sg_listen_fd);
        sg_listen_fd = -1;
        return OPRT_COM_ERROR;
    }

    if (listen(sg_listen_fd, APP_LOCAL_CTL_BACKLOG) != 0) {
        PR_ERR("local ctl: listen() failed, errno=%d", errno);
        (void)close(sg_listen_fd);
        sg_listen_fd = -1;
        return OPRT_COM_ERROR;
    }

    /* 板上脚本可能以别的用户跑：放开权限 */
    if (chmod(APP_LOCAL_CTL_SOCK, 0666) != 0) {
        PR_WARN("local ctl: chmod(%s) failed, errno=%d", APP_LOCAL_CTL_SOCK, errno);
    }

    sg_running = TRUE;

    cfg.thrdname   = "app_local_ctl";
    cfg.priority   = APP_LOCAL_CTL_THRD_PRIO;
    cfg.stackDepth = APP_LOCAL_CTL_THRD_STACK;
    cfg.psram_mode = 0;

    if (OPRT_OK != tal_thread_create_and_start(&sg_thrd, NULL, NULL, __ctl_task, NULL, &cfg)) {
        PR_ERR("local ctl: create thread failed");
        sg_running = FALSE;
        (void)close(sg_listen_fd);
        sg_listen_fd = -1;
        return OPRT_COM_ERROR;
    }

    PR_NOTICE("local ctl: ready at %s (echo text to let AI speak)", APP_LOCAL_CTL_SOCK);
    return OPRT_OK;
}
