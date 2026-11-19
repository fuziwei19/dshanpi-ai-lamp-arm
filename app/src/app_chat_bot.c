/**
 * @file app_chat_bot.c
 * @brief app_chat_bot module is used to
 * @version 0.1
 * @date 2025-03-25
 */

#include "tal_api.h"

#include "netmgr.h"

#include "ai_chat_main.h"
#include "app_chat_bot.h"
#include "app_lowpower.h"
#include "app_mcp_tools.h"
#include "app_local_ctl.h"
/* ==== emo-lamp / vision ==== */
#include "app_emo_lamp.h"
#include "app_vision.h"
#include "app_feishu.h"

#include "cJSON.h"
#include "ai_manage_mode.h"

#if defined(ENABLE_WIFI) && (ENABLE_WIFI == 1)
#include "tkl_wifi.h"
#endif

#if defined(ENABLE_PRINTER) && (ENABLE_PRINTER == 1)
#include "app_printer.h"
#endif

/***********************************************************
************************macro define************************
***********************************************************/
#define PRINTF_FREE_HEAP_TTIME (10 * 1000)
/* Net-status refresh cadence. Under idle low-power, stretch it way out: in ULP the screen is
   usually suspended and a 1s timer would keep waking the CPU for a refresh nobody sees. */
#if defined(ENABLE_APP_LOWPOWER) && (ENABLE_APP_LOWPOWER == 1)
#define DISP_NET_STATUS_TIME (30 * 1000)
#else
#define DISP_NET_STATUS_TIME (1 * 1000)
#endif

/***********************************************************
***********************typedef define***********************
***********************************************************/

/***********************************************************
***********************const declaration********************
***********************************************************/

/***********************************************************
***********************variable define**********************
***********************************************************/
static TIMER_ID sg_printf_heap_tm;

#if defined(ENABLE_COMP_AI_DISPLAY) && (ENABLE_COMP_AI_DISPLAY == 1)
static AI_UI_WIFI_STATUS_E sg_wifi_status = AI_UI_WIFI_STATUS_DISCONNECTED;
static TIMER_ID            sg_disp_status_tm;
#endif


/***********************************************************
***********************local define************************
***********************************************************/
/* DshanPi-A1 has no physical button, and the tuya CLI owns stdin
 * (tkl_uart opens /dev/stdin), so the keyboard "press S" button can
 * never see a key. Default to the wake-word mode instead.
 * 0 = hold(push-to-talk) 1 = one-shot 2 = wakeup(word) 3 = free
 */
#define APP_FORCED_CHAT_MODE   AI_CHAT_MODE_WAKEUP

#define AI_CHAT_PAR_KEY        "ty_ai_chat_par"

/**
 * @brief Force the chat mode stored in KV so the board always comes up in a
 *        mode that can actually be triggered on this hardware.
 */
static void __force_chat_mode(void)
{
    char     buf[64] = {0};
    uint8_t *old    = NULL;
    size_t   len    = 0;
    int      vol    = 59;

    if (OPRT_OK == tal_kv_get(AI_CHAT_PAR_KEY, &old, &len) && NULL != old) {
        cJSON *root = cJSON_Parse((const char *)old);
        if (NULL != root) {
            cJSON *v = cJSON_GetObjectItem(root, "volume");
            if (NULL != v && v->valueint >= 0 && v->valueint <= 100) {
                vol = v->valueint;
            }
            cJSON_Delete(root);
        }
        tal_kv_free(old);
    }

    snprintf(buf, sizeof(buf), "{\"volume\": %d, \"chat_mode\":%d}", vol, APP_FORCED_CHAT_MODE);
    tal_kv_set(AI_CHAT_PAR_KEY, (const uint8_t *)buf, strlen(buf));
    PR_NOTICE("ai chat mode forced to %d -> %s", APP_FORCED_CHAT_MODE, buf);
}

/***********************************************************
***********************function define**********************
***********************************************************/
#if defined(ENABLE_COMP_AI_DISPLAY) && (ENABLE_COMP_AI_DISPLAY == 1)
extern void app_ui_action_register(void);
#endif

static void __printf_free_heap_tm_cb(TIMER_ID timer_id, void *arg)
{
#if defined(ENABLE_EXT_RAM) && (ENABLE_EXT_RAM == 1)
    uint32_t free_heap       = tal_system_get_free_heap_size();
    uint32_t free_psram_heap = tal_psram_get_free_heap_size();
    PR_INFO("Free heap size:%d, Free psram heap size:%d", free_heap, free_psram_heap);
#else
    uint32_t free_heap = tal_system_get_free_heap_size();
    PR_INFO("Free heap size:%d", free_heap);
#endif
}

#if defined(ENABLE_COMP_AI_DISPLAY) && (ENABLE_COMP_AI_DISPLAY == 1)
static void __display_net_status_update(void)
{
    AI_UI_WIFI_STATUS_E wifi_status = AI_UI_WIFI_STATUS_DISCONNECTED;
    netmgr_status_e     net_status  = NETMGR_LINK_DOWN;

    netmgr_conn_get(NETCONN_AUTO, NETCONN_CMD_STATUS, &net_status);
    if (net_status == NETMGR_LINK_UP) {
#if defined(ENABLE_WIFI) && (ENABLE_WIFI == 1)
        // get rssi
        int8_t rssi = 0;
#ifndef PLATFORM_T5
        // BUG: Getting RSSI causes a crash on T5 platform
        tkl_wifi_station_get_conn_ap_rssi(&rssi);
#endif
        if (rssi >= -60) {
            wifi_status = AI_UI_WIFI_STATUS_GOOD;
        } else if (rssi >= -70) {
            wifi_status = AI_UI_WIFI_STATUS_FAIR;
        } else {
            wifi_status = AI_UI_WIFI_STATUS_WEAK;
        }
#else
        wifi_status = AI_UI_WIFI_STATUS_GOOD;
#endif
    } else {
        wifi_status = AI_UI_WIFI_STATUS_DISCONNECTED;
    }

    if (wifi_status != sg_wifi_status) {
        sg_wifi_status = wifi_status;
        ai_ui_disp_msg(AI_UI_DISP_NETWORK, (uint8_t *)&wifi_status, sizeof(AI_UI_WIFI_STATUS_E));
    }
}

static void __display_status_tm_cb(TIMER_ID timer_id, void *arg)
{
    __display_net_status_update();
}

#endif

static void __ai_chat_handle_event(AI_NOTIFY_EVENT_T *event)
{
    app_lowpower_feed_event(event); // power mgmt: activity / key-mode gate

    /* 飞书 IM：累积本轮 AI 文本，AI_USER_EVT_END 时投进出站信箱回发飞书。
       非阻塞：只 memcpy + 放信号量，真正的 HTTPS 发送在 app_feishu 的出站线程里。 */
    app_feishu_on_ai_event(event);

    switch(event->type) {
        #if defined(ENABLE_PRINTER) && (ENABLE_PRINTER == 1)
        case AI_USER_EVT_GENERATE_PICTURE:
        case AI_USER_EVT_GET_PICTURE_FROM_APP: {
            #if defined(ENABLE_COMP_AI_PICTURE) && (ENABLE_COMP_AI_PICTURE == 1)
            app_print_img_from_album((const char *)event->data);
            #endif
        } break;
        #endif
        case AI_USER_EVT_EMOTION:
        case AI_USER_EVT_LLM_EMOTION: {
            AI_NOTIFY_EMO_T *emo = (AI_NOTIFY_EMO_T *)(event->data);
            if (emo != NULL && emo->name != NULL) {
                /* 非阻塞：只记录情绪名并唤醒灯效线程（毫秒级返回） */
                app_emo_lamp_notify(emo->name);
            }
        } break;

        case AI_USER_EVT_ASR_OK: {
            AI_NOTIFY_TEXT_T *t = (AI_NOTIFY_TEXT_T *)event->data;
            if (t != NULL && t->datalen > 0 && t->data != NULL) {
                /* 命中「看一下 / 前面有什么 / 拍照」等关键词时才真正拍照 */
                app_vision_maybe_capture(t->data, (int)t->datalen);
            }
        } break;

        default:
        break;
    }

}

void app_chat_bot_set_online(BOOL_T online)
{
    app_lowpower_set_online(online);
}

OPERATE_RET app_chat_bot_init(void)
{
    OPERATE_RET rt = OPRT_OK;

    __force_chat_mode();

    AI_CHAT_MODE_CFG_T ai_chat_cfg = {
        .default_mode = AI_CHAT_MODE_HOLD, // push-to-talk; idle low-power gates on this mode
        .default_vol  = 70,
        .evt_cb       = __ai_chat_handle_event,
    };

    /* [PATCH-11-03-FREE120-KV] 免唤醒连续对话（唤醒一次 -> 连续多轮 -> 空闲 120s 回待命）
       ai_chat_init() 和 MQTT 连上时的 ai_mode_init() 都以 KV 里的 chat_mode 为准，
       所以这里先把 KV 修正成 3(FREE)，模式选择就自然生效，不用抢初始化时序。
       （WAKEUP=2 的行为是「每说一句都要重新喊唤醒词」；FREE=3 才是连续对话） */
    {
        uint8_t *kv_val  = NULL;
        size_t   kv_len  = 0;
        cJSON   *kv_root = NULL;
        char     kv_buf[96];
        bool     changed = false;

        if (OPRT_OK == tal_kv_get("ty_ai_chat_par", &kv_val, &kv_len) && kv_val) {
            kv_root = cJSON_Parse((const char *)kv_val);
            tal_kv_free(kv_val);
        }

        if (kv_root) {
            cJSON *m = cJSON_GetObjectItem(kv_root, "chat_mode");
            if (m && (m->valueint == (int)AI_CHAT_MODE_FREE)) {
                /* 已经是 FREE，什么都不用做 */
            } else {
                if (m) {
                    cJSON_SetNumberValue(m, (int)AI_CHAT_MODE_FREE);
                } else {
                    cJSON_AddNumberToObject(kv_root, "chat_mode", (int)AI_CHAT_MODE_FREE);
                }
                char *js = cJSON_PrintUnformatted(kv_root);
                if (js) {
                    tal_kv_set("ty_ai_chat_par", (const uint8_t *)js, strlen(js));
                    cJSON_free(js);
                    changed = true;
                }
            }
            cJSON_Delete(kv_root);
        } else {
            snprintf(kv_buf, sizeof(kv_buf),
                     "{\"volume\": %d, \"chat_mode\": %d}",
                     70, (int)AI_CHAT_MODE_FREE);
            tal_kv_set("ty_ai_chat_par", (const uint8_t *)kv_buf, strlen(kv_buf));
            changed = true;
        }

        if (changed) {
            PR_NOTICE("app: chat_mode -> FREE(3)，唤醒一次后可连续对话，空闲 120s 回待命");
        } else {
            PR_NOTICE("app: chat_mode 已是 FREE(3)");
        }
    }

    TUYA_CALL_ERR_RETURN(ai_chat_init(&ai_chat_cfg));


    app_lowpower_init();

#if defined(ENABLE_COMP_AI_DISPLAY) && (ENABLE_COMP_AI_DISPLAY == 1)
    app_ui_action_register();
#endif

#if defined(ENABLE_COMP_AI_VIDEO) && (ENABLE_COMP_AI_VIDEO == 1)
    TUYA_CALL_ERR_LOG(ai_video_init());
#endif

#if defined(ENABLE_COMP_AI_MCP) && (ENABLE_COMP_AI_MCP == 1)
    TUYA_CALL_ERR_RETURN(ai_mcp_init());
    /* 应用侧端侧 MCP 工具（灯环 / 人脸跟随 / 迎宾）。
       必须排在 ai_mcp_init() 之后订阅：MQTT 连上时 ai_mcp 先把 MCP server 建好，
       我们的回调才会在 server 就绪之后注册工具。 */
    TUYA_CALL_ERR_LOG(app_mcp_tools_init());
#endif

    /* 本机控制通道：给板上的独立进程（视觉迎宾 / 人脸跟随 Python 脚本）留一个
       /run/tuya_chatbot.sock，往里写一行文本就等于让 AI 说一句话
       （内部调用 ai_agent_send_text）。 */
    TUYA_CALL_ERR_LOG(app_local_ctl_init());

    /* 情绪灯效：云端每轮对话回的 emotion -> 灯环颜色。
       全异步：事件回调只写缓冲 + 放信号量，真正的灯控在独立线程里做，
       2s 后自动恢复原灯效，绝不占住对话流程。 */
    TUYA_CALL_ERR_LOG(app_emo_lamp_init());

    /* AI 图片理解：拍照 -> tuya_ai_image_input + 文本引导语 送给多模态 agent。
       关键词兜底（"看一下前面有什么"）与 MCP 工具 device_take_photo 共用同一工作线程。 */
    TUYA_CALL_ERR_LOG(app_vision_init());

    /* 飞书 IM 频道：飞书发文字 <-> 云端 AI。
       入站：IM message_bus -> ai_agent_send_text()；
       出站：AI_USER_EVT_TEXT_STREAM_* -> 累积 -> feishu_send_message()。
       凭证在 /etc/feishu.json（或 include 里的 im_secrets.h）；没配就是空转，不报错。 */
    TUYA_CALL_ERR_LOG(app_feishu_init());

#if defined(ENABLE_COMP_AI_PICTURE) && (ENABLE_COMP_AI_PICTURE == 1)
    TUYA_CALL_ERR_RETURN(ai_picture_init());
#endif

    // Free heap size
    tal_sw_timer_create(__printf_free_heap_tm_cb, NULL, &sg_printf_heap_tm);
    tal_sw_timer_start(sg_printf_heap_tm, PRINTF_FREE_HEAP_TTIME, TAL_TIMER_CYCLE);

#if defined(ENABLE_COMP_AI_DISPLAY) && (ENABLE_COMP_AI_DISPLAY == 1)
    ai_ui_disp_msg(AI_UI_DISP_NETWORK, (uint8_t *)&sg_wifi_status, sizeof(AI_UI_WIFI_STATUS_E));

    ai_ui_disp_msg(AI_UI_DISP_STATUS, (uint8_t *)INITIALIZING, strlen(INITIALIZING));
    ai_ui_disp_msg(AI_UI_DISP_EMOTION, (uint8_t *)EMOJI_NEUTRAL, strlen(EMOJI_NEUTRAL));

    // display status update
    tal_sw_timer_create(__display_status_tm_cb, NULL, &sg_disp_status_tm);
    tal_sw_timer_start(sg_disp_status_tm, DISP_NET_STATUS_TIME, TAL_TIMER_CYCLE);
#endif

    return OPRT_OK;
}
