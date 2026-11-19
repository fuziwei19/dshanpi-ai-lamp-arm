/**
 * @file app_local_ctl.h
 * @brief 本机控制通道：把 AI 的「文本对话入口」暴露给板上的本地进程
 *
 * @copyright Copyright (c) 2026
 */

#ifndef __APP_LOCAL_CTL_H__
#define __APP_LOCAL_CTL_H__

#include "tuya_cloud_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 启动本机控制 socket 服务（监听 /run/tuya_chatbot.sock）
 *
 * 服务做的事：收到一行文本 -> 交给 ai_agent_send_text() -> AI 生成回复并 TTS 说出。
 * 用于让板上的独立进程（如视觉迎宾 Python 脚本）触发 AI 主动说话。
 *
 * 幂等：重复调用只创建一次。
 *
 * @return OPERATE_RET OPRT_OK 成功
 */
OPERATE_RET app_local_ctl_init(void);

#ifdef __cplusplus
}
#endif

#endif /* __APP_LOCAL_CTL_H__ */
