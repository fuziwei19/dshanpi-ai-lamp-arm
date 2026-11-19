/**
 * @file app_mcp_tools.h
 * @brief 应用侧 MCP 工具：WS2812 灯环控制 + 机械臂人脸跟随 / 迎宾
 *
 * @copyright Copyright (c) 2021-2025 Tuya Inc. All Rights Reserved.
 */

#ifndef __APP_MCP_TOOLS_H__
#define __APP_MCP_TOOLS_H__

#include "tuya_cloud_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 注册本应用的端侧 MCP 工具
 *
 * @note 必须在 ai_mcp_init() 之后调用：本函数只是订阅 MQTT_CONNECTED，
 *       等 ai_mcp 的 MCP server 建好之后才真正注册工具（与
 *       your_otto_robot/src/otto/otto_mcp_tools.c 同一套路）。
 *
 * @return OPERATE_RET 订阅结果
 */
OPERATE_RET app_mcp_tools_init(void);

#ifdef __cplusplus
}
#endif

#endif /* __APP_MCP_TOOLS_H__ */
