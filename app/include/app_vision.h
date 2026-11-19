/**
 * @file app_vision.h
 * @brief AI 图片理解：拍一张照，连同引导语一起送给云端多模态 agent
 *
 * 触发方式（两条路，互补）
 * ----------------------
 * 1) 云端驱动：端侧 MCP 工具 `device_take_photo`。云端 LLM 判断用户要看画面时
 *    调用它，端侧拍照后把图塞回当前对话 —— 这是最正的路径（标准多模态 tool call）。
 * 2) 端侧兜底：监听 ASR 文本，命中「看一下 / 前面有什么 / 拍照 / camera」等关键词
 *    就地起任务。万一云端没主动调工具，用户照样能看图得到回答。
 *
 * 两条路都走同一个**独立工作线程**：拍照（约 0.3~1s）+ 上传（tuya_ai_image_input）
 * 都不在对话主线程 / 事件回调里做，避免卡住会话。
 *
 * @copyright Copyright (c) 2026
 */

#ifndef __APP_VISION_H__
#define __APP_VISION_H__

#include "tuya_cloud_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 初始化图片理解模块（建工作线程）。幂等。
 */
OPERATE_RET app_vision_init(void);

/**
 * @brief 请求拍一张照并送给 AI（**非阻塞**，立即返回）
 *
 * @param[in] user_text 可选的用户原话（会拼进引导语，让模型知道用户问了什么）；
 *                      传 NULL 表示无上下文（MCP 工具触发时就是这种）
 * @return BOOL_T TRUE=已受理；FALSE=正忙 / 未启用 / 距上次太近被节流
 */
BOOL_T app_vision_request(const char *user_text);

/**
 * @brief 请求拍一张照，并**完整指定**送给云端的引导语（**非阻塞**）
 *
 * 与 app_vision_request() 的区别：后者会把 user_text 套进固定模板
 * （「用户刚刚说：… 请你结合这张实时画面，用一到两句自然口语回答他」），
 * 适合回答提问；本接口不套模板，适合「让它主动自我介绍」这类需要
 * 调用方把话说全的场景。
 *
 * @param[in] prompt 完整的引导语（会原样送给云端）
 * @return BOOL_T TRUE=已受理；FALSE=正忙 / 未启用 / 距上次太近被节流
 */
BOOL_T app_vision_request_prompt(const char *prompt);

/**
 * @brief ASR 文本命中关键词时自动请求拍照（**非阻塞**）
 *
 * @param[in] text  ASR 文本（不必以 '\0' 结尾）
 * @param[in] len   文本长度
 * @return BOOL_T TRUE=命中并已受理
 */
BOOL_T app_vision_maybe_capture(const char *text, int len);

/**
 * @brief 运行时开关「关键词自动拍照」
 */
void app_vision_set_auto_enabled(BOOL_T on);

#ifdef __cplusplus
}
#endif

#endif /* __APP_VISION_H__ */
