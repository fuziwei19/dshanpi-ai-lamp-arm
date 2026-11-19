/**
 * @file app_feishu.h
 * @brief 飞书（Feishu / Lark）IM 频道 <-> 云端 AI 双向桥
 *
 * 来自 tuya/TClaw 的 IM 组件（IM/channels/feishu_bot.c），只取 feishu 一条链路，
 * 用长连接（WebSocket + Protobuf）收消息、用 REST API 发消息。
 *
 * 数据流
 * ------
 *   入站：飞书用户发文字 -> IM message_bus -> ai_agent_send_text()   -> 云端 AI
 *   出站：云端 AI 文字流(AI_USER_EVT_TEXT_STREAM_*) -> 累积
 *         -> 文字流闭合 + 会话状态回落 IDLE（见 app_feishu.c 顶部说明）
 *         -> 出站线程 -> feishu_send_message()                        -> 飞书
 *
 *   拍照：飞书消息命中「拍照/看看」等关键词 -> app_vision_request() 拍一张
 *         -> app_vision.c 里回调 app_feishu_on_photo(jpeg,len)
 *         -> 出站线程 -> feishu_send_image()（上传换 image_key 再发） -> 飞书
 *         同一张 JPEG 也照常送给云端多模态 AI，所以用户会先收到照片、再收到 AI 的描述。
 *
 * 设计要点：**事件回调里绝不做网络 IO**
 *   - AI_USER_EVT_TEXT_STREAM_* 只往内存缓冲里 memcpy（微秒级）
 *   - 回合结束后只把文本投进「单槽信箱」+ 放信号量
 *   - app_feishu_on_photo() 只 memcpy 一份 JPEG，上传同样在出站线程做
 *   - 真正的 HTTP 发送由独立线程（feishu_out）做
 *   否则飞书发送的 1~2s 阻塞会卡住整条 AI 事件流水线。
 *
 * @copyright Copyright (c) 2026
 */

#ifndef __APP_FEISHU_H__
#define __APP_FEISHU_H__

#include "tal_api.h"
#include "ai_user_event.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 初始化飞书桥（幂等）
 *
 * 依次做：建同步原语 -> 读 /etc/feishu.json -> message_bus_init / http_proxy_init
 * -> feishu_bot_init -> 起 入站桥线程 + 出站发送线程。
 *
 * 真正的 feishu_bot_start()（要拿 tenant token + WS 端点，依赖公网）由桥线程
 * 反复重试，直到成功 —— 因为启动时刻网络往往还没就绪。
 *
 * @return OPERATE_RET OPRT_OK 表示桥已起来（不代表已经连上飞书）
 */
OPERATE_RET app_feishu_init(void);

/**
 * @brief 供 AI 事件回调调用（在 app_chat_bot.c 的 __ai_chat_handle_event 里）
 *
 * 只应传 TEXT_STREAM_* / END / ABORT 这几类事件；其它类型会被直接忽略。
 * 非飞书触发的那一轮对话（比如语音唤醒）不会被转发。
 */
void app_feishu_on_ai_event(AI_NOTIFY_EVENT_T *event);

/** @brief 飞书 WS 是否已经连上（成功拿到 tenant token 并起了 WS 线程） */
BOOL_T app_feishu_is_started(void);

/** @brief app_id/app_secret 是否都已经配置 */
BOOL_T app_feishu_is_configured(void);

/** @brief 最近一次收到消息的会话（open_id 或 oc_xxx）；没有则返回空串 */
const char *app_feishu_chat_id(void);

/** @brief app_id 明文（用于状态查询），未配置时返回空串 */
const char *app_feishu_app_id(void);

/**
 * @brief 主动往飞书发一条文本（诊断用）
 *
 * @param[in] chat_id 目标会话；传 NULL 或空串则用最近一次收到消息的会话
 * @param[in] text    文本内容
 * @return OPERATE_RET OPRT_OK 表示已投进出站信箱（不代表已送达）
 */
OPERATE_RET app_feishu_send_text(const char *chat_id, const char *text);

/**
 * @brief 交给桥接层一张刚拍到的 JPEG —— 由 app_vision / 拍照工具回调
 *
 * **非阻塞**：只把这帧 memcpy 一份进「待发照片槽」+ 放信号量，真正的上传在
 * 出站线程里做（同上：绝不阻塞拍照/事件线程）。
 *
 * 只有「当前有一轮飞书对话在进行」时才会收下（s_pending），其它情况（语音唤醒、
 * 本机命令触发的拍照）直接丢弃 —— 保证照片不会串味发到飞书。
 * 一轮里只保留第一张（后面的丢弃）。
 *
 * @param[in] jpeg JPEG 数据（函数内会拷贝，调用方可立即释放）
 * @param[in] len  JPEG 长度
 */
void app_feishu_on_photo(const uint8_t *jpeg, uint32_t len);

/**
 * @brief 主动往飞书发一张图片（诊断用，不经过 AI）
 *
 * @param[in] chat_id 目标会话；NULL/空串则用最近一次收到消息的会话
 * @param[in] jpeg    JPEG 数据
 * @param[in] len     JPEG 长度
 * @return OPERATE_RET OPRT_OK 表示已投进待发槽（不代表已送达）
 */
OPERATE_RET app_feishu_send_image(const char *chat_id, const uint8_t *jpeg, uint32_t len);

/**
 * @brief 自测：往 IM 消息总线里灌一条「假的入站消息」
 *
 * 用途：在还没拿到飞书 app_id/app_secret 时，验证「总线 -> 等空闲 -> ai_agent_send_text
 * -> 文字流累积 -> 回合结束判定」这半条链路（不碰任何网络）。
 * 日志里应能看到 inbound -> ai_agent_send_text -> turn finished (stream closed + ai idle)。
 *
 * 出站会因为 bot 未启动而 drop（日志 feishu: bot not started），属预期。
 *
 * @param[in] chat_id 目标会话；NULL/空串时用 "ou_test_fake"
 * @param[in] text    文本内容
 * @return OPERATE_RET OPRT_OK 表示已投进总线队列
 */
OPERATE_RET app_feishu_test_inject(const char *chat_id, const char *text);

/**
 * @brief 清空已保存的飞书凭证（KV 里写空串）
 *
 * 用途：填错凭证后不用重刷固件也能恢复。清空后 app_feishu_is_configured() 变 false，
 * 出站线程不再尝试连飞书（也不会刷错误日志）。需要重启进程才彻底生效。
 *
 * @note 只清 app_id/app_secret，不动 /etc/feishu.json —— 如果那个文件还在，
 *       下次启动仍会从它加载。
 */
OPERATE_RET app_feishu_clear_credentials(void);

#ifdef __cplusplus
}
#endif

#endif /* __APP_FEISHU_H__ */
