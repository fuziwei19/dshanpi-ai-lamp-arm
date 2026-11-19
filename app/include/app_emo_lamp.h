/**
 * @file app_emo_lamp.h
 * @brief 情绪灯效：把云端返回的 emotion 字段映射成 WS2812 灯环颜色
 *
 * 为什么单独做一个模块
 * ------------------
 * 云端**每次对话都会回一个情绪值**，如果收到就同步点亮、等 2 秒再熄灭，
 * 对话主流程会被反复占住（排队、卡顿）。所以这里把「显示情绪」做成一个
 * **完全异步的覆盖层**：
 *
 *   - 事件回调里只做两件极快的事：拷名字到全局缓冲 + 释放一个信号量（零 IO，微秒级）
 *   - 真正的发灯指令、2 秒计时、恢复原灯效，全部在独立线程里做
 *   - 2 秒内又来新情绪 → **立刻刷新颜色并重新计时**（debounce），不会堆积排队
 *
 * 恢复语义：覆盖前先读一次灯环当前状态存为「基础灯效」，2 秒后写回去；
 *           读不到就退化为关灯。
 *
 * @copyright Copyright (c) 2026
 */

#ifndef __APP_EMO_LAMP_H__
#define __APP_EMO_LAMP_H__

#include "tuya_cloud_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 初始化情绪灯效模块（建线程 / 信号量 / 读配置）。幂等。
 *
 * 配置（可选）：/etc/emo-lamp.json
 *   {"enabled": true, "hold_ms": 2000}
 * 文件不存在则用编译期默认值（enabled=true, hold_ms=2000）。
 *
 * @return OPERATE_RET OPRT_OK 成功
 */
OPERATE_RET app_emo_lamp_init(void);

/**
 * @brief 通知一个情绪（**非阻塞**，可直接在事件回调里调）
 *
 * @param[in] emotion_name 云端情绪名，如 "ANGRY" / "HAPPY" / "SAD"（大小写不敏感）
 */
void app_emo_lamp_notify(const char *emotion_name);

/**
 * @brief 运行时开关情绪灯效（false 时 notify 直接返回）
 */
void app_emo_lamp_set_enabled(BOOL_T on);

/**
 * @brief 当前是否启用
 */
BOOL_T app_emo_lamp_is_enabled(void);

/**
 * @brief 最近一次实际点亮的颜色（"#RRGGBB"，给日志 / 查询用）
 */
const char *app_emo_lamp_last_color(void);

#ifdef __cplusplus
}
#endif

#endif /* __APP_EMO_LAMP_H__ */
