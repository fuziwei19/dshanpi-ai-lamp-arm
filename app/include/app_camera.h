/**
 * @file app_camera.h
 * @brief 摄像头拍照（V4L2 / UVC），给「AI 图片理解」提供一张 JPEG 快照
 *
 * 设计要点
 * --------
 * - 直接用 TuyaOpen LINUX 平台层已经实现好的 tkl_camera_v4l2_*（内部就是标准
 *   V4L2 ioctl + mmap 四缓冲），不再自己写一遍 ioctl。
 * - 「抓一帧就关」：每次 snapshot 都 open -> start -> 取帧 -> stop -> close，
 *   不长期占用 /dev/video*。板上同时还有 Python 视觉服务（迎宾 / 人脸跟随），
 *   它们也用同一个 UVC 摄像头，谁先占谁先用 —— 拍照失败时错误信息会说明这一点。
 * - 输出的是 MJPEG（UVC 直接给的 JPEG 流），不需要软件编码，可直接喂给
 *   tuya_ai_image_input()。
 *
 * @copyright Copyright (c) 2026
 */

#ifndef __APP_CAMERA_H__
#define __APP_CAMERA_H__

#include "tuya_cloud_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 初始化拍照模块（读设备节点配置）。幂等。
 *
 * 设备节点优先取 /etc/ai-camera.dev 的第一行（纯文本，如 "/dev/video12"），
 * 文件不存在则用编译期默认值（/dev/video12）。
 *
 * @return OPERATE_RET OPRT_OK
 */
OPERATE_RET app_camera_init(void);

/**
 * @brief 抓一帧 JPEG
 *
 * 成功后 *jpeg 指向一块用 tal_malloc 分配的内存（调用方用 app_camera_free 释放），
 * *len 是 JPEG 字节数。失败时 *jpeg == NULL。
 *
 * @param[out] jpeg 返回 JPEG 数据
 * @param[out] len  返回 JPEG 长度
 * @return OPERATE_RET OPRT_OK 成功；OPRT_TIMEOUT 取帧超时；其它为打开/配置失败
 */
OPERATE_RET app_camera_snapshot(uint8_t **jpeg, uint32_t *len);

/**
 * @brief 释放 app_camera_snapshot 返回的内存
 */
void app_camera_free(uint8_t *jpeg);

/**
 * @brief 最近一次失败的可读原因（给日志 / MCP 回复用）
 */
const char *app_camera_last_error(void);

/**
 * @brief 当前使用的设备节点（如 "/dev/video12"）
 */
const char *app_camera_device(void);

#ifdef __cplusplus
}
#endif

#endif /* __APP_CAMERA_H__ */
