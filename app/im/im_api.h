#ifndef __IM_API_H__
#define __IM_API_H__

/*
 * IM 组件聚合头（your_chat_bot 版本：只保留 feishu 一条链路）。
 *
 * 上游完整组件见 https://github.com/tuya/TClaw/tree/master/IM
 * （telegram / discord / weixin / qqbot 未纳入本 app 的编译，源文件如需启用，
 *   把对应 .c 加进 im/CMakeLists.txt 并在下方补上 header 即可。）
 */

#include "im_platform.h"
#include "im_config.h"
#include "bus/message_bus.h"
#include "channels/feishu_bot.h"
#include "proxy/http_proxy.h"

#endif /* __IM_API_H__ */
