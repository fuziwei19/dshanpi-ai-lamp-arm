#ifndef __IM_SECRETS_H__
#define __IM_SECRETS_H__

/*
 * 编译期密钥（可选）。
 *
 * 优先级：运行时 /etc/feishu.json  >  本文件  >  NVS(KV)
 * （其实 /etc/feishu.json 与 NVS 最终都落到同一组静态变量上，
 *   具体顺序见 src/app_feishu.c 的 __load_conf() / feishu_bot_init()。）
 *
 * 多数情况下**不需要**改这个文件：板子上写一个 /etc/feishu.json 就够了：
 *
 *   echo '{"app_id":"cli_xxxxxxxx","app_secret":"yyyyyyyy"}' > /etc/feishu.json
 *   chmod 600 /etc/feishu.json
 *   # 然后重启 app
 *
 * 如果你更想在固件里固化（比如批量出货），把下面两行的值填上即可。
 */

/* 飞书自建应用凭证（开放平台 -> 开发者后台 -> 你的应用 -> 凭证与基础信息） */
#define IM_SECRET_FS_APP_ID        ""
#define IM_SECRET_FS_APP_SECRET    ""

/*
 * 白名单：只接受这些发送者的消息（open_id / user_id / union_id，逗号分隔）。
 * 留空 = 不限制（任何能跟机器人对话的人都行）。
 * 首次不知道自己的 open_id 时，留空跑一次，日志里会打印
 *   feishu: inbound chat=<open_id | oc_xxx>
 * 把那个 open_id 填进来即可。
 */
#define IM_SECRET_FS_ALLOW_FROM    ""

/* 本组件只集成 feishu 一条链路 */
#define IM_SECRET_CHANNEL_MODE     "feishu"

/* 代理（可选，一般不需要） */
#define IM_SECRET_PROXY_HOST       ""
#define IM_SECRET_PROXY_PORT       ""
#define IM_SECRET_PROXY_TYPE       "http"

#endif /* __IM_SECRETS_H__ */
