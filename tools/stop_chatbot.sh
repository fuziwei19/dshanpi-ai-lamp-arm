#!/bin/sh
# stop_chatbot.sh — 清掉所有 your_chat_bot 残留实例
#
# 匹配必须用 /proc/<pid>/comm 而不是 readlink /proc/<pid>/exe：
# 二进制被覆盖后 exe 会带 " (deleted)" 后缀，用 exe 匹配会静默漏掉旧实例，
# 导致新实例抢不到 ALSA / 5055 端口。
#
# 两级杀法（2026-11-03 加）：
#   SIGTERM 先礼后兵 -> 等 10 秒 -> 还活着就 SIGKILL。
#   原来只发 SIGTERM、等 10 秒就 exit 0，一旦 app 没在 10 秒内退干净，
#   systemd 立刻拉起新实例，新实例开 /dev/snd/pcmC0D0p 得到 EBUSY：
#       Audio playback device 'plughw:0,0' not available: Device or resource busy
#       app_chat_bot_init failed
#   而 app_chat_bot_init 是在 ai_mcp_init() / app_feishu_init() 之前失败的，
#   于是 **MCP 工具和飞书全都没起来**，但进程还活着、AI 对话看似正常 —— 很唬人。

count_alive() {
    left=0
    for c in /proc/[0-9]*/comm; do
        name=$(cat "$c" 2>/dev/null) || continue
        case "$name" in your_chat_bot*) left=$((left + 1));; esac
    done
    echo "$left"
}

# ---- 第一级：SIGTERM ----
n=0
for c in /proc/[0-9]*/comm; do
    pid=${c#/proc/}; pid=${pid%/comm}
    name=$(cat "$c" 2>/dev/null) || continue
    case "$name" in
        your_chat_bot*)
            echo "  TERM $pid ($name)"
            kill "$pid" 2>/dev/null
            n=$((n + 1))
            ;;
    esac
done
[ "$n" = "0" ] && echo "  （没有实例在跑）"

# ---- 等最多 10 秒让它自己退 ----
i=0
while [ "$i" -lt 10 ]; do
    [ "$(count_alive)" = "0" ] && break
    sleep 1
    i=$((i + 1))
done

# ---- 第二级：还赖着就 SIGKILL ----
if [ "$(count_alive)" != "0" ]; then
    echo "  !! 10 秒没退干净，改用 SIGKILL"
    for c in /proc/[0-9]*/comm; do
        pid=${c#/proc/}; pid=${pid%/comm}
        name=$(cat "$c" 2>/dev/null) || continue
        case "$name" in
            your_chat_bot*)
                echo "  KILL $pid ($name)"
                kill -9 "$pid" 2>/dev/null
                ;;
        esac
    done
    sleep 2
fi

if [ "$(count_alive)" != "0" ]; then
    echo "  !! 仍有残留实例，ALSA 可能被占，新实例会起不来"
else
    echo "  已清干净"
fi
exit 0
