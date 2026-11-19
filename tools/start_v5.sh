#!/bin/sh
# start_v5.sh — 干净重启 your_chat_bot（带 情绪灯效 + AI 图片理解）
#   停进程用 /proc/<pid>/comm 匹配（值形如 your_chat_bot_1，不受 " (deleted)" 影响，
#   也不会误伤本脚本自身），然后启动一次并验证新功能是否就绪。
APP=/home/yuxuan/Documents/TuyaOpen/apps/tuya.ai/your_chat_bot
DIST=$APP/dist/your_chat_bot_1.0.1
LOG=/tmp/chatbot_v5.log

echo "=== 1) 停掉所有实例 ==="
n=0
for c in /proc/[0-9]*/comm; do
    pid=${c#/proc/}; pid=${pid%/comm}
    name=$(cat "$c" 2>/dev/null) || continue
    case "$name" in
        your_chat_bot*)
            echo "  kill $pid ($name)"
            kill "$pid" 2>/dev/null
            n=$((n + 1))
            ;;
    esac
done
[ "$n" = "0" ] && echo "  （没有实例在跑）"

i=0
while [ "$i" -lt 10 ]; do
    left=0
    for c in /proc/[0-9]*/comm; do
        name=$(cat "$c" 2>/dev/null) || continue
        case "$name" in your_chat_bot*) left=$((left + 1));; esac
    done
    [ "$left" = "0" ] && break
    sleep 1
    i=$((i + 1))
done

echo "=== 2) 复查资源 ==="
for c in /proc/[0-9]*/comm; do
    name=$(cat "$c" 2>/dev/null) || continue
    case "$name" in your_chat_bot*) echo "  !! 仍在跑: ${c#/proc/} $name";; esac
done
ss -lntp 2>/dev/null | grep -F 5055 || echo "  5055 空闲"
fuser /dev/snd/pcmC0D0p /dev/snd/pcmC0D0c >/dev/null 2>&1 && echo "  !! ALSA 仍被占用" || echo "  ALSA 空闲"

echo "=== 3) 启动（cd 到 app 目录，KV 是相对 cwd 的）==="
cd "$APP" || exit 1
: > "$LOG"
setsid nohup ./run_chatbot.sh >> "$LOG" 2>&1 < /dev/null &
sleep 35

echo "=== 4) 验证 ==="
printf '  Added tool 条数: %s\n' "$(grep -ac 'Added tool' "$LOG")"
grep -ao 'Added tool: device_[a-z_]*' "$LOG" | sort -u | sed 's/^/    /'
echo "  app_mcp 汇总:"
grep -ao 'app_mcp: tools registered.*' "$LOG" | sed 's/^/    /'
echo "  本机通道:"
grep -an 'local ctl' "$LOG" | sed 's/^/    /'
echo "  情绪灯效:"
grep -a 'emo_lamp' "$LOG" | sed 's/^/    /'
echo "  图片理解:"
grep -a 'vision:\|app_camera' "$LOG" | sed 's/^/    /'
echo "  socket:"
ls -la /run/tuya_chatbot.sock 2>&1 | sed 's/^/    /'
echo "  音频:"
grep -a 'ALSA capture device setup\|ALSA playback device setup\|not available' "$LOG" | sed 's/^/    /'
echo "  进程:"
for c in /proc/[0-9]*/comm; do
    pid=${c#/proc/}; pid=${pid%/comm}
    name=$(cat "$c" 2>/dev/null) || continue
    case "$name" in your_chat_bot*) echo "    PID $pid $name";; esac
done
