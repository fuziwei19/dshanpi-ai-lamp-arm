#!/bin/bash
# greet_precheck.sh — 迎宾模式启动前的快速自检（由 ExecStartPre 调用）
# ① 日志轮转 ② 确认灯效守护 / 摄像头在位，失败立刻给明确原因
LOG=/var/log/xiaozhi-linux/vision_greet.log

mkdir -p /var/log/xiaozhi-linux

if [ -f "$LOG" ] && [ "$(stat -c%s "$LOG" 2>/dev/null || echo 0)" -gt 5000000 ]; then
    mv -f "$LOG" "$LOG.1"
fi

echo "================ precheck $(date -Is) ================"

if ! ls /dev/video* >/dev/null 2>&1; then
    echo "[precheck] 失败：找不到任何摄像头节点 /dev/video*"
    exit 1
fi

# 灯效守护：没有灯也能跑（只是没光效），所以只警告不阻断
if ! systemctl is-active --quiet ws2812d.service; then
    echo "[precheck] 警告：ws2812d 未运行，灯环将无效果（systemctl start ws2812d）"
fi

echo "[precheck] ok  cams=$(ls /dev/video* 2>/dev/null | tr '\n' ' ')"
exit 0
