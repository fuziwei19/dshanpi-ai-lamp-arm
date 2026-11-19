#!/bin/bash
# face_follow_precheck.sh — 人脸跟随启动前的快速自检（由 ExecStartPre 调用）
# 作用：① 日志轮转 ② 机械臂串口/摄像头存在性检查，失败则立刻给出明确原因，
#       避免"启动了但马上崩"却不知道为什么。
LOG=/var/log/xiaozhi-linux/face_follow.log

mkdir -p /var/log/xiaozhi-linux

# 日志超过 5MB 就轮转一份，避免无限增长
if [ -f "$LOG" ] && [ "$(stat -c%s "$LOG" 2>/dev/null || echo 0)" -gt 5000000 ]; then
    mv -f "$LOG" "$LOG.1"
fi

echo "================ precheck $(date -Is) ================"

# 机械臂串口是硬性依赖
if [ ! -e /dev/ttyACM0 ]; then
    echo "[precheck] 失败：找不到机械臂串口 /dev/ttyACM0（USB 线松了或机械臂没上电？）"
    exit 1
fi

# 摄像头（具体节点号交给跟随脚本自动探测，这里只确认至少有一个）
if ! ls /dev/video* >/dev/null 2>&1; then
    echo "[precheck] 失败：找不到任何摄像头节点 /dev/video*"
    exit 1
fi

echo "[precheck] ok  arm=/dev/ttyACM0  cams=$(ls /dev/video* 2>/dev/null | tr '\n' ' ')"
exit 0
