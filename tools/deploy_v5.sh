#!/bin/sh
# deploy_v5.sh — 同步新产物进 dist -> 校验新符号 -> 重启并验证
APP=/home/yuxuan/Documents/TuyaOpen/apps/tuya.ai/your_chat_bot
DIST=$APP/dist/your_chat_bot_1.0.1
BUILD=$APP/.build/bin

echo "=== 1) 同步产物进 dist ==="
cp -f "$BUILD/your_chat_bot_1.0.1.elf"     "$DIST/your_chat_bot_1.0.1.elf"
cp -f "$BUILD/your_chat_bot_QIO_1.0.1.bin" "$DIST/your_chat_bot_QIO_1.0.1.bin"
ls -l "$DIST"

echo "=== 2) 新符号校验（确认改动真进了二进制）==="
for s in app_emo_lamp_notify app_vision_request device_take_photo emo-lamp ai-camera; do
    printf '  %-22s -> %s 次\n' "$s" "$(grep -ac -- "$s" "$DIST/your_chat_bot_1.0.1.elf")"
done

echo "=== 3) 重启 ==="
sh /root/start_v5.sh
