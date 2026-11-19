#!/bin/bash
# 等 ws2812 字符设备出现，最多 10s；无论如何 exit 0
i=0
while [ $i -lt 20 ] && [ ! -e /dev/ws2812-leds ]; do
    i=$((i + 1))
    sleep 0.5
done
exit 0
