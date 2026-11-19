#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""led_client.py — 灯环客户端（复用常驻守护 ws2812d）

只做两件事：
  1. 薄封装 socket 协议（set / on / off / state）
  2. 提供几个「编排好的光效动作」：渐亮(fade_in) / 渐暗(fade_out) / 闪烁(flash) / 流水一圈(sweep)

注意：驱动在 close(fd) 时会自动关灯，所以真正的「持灯」是 ws2812d 守护进程，
本模块只发指令，进程退出不会影响灯的状态。
"""
import json
import socket
import time

SOCK_PATH = '/run/ws2812d.sock'


def send(req, timeout=4.0):
    """和守护进程通信，返回 {'ok': bool, ...}"""
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect(SOCK_PATH)
    except OSError as e:
        return {'ok': False, 'error': '灯效守护未运行(%s)：systemctl start ws2812d' % e}
    try:
        s.sendall((json.dumps(req, ensure_ascii=False) + '\n').encode('utf-8'))
        buf = b''
        while b'\n' not in buf:
            ch = s.recv(4096)
            if not ch:
                break
            buf += ch
        return json.loads(buf.decode('utf-8').strip())
    except Exception as e:
        return {'ok': False, 'error': str(e)}
    finally:
        s.close()


def set_light(on=True, color=None, brightness=None, effect=None, timeout=4.0):
    req = {'cmd': 'set', 'on': bool(on)}
    if color is not None:
        req['color'] = color
    if brightness is not None:
        req['brightness'] = int(brightness)
    if effect is not None:
        req['effect'] = effect
    return send(req, timeout=timeout)


def off():
    return send({'cmd': 'off'})


def state():
    return send({'cmd': 'state'})


# --------------------------------------------------------------------------
# 编排好的光效动作（每个都会阻塞到做完，供动作线程串行调用）
# --------------------------------------------------------------------------

def fade_in(color, target=55, steps=10, delay=0.05):
    """从 0 渐亮到 target%（迎宾用，柔和不突兀）"""
    for k in range(1, steps + 1):
        set_light(True, color=color, brightness=int(round(target * k / steps)))
        time.sleep(delay)
    return set_light(True, color=color, brightness=target)


def fade_out(color='#FFB45A', start=55, steps=10, delay=0.06):
    """渐暗到熄灭"""
    for k in range(steps - 1, -1, -1):
        set_light(True, color=color, brightness=int(round(start * k / steps)))
        time.sleep(delay)
    return off()


def flash(color, times=2, level=60, on_s=0.13, off_s=0.13):
    """闪 N 下（告别用）"""
    for _ in range(times):
        set_light(True, color=color, brightness=level)
        time.sleep(on_s)
        set_light(True, color=color, brightness=0)
        time.sleep(off_s)
    return set_light(True, color=color, brightness=level)


def sweep(color, hold_s=0.9, level=55):
    """流水跑一圈再回到常亮（"注意到你了"的仪式感）"""
    set_light(True, color=color, brightness=level, effect='chase')
    time.sleep(hold_s)
    return set_light(True, color=color, brightness=level, effect='static')


def idle(color, brightness):
    """回到"有人在场"的常态光"""
    return set_light(True, color=color, brightness=brightness, effect='static')
