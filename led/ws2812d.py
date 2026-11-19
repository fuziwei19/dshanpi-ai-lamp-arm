#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ws2812d — WS2812 灯环常驻守护进程（DshanPi-A1 / RK3576）

为什么必须是守护进程
--------------------
内核驱动 ws2812_drv.c 的 release 回调在 close(fd) 时会把所有灯珠清零：
    static int ws2812_release(...) { memset(dev->rgb_data, 0, dev->led_num*3); ... }
所以「开灯」不能靠一次性 CLI（写完就退出 = 立刻关灯），
必须有一个进程长期持有 /dev/ws2812-leds 的句柄。

设备写入协议（来自 ws2812_drv.c）
--------------------------------
    write(fd, buf, N*3)
    * 每颗灯 3 字节，顺序是 G, R, B（不是 RGB！）
    * 灯珠数量 N = 写入字节数 / 3（驱动会按长度重新分配缓冲）
    * SPI 侧：1 个 WS2812 bit 用 1 个 SPI 字节编码（1->0xFC, 0->0x30），
      6.67MHz 下正好 1.2us/bit，复位用 50 个 0 字节
    * 写入是异步工作队列，连续快写会被合并，所以高刷动画是安全的

对外接口
--------
Unix socket /run/ws2812d.sock，行分隔 JSON，一条命令一行、回一行 JSON：
    {"cmd":"on"}
    {"cmd":"off"}
    {"cmd":"toggle"}
    {"cmd":"set","color":"#FF8800","brightness":60,"effect":"static"}
    {"cmd":"state"}
    {"cmd":"quit"}          # 关灯并退出守护进程

用法：
    python3 ws2812d.py                  # 前台运行
    python3 ws2812d.py --selftest       # 硬件自检：红/绿/蓝/白/流水 依次演示
    python3 ws2812d.py --count 35       # 指定灯珠数量（默认 35）
"""
import argparse
import json
import math
import os
import signal
import socket
import socketserver
import sys
import threading
import time

DEV = '/dev/ws2812-leds'
SOCK_PATH = '/run/ws2812d.sock'
DEFAULT_COUNT = 24   # 机械臂上的环形 WS2812 实际灯珠数
FPS = 30.0
GAMMA = 1.8          # 亮度伽马，让低亮段调节更细腻

# ---- 颜色名表（中文/英文 -> RGB）----
COLOR_NAMES = {
    '红': (255, 0, 0), '红色': (255, 0, 0), 'red': (255, 0, 0),
    '绿': (0, 255, 0), '绿色': (0, 255, 0), 'green': (0, 255, 0),
    '蓝': (0, 0, 255), '蓝色': (0, 0, 255), 'blue': (0, 0, 255),
    '黄': (255, 200, 0), '黄色': (255, 200, 0), 'yellow': (255, 200, 0),
    '紫': (160, 0, 255), '紫色': (160, 0, 255), 'purple': (160, 0, 255),
    '青': (0, 255, 200), '青色': (0, 255, 200), 'cyan': (0, 255, 200),
    '橙': (255, 100, 0), '橙色': (255, 100, 0), 'orange': (255, 100, 0),
    '粉': (255, 60, 120), '粉色': (255, 60, 120), 'pink': (255, 60, 120),
    '白': (255, 255, 255), '白色': (255, 255, 255), 'white': (255, 255, 255),
    '暖白': (255, 180, 90), '暖光': (255, 180, 90), 'warm': (255, 180, 90),
    '冷白': (200, 230, 255), 'cool': (200, 230, 255),
    '彩虹': None,   # 特殊：交给 effect=rainbow
    '氛围': (255, 120, 200),
}

EFFECTS = ('static', 'breathing', 'rainbow', 'chase', 'blink')
EFFECT_ALIAS = {
    '常亮': 'static', '静态': 'static', '不动': 'static', 'solid': 'static',
    '呼吸': 'breathing', '呼吸灯': 'breathing', 'breath': 'breathing',
    '彩虹': 'rainbow', '流光': 'rainbow', 'rainbow': 'rainbow',
    '流水': 'chase', '跑马灯': 'chase', '追逐': 'chase', 'chase': 'chase',
    '闪烁': 'blink', '闪': 'blink', 'blink': 'blink',
}


def parse_color(s):
    """把颜色字符串解析成 (r,g,b)；识别不出返回 None"""
    if s is None:
        return None
    if isinstance(s, (list, tuple)) and len(s) == 3:
        return tuple(int(max(0, min(255, v))) for v in s)
    t = str(s).strip()
    low = t.lower()
    if low in COLOR_NAMES and COLOR_NAMES[low] is not None:
        return COLOR_NAMES[low]
    if t in COLOR_NAMES and COLOR_NAMES[t] is not None:
        return COLOR_NAMES[t]
    if low.startswith('#'):
        low = low[1:]
    if len(low) == 6:
        try:
            return (int(low[0:2], 16), int(low[2:4], 16), int(low[4:6], 16))
        except ValueError:
            return None
    if len(low) == 3:
        try:
            return tuple(int(c * 2, 16) for c in low)
        except ValueError:
            return None
    return None


def parse_effect(s):
    if s is None:
        return None
    t = str(s).strip().lower()
    if t in EFFECTS:
        return t
    return EFFECT_ALIAS.get(str(s).strip(), EFFECT_ALIAS.get(t))


def scale(rgb, brightness, gamma=GAMMA):
    """按亮度(0~100)缩放 RGB，带伽马；非零颜色保证不会缩成全黑"""
    k = max(0.0, min(100.0, float(brightness))) / 100.0
    k = k ** gamma
    out = [min(255, int(v * k + 0.5)) for v in rgb]
    if max(rgb) > 0 and max(out) == 0:
        # 低亮度下避免因量化直接变黑
        i = rgb.index(max(rgb))
        out[i] = 1
    return tuple(out)


def hsv_to_rgb(h, s, v):
    i = int(h * 6.0) % 6
    f = h * 6.0 - int(h * 6.0)
    p, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
    r, g, b = [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][i]
    return int(r * 255), int(g * 255), int(b * 255)


class State(object):
    def __init__(self, count, color=(255, 180, 90), brightness=60):
        # 注意：用 RLock。update() 会在持锁状态下取快照，普通 Lock 会自死锁
        self.lock = threading.RLock()
        self.count = count
        self.on = False
        self.color = tuple(color)
        self.brightness = brightness
        self.effect = 'static'
        self.version = 0        # 每次状态变化 +1，用于跳过重复渲染

    def _snap(self):
        """必须在已持锁的情况下调用"""
        return dict(on=self.on, color=self.color,
                    brightness=self.brightness, effect=self.effect,
                    count=self.count, version=self.version)

    def snapshot(self):
        with self.lock:
            return self._snap()

    def update(self, d):
        with self.lock:
            changed = False
            if 'on' in d and d['on'] is not None:
                self.on = bool(d['on']); changed = True
            if 'toggle' in d and d['toggle']:
                self.on = not self.on; changed = True
            c = parse_color(d.get('color'))
            if c is not None:
                self.color = c; changed = True
            if d.get('brightness') is not None:
                self.brightness = int(max(0, min(100, int(d['brightness'])))); changed = True
            e = parse_effect(d.get('effect'))
            if e is not None:
                self.effect = e; changed = True
            if changed:
                self.version += 1
            return self._snap()


class Strip(object):
    """持有 /dev/ws2812-leds 的句柄；断线自动重开"""

    def __init__(self, count, path=DEV):
        self.path = path
        self.count = count
        self.fd = None
        self.fail = 0

    def ensure(self):
        if self.fd is not None:
            return True
        try:
            self.fd = os.open(self.path, os.O_WRONLY)
            self.fail = 0
            # 首帧全灭，避免上电残留
            self.write([(0, 0, 0)] * self.count)
            log('[strip] 已打开 %s (LED=%d)' % (self.path, self.count))
            return True
        except OSError as e:
            self.fd = None
            if self.fail < 3 or self.fail % 20 == 0:
                log('[strip] 打开 %s 失败: %s（等待驱动加载…）' % (self.path, e))
            self.fail += 1
            return False

    def write(self, pixels):
        if self.fd is None and not self.ensure():
            return False
        # 驱动要求 G, R, B 顺序
        buf = bytearray()
        for (r, g, b) in pixels:
            buf.append(g & 0xFF)
            buf.append(r & 0xFF)
            buf.append(b & 0xFF)
        try:
            os.write(self.fd, bytes(buf))
            return True
        except OSError as e:
            log('[strip] 写入失败: %s（重开设备）' % e)
            try:
                os.close(self.fd)
            except OSError:
                pass
            self.fd = None
            return False

    def close(self):
        if self.fd is not None:
            try:
                os.write(self.fd, bytes(3 * self.count))   # 全灭
            except OSError:
                pass
            try:
                os.close(self.fd)      # 驱动的 release 也会清零
            except OSError:
                pass
            self.fd = None


def frame(st, strip, t0):
    """按当前状态生成一帧像素"""
    n = st.count
    t = time.time() - t0
    if not st.on or st.brightness <= 0:
        return [(0, 0, 0)] * n
    if st.effect == 'rainbow':
        return [hsv_to_rgb((i / float(n) + t / 6.0) % 1.0, 1.0,
                           st.brightness / 100.0) for i in range(n)]
    if st.effect == 'chase':
        head = int(t * n / 2.0) % n
        tail = max(2, n // 4)
        out = []
        for i in range(n):
            d = (i - head) % n
            k = max(0.0, 1.0 - d / float(tail))
            out.append(scale(st.color, st.brightness * (0.25 + 0.75 * k)))
        return out
    if st.effect == 'blink':
        return [scale(st.color, st.brightness) if (int(t * 2) % 2 == 0)
                else (0, 0, 0)] * n
    if st.effect == 'breathing':
        k = 0.15 + 0.85 * (0.5 + 0.5 * math.sin(t * 2 * math.pi / 3.0))
        return [scale(st.color, st.brightness * k)] * n
    return [scale(st.color, st.brightness)] * n


def render_loop(st, strip, stop_evt):
    t0 = time.time()
    last_version = -1
    last_effect = None
    dt = 1.0 / FPS
    while not stop_evt.is_set():
        s = st.snapshot()
        animated = s['effect'] != 'static'
        need = (s['version'] != last_version) or animated or (s['effect'] != last_effect)
        if need:
            strip.write(frame(st, strip, t0))
            last_version = s['version']
            last_effect = s['effect']
        # 灯灭且无动画时降频，别空转烧 CPU
        time.sleep(dt if (animated and s['on']) else 0.1)


LOG_T0 = time.time()


def log(msg):
    sys.stdout.write('[%7.3f] %s\n' % (time.time() - LOG_T0, msg))
    sys.stdout.flush()


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        st = self.server.state
        for line in self.rfile:
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line.decode('utf-8'))
            except Exception as e:
                self.reply({'ok': False, 'error': 'bad json: %s' % e})
                continue
            cmd = str(req.get('cmd', 'set')).lower()
            if cmd == 'state':
                s = st.snapshot()
                s['color_hex'] = '#%02X%02X%02X' % s['color']
                s['color'] = list(s['color'])
                self.reply({'ok': True, 'state': s})
            elif cmd == 'quit':
                self.reply({'ok': True, 'msg': 'bye'})
                self.server.stop_evt.set()
                return
            else:
                if cmd == 'off':
                    req.update(on=False)
                elif cmd == 'on':
                    req.update(on=True)
                elif cmd == 'toggle':
                    req.update(toggle=True)
                s = st.update(req)
                s['color_hex'] = '#%02X%02X%02X' % s['color']
                s['color'] = list(s['color'])
                self.reply({'ok': True, 'state': s})

    def reply(self, obj):
        try:
            self.wfile.write((json.dumps(obj, ensure_ascii=False) + '\n').encode('utf-8'))
            self.wfile.flush()
        except Exception:
            pass


class Server(socketserver.ThreadingUnixStreamServer):
    allow_reuse_address = True
    daemon_threads = True


def selftest(strip, count):
    """硬件自检：依次红/绿/蓝/白/流水，确认接线与灯珠数正确"""
    log('自检开始：红 -> 绿 -> 蓝 -> 白 -> 流水（每步 1.2s）')
    for name, rgb in (('红', (255, 0, 0)), ('绿', (0, 255, 0)),
                      ('蓝', (0, 0, 255)), ('白', (255, 255, 255))):
        log('  %s' % name)
        strip.write([rgb] * count)
        time.sleep(1.2)
    log('  流水')
    for k in range(count * 2):
        px = [(0, 0, 0)] * count
        px[k % count] = (180, 0, 180)
        strip.write(px)
        time.sleep(0.05)
    strip.write([(0, 0, 0)] * count)
    log('自检结束（如果颜色不对，检查 GRB 顺序与灯珠数）')


def main():
    ap = argparse.ArgumentParser(description='WS2812 灯环守护进程')
    ap.add_argument('--count', type=int, default=DEFAULT_COUNT, help='灯珠数量(默认35)')
    ap.add_argument('--sock', default=SOCK_PATH, help='Unix socket 路径')
    ap.add_argument('--color', default='#FFB45A', help='初始颜色')
    ap.add_argument('--brightness', type=int, default=60, help='初始亮度 0~100')
    ap.add_argument('--on', action='store_true', help='启动即点亮')
    ap.add_argument('--selftest', action='store_true', help='只做硬件自检后退出')
    opt = ap.parse_args()

    strip = Strip(opt.count)
    stop_evt = threading.Event()

    if opt.selftest:
        if not strip.ensure():
            log('无法打开 %s：驱动没加载？' % DEV)
            return 1
        try:
            selftest(strip, opt.count)
        finally:
            strip.close()
        return 0

    st = State(opt.count, parse_color(opt.color) or (255, 180, 90), opt.brightness)
    if opt.on:
        st.update({'on': True})

    # 信号处理：退出前务必关灯
    def _sig(signum, frame_):
        log('收到信号 %d，关灯退出' % signum)
        stop_evt.set()
    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    if os.path.exists(opt.sock):
        try:
            os.unlink(opt.sock)
        except OSError:
            pass
    srv = Server(opt.sock, Handler)
    srv.state = st
    srv.stop_evt = stop_evt
    os.chmod(opt.sock, 0o666)
    threading.Thread(target=srv.serve_forever, kwargs={'poll_interval': 0.2},
                     daemon=True).start()
    log('守护进程就绪：socket=%s  LED=%d  亮度=%d%%  等待命令…'
        % (opt.sock, opt.count, opt.brightness))

    rt = threading.Thread(target=render_loop, args=(st, strip, stop_evt), daemon=True)
    rt.start()

    try:
        while not stop_evt.is_set():
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        stop_evt.set()
        try:
            srv.shutdown()
        except Exception:
            pass
        strip.close()
        try:
            os.unlink(opt.sock)
        except OSError:
            pass
        log('已退出，灯已关闭')
    return 0


if __name__ == '__main__':
    sys.exit(main())
