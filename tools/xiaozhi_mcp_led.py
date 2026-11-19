#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""xiaozhi_mcp_led.py — 给小智（xiaozhi）AI 用的 MCP 工具集

包含两组工具：
  1) WS2812 灯环：开/关、颜色、亮度、灯效、状态查询
  2) 机械臂人脸跟随：启动 / 停止 / 查询状态（由 face-follow.service 承载）

工作原理
--------
小智官方支持「MCP 接入点」：本地跑一个 MCP server，通过 WebSocket 连到小智云端的
接入点（在小智控制台 xiaozhi.me 里获取）。之后云端的 LLM 就能在对话中调用这里的工具，
实现"说一句话 -> 灯变色 / 机械臂开始跟着你转"。

本文件只提供工具（stdio 传输），需要用 mcp_pipe.py 桥接到小智的 MCP 接入点：
    export MCP_ENDPOINT='wss://...（小智控制台里的接入点地址）'
    python3 mcp_pipe.py xiaozhi_mcp_led.py

灯效真正干活的是常驻守护进程 ws2812d（见 ws2812d.py）：驱动在 close(fd) 时会自动
关灯，所以必须由守护进程长期持有 /dev/ws2812-leds。

人脸跟随真正干活的是 systemd 单元 face-follow.service（跑
yolov8pose_head_follow_so101.py）。这里只做 start/stop：
  * 用 SIGINT 停止（单元里 KillSignal=SIGINT）恰好等价于终端 Ctrl+C，
    跟随脚本会平滑回到 PARK 收纳点并释放舵机力矩；
  * 单元不是 enable 的，开机不会自动动机械臂。

注意：MCP 的 stdout 被协议占用，调试只能用 logger，不能 print。
"""
import json
import logging
import os
import shutil
import socket
import subprocess
import time

from mcp.server.fastmcp import FastMCP

SOCK_PATH = os.environ.get('WS2812_SOCK', '/run/ws2812d.sock')

# ---- 人脸跟随相关路径 ----
FOLLOW_UNIT = 'face-follow.service'
FOLLOW_CFG = '/etc/face-follow.json'
FOLLOW_LOG = '/var/log/xiaozhi-linux/face_follow.log'

# ---- 迎宾模式相关路径 ----
GREET_UNIT = 'vision-greet.service'
GREET_CFG = '/etc/vision-greet.json'
GREET_LOG = '/var/log/xiaozhi-linux/vision_greet.log'

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('xiaozhi_led')

mcp = FastMCP('台灯与机械臂控制')


def _send(req, timeout=4.0):
    """和 ws2812d 守护进程通信，返回 (ok, 状态字典或错误信息)"""
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect(SOCK_PATH)
    except OSError as e:
        log.error('连接 %s 失败: %s', SOCK_PATH, e)
        return False, ('灯效守护进程没在运行（%s）。请先启动：'
                       'systemctl start ws2812d' % e)
    try:
        s.sendall((json.dumps(req, ensure_ascii=False) + '\n').encode('utf-8'))
        buf = b''
        while b'\n' not in buf:
            chunk = s.recv(4096)
            if not chunk:
                break
            buf += chunk
        resp = json.loads(buf.decode('utf-8').strip())
    except Exception as e:
        log.error('与守护进程通信失败: %s', e)
        return False, '与灯效守护进程通信失败: %s' % e
    finally:
        s.close()
    if not resp.get('ok'):
        return False, resp.get('error', '未知错误')
    return True, resp.get('state', {})


def _brief(state):
    if not isinstance(state, dict):
        return str(state)
    return ('现在灯是%s，颜色%s，亮度%d%%，灯效%s。'
            % ('开着的' if state.get('on') else '关着的',
               state.get('color_hex', state.get('color')),
               state.get('brightness', 0),
               state.get('effect', 'static')))


def _ok(state):
    return json.dumps({'success': True, 'state': state,
                       'message': _brief(state)}, ensure_ascii=False)


def _err(msg):
    return json.dumps({'success': False, 'error': str(msg)}, ensure_ascii=False)


@mcp.tool()
def set_light_power(on: bool) -> str:
    """打开或关闭台灯（WS2812 灯环）。

    当用户说「开灯」「关灯」「把灯打开」「关一下灯」「打开台灯」「关闭灯环」时调用。

    参数 on: true = 开灯, false = 关灯
    """
    ok, r = _send({'cmd': 'set', 'on': bool(on)})
    return _ok(r) if ok else _err(r)


@mcp.tool()
def set_light_color(color: str) -> str:
    """设置台灯（WS2812 灯环）的颜色，灯会自动点亮。

    当用户说「把灯调成红色」「换成蓝色」「来个暖白的光」「设成紫色」时调用。

    参数 color: 颜色名称或十六进制色值。支持中文名：
      红/绿/蓝/黄/紫/青/橙/粉/白/暖白/冷白，
      也可以传 '#FF8800' 这样的十六进制值。
    """
    ok, r = _send({'cmd': 'set', 'color': color, 'on': True})
    return _ok(r) if ok else _err(r)


@mcp.tool()
def set_light_brightness(percent: int) -> str:
    """调节台灯（WS2812 灯环）的亮度。

    当用户说「调亮一点」「调暗一点」「亮度调到 50%」「太亮了」时调用。
    用户说「调亮一点」这类模糊说法时，可以在当前亮度基础上加/减 20 左右。

    参数 percent: 亮度百分比，0~100 的整数（0 等于关灯，100 最亮）。
    """
    try:
        p = int(percent)
    except Exception:
        return _err('亮度必须是 0~100 的整数')
    p = max(0, min(100, p))
    ok, r = _send({'cmd': 'set', 'brightness': p,
                   'on': True if p > 0 else False})
    return _ok(r) if ok else _err(r)


@mcp.tool()
def set_light_effect(effect: str) -> str:
    """设置台灯（WS2812 灯环）的动态灯效。

    当用户说「灯呼吸起来」「来个彩虹效果」「跑马灯」「让它闪一闪」时调用。

    参数 effect: 灯效名称。支持：
      常亮(静态)、呼吸、彩虹、流水(跑马灯)、闪烁
    """
    ok, r = _send({'cmd': 'set', 'effect': effect, 'on': True})
    return _ok(r) if ok else _err(r)


@mcp.tool()
def control_light(on: bool = True, color: str = '',
                  brightness: int = -1, effect: str = '') -> str:
    """一次性控制台灯（WS2812 灯环）的开关、颜色、亮度和灯效。

    当用户一句话里同时说了多项要求时用这个工具，例如
    「把灯调成蓝色的，亮度 30%」「开灯，暖白，亮一点」。

    参数:
      on: true 开灯 / false 关灯
      color: 颜色名（红/绿/蓝/黄/紫/青/橙/粉/白/暖白/冷白）或 '#RRGGBB'，留空表示不改
      brightness: 亮度 0~100，-1 表示不改
      effect: 灯效（常亮/呼吸/彩虹/流水/闪烁），留空表示不改
    """
    req = {'cmd': 'set', 'on': bool(on)}
    if color:
        req['color'] = color
    if brightness is not None and int(brightness) >= 0:
        req['brightness'] = max(0, min(100, int(brightness)))
    if effect:
        req['effect'] = effect
    ok, r = _send(req)
    return _ok(r) if ok else _err(r)


@mcp.tool()
def get_light_state() -> str:
    """查询台灯（WS2812 灯环）当前的状态：开关、颜色、亮度、灯效。

    当用户问「灯现在是什么颜色」「灯开着吗」「亮度多少」时调用。
    """
    ok, r = _send({'cmd': 'state'})
    return _ok(r) if ok else _err(r)


# ---------------------------------------------------------------------------
# 机械臂人脸跟随（真正干活的是 systemd 单元 face-follow.service）
# ---------------------------------------------------------------------------

def _systemctl(args, timeout=60):
    """调用 systemctl，返回 (rc, stdout, stderr)"""
    exe = shutil.which('systemctl') or '/usr/bin/systemctl'
    p = subprocess.run([exe] + list(args), capture_output=True,
                       text=True, timeout=timeout)
    return p.returncode, (p.stdout or '').strip(), (p.stderr or '').strip()


def _unit_state(unit):
    """systemd is-active 状态：active/inactive/failed/activating/deactivating"""
    try:
        _, out, _ = _systemctl(['is-active', unit], timeout=10)
    except Exception as e:
        log.error('查询 %s 状态失败: %s', unit, e)
        return 'unknown'
    return out or 'unknown'


def _unit_pid(unit):
    try:
        _, out, _ = _systemctl(['show', '-p', 'MainPID', '--value', unit],
                               timeout=10)
        return out or '0'
    except Exception:
        return '0'


def _tail_file(path, n=12):
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            lines = [l.rstrip('\n') for l in f.readlines()[-n:]]
        return [l for l in lines if l.strip()]
    except FileNotFoundError:
        return ['(还没有日志文件 %s)' % path]
    except Exception as e:
        return ['(读取日志失败: %s)' % e]


def _file_size(path):
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def _read_since(path, offset, limit=20000):
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            f.seek(offset)
            return f.read(limit)
    except Exception:
        return ''


# --- 下面几个是「人脸跟随」的薄封装（历史代码沿用）---
def _follow_state():
    return _unit_state(FOLLOW_UNIT)


def _follow_pid():
    return _unit_pid(FOLLOW_UNIT)


def _tail_log(n=12):
    return _tail_file(FOLLOW_LOG, n)


def _log_offset():
    return _file_size(FOLLOW_LOG)


def _log_since(offset, limit=20000):
    return _read_since(FOLLOW_LOG, offset, limit)


def _fjson(payload):
    return json.dumps(payload, ensure_ascii=False)


_RUNNING_STATES = ('active', 'activating', 'deactivating')


@mcp.tool()
def start_face_follow(aim: str = 'head', band: float = 0.0,
                      dry_run: bool = False) -> str:
    """启动机械臂的「人脸跟随」：摄像头找到人脸后，机械臂自动转动对准你。

    当用户说「启动人脸跟随」「开始跟随我」「让机械臂跟着我」「打开追踪」
    「机械臂转过来对着我」「跟着我的脸」时调用。
    用户说「停止跟随」「别跟着我了」「关掉追踪」时改用 stop_face_follow。

    重要：调用成功后机械臂会立即开始运动（先平滑摆一下再进入跟随），
    请在回复里提醒用户注意周围安全、别把手放在机械臂活动范围内。

    参数:
      aim: 'head' = 对准人脸（默认）/ 'body' = 对准人体框中心
      band: 允许带，0.05~0.6，越大越"懒得动"、越稳，默认 0.25；传 0 表示用默认值
      dry_run: true = 只跑视觉识别、不驱动机械臂（演示或自检用），默认 false
    """
    if _follow_state() in _RUNNING_STATES:
        return _fjson({'success': True, 'already_running': True,
                       'message': '人脸跟随已经在运行了，机械臂正在跟着画面里的人脸转，'
                                  '不用重复启动。'})

    aim_v = 'body' if str(aim).strip().lower() == 'body' else 'head'
    try:
        band_v = float(band)
    except (TypeError, ValueError):
        band_v = 0.0
    if not (band_v > 0):
        band_v = 0.25
    band_v = max(0.05, min(0.6, band_v))

    cfg = {'aim': aim_v, 'band': round(band_v, 2),
           'cam_id': 12, 'dry_run': bool(dry_run)}
    try:
        with open(FOLLOW_CFG, 'w', encoding='utf-8') as f:
            json.dump(cfg, f, ensure_ascii=False)
    except Exception as e:
        log.error('写入 %s 失败: %s', FOLLOW_CFG, e)
        return _err('写入跟随参数失败：%s' % e)

    off = _log_offset()                   # 只关心本次启动新产生的日志
    try:
        rc, out, err = _systemctl(['start', FOLLOW_UNIT], timeout=60)
    except subprocess.TimeoutExpired:
        return _err('启动人脸跟随超时（60 秒），请稍后用状态查询确认。')
    except Exception as e:
        return _err('启动人脸跟随时出错：%s' % e)

    time.sleep(4.0)                       # 等 precheck + 解释器启动
    state = _follow_state()
    newlog = _log_since(off)

    # 进程虽然起来了但已经抛异常（串口被占、模型缺失…）→ 立刻报错，别假装成功
    if state in _RUNNING_STATES and 'Traceback' in newlog:
        log.error('启动 %s 后立即报错', FOLLOW_UNIT)
        return _err('人脸跟随启动后立即报错：\n%s' % newlog[-1500:])

    if state in _RUNNING_STATES:
        if dry_run:
            extra = '这是演示模式，机械臂不会动，只做视觉识别。'
        else:
            extra = ('机械臂正在初始化（加载模型约需 10~25 秒，之后开始跟着你转）。'
                     '请注意安全，不要把手放在机械臂活动范围内。')
        return _fjson({'success': True, 'state': state, 'pid': _follow_pid(),
                       'params': cfg,
                       'message': '人脸跟随已启动。%s想让它停下就说「停止跟随」。' % extra})

    detail = err or out or ''
    tail = '\n'.join(_tail_log(10))
    log.error('启动 %s 失败 state=%s rc=%s err=%s', FOLLOW_UNIT, state, rc, detail)
    return _err('人脸跟随启动失败（服务状态 %s）%s\n最近日志：\n%s'
                % (state, ('：' + detail) if detail else '', tail))


@mcp.tool()
def stop_face_follow() -> str:
    """停止机械臂的「人脸跟随」：机械臂会平滑回到收纳位置并释放力矩（可用手自由摆动）。

    当用户说「停止跟随」「别跟着我了」「关掉追踪」「停下来」「机械臂回原位」
    「结束跟随」时调用。
    """
    state = _follow_state()
    if state not in _RUNNING_STATES:
        return _fjson({'success': True, 'already_stopped': True,
                       'message': '人脸跟随当前没有在运行，机械臂处于待机状态。'})

    try:
        rc, out, err = _systemctl(['stop', FOLLOW_UNIT], timeout=60)
    except subprocess.TimeoutExpired:
        return _err('停止人脸跟随超时（60 秒），机械臂可能仍在运动，请检查。')
    except Exception as e:
        return _err('停止人脸跟随时出错：%s' % e)

    time.sleep(0.5)
    state = _follow_state()
    tail = _tail_log(15)
    if state in _RUNNING_STATES:
        return _err('停止失败，服务仍是 %s 状态。\n最近日志：\n%s'
                    % (state, '\n'.join(tail)))
    released = any('力矩已释放' in l for l in tail)
    if released:
        msg = '已停止人脸跟随，机械臂已平滑回到收纳位并释放力矩（现在可以用手自由摆动）。'
    else:
        msg = ('已停止人脸跟随。没读到归位日志——如果刚才跑的是演示模式（dry_run）'
               '则属正常；否则请确认机械臂是否已回到收纳位。')
    return _fjson({'success': True, 'state': state,
                   'torque_released': released,
                   'message': msg})


@mcp.tool()
def get_face_follow_status() -> str:
    """查询机械臂「人脸跟随」当前状态（在不在跟随、跑的是哪种模式）。

    当用户问「机械臂在跟着我吗」「跟随开着吗」「追踪状态如何」「人脸跟随还开着吗」时调用。
    """
    state = _follow_state()
    if state in _RUNNING_STATES:
        params = ''
        for l in reversed(_tail_log(120)):
            if l.startswith('[launch]'):
                params = l
                break
        pid = _follow_pid()
        return _fjson({'success': True, 'running': True, 'state': state,
                       'pid': pid, 'launch': params,
                       'message': '人脸跟随正在运行中（服务状态 %s，进程号 %s）。'
                                  '想让它停下就说「停止跟随」。' % (state, pid)})
    return _fjson({'success': True, 'running': False, 'state': state,
                   'message': '人脸跟随当前没有在运行（服务状态 %s）。'
                              '想启动就说「启动人脸跟随」。' % state})


# ---------------------------------------------------------------------------
# 迎宾模式（真正干活的是 systemd 单元 vision-greet.service）
# ---------------------------------------------------------------------------

def _greet_default_cfg(dry_run=False):
    return {'cam_id': 12, 'dry_run': bool(dry_run), 'calib': False,
            'absent_sec': 8.0, 'greet_cooldown': 25.0, 'wave_cooldown': 12.0,
            'idle_bright': 35, 'nod_amp': 13.0, 'wave_deg': 20.0,
            'debug_every': 0.0}


@mcp.tool()
def start_greeting(dry_run: bool = False) -> str:
    """启动「迎宾模式」：摄像头看到人来就打招呼，挥手就告别，人走了自动省电。

    当用户说「开始迎宾」「打开迎宾模式」「开启迎宾」「有人来了打招呼」
    「进入迎宾模式」时调用；用户说「关闭迎宾」「停止迎宾」时改用 stop_greeting。

    行为：
      · 人出现并停留一会 → 灯环渐亮 + 流水一圈 + 机械臂抬头点头
      · 举手左右摆动     → 灯环闪两下 + 机械臂招手
      · 人离开一段时间   → 灯环熄灭 + 机械臂回收纳位并释放力矩

    重要：启动后机械臂会先回 Home 姿态再做动作，请在回复里提醒用户注意安全、
    别把手放在机械臂活动范围内。

    参数:
      dry_run: true = 只跑视觉和灯效、机械臂完全不动（演示/自检用），默认 false

    注意：迎宾与「人脸跟随」互斥（都要独占机械臂和摄像头），启动迎宾会自动停掉跟随。
    """
    if _unit_state(GREET_UNIT) in _RUNNING_STATES:
        return _fjson({'success': True, 'already_running': True,
                       'message': '迎宾模式已经在运行了，不用重复启动。'})

    stopped_follow = False
    if _unit_state(FOLLOW_UNIT) in _RUNNING_STATES:
        try:
            _systemctl(['stop', FOLLOW_UNIT], timeout=60)
            stopped_follow = True
        except Exception as e:
            log.warning('停止人脸跟随失败: %s', e)

    cfg = _greet_default_cfg(dry_run)
    try:
        with open(GREET_CFG, 'w', encoding='utf-8') as f:
            json.dump(cfg, f, ensure_ascii=False)
    except Exception as e:
        log.error('写入 %s 失败: %s', GREET_CFG, e)
        return _err('写入迎宾参数失败：%s' % e)

    off = _file_size(GREET_LOG)
    try:
        rc, out, err = _systemctl(['start', GREET_UNIT], timeout=60)
    except subprocess.TimeoutExpired:
        return _err('启动迎宾模式超时（60 秒），请稍后用状态查询确认。')
    except Exception as e:
        return _err('启动迎宾模式出错：%s' % e)

    time.sleep(4.0)
    state = _unit_state(GREET_UNIT)
    newlog = _read_since(GREET_LOG, off)
    if state in _RUNNING_STATES and 'Traceback' in newlog:
        log.error('启动 %s 后立即报错', GREET_UNIT)
        return _err('迎宾模式启动后立即报错：\n%s' % newlog[-1500:])

    if state in _RUNNING_STATES:
        tip = ('这是演示模式，机械臂不会动。' if dry_run
               else '机械臂正在初始化（约 15~25 秒），之后先回 Home 姿态待机。')
        extra = '（已先停止人脸跟随）' if stopped_follow else ''
        return _fjson({'success': True, 'state': state,
                       'pid': _unit_pid(GREET_UNIT), 'params': cfg,
                       'stopped_face_follow': stopped_follow,
                       'message': '迎宾模式已启动%s。%s'
                                  '现在走到摄像头前会打招呼，举手摆动会告别，'
                                  '人走后会自动省电。想关掉就说「关闭迎宾」。'
                                  % (extra, tip)})

    detail = err or out or ''
    tail = '\n'.join(_tail_file(GREET_LOG, 10))
    log.error('启动 %s 失败 state=%s rc=%s err=%s', GREET_UNIT, state, rc, detail)
    return _err('迎宾模式启动失败（服务状态 %s）%s\n最近日志：\n%s'
                % (state, ('：' + detail) if detail else '', tail))


@mcp.tool()
def stop_greeting() -> str:
    """停止「迎宾模式」：机械臂会平滑回到收纳位并释放力矩。

    当用户说「关闭迎宾」「停止迎宾」「别打招呼了」「退出迎宾模式」时调用。
    """
    state = _unit_state(GREET_UNIT)
    if state not in _RUNNING_STATES:
        return _fjson({'success': True, 'already_stopped': True,
                       'message': '迎宾模式当前没有在运行。'})

    try:
        rc, out, err = _systemctl(['stop', GREET_UNIT], timeout=60)
    except subprocess.TimeoutExpired:
        return _err('停止迎宾模式超时（60 秒），请检查机械臂状态。')
    except Exception as e:
        return _err('停止迎宾模式出错：%s' % e)

    time.sleep(0.5)
    state = _unit_state(GREET_UNIT)
    tail = _tail_file(GREET_LOG, 15)
    if state in _RUNNING_STATES:
        return _err('停止失败，服务仍是 %s 状态。\n最近日志：\n%s'
                    % (state, '\n'.join(tail)))
    released = any('力矩已释放' in l for l in tail)
    return _fjson({'success': True, 'state': state, 'torque_released': released,
                   'message': '已关闭迎宾模式，机械臂平滑回到收纳位'
                              + ('并释放力矩。' if released else '。')})


@mcp.tool()
def get_greeting_status() -> str:
    """查询「迎宾模式」当前状态（在不在运行、最近打了几次招呼）。

    当用户问「迎宾模式开着吗」「迎宾状态怎么样」「有人来打招呼了吗」时调用。
    """
    state = _unit_state(GREET_UNIT)
    if state in _RUNNING_STATES:
        tail = _tail_file(GREET_LOG, 300)
        params = ''
        for l in reversed(tail):
            if l.startswith('[launch]'):
                params = l
                break
        evts = [l for l in tail if l.startswith('[evt]')]
        pid = _unit_pid(GREET_UNIT)
        return _fjson({'success': True, 'running': True, 'state': state, 'pid': pid,
                       'launch': params, 'recent_events': evts[-8:],
                       'message': '迎宾模式正在运行中（进程 %s），最近事件 %d 条。'
                                  '想关闭就说「关闭迎宾」。' % (pid, len(evts))})
    return _fjson({'success': True, 'running': False, 'state': state,
                   'message': '迎宾模式当前没有在运行（服务状态 %s）。'
                              '想开启就说「开始迎宾」。' % state})


if __name__ == '__main__':
    log.info('启动 xiaozhi LED MCP 工具（stdio），socket=%s', SOCK_PATH)
    mcp.run(transport='stdio')
