#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""greet_launch.py — 迎宾模式的「参数校验 + 启动器」

由 systemd 单元 vision-greet.service 调用（ExecStart）。
从 /etc/vision-greet.json 读参数（该文件由小智 MCP 工具写入），严格校验后
用 os.execv 以 argv 列表启动 vision_interact.py（不做 shell 拼接，防注入）。

与 face_follow_launch.py 同一套路：execv 让迎宾程序直接成为单元主进程，
于是 systemd 的 KillSignal=SIGINT 就等于终端 Ctrl+C，会走到
「机械臂回收纳位 + 释放力矩」的优雅收尾。
"""
import json
import os
import sys

CFG_PATH = '/etc/vision-greet.json'
PYTHON = '/root/miniconda3/envs/lerobot/bin/python'
TARGET = '/root/vision_interact.py'

DEFAULTS = {
    'cam_id': 12,
    'dry_run': False,          # True = 不连机械臂（只跑视觉+灯效）
    'calib': False,            # True = 只打印判据数值，不做任何反馈
    'absent_sec': 8.0,         # 无人多久 -> 告别/省电
    'greet_cooldown': 25.0,    # 两次迎宾最小间隔
    'wave_cooldown': 12.0,     # 两次挥手告别的间隔
    'idle_bright': 35,         # 在场常态亮度
    'nod_amp': 13.0,           # 点头幅度（度，保留兼容）
    'wave_deg': 20.0,          # 招手幅度（度）
    'pan_amp': 16.0,           # 迎宾「左右小幅旋转」幅度（度）
    'pan_cycles': 2,           # 左右摆动来回次数
    'greet_say': '打个招呼',    # 迎宾时让 AI 说的话（走 ai_agent_send_text）
    'say_enabled': True,       # False = 只动机械臂，不叫 AI 说话
    'debug_every': 0.0,        # >0 时每 N 秒存调试图
}


def load_cfg(path=CFG_PATH):
    cfg = dict(DEFAULTS)
    try:
        with open(path, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            for k in DEFAULTS:
                if k in raw:
                    cfg[k] = raw[k]
    except FileNotFoundError:
        print('[launch] %s 不存在，用默认参数' % path, flush=True)
    except Exception as e:
        print('[launch] 读取 %s 失败(%s)，用默认参数' % (path, e), flush=True)
    return cfg


def clamp_float(v, lo, hi, default):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    if f != f:
        return default
    return max(lo, min(hi, f))


def clamp_int(v, lo, hi, default):
    try:
        i = int(v)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, i))


def sanitize_say(v, default='打个招呼', max_len=200):
    """把迎宾语收紧成一行短文本（execv 不经 shell，但还是要防超长/换行）"""
    if not isinstance(v, str):
        return default
    s = ' '.join(v.split())          # 去掉换行、制表符，合并连续空白
    s = s.strip()
    if not s:
        return default
    return s[:max_len]


def build_argv(cfg):
    argv = [PYTHON, TARGET,
            '--cam-id', str(clamp_int(cfg.get('cam_id'), 0, 31, 12)),
            '--absent-sec', '%.1f' % clamp_float(cfg.get('absent_sec'), 2.0, 600.0, 8.0),
            '--greet-cooldown', '%.1f' % clamp_float(cfg.get('greet_cooldown'), 0.0, 3600.0, 25.0),
            '--wave-cooldown', '%.1f' % clamp_float(cfg.get('wave_cooldown'), 0.0, 3600.0, 12.0),
            '--idle-bright', str(clamp_int(cfg.get('idle_bright'), 0, 100, 35)),
            '--nod-amp', '%.1f' % clamp_float(cfg.get('nod_amp'), 3.0, 30.0, 13.0),
            '--wave-deg', '%.1f' % clamp_float(cfg.get('wave_deg'), 5.0, 40.0, 20.0),
            '--pan-amp', '%.1f' % clamp_float(cfg.get('pan_amp'), 3.0, 40.0, 16.0),
            '--pan-cycles', str(clamp_int(cfg.get('pan_cycles'), 1, 6, 2)),
            '--greet-say', sanitize_say(cfg.get('greet_say')),
            ]
    if not cfg.get('say_enabled', True):
        argv.append('--no-say')
    dbg = clamp_float(cfg.get('debug_every'), 0.0, 3600.0, 0.0)
    if dbg > 0:
        argv += ['--debug-every', '%.1f' % dbg]
    if cfg.get('dry_run'):
        argv.append('--dry-run')
    if cfg.get('calib'):
        argv.append('--calib')
    return argv


def main():
    cfg = load_cfg()
    argv = build_argv(cfg)
    print('[launch] %s' % ' '.join(argv), flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os.execv(PYTHON, argv)


if __name__ == '__main__':
    main()
