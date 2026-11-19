#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""face_follow_launch.py — 人脸跟随的「参数校验 + 启动器」

由 systemd 单元 face-follow.service 调用（ExecStart）。
职责：从 /etc/face-follow.json 读取参数（该文件由小智 MCP 工具写入），
严格校验后用 os.execv 以 **argv 列表** 方式启动 yolov8pose_head_follow_so101.py，
不做任何 shell 字符串拼接 —— 彻底杜绝语音内容导致的命令注入。

为什么要多这一层：
  1. systemd 的 ExecStart 无法使用运行时动态参数，而 os.execv 可以；
  2. os.execv 会让跟随脚本直接成为该单元的主进程，
     于是 systemd 的 KillSignal=SIGINT 就等于在终端里按 Ctrl+C，
     跟随脚本的 KeyboardInterrupt 分支会被触发 ->
     平滑移动回 PARK 收纳点 + 释放舵机力矩（而不是被硬杀导致手臂僵住）。
"""
import json
import os
import sys

CFG_PATH = '/etc/face-follow.json'
PYTHON = '/root/miniconda3/envs/lerobot/bin/python'
FOLLOW_PY = '/root/yolov8pose_head_follow_so101.py'

DEFAULTS = {
    'aim': 'head',      # head=对准人脸关键点 / body=对准人体框中心
    'band': 0.25,       # 允许带（画面半宽的百分比），越大越稳
    'cam_id': 12,       # 摄像头节点号
    'dry_run': False,   # 只跑视觉，不驱动机械臂
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
        print('[launch] %s 不存在，使用默认参数' % path, flush=True)
    except Exception as e:
        print('[launch] 读取 %s 失败(%s)，使用默认参数' % (path, e), flush=True)
    return cfg


def clamp_float(v, lo, hi, default):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    if f != f:                      # NaN
        return default
    return max(lo, min(hi, f))


def clamp_int(v, lo, hi, default):
    try:
        i = int(v)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, i))


def build_argv(cfg):
    """把配置翻译成 argv 列表（每个参数独立成 token，无 shell 参与）"""
    aim = 'body' if str(cfg.get('aim', 'head')).strip().lower() == 'body' else 'head'
    band = clamp_float(cfg.get('band'), 0.05, 0.60, DEFAULTS['band'])
    cam = clamp_int(cfg.get('cam_id'), 0, 31, DEFAULTS['cam_id'])

    argv = [PYTHON, FOLLOW_PY,
            '--aim', aim,
            '--band', '%.2f' % band,
            '--cam-id', str(cam),
            '--exit-pose', 'park']      # 退出必定平滑回收纳点
    if cfg.get('dry_run'):
        argv.append('--dry-run')        # 只跑视觉，机械臂完全不动
    return argv


def main():
    cfg = load_cfg()
    argv = build_argv(cfg)
    print('[launch] %s' % ' '.join(argv), flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os.execv(PYTHON, argv)              # 不返回：本进程被跟随脚本替换


if __name__ == '__main__':
    main()
