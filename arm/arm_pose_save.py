#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""arm_pose_save.py — 读取机械臂「此刻的真实姿态」，存成迎宾模式的基准位

用法
----
    python3 arm_pose_save.py                 # 读当前姿态 -> 写 /etc/greet-pose.json
    python3 arm_pose_save.py --out /tmp/x.json
    python3 arm_pose_save.py --print-only    # 只打印，不写文件

保存下来的 `base_pose` 会被 vision_interact.py 当迎宾基准位用：
    迎宾启动 -> 机械臂转到这个姿态
    有人靠近 -> 在这个姿态上做小幅左右旋转 + 让 AI 打招呼

⚠️ /dev/ttyACM0 是独占的
    人脸跟随 / 迎宾服务在跑的时候读不了（会报 Device or resource busy）：
        systemctl stop face-follow vision-greet
    只读不动作：连上读 6 个关节角度，读完保持力矩断开，机械臂不会动。
"""
import argparse
import datetime
import json
import os
import sys

from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.robots.so101_follower.so101_follower import SO101Follower

DEFAULT_OUT = '/etc/greet-pose.json'


def read_pose(port, robot_id):
    """连上机械臂读一次当前姿态（度）。返回 dict；失败抛异常。"""
    cfg = SO101FollowerConfig(port=port, id=robot_id, cameras={}, use_degrees=True)
    robot = SO101Follower(cfg)
    robot.connect()
    try:
        names = list(robot.action_features.keys())
        obs = robot.get_observation()
        return {n: round(float(obs.get(n, 0.0)), 4) for n in names}
    finally:
        try:
            robot.bus.disconnect(disable_torque=False)   # 保持力矩：读姿态不该让臂软下来
        except Exception as e:
            print('[warn] 断开串口失败: %r' % e, file=sys.stderr)


def load_existing(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        return raw if isinstance(raw, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception as e:
        print('[warn] 读取 %s 失败(%r)，将覆盖重建' % (path, e), file=sys.stderr)
        return {}


def main():
    ap = argparse.ArgumentParser(description='读取机械臂当前姿态并保存为迎宾基准位')
    ap.add_argument('--port', default='/dev/ttyACM0')
    ap.add_argument('--robot-id', default='my_awesome_follower_arm')
    ap.add_argument('--out', default=DEFAULT_OUT)
    ap.add_argument('--print-only', action='store_true', help='只打印不写文件')
    opt = ap.parse_args()

    try:
        pose = read_pose(opt.port, opt.robot_id)
    except Exception as e:
        msg = str(e)
        print('[error] 读取机械臂姿态失败: %s' % msg, file=sys.stderr)
        if 'busy' in msg.lower() or 'Resource temporarily unavailable' in msg:
            print('        串口被占用 —— 先停掉占用它的服务：', file=sys.stderr)
            print('        systemctl stop face-follow vision-greet', file=sys.stderr)
        print('        也可能只是偶发通讯超时，直接重跑本脚本即可。', file=sys.stderr)
        return 1

    print('机械臂当前姿态（度）:')
    for n, v in pose.items():
        print('    %-20s %8.2f' % (n, v))
    print('PYDICT = {%s}' % ', '.join("'%s': %s" % (n, v) for n, v in pose.items()))

    if opt.print_only:
        return 0

    doc = load_existing(opt.out)
    doc['base_pose'] = pose
    doc['saved_at'] = datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
    doc['source'] = 'arm_pose_save.py'

    tmp = opt.out + '.tmp'
    try:
        os.makedirs(os.path.dirname(opt.out) or '.', exist_ok=True)
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)
            f.write('\n')
        os.replace(tmp, opt.out)          # 原子替换，避免迎宾进程读到半个文件
    except Exception as e:
        print('[error] 写入 %s 失败: %r' % (opt.out, e), file=sys.stderr)
        return 1

    print('已保存迎宾基准位 -> %s' % opt.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
