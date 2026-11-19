#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""check_shake_only_pan.py — 验证「摇摇头」只驱动 shoulder_pan（5 号舵机）

做法：把 arm_actions.smooth_goto 换成只记录、不下发的桩
（build_pose/smooth_goto 都是 torch tensor[1,6]，按 names 顺序排列），
然后跑真的 arm.shake_head()，逐帧比对目标姿态与基准位的差。
除了 shoulder_pan，其它关节的差都应该 < 0.01°。

用法（板上）：cd /root/gesture && python check_shake_only_pan.py
"""
import sys

sys.path.insert(0, '/root')

import arm_actions as arm                                     # noqa: E402
from yolov8pose_head_follow_so101 import build_pose           # noqa: E402

BASE = {
    'shoulder_pan.pos': 1.98,
    'shoulder_lift.pos': -77.05,
    'elbow_flex.pos': 14.47,
    'wrist_flex.pos': 51.16,
    'wrist_roll.pos': 19.47,
    'gripper.pos': 0.47,
}

names = sorted(BASE)
frames = []
# pan_wiggle 是位置参数调用：smooth_goto(robot, names, target_tensor, step_s, steps)
arm.smooth_goto = lambda *a, **kw: frames.append(list(a[2][0].tolist()))

baseline = list(build_pose(dict(BASE), names)[0].tolist())

arm.shake_head(None, names, base=dict(BASE), amp=10.0, cycles=2)

print('共下发 %d 帧目标姿态（每帧 6 个关节，顺序 %s）' % (len(frames), '/'.join(names)))
print()
bad = 0
for i, vec in enumerate(frames):
    diff = {n: round(v - b, 2) for n, v, b in zip(names, vec, baseline)
            if abs(v - b) > 0.01}
    others = {k: v for k, v in diff.items() if k != 'shoulder_pan.pos'}
    if others:
        bad += 1
    print('  %2d 帧  变化: %-24s  其它关节: %s'
          % (i, diff or '（回基准位）', others or '无'))

print()
print('结论: %s' % ('只动了 shoulder_pan（5 号舵机）✅' if bad == 0
                   else '有 %d 帧动了别的关节 ❌' % bad))
print('舵机编号: shoulder_pan = 5 号 = 「转灯头」那一路')
sys.exit(0 if bad == 0 else 1)
