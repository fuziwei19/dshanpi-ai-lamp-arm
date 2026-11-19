#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test_wave_channels.py — 挥手双通道的离线回归测试

为什么要有它：挥手这件事我们改过两次、每次都出过「真机上没反应」的问题。
用合成数据把两条通道都跑一遍，改完代码立刻能验证，不必等真人站到镜头前。

跑法：  python3 test_wave_channels.py
        （只依赖 greet_logic.py，不需要 cv2 / numpy / 摄像头 / 机械臂）

覆盖：
  1 通道 B（手部）：person=False + kpts=None，只有手掌中心 x 往复 → 必须出 wave
  2 通道 A（人体）：人完整入镜 + 举手 + 腕 x 往复          → 必须出 wave
  3 反例：手掌中心 x 基本不动                              → 不能出 wave
  4 反例：单帧位置突跳（纹理误检的典型形态）               → 必须清历史、不出 wave
  5 反例：hand_big=False 时手部通道不参与                   → 不能出 wave
  6 手在镜头前也应刷新 _last_seen，避免被判「离开」         → 不能出 leave
"""
import sys
import os
import importlib.util

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, '/root')


def _load_greet_logic():
    """板上叫 greet_logic.py；本地开发副本叫 cur_greet_logic.py。都兼容。"""
    for name in ('greet_logic', 'cur_greet_logic'):
        try:
            return importlib.import_module(name)
        except ImportError:
            continue
    for fn in ('greet_logic.py', 'cur_greet_logic.py'):
        p = os.path.join(_HERE, fn)
        if os.path.exists(p):
            spec = importlib.util.spec_from_file_location('greet_logic', p)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    raise SystemExit('找不到 greet_logic.py / cur_greet_logic.py')


G = _load_greet_logic()       # noqa: E402


CFG = dict(G.DEFAULTS)
CFG['wave_cooldown'] = 0.0        # 测试里关掉冷却，方便连续观察
CFG['present_frames'] = 3
CFG['present_hold'] = 0.0
CFG['absent_sec'] = 1.0


def _mk_kpts(hand_x, shoulder_w=80.0, raised=True, conf=0.9):
    """合成 17 点：双肩固定，右腕跟着 hand_x 左右摆动（letterbox 尺度）

    刻意只用「列表套列表 + 可解包」的结构，和模型返回的 (17,3) 数组行为一致，
    这样测试脚本不需要 numpy（本地 PC 上可能没装）。
    """
    kp = [[0.0, 0.0, 0.0] for _ in range(17)]
    kp[5] = [160 - shoulder_w / 2.0, 150, conf]
    kp[6] = [160 + shoulder_w / 2.0, 150, conf]
    wy = 150 - (0.5 * shoulder_w if raised else -0.1 * shoulder_w)
    kp[10] = [160 + hand_x * shoulder_w, wy, conf]    # 右腕
    return kp


def _oscillate(n=40, amp=0.35, period=12.0, center=0.0):
    import math
    for i in range(n):
        yield center + amp * math.sin(2.0 * math.pi * i / period)


def run_case(name, feed, expect_wave, expect_leave=False):
    det = G.EventDetector(CFG)
    got = set()
    now = 1000.0
    for hx, person, big, kpts in feed:
        now += 0.05
        for ev in det.update(person, kpts, 0.25 if person else 0.0, now,
                             hand_x=hx, hand_big=big):
            got.add(ev)
    wave_ok = ('wave' in got) == expect_wave
    leave_ok = ('leave' in got) == expect_leave
    flag = 'PASS' if (wave_ok and leave_ok) else 'FAIL'
    print('  %-46s wave=%-5s leave=%-5s  %s'
          % (name, 'wave' in got, 'leave' in got, flag))
    if not wave_ok:
        print('    !! 期望 wave=%s，实际事件=%s' % (expect_wave, sorted(got)))
    return wave_ok and leave_ok


def main():
    ok = True
    print('挥手双通道回归测试（wave_window=%.1fs reversals=%d amp=%.2f jump=%.2f）'
          % (CFG['wave_window'], CFG['wave_reversals'], CFG['wave_amp'],
             CFG['hand_jump']))

    # 1) 通道 B：完全没有人（person=False），只有手在镜头前来回摆
    feed = [(x, False, True, None) for x in _oscillate(amp=0.30, period=10.0)]
    ok &= run_case('1 手部通道：无人 + 手往复（手挡住身体）', feed, True)

    # 2) 通道 A：人完整入镜 + 举手 + 腕左右摆
    feed = [(None, True, False, _mk_kpts(x, raised=True))
            for x in _oscillate(amp=0.30, period=10.0)]
    ok &= run_case('2 人体通道：有 + 举手 + 腕往复', feed, True)

    # 3) 反例：手基本不动 —— 不能误判成挥手
    feed = [(0.50, False, True, None) for _ in range(40)]
    ok &= run_case('3 反例：手不动（不应挥手）', feed, False)

    # 4) 反例：单帧突跳（纹理误检形态）应被 hand_jump 拦掉
    hs = []
    for i in range(20):
        hs.append(0.20 if i % 2 == 0 else 0.90)      # 每帧 0.7 的跳变
    feed = [(x, False, True, None) for x in hs]
    ok &= run_case('4 反例：位置突跳（不应挥手）', feed, False)

    # 5) 反例：没有「够大够真的手」时手部通道不参与
    feed = [(x, False, False, None) for x in _oscillate(amp=0.30, period=10.0)]
    ok &= run_case('5 反例：hand_big=False（不应挥手）', feed, False)

    # 6) 手一直在镜头前，应该持续刷新 _last_seen，不能判离开
    feed = [(0.5, False, True, None) for _ in range(40)]
    ok &= run_case('6 手在镜头前不应判「离开」', feed, False, expect_leave=False)

    print('selftest: %s' % ('PASS' if ok else 'FAIL'))
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
