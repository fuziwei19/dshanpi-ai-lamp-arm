#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""greet_logic.py — 迎宾交互的「纯逻辑」层（无硬件依赖，可离线测试）

包含：
  * hand_raise_metric()  举手判据
  * count_reversals()    挥手判据（方向反转计数）
  * EventDetector        状态机 + 时序去抖 -> 产生 greet / wave / leave 事件
  * DEFAULTS             默认参数

刻意不 import cv2 / torch / lerobot / rknnlite，方便在 PC 上直接跑单元测试。
"""
import collections

# COCO 17 关键点（0-based）
LS, RS, LW, RW = 5, 6, 9, 10     # 左肩 / 右肩 / 左腕 / 右腕

DEFAULTS = {
    'present_frames': 8,        # 连续多少帧检测到人 → 判定"在场"
    'present_hold': 0.8,        # 在场后至少停留多少秒才打招呼
    'absent_sec': 8.0,          # 无人超过多少秒 → 告别/省电
    'greet_cooldown': 25.0,     # 两次迎宾最小间隔
    'wave_cooldown': 12.0,      # 两次挥手告别的间隔
    'wave_amp': 0.12,           # 挥手判定：x 摆幅（相对肩宽）
    'wave_reversals': 2,        # 挥手判定：方向反转次数
    'wave_window': 2.0,         # 挥手判定：时间窗口（秒）
    'wave_raise': 0.25,         # 举手判定：手腕需高于肩 0.25×肩宽
    'min_box': 0.06,            # 人体框面积占画面比例下限
    # 挥手「手部通道」：手举到镜头前会把身体挡住/挤出画面，人体姿态就废了，
    # 这时改用「手掌中心 x 的左右往复」判定（复用同一个 count_reversals）。
    'hand_jump': 0.35,          # 手掌中心单帧位移超过该值 → 视为不可信，清空历史
}


def hand_raise_metric(kpts, conf_thr=0.5, raise_ratio=0.25):
    """判断「是否有一只手举过肩」，返回 (raised, x_norm, rise, shoulder_w)

    x_norm = (腕 x - 肩中点 x) / 肩宽 —— 归一化后与人离镜头远近、整体左右平移无关，
    挥手检测就是看这个值在时间上的往复。
    """
    if kpts is None:
        return False, None, 0.0, 0.0
    sw = abs(float(kpts[LS][0]) - float(kpts[RS][0]))
    if sw < 8.0:
        return False, None, 0.0, 0.0
    sx_mid = (float(kpts[LS][0]) + float(kpts[RS][0])) / 2.0
    best = None
    for wi, si in ((LW, LS), (RW, RS)):
        if float(kpts[wi][2]) > conf_thr and float(kpts[si][2]) > conf_thr:
            rise = float(kpts[si][1]) - float(kpts[wi][1])   # y 向下，正数 = 腕高于肩
            if rise > raise_ratio * sw:
                if best is None or rise > best[0]:
                    best = (rise, float(kpts[wi][0]))
    if best is None:
        return False, None, 0.0, sw
    return True, (best[1] - sx_mid) / sw, best[0], sw


def count_reversals(xs, min_amp):
    """统计序列中的方向反转次数（单次摆动幅度需超过 min_amp 才计入）

    挥手 = 若干次「左→右→左」的方向反转；比看绝对位置稳（不受人整体左右站位影响）。
    """
    if xs is None or len(xs) < 2:
        return 0
    turns = 0
    dirn = 0
    last = xs[0]
    for x in xs[1:]:
        d = x - last
        if d > min_amp:
            if dirn == -1:
                turns += 1
            dirn = 1
            last = x
        elif d < -min_amp:
            if dirn == 1:
                turns += 1
            dirn = -1
            last = x
    return turns


class EventDetector:
    """把「每帧的检测结果」变成「交互事件」

    状态：不在场 <-> 在场（连续 N 帧去抖）
    事件：
      appear  刚被判为在场（内部用，不对外触发动作）
      greet   在场且停留够久 -> 打招呼（带冷却，一轮只触发一次）
      wave    举手 + 左右摆动 -> 告别（带冷却）
      leave   不在场超过 absent_sec -> 省电
    """

    def __init__(self, cfg=None):
        c = dict(DEFAULTS)
        if cfg:
            c.update({k: v for k, v in cfg.items() if k in DEFAULTS})
        self.cfg = c
        self.present = False
        self._hit = 0
        self._miss = 0
        self._present_since = 0.0
        self._last_seen = 0.0
        self._greeted = True        # 初始 True：开机时不需要迎宾
        self._leave_fired = True    # 初始 True：开机时不需要告别
        self._last_greet = -1e9
        self._last_wave = -1e9
        self._wrist = collections.deque()
        self._hand = collections.deque()       # 手部通道：(time, 手掌中心 x)
        self.calib = {}

    def update(self, person, kpts, box_area_ratio, now, hand_x=None, hand_big=False):
        cfg = self.cfg
        events = []

        # --- 在场去抖 ---
        if person:
            self._hit += 1
            self._miss = 0
            self._last_seen = now
        else:
            self._miss += 1
            self._hit = 0

        if not self.present and self._hit >= cfg['present_frames']:
            self.present = True
            self._present_since = now
            self._greeted = False
            self._leave_fired = False
            self._wrist.clear()
            events.append('appear')
        elif self.present and self._miss >= cfg['present_frames']:
            self.present = False
            self._wrist.clear()

        # --- 迎宾 ---
        if (self.present and not self._greeted
                and (now - self._present_since) >= cfg['present_hold']):
            self._greeted = True
            if (now - self._last_greet) >= cfg['greet_cooldown']:
                self._last_greet = now
                events.append('greet')

        # --- 挥手 ---
        # 通道 A（准）：人体姿态的「腕高于肩 + 腕 x 左右往复」。
        #   要求人完整入镜、姿态关键点可信，是标准场景。
        raised, x_norm, rise, sw = hand_raise_metric(
            kpts, raise_ratio=cfg['wave_raise'])
        turns_torso = 0
        if self.present and kpts is not None:
            if raised:
                self._wrist.append((now, x_norm))
            while self._wrist and now - self._wrist[0][0] > cfg['wave_window']:
                self._wrist.popleft()
            if len(self._wrist) >= 4:
                turns_torso = count_reversals([p[1] for p in self._wrist],
                                              cfg['wave_amp'])
        else:
            self._wrist.clear()

        # 通道 B（兜底）：手部 21 点的「手掌中心 x 左右往复」。
        #   为什么需要它：人站在台灯前挥手时，手往往离镜头很近，
        #   身体被手挡住（或超出画面）→ yolov8-pose 直接检不到人 →
        #   通道 A 连 kpts 都没有（sw=0）→ 挥手永远判不出来。
        #   这里完全不依赖人体姿态，只要有只「够大且高置信」的手在画面里就行。
        turns_hand = 0
        if hand_big and hand_x is not None:
            if self._hand and abs(hand_x - self._hand[-1][1]) > cfg['hand_jump']:
                self._hand.clear()        # 单帧位置突跳 = 不可信，重新累积
            self._hand.append((now, hand_x))
            while self._hand and now - self._hand[0][0] > cfg['wave_window']:
                self._hand.popleft()
            if len(self._hand) >= 4:
                turns_hand = count_reversals([p[1] for p in self._hand],
                                             cfg['wave_amp'])
        else:
            self._hand.clear()

        turns = max(turns_torso, turns_hand)
        wave_ok = ((self.present and turns_torso >= cfg['wave_reversals'])
                   or (hand_big and turns_hand >= cfg['wave_reversals']))
        if wave_ok and (now - self._last_wave) >= cfg['wave_cooldown']:
            self._last_wave = now
            self._wrist.clear()
            self._hand.clear()
            events.append('wave')

        # 手在镜头前也算「人还在」：否则人站着只伸手时会被判离开 -> 收纳归位，
        # 镜头跟着朝下，就再也看不见他了（实测会卡死在这个状态）。
        if hand_big and hand_x is not None:
            self._last_seen = now

        # --- 离开 ---
        if (not self.present and not self._leave_fired
                and self._last_seen > 0
                and (now - self._last_seen) >= cfg['absent_sec']):
            self._leave_fired = True
            events.append('leave')

        self.calib = dict(present=self.present, hit=self._hit, miss=self._miss,
                          raised=raised, x_norm=x_norm, rise=rise, sw=sw,
                          turns=turns, turns_torso=turns_torso,
                          turns_hand=turns_hand,
                          wrist_n=len(self._wrist),
                          hand_n=len(self._hand), hand_big=hand_big,
                          hand_x=hand_x,
                          silent=(round(now - self._last_seen, 1)
                                  if self._last_seen else -1))
        return events
