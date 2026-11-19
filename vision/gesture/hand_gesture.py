#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""hand_gesture.py — 从 21 个手部关键点判断手势

关键点沿用 MediaPipe Hands 的 21 点定义（0-based）：
     0  腕
  1..4   拇指  CMC / MCP / IP / TIP
  5..8   食指  MCP / PIP / DIP / TIP
  9..12  中指  MCP / PIP / DIP / TIP
 13..16  无名指 MCP / PIP / DIP / TIP
 17..20  小指  MCP / PIP / DIP / TIP

判据只用**几何**，不训练模型：
  * 手掌尺寸基准 = 腕(0) 到 中指根(9) 的距离（对距离远近不敏感）
  * 手指伸直     = 指尖到该指根的距离 > k × 手掌尺寸
  * 手掌张开     = 五指都伸直
  * OK           = 拇指尖与食指尖捏合，且中指/无名指/小指伸直

刻意不 import cv2 / numpy 之外的东西，方便离线跑单元测试：
    python3 hand_gesture.py --selftest
"""

import numpy as np

# 关键点索引
WRIST = 0
THUMB_TIP = 4
INDEX_MCP, INDEX_TIP = 5, 8
MIDDLE_MCP, MIDDLE_TIP = 9, 12
RING_MCP, RING_TIP = 13, 16
PINKY_MCP, PINKY_TIP = 17, 20

# 每根手指的 (指根, 指尖)
FINGERS = (
    ('thumb', 1, THUMB_TIP),
    ('index', INDEX_MCP, INDEX_TIP),
    ('middle', MIDDLE_MCP, MIDDLE_TIP),
    ('ring', RING_MCP, RING_TIP),
    ('pinky', PINKY_MCP, PINKY_TIP),
)

# 可调阈值
EXTEND_RATIO = 1.30      # 指尖离指根超过 手掌尺寸×该值 视为伸直
PINCH_RATIO = 0.45       # 拇指尖与食指尖距离小于 手掌尺寸×该值 视为捏合


def palm_size(lms):
    """手掌尺寸基准：腕 -> 中指根的距离。"""
    return float(np.linalg.norm(lms[MIDDLE_MCP] - lms[WRIST]))


def finger_extended(lms, base_idx, tip_idx, size, ratio=EXTEND_RATIO):
    d = float(np.linalg.norm(lms[tip_idx] - lms[base_idx]))
    return (d > ratio * size), d / (size if size > 1e-6 else 1.0)


def classify(lms):
    """判断手势。返回 (名称, 详情 dict)

    lms: shape (21, 2) 或 (21, 3)，只用前两列（x, y）。
    名称 ∈ {'palm', 'ok', 'fist', 'point', 'unknown'}
    """
    lms = np.asarray(lms, dtype=np.float32)
    if lms.ndim != 2 or lms.shape[0] < 21 or lms.shape[1] < 2:
        return 'unknown', {'err': 'landmark shape %r' % (lms.shape,)}

    xy = lms[:, :2]
    size = palm_size(xy)
    if size <= 1e-6:
        return 'unknown', {'err': 'degenerate hand (palm size ~0)'}

    ext = {}
    for name, b, t in FINGERS:
        ext[name], ext[name + '_r'] = finger_extended(xy, b, t, size)

    pinch_d = float(np.linalg.norm(xy[THUMB_TIP] - xy[INDEX_TIP])) / size

    n_ext = sum(1 for name, _, _ in FINGERS if ext[name])
    detail = dict(ext, palm_size=size, pinch_r=pinch_d, n_extended=n_ext)

    # OK：拇指与食指捏成一个圈，其余三指张开
    if (pinch_d < PINCH_RATIO and ext['middle'] and ext['ring'] and ext['pinky']):
        return 'ok', detail
    # 手掌：五指全部张开
    if n_ext == 5:
        return 'palm', detail
    # 握拳：几乎没有伸直的手指
    if n_ext == 0:
        return 'fist', detail
    # 只伸食指：通常用来「指」
    if ext['index'] and n_ext == 1:
        return 'point', detail

    return 'unknown', detail


def palm_center_x(lms):
    """手掌中心的归一化 x（用于横向跟随）。"""
    xy = np.asarray(lms, dtype=np.float32)[:, :2]
    pts = [WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]
    return float(np.mean([xy[p][0] for p in pts]))


# ---------------------------------------------------------------- 自检

def _mk_hand(kind='palm'):
    """构造一串合成关键点，用于离线验证分类逻辑（不依赖摄像头/模型）。"""
    p = np.zeros((21, 2), np.float32)
    # 手腕在下方，手掌朝上
    p[WRIST] = [0.50, 0.90]
    p[1] = [0.40, 0.82]; p[2] = [0.34, 0.74]; p[3] = [0.30, 0.66]
    p[INDEX_MCP] = [0.46, 0.74]; p[6] = [0.45, 0.64]
    p[7] = [0.45, 0.56]
    p[MIDDLE_MCP] = [0.52, 0.72]; p[10] = [0.52, 0.60]
    p[11] = [0.52, 0.51]
    p[RING_MCP] = [0.58, 0.74]; p[14] = [0.59, 0.63]
    p[15] = [0.59, 0.55]
    p[PINKY_MCP] = [0.64, 0.78]; p[18] = [0.66, 0.69]
    p[19] = [0.67, 0.63]

    if kind == 'palm':
        # 五指全部伸直（指尖远离指根）
        p[THUMB_TIP] = [0.24, 0.56]
        p[INDEX_TIP] = [0.44, 0.44]
        p[MIDDLE_TIP] = [0.52, 0.36]
        p[RING_TIP] = [0.60, 0.42]
        p[PINKY_TIP] = [0.69, 0.54]
    elif kind == 'ok':
        # 拇指尖与食指尖捏合，其余三指伸直
        p[THUMB_TIP] = [0.47, 0.47]
        p[INDEX_TIP] = [0.48, 0.46]
        p[MIDDLE_TIP] = [0.52, 0.36]
        p[RING_TIP] = [0.60, 0.42]
        p[PINKY_TIP] = [0.69, 0.54]
    elif kind == 'fist':
        # 全部蜷起：指尖贴回指根附近
        p[THUMB_TIP] = [0.36, 0.76]
        p[INDEX_TIP] = [0.47, 0.72]
        p[MIDDLE_TIP] = [0.53, 0.70]
        p[RING_TIP] = [0.59, 0.72]
        p[PINKY_TIP] = [0.65, 0.76]
    return p


def _selftest():
    cases = [('palm', 'palm'), ('ok', 'ok'), ('fist', 'fist')]
    ok = True
    print('手掌尺寸基准 EXTEND_RATIO=%.2f  PINCH_RATIO=%.2f'
          % (EXTEND_RATIO, PINCH_RATIO))
    for kind, expect in cases:
        got, d = classify(_mk_hand(kind))
        good = (got == expect)
        ok = ok and good
        print('  %-6s -> %-8s %s  (伸直%d指, 捏合比%.2f)'
              % (kind, got, 'OK ' if good else 'FAIL', d.get('n_extended', -1),
                 d.get('pinch_r', -1)))
    print('selftest: %s' % ('PASS' if ok else 'FAIL'))
    return 0 if ok else 1


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='手势分类（几何判据）自检')
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(_selftest())
    ap.print_help()
