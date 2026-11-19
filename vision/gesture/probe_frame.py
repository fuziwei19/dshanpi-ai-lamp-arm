#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""probe_frame.py — 单帧探针：拿一张静态图直接喂手部模型，看它到底能检出什么

用途：当「明明手就在画面里，程序却说没检测到」时，用它把问题钉死在
      「模型看不见」还是「业务代码没跑通」。

  python3 probe_frame.py /tmp/x.jpg              # 默认阈值
  python3 probe_frame.py /tmp/x.jpg --thr 0.3    # 放宽手掌检测阈值再试
  python3 probe_frame.py /tmp/x.jpg --save /tmp/out.jpg   # 画出检出结果

会打印：
  * 手掌检测器检出几只、每只的 score 与 bbox（含归一化尺寸）
  * 每只手的 21 点置信度、手势分类、伸直指数量、捏合比
  * 人体姿态模型检不检得到人（框面积比）
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, '/root/gesture')
sys.path.insert(0, '/root')

PALM = '/root/gesture/palm_detection_mediapipe_2023feb.onnx'
HAND = '/root/gesture/handpose_estimation_mediapipe_2023feb.onnx'
POSE = '/root/rk_lerobot_tools/model/yolov8_pose_fp.rknn'


def fix_logging():
    """修回被 rknnlite 破坏的 logging 级别表。

    ⚠️ 板上实测：`import rknnlite.api` 会把 `logging._nameToLevel` 洗成
    只剩单字母的 {'C':50,'E':40,'W':30,...}，标准名 'WARNING' 等全部消失。
    此后任何 `logger.setLevel("WARNING")` 都会抛
        ValueError: Unknown level: 'WARNING'
    —— torch/fx/passes/utils/matcher_utils.py 就在 import 时这么干，
    所以「先 import rknnlite、后 import torch」必定崩，
    而且报错信息看起来像环境变量问题，非常容易误判。

    两种解法：① 让 torch/lerobot 先 import；② 事后把名字登记回来（本函数）。
    """
    import logging
    for name, val in (('CRITICAL', 50), ('FATAL', 50), ('ERROR', 40),
                      ('WARN', 30), ('WARNING', 30), ('INFO', 20),
                      ('DEBUG', 10), ('NOTSET', 0)):
        logging.addLevelName(val, name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('img')
    ap.add_argument('--thr', type=float, default=0.65, help='手掌检测 score 阈值')
    ap.add_argument('--conf', type=float, default=0.8, help='21 点置信度阈值')
    ap.add_argument('--save', default='')
    ap.add_argument('--no-pose', action='store_true', help='跳过人体姿态（省时间）')
    opt = ap.parse_args()

    img = cv2.imread(opt.img)
    if img is None:
        print('读不到图片: %s' % opt.img)
        return 1
    H, W = img.shape[:2]
    print('图片 %s  尺寸 %dx%d' % (os.path.basename(opt.img), W, H))

    from mp_palmdet import MPPalmDet
    from mp_handpose import MPHandPose
    import hand_gesture

    palm = MPPalmDet(PALM, scoreThreshold=opt.thr)
    hp = MPHandPose(HAND, confThreshold=opt.conf)

    palms = palm.infer(img)
    n = 0 if palms is None else len(palms)
    print('\n[手掌检测] thr=%.2f -> 检出 %d 只' % (opt.thr, n))
    if n == 0:
        print('  → 模型在这个阈值下看不到手。试 `--thr 0.3`；若仍为 0，')
        print('    说明手确实不在画面里（或太小/太边缘/太糊）。')

    vis = img.copy()
    for i, p in enumerate(palms if n else []):
        x1, y1, x2, y2 = float(p[0]), float(p[1]), float(p[2]), float(p[3])
        score = float(p[18])
        bw, bh = (x2 - x1) / W, (y2 - y1) / H
        print('  #%d score=%.3f  bbox=(%.0f,%.0f,%.0f,%.0f)  归一化 宽%.3f 高%.3f'
              % (i, score, x1, y1, x2, y2, bw, bh))
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        r = hp.infer(img, p)
        if r is None or len(r) < 132:
            print('     21 点失败（手可能太近/太偏/被裁切）')
            continue
        lms = np.array(r[4:67], dtype=np.float32).reshape(21, 3)
        norm = np.stack([lms[:, 0] / W, lms[:, 1] / H], axis=1)
        g, d = hand_gesture.classify(norm)
        print('     21 点 conf=%.3f -> 手势=%s  伸直%d指  捏合比%.2f  掌尺寸%.3f'
              % (float(r[131]), g, d.get('n_extended', -1),
                 d.get('pinch_r', -1.0), d.get('palm_size', 0.0)))
        for j, (x, y) in enumerate(norm):
            cv2.circle(vis, (int(x * W), int(y * H)), 3, (0, 200, 255), -1)

    if not opt.no_pose:
        try:
            # 顺序要紧：yolov8pose 会牵出 torch/lerobot，必须在 rknnlite 之前 import
            from yolov8pose_head_follow_so101 import (decode, letterbox,
                                                      CONF_THRESH, draw)
            from rknnlite.api import RKNNLite
            fix_logging()          # 万一已经被 rknnlite 洗过，这里补回来
            rk = RKNNLite()
            if rk.load_rknn(POSE) == 0:
                rk.init_runtime()
                lb, r, ox, oy = letterbox(img, 320)
                outs = rk.inference(inputs=[np.expand_dims(lb[..., ::-1], 0)],
                                    data_format='nhwc')
                dets = decode(outs, CONF_THRESH, 320) if outs else []
                print('\n[人体姿态] 检出 %d 人' % len(dets))
                for box, sc, kp in dets:
                    ar = ((box[2] - box[0]) * (box[3] - box[1])) / float(320 * 320)
                    print('  score=%.2f  框面积比=%.4f (>=0.06 才当"有人")' % (sc, ar))
                    draw(vis, box, sc, kp, r, ox, oy)
                if not dets:
                    print('  → 姿态模型看不到人（人不在画面里 / 被挡 / 太近超出画面）')
                rk.release()
        except Exception as e:
            print('\n[人体姿态] 跳过/失败: %r' % e)

    if opt.save:
        cv2.imwrite(opt.save, vis)
        print('\n已保存调试图: %s' % opt.save)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
