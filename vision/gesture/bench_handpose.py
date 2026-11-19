#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""bench_handpose.py — 测 opencv_zoo 的 palm_det + handpose（21 点）在 RK3576 上的速度

用法：
  /root/miniconda3/envs/lerobot/bin/python bench_handpose.py --cam 11 --n 30
  /root/miniconda3/envs/lerobot/bin/python bench_handpose.py --cam 11 --show   # 叠加关键点存图
"""
import argparse
import time

import cv2 as cv
import numpy as np

from mp_palmdet import MPPalmDet
from mp_handpose import MPHandPose

PALM_ONNX = '/root/gesture/palm_detection_mediapipe_2023feb.onnx'
POSE_ONNX = '/root/gesture/handpose_estimation_mediapipe_2023feb.onnx'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cam', type=int, default=11)
    ap.add_argument('--n', type=int, default=30)
    ap.add_argument('--show', action='store_true', help='把最后一帧画上关键点存到 /tmp/handpose.jpg')
    opt = ap.parse_args()

    t0 = time.time()
    palm_det = MPPalmDet(PALM_ONNX, scoreThreshold=0.5)
    handpose = MPHandPose(POSE_ONNX, confThreshold=0.8)
    print('模型加载 %.2fs' % (time.time() - t0), flush=True)

    cap = cv.VideoCapture(opt.cam)
    if not cap.isOpened():
        print('摄像头 /dev/video%d 打不开' % opt.cam)
        return
    print('摄像头就绪', flush=True)

    n_hands_total = 0
    n_ok = 0
    t0 = time.time()
    last = None
    for i in range(opt.n):
        ok, frame = cap.read()
        if not ok:
            break
        if frame is None:
            continue
        palms = palm_det.infer(frame)
        for p in palms:
            r = handpose.infer(frame, p)
            if r is None or len(r) == 0:
                continue
            n_hands_total += 1
            conf = r[-1]
            if conf > 0.8:
                n_ok += 1
            last = (frame.copy(), r)
    dt = (time.time() - t0) / max(1, opt.n)
    print('%.1f ms/frame -> %.1f fps' % (dt * 1000, 1.0 / dt), flush=True)
    print('检测到手 %d 次（置信>0.8 的 %d 次）' % (n_hands_total, n_ok), flush=True)

    if opt.show and last is not None:
        frame, r = last
        lms = np.array(r[4:67]).reshape(21, 3)  # landmarks 见 _postprocess 返回布局
        H, W = frame.shape[:2]
        for (x, y, _z) in lms:
            cv.circle(frame, (int(x * W), int(y * H)), 3, (0, 255, 0), -1)
        cv.imwrite('/tmp/handpose.jpg', frame)
        print('已存 /tmp/handpose.jpg（关键点已按归一化坐标画上）')

    cap.release()


if __name__ == '__main__':
    main()
