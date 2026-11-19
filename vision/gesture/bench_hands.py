#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""bench_hands.py — 测 MediaPipe HandLandmarker 在 RK3576 上的推理速度

用法：
  /root/miniconda3/envs/lerobot/bin/python /root/gesture/bench_hands.py           # 纯合成图
  /root/miniconda3/envs/lerobot/bin/python /root/gesture/bench_hands.py --cam 12  # 真实摄像头
"""
import argparse
import time

import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL = '/root/gesture/hand_landmarker.task'


def make_detector(num_hands=2):
    bo = mp_python.BaseOptions(model_asset_path=MODEL,
                               delegate=mp_python.BaseOptions.Delegate.CPU)
    opt = vision.HandLandmarkerOptions(base_options=bo, num_hands=num_hands,
                                       running_mode=vision.RunningMode.VIDEO)
    t0 = time.time()
    det = vision.HandLandmarker.create_from_options(opt)
    return det, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cam', type=int, default=-1, help='摄像头号；-1 = 用合成图')
    ap.add_argument('--n', type=int, default=30)
    opt = ap.parse_args()

    det, init_s = make_detector()
    print('init %.2fs (CPU, num_hands=2)' % init_s, flush=True)

    cap = None
    if opt.cam >= 0:
        import cv2
        cap = cv2.VideoCapture(opt.cam)
        if not cap.isOpened():
            print('摄像头 /dev/video%d 打不开' % opt.cam)
            return
        print('摄像头 /dev/video%d 已打开' % opt.cam, flush=True)

    def next_frame(i):
        if cap is not None:
            ok, f = cap.read()
            if not ok:
                return None
            return f
        img = np.zeros((480, 640, 3), np.uint8)
        img[:] = (120, 120, 120)
        return img

    # 预热
    for i in range(5):
        f = next_frame(i)
        if f is None:
            break
        mi = mp.Image(image_format=mp.ImageFormat.SRGB, data=f[:, :, ::-1])
        det.detect_for_video(mi, int(time.time() * 1000) + i * 33)

    n = opt.n
    t0 = time.time()
    got = 0
    for i in range(n):
        f = next_frame(i)
        if f is None:
            break
        mi = mp.Image(image_format=mp.ImageFormat.SRGB, data=f[:, :, ::-1])
        r = det.detect_for_video(mi, int(time.time() * 1000) + i * 33)
        got += len(r.hand_landmarks)
    dt = (time.time() - t0) / max(1, n)
    print('%.1f ms/frame -> %.1f fps   (检测到手 %d 次)' % (dt * 1000, 1.0 / dt, got),
          flush=True)

    if cap is not None:
        cap.release()


if __name__ == '__main__':
    main()
