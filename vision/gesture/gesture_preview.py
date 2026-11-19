#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gesture_preview.py — 手势 / 挥手 实时调试预览

在浏览器里直接看到摄像头画面 + 21 个手部关键点 + 实时判据，
用来确认「我做的手势到底被识别成了什么」。

用法（先停掉迎宾服务，它占着摄像头）：
    systemctl stop vision-greet
    cd /root/gesture && nohup /root/miniconda3/envs/lerobot/bin/python \
        gesture_preview.py --cam 11 --port 8080 > /tmp/gesture_preview.log 2>&1 &
    # 浏览器打开 http://<板子IP>:8080

画面上的状态栏（用英文避免 OpenCV 字体不支持中文）：
    PERSON  是否检测到人（人体框面积 >= 6%）
    RAISED  手是否举过肩        RISE  手腕高出肩膀多少像素
    TURNS   2 秒窗口内挥手的方向反转次数（>=2 判定为挥手）
    GEST    手势名   FING 伸直几根手指   PINCH 拇指-食指距离/手掌尺寸
    FOLLOW  是否在跟随（手掌）   NOD 是否触发点头（OK）
"""
import argparse
import collections
import os
import threading
import time

import cv2
import numpy as np

sys_path_done = False


def setup_path():
    global sys_path_done
    if sys_path_done:
        return
    import sys
    for p in ('/root', '/root/gesture'):
        if p not in sys.path:
            sys.path.insert(0, p)
    sys_path_done = True


setup_path()

from rknnlite.api import RKNNLite                      # noqa: E402
from greet_logic import (                              # noqa: E402
    hand_raise_metric, count_reversals)
from mp_palmdet import MPPalmDet                       # noqa: E402
from mp_handpose import MPHandPose                     # noqa: E402
import hand_gesture                                    # noqa: E402

# ---- 下面四个函数从 yolov8pose_head_follow_so101.py 原样搬过来 ----
# 目的：预览脚本不需要 import 那个模块，也就不会牵出 torch / lerobot
# （该环境下 torch 导入会因 PYTORCH_MATCHER_LOGLEVEL 被设成带引号的值而崩）。
NMS_THRESH = 0.45
SKELETON = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13],
            [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
            [2, 4], [3, 5], [4, 6], [5, 7]]


def open_camera(prefer, w=640, h=480):
    def try_open(i):
        c = cv2.VideoCapture(i)
        c.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        c.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        if c.isOpened():
            ok, f = c.read()
            if ok and f is not None:
                return c, i
        c.release()
        return None, None

    c, i = try_open(prefer)
    if c is not None:
        return c, i
    cands = []
    for k in range(0, 24):
        try:
            with open('/sys/class/video4linux/video%d/name' % k) as fh:
                nm = fh.read().strip().lower()
        except Exception:
            continue
        if any(s in nm for s in ('camera', 'uvc', 'webcam')):
            cands.append(k)
    print('/dev/video%d 不可用, 扫描到候选: %s' % (prefer, cands), flush=True)
    for k in cands:
        if k == prefer:
            continue
        c, i = try_open(k)
        if c is not None:
            print('自动选中摄像头 /dev/video%d' % i, flush=True)
            return c, i
    return None, None


def letterbox(img, size, color=114):
    h, w = img.shape[:2]
    r = min(size / w, size / h)
    nw, nh = int(round(w * r)), int(round(h * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), color, dtype=np.uint8)
    ox, oy = (size - nw) // 2, (size - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = resized
    return canvas, r, ox, oy


def nms(boxes, scores, thr):
    idx = np.argsort(scores)[::-1]
    keep = []
    while len(idx) > 0:
        i = idx[0]
        keep.append(int(i))
        if len(idx) == 1:
            break
        rest = idx[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        iw = np.maximum(0.0, xx2 - xx1)
        ih = np.maximum(0.0, yy2 - yy1)
        inter = iw * ih
        a1 = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        a2 = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / (a1 + a2 - inter + 1e-9)
        idx = rest[iou <= thr]
    return keep


def decode(out, conf_th, size):
    if out is None:
        return []
    a = np.asarray(out)
    if a.size == 0 or a.ndim == 0:
        return []
    while a.ndim > 2:
        a = a[0]
    pred = a
    conf = pred[4]
    idx = np.where(conf > conf_th)[0]
    if len(idx) == 0:
        return []
    cx, cy, w, h = pred[0, idx], pred[1, idx], pred[2, idx], pred[3, idx]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    scores = conf[idx]
    kps = pred[5:].reshape(17, 3, -1)[:, :, idx].transpose(2, 0, 1)
    keep = nms(boxes, scores, NMS_THRESH)
    return [(boxes[i], float(scores[i]), kps[i]) for i in keep]


def draw(img, box, score, kpts, r, ox, oy):
    x1 = int((box[0] - ox) / r); y1 = int((box[1] - oy) / r)
    x2 = int((box[2] - ox) / r); y2 = int((box[3] - oy) / r)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, 'person %.2f' % score, (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
    pts = kpts.copy()
    pts[:, 0] = (pts[:, 0] - ox) / r
    pts[:, 1] = (pts[:, 1] - oy) / r
    for p in pts:
        if p[2] > 0.5 and p[0] > 0 and p[1] > 0:
            cv2.circle(img, (int(p[0]), int(p[1])), 4, (0, 0, 255), -1)
    for a, b in SKELETON:
        pa, pb = pts[a - 1], pts[b - 1]
        if pa[2] > 0.5 and pb[2] > 0.5 and pa[0] > 0 and pa[1] > 0 \
                and pb[0] > 0 and pb[1] > 0:
            cv2.line(img, (int(pa[0]), int(pa[1])),
                     (int(pb[0]), int(pb[1])), (255, 128, 0), 2)
    return pts

POSE_RKNN = '/root/rk_lerobot_tools/model/yolov8_pose_fp.rknn'
PALM_ONNX = '/root/gesture/palm_detection_mediapipe_2023feb.onnx'
HAND_ONNX = '/root/gesture/handpose_estimation_mediapipe_2023feb.onnx'

SIZE = 640
CONF = 0.5
MIN_BOX = 0.06
WAVE_WINDOW = 2.0
WAVE_AMP = 0.12
WAVE_RAISE = 0.25

# 手部骨架连线（21 点）
HAND_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),
)

LS, RS, LW, RW = 5, 6, 9, 10     # COCO：左肩 / 右肩 / 左腕 / 右腕

_latest = {'jpg': None, 'lock': threading.Lock(), 'stat': {}}


def detect_hand(palm_det, handpose, frame):
    """返回 (手势名, 归一化x, 置信, 详情, 21点归一化坐标 or None)"""
    try:
        palms = palm_det.infer(frame)
    except Exception:
        return 'none', None, 0.0, {}, None
    if palms is None or len(palms) == 0:
        return 'none', None, 0.0, {}, None

    H, W = frame.shape[:2]
    best = None
    for p in palms:
        try:
            r = handpose.infer(frame, p)
        except Exception:
            continue
        if r is None or len(r) < 132:
            continue
        conf = float(r[131])
        if conf < 0.8:
            continue
        lms = np.array(r[4:67], dtype=np.float32).reshape(21, 3)
        norm = np.stack([lms[:, 0] / max(1, W), lms[:, 1] / max(1, H)], axis=1)
        gname, detail = hand_gesture.classify(norm)
        if best is None or conf > best[2]:
            best = (gname, float(hand_gesture.palm_center_x(norm)), conf, detail, norm)
    if best is None:
        return 'none', None, 0.0, {}, None
    return best


def draw_hand(frame, norm, color=(0, 255, 0)):
    H, W = frame.shape[:2]
    pts = [(int(x * W), int(y * H)) for x, y in norm]
    for a, b in HAND_EDGES:
        cv2.line(frame, pts[a], pts[b], color, 2, cv2.LINE_AA)
    for (x, y) in pts:
        cv2.circle(frame, (x, y), 3, (0, 200, 255), -1, cv2.LINE_AA)


def put(frame, y, text, color=(255, 255, 255), scale=0.6):
    """画面上写字（带描边，保证深浅背景都看得清）"""
    cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, 1, cv2.LINE_AA)


def vision_loop(opt):
    rknn = RKNNLite()
    if rknn.load_rknn(POSE_RKNN) != 0:
        print('pose 模型加载失败', flush=True)
        return
    rknn.init_runtime()
    palm_det = MPPalmDet(PALM_ONNX, scoreThreshold=opt.palm_thr)
    handpose = MPHandPose(HAND_ONNX, confThreshold=0.8)

    cap, idx = open_camera(opt.cam)
    if cap is None:
        print('摄像头打不开', flush=True)
        return
    print('摄像头 /dev/video%d 就绪' % idx, flush=True)

    wrist_hist = collections.deque()
    fps_t0, fps_n = time.time(), 0
    fps = 0.0

    while True:
        # ---- CAM_WANT_FLAG: your_chat_bot 要拍照，先把摄像头交出去 ----
        if os.path.exists('/run/cam-want'):
            if cap is not None:
                cap.release()
                cap = None
                print('[cam] 让出摄像头', flush=True)
            time.sleep(0.1)
            continue
        if cap is None:
            cap, idx = open_camera(opt.cam)
            if cap is None:
                time.sleep(0.5)
                continue
            print('[cam] 拿回摄像头 /dev/video%d' % idx, flush=True)

        ok, frame = cap.read()
        if not ok or frame is None:
            time.sleep(0.1)
            continue
        if opt.rot:
            frame = cv2.rotate(frame, cv2.ROTATE_180)

        # ---- 人体姿态（NPU）----
        lb, ratio, ox, oy = letterbox(frame, SIZE)
        inp = np.expand_dims(lb[..., ::-1], 0)
        outs = rknn.inference(inputs=[inp], data_format='nhwc')
        dets = decode(outs, CONF, SIZE) if outs else []
        person, kpts, area = False, None, 0.0
        if dets:
            best = None
            for box, score, kp in dets:
                bw = max(0.0, box[2] - box[0])
                bh = max(0.0, box[3] - box[1])
                ar = (bw * bh) / float(SIZE * SIZE)
                if best is None or ar > best[0]:
                    best = (ar, box, score, kp)
            area, box, score, kp = best
            if area >= MIN_BOX:
                person, kpts = True, kp
            draw(frame, box, score, kp, ratio, ox, oy)

        # ---- 挥手判据 ----
        raised, x_norm, rise, sw = hand_raise_metric(kpts, raise_ratio=WAVE_RAISE)
        now = time.time()
        if raised and x_norm is not None:
            wrist_hist.append((now, x_norm))
        while wrist_hist and now - wrist_hist[0][0] > WAVE_WINDOW:
            wrist_hist.popleft()
        turns = 0
        if len(wrist_hist) >= 2:
            turns = count_reversals([p[1] for p in wrist_hist], WAVE_AMP)

        # ---- 手势 ----
        gname, gx, gconf, detail, norm = detect_hand(palm_det, handpose, frame)
        if norm is not None:
            draw_hand(frame, norm)

        # ---- 状态栏 ----
        y = 24
        put(frame, y, 'PERSON=%s  AREA=%.3f (need >= %.2f)'
            % (person, area, MIN_BOX),
            (80, 255, 120) if person else (120, 120, 255))
        y += 24
        put(frame, y, 'RAISED=%s  RISE=%.0f  SW=%.0f  TURNS=%d  (wave need TURNS>=2)'
            % (raised, rise, sw, turns),
            (80, 255, 120) if turns >= 2 else (200, 200, 200))
        y += 24
        put(frame, y, 'SW=%.0f(need>=8)  SHOULDER c=%.2f/%.2f  WRIST c=%.2f/%.2f'
            % (sw,
               (float(kpts[LS][2]) if kpts is not None else -1),
               (float(kpts[RS][2]) if kpts is not None else -1),
               (float(kpts[LW][2]) if kpts is not None else -1),
               (float(kpts[RW][2]) if kpts is not None else -1)),
            (80, 255, 120) if sw >= 8.0 else (120, 120, 255))
        y += 24
        gc = {'palm': (80, 255, 120), 'ok': (80, 200, 255)}.get(gname, (200, 200, 200))
        put(frame, y, 'GEST=%s  FING=%s  PINCH=%s  CONF=%.2f'
            % (gname, detail.get('n_extended', '-'),
               ('%.2f' % detail['pinch_r']) if 'pinch_r' in detail else '-', gconf),
            gc)
        y += 24
        put(frame, y, 'FPS=%.1f  cam=%d' % (fps, idx), (180, 180, 180))

        # 触发提示
        if gname == 'palm':
            put(frame, 160, '>> PALM: would FOLLOW (pan)', (80, 255, 120), 0.7)
        elif gname == 'ok':
            put(frame, 160, '>> OK: would NOD', (80, 200, 255), 0.7)
        if turns >= 2:
            put(frame, 188, '>> WAVE detected', (80, 255, 120), 0.7)

        ok_jpg, jpg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if ok_jpg:
            with _latest['lock']:
                _latest['jpg'] = jpg.tobytes()

        fps_n += 1
        if now - fps_t0 >= 2.0:
            fps = fps_n / (now - fps_t0)
            fps_t0, fps_n = now, 0
            print('[fps] %.1f  person=%s area=%.3f  gest=%s conf=%.2f  '
                  'sw=%.0f turns=%d' % (fps, person, area, gname, gconf, sw, turns),
                  flush=True)


PAGE = b"""<html><head><meta charset="utf-8"><title>gesture debug</title>
<style>body{background:#111;color:#eee;font-family:sans-serif;margin:16px}
img{max-width:100%;border:1px solid #444}
code{color:#8ef}</style></head><body>
<h2>Gesture / Wave live debug</h2>
<img src="/stream">
<p><b>PERSON</b> = human detected (body box &ge; 6% of frame). Hand gesture only runs when PERSON=1 in the service.</p>
<p><b>RAISED</b> = hand above shoulder &nbsp; <b>TURNS</b> = wave direction reversals in 2s (need &ge;2)</p>
<p><b>SW</b> = shoulder width in px (needs &ge;8) &nbsp; <b>SHOULDER c / WRIST c</b> = keypoint confidence (needs &gt;0.5). Wave needs SW ok + wrist c ok + RISE &gt; 0.25&times;SW.</p>
<p><b>GEST</b> = palm / ok / fist / point &nbsp; <b>FING</b> = extended fingers &nbsp; <b>PINCH</b> = thumb-index gap / palm size (small = OK)</p>
</body></html>"""


class Handler(object):
    pass


def serve(port):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path.startswith('/stream'):
                self.send_response(200)
                self.send_header('Age', '0')
                self.send_header('Cache-Control', 'no-cache, private')
                self.send_header('Pragma', 'no-cache')
                self.send_header('Content-Type',
                                 'multipart/x-mixed-replace; boundary=--jpgboundary')
                self.end_headers()
                try:
                    while True:
                        with _latest['lock']:
                            jpg = _latest['jpg']
                        if jpg:
                            self.wfile.write(b'--jpgboundary\r\n')
                            self.wfile.write(b'Content-Type: image/jpeg\r\n')
                            self.wfile.write(('Content-Length: %d\r\n\r\n'
                                              % len(jpg)).encode())
                            self.wfile.write(jpg)
                            self.wfile.write(b'\r\n')
                        time.sleep(0.05)
                except Exception:
                    return
            else:
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.end_headers()
                self.wfile.write(PAGE)

    srv = ThreadingHTTPServer(('0.0.0.0', port), H)
    print('preview on http://0.0.0.0:%d' % port, flush=True)
    srv.serve_forever()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cam', type=int, default=11)
    ap.add_argument('--port', type=int, default=8080)
    ap.add_argument('--rot', action='store_true', help='画面倒 180 度')
    ap.add_argument('--palm-thr', type=float, default=0.65)
    opt = ap.parse_args()

    threading.Thread(target=vision_loop, args=(opt,), daemon=True).start()
    serve(opt.port)


if __name__ == '__main__':
    main()
