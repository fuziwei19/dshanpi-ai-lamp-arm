#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
yolov8pose_head_follow_so101.py  (DshanPi-A1 / RK3576 版)

基于 100ask 教程《TuyaDeskLamp/LeRobot》【拓展】YOLOV8-POSE实现跟随 一节改编：
  * 模型为 ultralytics 标准导出 + FP16 转换的单输出 (1,56,N)，非官方 4 输出
    —— 因此后处理改为标准 YOLOv8-Pose 单输出解码
  * 标定 id 使用 my_awesome_follower_arm
  * 无显示器环境用 --debug-every 周期性存帧代替 --show

用法:
    python3 yolov8pose_head_follow_so101.py                # 默认 headless
    python3 yolov8pose_head_follow_so101.py --show         # 有桌面时显示预览
    默认对准「头部关键点(鼻子/双眼)」，--aim body 可退回人体框中心

抖动抑制（重要）:
    默认 --band 0.25：对准点只要落在画面中心 ±25% 的「允许带」内就完全不动，
    不追求精确压住十字 -> 人脸在画面里就够了，机械臂不会来回微调。
    上下抖动偏大时按这个顺序调：
      1) --band 0.35            放松允许带（最有效）
      2) --v-gain 0.3           减小垂直增益
      3) --max-speed 12         降低最大角速度
    控制律默认 prop（以实际角度为基准的比例控制），不会积分饱和、不来回过冲；
    需要复现旧行为用 --mode accum。
    Ctrl+C 退出：机械臂平滑移动到「固定归位点(PARK)」后释放舵机力矩
"""
import time
import math
import argparse
import numpy as np
import cv2
import torch

from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.robots.so101_follower.so101_follower import SO101Follower
from rknnlite.api import RKNNLite as RKNN

CLASSES = ['person']
NMS_THRESH = 0.45
CONF_THRESH = 0.5

# ---- 跟随控制参数（与教程一致，抖动抑制见下）----
EMA_ALPHA = 0.15          # 水平方向 EMA 平滑系数
EMA_ALPHA_Y = 0.06        # 垂直方向 EMA 平滑系数（更慢 -> 抑制上下抖动）
DEAD_ZONE_X = 25          # 旧参数（accum 模式兼容用）
DEAD_ZONE_Y = 35
BAND_FRAC = 0.25          # 默认允许带：画面半宽的 25%（人脸落在此带内就完全不动）
PAN_GAIN = 0.01
SHOULDER_GAIN = 0.01
ELBOW_GAIN = 0.01
ELBOW_V_SHARE = 0.35      # 垂直方向 elbow 的分担比例（越小「上下摆幅」越小）

MISS_TO_SEARCH = 8
HIT_TO_CONFIRM_TRACK = 3
LOST_TIMEOUT = 3.0
SEARCH_PAN_RANGE = 45.0
SEARCH_STEP_DEG = 2.0

HOME_POSE_DEG = {
    'shoulder_pan.pos': 0.0,
    'shoulder_lift.pos': 30.0,
    'elbow_flex.pos': -65.0,
    'wrist_flex.pos': 50.0,
    'wrist_roll.pos': 20.0,
    'gripper.pos': 0.0,
}
ZERO_POSE_DEG = {
    'shoulder_pan.pos': -10.0,
    'shoulder_lift.pos': -5.0,
    'elbow_flex.pos': -5.0,
    'wrist_flex.pos': 70.0,
    'wrist_roll.pos': 10.0,
    'gripper.pos': 0.0,
}

# ---- 退出固定归位点（PARK）----
# 机械臂「收纳 / 0 点」姿态：松手后自然折叠落到的那组角度
# （LeRobot 单位，与脚本 action 空间一致；多次实测基本一致 = 稳定重力平衡点）
# Ctrl+C / 程序结束时平滑移动到这里，然后释放舵机力矩。
# 顺序: pan, lift, elbow, wrist_flex, wrist_roll, gripper
PARK_POSE_DEG = {
    'shoulder_pan.pos': -2.33,
    'shoulder_lift.pos': -101.76,
    'elbow_flex.pos': 93.89,
    'wrist_flex.pos': 73.41,
    'wrist_roll.pos': 19.03,
    'gripper.pos': 0.4725,
}
PARK_ORDER = ['shoulder_pan.pos', 'shoulder_lift.pos', 'elbow_flex.pos',
              'wrist_flex.pos', 'wrist_roll.pos', 'gripper.pos']
EXIT_MOVE_DURATION = 4.0   # 退出时移动到归位点的耗时（秒），越慢越安全

SKELETON = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13],
            [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
            [2, 4], [3, 5], [4, 6], [5, 7]]

# 画面旋转修正表：key = 逆时针角度（用于矫正摄像头安装方向）
ROT_MAP = {
    90: cv2.ROTATE_90_COUNTERCLOCKWISE,   # 逆时针90°（矫正被顺时针装反90°的摄像头）
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_CLOCKWISE,
}


class SlewLimiter:
    """关节速度限幅器（安全保护）

    限制所有关节每秒最大角度变化量，避免「画面边缘出现目标 → 一帧跳几度 →
    机械臂猛地甩过去」造成伤人/撞物。0 表示不限速。
    """

    def __init__(self, max_deg_per_sec):
        self.max = float(max_deg_per_sec)
        self.last = None
        self.t = None

    def reset(self, action):
        self.last = action.clone()
        self.t = None

    def apply(self, target, now):
        if self.max <= 0:
            return target
        if self.last is None:
            self.last = target.clone()
            self.t = now
            return target
        # reset 后首帧 dt=0 -> 保持在当前位置，随后按限速逐步逼近
        dt = (now - self.t) if self.t is not None else 0.0
        dt = max(dt, 0.0)
        max_delta = self.max * dt
        delta = torch.clamp(target - self.last, -max_delta, max_delta)
        out = self.last + delta
        self.last = out.clone()
        self.t = now
        return out


class MjpegServer:
    """极简 MJPEG 串流服务器：浏览器打开 http://<板子IP>:<port> 即可实时看画面"""

    def __init__(self, port):
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        outer = self
        outer.frame = None

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path != '/':
                    self.send_response(404); self.end_headers(); return
                self.send_response(200)
                self.send_header('Content-Type',
                                 'multipart/x-mixed-replace; boundary=frame')
                self.end_headers()
                try:
                    while True:
                        f = outer.frame
                        if f is not None:
                            ok, buf = cv2.imencode('.jpg', f,
                                                   [cv2.IMWRITE_JPEG_QUALITY, 80])
                            if ok:
                                data = buf.tobytes()
                                self.wfile.write(
                                    b'--frame\r\nContent-Type: image/jpeg\r\n'
                                    b'Content-Length: ' + str(len(data)).encode() +
                                    b'\r\n\r\n' + data + b'\r\n')
                        time.sleep(0.04)
                except Exception:
                    pass

            def log_message(self, *a):
                pass

        self.httpd = ThreadingHTTPServer(('0.0.0.0', port), Handler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def update(self, img):
        self.frame = img


def open_camera(prefer, w=640, h=480):
    """打开摄像头：优先用 prefer；不可用则按 sysfs 名称自动找 USB 摄像头节点。

    解决「摄像头重新插拔后设备号变化（video11 -> video12）」导致脚本必须
    手动改 --cam-id 的问题。返回 (cap, index) 或 (None, None)。
    """
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
    print('/dev/video%d 不可用, 按设备名扫描到候选节点: %s' % (prefer, cands))
    for k in cands:
        if k == prefer:
            continue
        c, i = try_open(k)
        if c is not None:
            print('自动选中摄像头 /dev/video%d' % i)
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
    """单输出 (56, N) -> [(box(x1,y1,x2,y2) in size-scale, score, kpts(17,3)), ...]"""
    if out is None:
        return []
    a = np.asarray(out)
    if a.size == 0 or a.ndim == 0:
        # NPU 推理被 Ctrl+C 打断时可能返回空结果
        return []
    while a.ndim > 2:
        a = a[0]
    pred = a                                     # (56, N)
    conf = pred[4]
    idx = np.where(conf > conf_th)[0]
    if len(idx) == 0:
        return []
    cx, cy, w, h = pred[0, idx], pred[1, idx], pred[2, idx], pred[3, idx]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    scores = conf[idx]
    kps = pred[5:].reshape(17, 3, -1)[:, :, idx].transpose(2, 0, 1)  # (K,17,3)
    keep = nms(boxes, scores, NMS_THRESH)
    res = []
    for i in keep:
        res.append((boxes[i], float(scores[i]), kps[i]))
    return res


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
        if pa[2] > 0.5 and pb[2] > 0.5 and pa[0] > 0 and pa[1] > 0 and pb[0] > 0 and pb[1] > 0:
            cv2.line(img, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), (255, 128, 0), 2)
    return pts


def pick_aim_point(box, kpts, thr=0.5):
    """挑出「要对准的画面点」——优先头部，让人脸落到画面中心。

    YOLOv8-Pose 的 17 个 COCO 关键点：
      0=nose, 1=left_eye, 2=right_eye, 3=left_ear, 4=right_ear,
      5=left_shoulder, 6=right_shoulder, ...

    优先级：鼻子 > 双眼中点 > 单眼 > 双肩中点(上抬到头部高度) > 人体框上部。
    返回 (x, y, 来源名)，坐标在 letterbox（模型）坐标系。
    """
    def ok(i):
        # 置信度够 且 坐标有效（模型对「没检出的关键点」常输出 (0,0,低conf)）
        return kpts[i][2] > thr and kpts[i][0] > 0 and kpts[i][1] > 0

    if ok(0):
        return float(kpts[0][0]), float(kpts[0][1]), 'nose'
    if ok(1) and ok(2):
        ex_ = float(kpts[1][0] + kpts[2][0]) / 2.0
        ey_ = float(kpts[1][1] + kpts[2][1]) / 2.0
        return ex_, ey_, 'eyes'
    if ok(1):
        return float(kpts[1][0]), float(kpts[1][1]), 'eye'
    if ok(2):
        return float(kpts[2][0]), float(kpts[2][1]), 'eye'
    if ok(5) and ok(6):
        # 只有肩膀时：用肩中点的水平位置 + 人体框上部 12% 处近似头部
        x = (kpts[5][0] + kpts[6][0]) / 2
        y = box[1] + 0.12 * (box[3] - box[1])
        return x, y, 'shoulders'
    # 兜底：人体框上部
    return (box[0] + box[2]) / 2, box[1] + 0.15 * (box[3] - box[1]), 'box'


def send_action_to_robot(robot, action_vec):
    """action_vec: torch.Tensor [1, D] -> dict 按 action_features 顺序发送"""
    names = list(robot.action_features.keys())
    arr = action_vec.detach().cpu().numpy().reshape(-1)
    n = min(len(names), len(arr))
    act = {names[i]: float(arr[i]) for i in range(n)}
    robot.send_action(act)


def build_pose(deg_map, names):
    v = torch.zeros(1, len(names), dtype=torch.float32)
    for i, n in enumerate(names):
        v[0, i] = float(deg_map.get(n, 0.0))
    return v


def read_pose_vec(robot, names):
    """读回机械臂「此刻的实际角度」-> tensor[1,D]（比例控制用基准，避免积分饱和）"""
    obs = robot.get_observation()
    v = torch.zeros(1, len(names), dtype=torch.float32)
    for i, n in enumerate(names):
        v[0, i] = float(obs.get(n, 0.0))
    return v


def follow_delta(ex_f, ey_f, band_x, band_y, pan_gain, shoulder_gain, elbow_gain,
                 sx=1.0, sy=1.0):
    """跟随控制律（纯函数，便于离线仿真验证）

    只对「越过允许带」的那部分误差产生增量：带内返回全 0（完全不动）。
    刚出带时增量≈0，越远越大 -> 出入口平滑，不会在带边顿一下。
    返回 (d_pan, d_lift, d_elbow)，单位：度。
    """
    d_pan = d_lift = d_elbow = 0.0
    if abs(ex_f) > band_x:
        d_pan = sx * math.copysign(abs(ex_f) - band_x, ex_f) * pan_gain
    if abs(ey_f) > band_y:
        s = sy * math.copysign(abs(ey_f) - band_y, ey_f)
        d_lift = s * shoulder_gain
        d_elbow = s * elbow_gain
    return d_pan, d_lift, d_elbow


def smooth_goto(robot, names, target, duration=2.0, steps=50):
    obs = robot.get_observation()
    cur = torch.zeros(1, len(names), dtype=torch.float32)
    for i, n in enumerate(names):
        cur[0, i] = float(obs.get(n, 0.0))
    for k in range(steps):
        a = (k + 1) / steps
        send_action_to_robot(robot, (1 - a) * cur + a * target)
        time.sleep(duration / steps)
    return (1 - a) * cur + a * target if steps else target


def exit_routine(robot, names, park_pose, exit_pose='park', hold=False,
                 duration=EXIT_MOVE_DURATION):
    """统一的退出处理：读当前姿态 → 移动到目标姿态 → 释放/保持力矩。

    同时供「主循环结束」和「回 Home 途中被打断」两条路径调用，
    保证任何退出方式下机械臂都会被妥善收起。
    """
    try:
        obs = robot.get_observation()
        cur = torch.zeros(1, len(names), dtype=torch.float32)
        for i, n in enumerate(names):
            cur[0, i] = float(obs.get(n, 0.0))
        print('退出时实际姿态(度):',
              {n: round(float(cur[0, i]), 2) for i, n in enumerate(names)})

        if exit_pose == 'current':
            final = cur                      # 原地保持：不产生任何多余动作
        elif exit_pose == 'home':
            final = build_pose(HOME_POSE_DEG, names)
        elif exit_pose == 'zero':
            final = build_pose(ZERO_POSE_DEG, names)
        else:                                # park：平滑移动到固定归位点
            final = build_pose(park_pose, names)

        if exit_pose == 'current':
            send_action_to_robot(robot, final)        # 只重申一次当前姿态
        else:
            print('正在平滑移动到%s点（%.1f 秒）...' %
                  ('固定归位' if exit_pose == 'park' else '归零', duration))
            smooth_goto(robot, names, final, duration=duration, steps=60)
            obs2 = robot.get_observation()            # 停稳后确认实际到位
            print('到位后实际姿态:',
                  {n: round(float(obs2.get(n, 0.0)), 2) for n in names})

        # 关串口：默认释放力矩（机械臂可自由摆动）；hold 则保留力矩锁定姿态
        if hold:
            robot.bus.disconnect(disable_torque=False)
            print('已锁定在归位点（舵机保持力矩）。'
                  '如需手动摆臂，运行: python /root/release_arm.py')
        else:
            robot.bus.disconnect(disable_torque=True)
            print('力矩已释放，机械臂现在可用手自由摆动。')
    except Exception as e:
        print('退出处理失败:', e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_path', default='/root/rk_lerobot_tools/model/yolov8_pose_fp.rknn')
    ap.add_argument('--cam-id', type=int, default=12,
                    help='摄像头节点号(默认12)。若该节点不可用会自动按设备名扫描其他节点')
    ap.add_argument('--size', type=int, default=320, help='模型输入边长')
    ap.add_argument('--fps', type=int, default=30)
    ap.add_argument('--port', default='/dev/ttyACM0')
    ap.add_argument('--robot-id', default='my_awesome_follower_arm')
    ap.add_argument('--rot', type=int, default=90, choices=[0, 90, 180, 270],
                    help='画面旋转修正（逆时针角度）：90=矫正顺时针装反90°的摄像头，0=不旋转')
    ap.add_argument('--band', type=float, default=BAND_FRAC, metavar='FRAC',
                    help='允许带（相对画面尺寸的比例，默认0.25）：对准点落在画面中心 ±FRAC 范围内就'
                         '完全不动，不追求精确对准十字。这是抑制抖动最有效的开关，调大更稳')
    ap.add_argument('--band-x', type=float, default=0.0, metavar='PX',
                    help='直接用像素指定水平允许带半宽（0=按 --band 自动算）')
    ap.add_argument('--band-y', type=float, default=0.0, metavar='PX',
                    help='直接用像素指定垂直允许带半高（0=按 --band 自动算），上下抖动大就调大它')
    ap.add_argument('--v-gain', type=float, default=0.5, metavar='K',
                    help='垂直（俯仰）额外增益，越小上下动作越轻柔（默认0.5）')
    ap.add_argument('--mode', choices=['prop', 'accum'], default='prop',
                    help='控制律：prop=比例控制(默认，以实际角度为基准，误差进带即停，'
                         '不会积分饱和->不来回过冲)；accum=旧版累加式(仅在需要复现旧行为时用)')
    ap.add_argument('--max-speed', type=float, default=20.0, metavar='DEG_S',
                    help='关节最大角速度(度/秒)，安全限速；越小越慢越安全，0=不限速（默认20）')
    ap.add_argument('--gain', type=float, default=0.6,
                    help='跟随增益系数(0~1)，越小反应越柔和（默认0.6）')
    ap.add_argument('--aim', choices=['head', 'body'], default='head',
                    help='对准目标：head=头部关键点(鼻子>双眼>肩膀)让人脸居中(默认)；'
                         'body=整个人体框中心（旧行为）')
    ap.add_argument('--invert-x', action='store_true',
                    help='反转水平跟随方向（若发现目标越追越偏就加这个）')
    ap.add_argument('--invert-y', action='store_true',
                    help='反转俯仰跟随方向（若发现人脸越追越跑偏就加这个）')
    ap.add_argument('--search-step', type=float, default=1.0, metavar='DEG',
                    help='搜索模式下每帧扫描角度（默认1.0，越大扫得越快）')
    ap.add_argument('--show', action='store_true')
    ap.add_argument('--http', type=int, default=0, metavar='PORT',
                    help='浏览器实时串流端口，如 --http 8080')
    ap.add_argument('--dry-run', action='store_true', help='只跑视觉不控机械臂')
    ap.add_argument('--diag', action='store_true', help='打印每轮各阶段耗时（性能排查用）')
    ap.add_argument('--exit-pose', choices=['park', 'current', 'zero', 'home'], default='park',
                    help='退出(Ctrl+C/异常)时停到哪：park=平滑移动到固定归位点(默认)；'
                         'current=原地保持不动；zero/home=平滑回到固定归零/Home 姿态')
    ap.add_argument('--park', default=None, metavar='P1,P2,P3,P4,P5,P6',
                    help='自定义固定归位点，6 个角度用逗号分隔，顺序 '
                         'pan,lift,elbow,wrist_flex,wrist_roll,gripper（默认用内置实测值）')
    ap.add_argument('--hold', action='store_true',
                    help='退出时保持舵机力矩锁定姿态（默认：到位后释放力矩，机械臂可用手自由摆动）')
    ap.add_argument('--release', action='store_true',
                    help='（兼容旧参数）等价于默认行为：退出时释放力矩')
    ap.add_argument('--debug-every', type=float, default=3.0, help='每 N 秒存一张调试图到 /tmp/follow_debug.jpg')
    opt = ap.parse_args()

    # 固定归位点：默认用内置实测值，--park 可覆盖
    park_pose = dict(PARK_POSE_DEG)
    if opt.park:
        vals = [float(x) for x in opt.park.replace(' ', '').split(',')]
        assert len(vals) == 6, '--park 需要 6 个数值: pan,lift,elbow,wrist_flex,wrist_roll,gripper'
        park_pose = dict(zip(PARK_ORDER, vals))
    print('退出归位点(PARK):', {k: round(v, 2) for k, v in park_pose.items()})
    hold_flag = bool(opt.hold and not opt.release)   # 是否保持力矩（默认释放）

    print('加载模型:', opt.model_path)
    rknn = RKNN(verbose=False)
    assert rknn.load_rknn(opt.model_path) == 0, 'load_rknn failed'
    assert rknn.init_runtime() == 0, 'init_runtime failed'
    print('模型就绪')

    cap, cam_idx = open_camera(opt.cam_id)
    assert cap is not None, '找不到可用摄像头（试过 /dev/video%d 及自动扫描）' % opt.cam_id
    opt.cam_id = cam_idx
    print('摄像头 /dev/video%d 就绪' % cam_idx)

    srv = None
    if opt.http:
        srv = MjpegServer(opt.http)
        print('串流已开启: http://0.0.0.0:%d  (浏览器打开 http://<板子IP>:%d)' %
              (opt.http, opt.http))

    robot = None
    names = []
    target_action = None
    try:
        if not opt.dry_run:
            cfg = SO101FollowerConfig(port=opt.port, id=opt.robot_id, cameras={}, use_degrees=True)
            robot = SO101Follower(cfg)
            robot.connect()
            names = list(robot.action_features.keys())
            print('机械臂已连接, joints =', names)
            target_action = smooth_goto(robot, names, build_pose(HOME_POSE_DEG, names))
            print('已回 Home 姿态')
        else:
            print('[DRY-RUN] 不连接机械臂')
    except KeyboardInterrupt:
        # 回 Home 途中被打断：也要归位 + 释放，不能把机械臂丢在半路
        print('\n回 Home 途中收到 Ctrl+C，先归位再退出...')
        if robot is not None:
            exit_routine(robot, names, park_pose, 'park', hold_flag)
        cap.release()
        cv2.destroyAllWindows()
        rknn.release()
        return

    limiter = SlewLimiter(opt.max_speed) if robot is not None else None
    if limiter is not None:
        limiter.reset(target_action)
        print('安全限速: %.0f 度/秒   跟随增益: %.2f   垂直增益: %.2f   控制律: %s' %
              (opt.max_speed, opt.gain, opt.v_gain, opt.mode))

    idx = {n: i for i, n in enumerate(names)}
    pan_center = HOME_POSE_DEG['shoulder_pan.pos']
    shoulder_center = HOME_POSE_DEG['shoulder_lift.pos']
    elbow_center = HOME_POSE_DEG['elbow_flex.pos']

    period = 1.0 / opt.fps
    miss = hit = 0
    cam_fail = 0
    last_hit = time.time()
    search_mode = False
    search_dir, search_pan = 1.0, pan_center
    ex_f = ey_f = 0.0
    last_debug = 0.0
    t_obs = 0.0
    band_printed = False
    fps_t0, fps_n = time.time(), 0

    try:
        while True:
            t0 = time.time()
            ok, frame = cap.read()
            t_rd = time.time()
            if not ok:
                cam_fail += 1
                if cam_fail % 20 == 1:
                    print('摄像头读帧失败 (连续 %d 次)' % cam_fail)
                if cam_fail >= 60:
                    print('摄像头连续读帧失败，程序退出')
                    break
                time.sleep(0.2)
                continue
            cam_fail = 0
            if opt.rot:
                frame = cv2.rotate(frame, ROT_MAP[opt.rot])
            h, w = frame.shape[:2]
            # 允许带（像素）：带内完全不动 -> 人脸只要落在带里就算「已对准」
            band_x = opt.band_x if opt.band_x > 0 else opt.band * w
            band_y = opt.band_y if opt.band_y > 0 else opt.band * h
            if not band_printed:
                print('画面 %dx%d（--rot %d 后）  允许带(±像素): 水平 %.0f  垂直 %.0f'
                      % (w, h, opt.rot, band_x, band_y))
                print('  -> 对准点落在此带内机械臂完全不动（--band/--band-x/--band-y 可调，调大更稳）')
                band_printed = True

            lb, r, ox, oy = letterbox(frame, opt.size)
            inp = np.expand_dims(lb[..., ::-1], 0)
            outs = rknn.inference(inputs=[inp], data_format='nhwc')
            t_npu = time.time()
            if outs is None:
                # Ctrl+C 打断 NPU 推理时 rknnlite 会吞掉中断并返回空值，
                # 这里主动退出循环，走统一的归位流程（避免要按两次 Ctrl+C）
                print('推理被中断，退出...')
                break
            dets = decode(outs, CONF_THRESH, opt.size)

            if dets:
                box, score, kpts = max(dets, key=lambda d: (d[0][2] - d[0][0]) * (d[0][3] - d[0][1]))
                pts = draw(frame, box, score, kpts, r, ox, oy)

                # —— 对准点：默认头部（鼻子/双眼），--aim body 时退回人体框中心 ——
                if opt.aim == 'head':
                    try:
                        ax, ay, aim_src = pick_aim_point(box, kpts)
                    except Exception as e:
                        # 单帧关键点异常不应该终结整个跟随流程：退回人体框上部
                        ax = (box[0] + box[2]) / 2
                        ay = box[1] + 0.15 * (box[3] - box[1])
                        aim_src = 'box(fallback)'
                        print('取头部点失败(%s)，本帧退回框中心' % e)
                else:
                    ax, ay, aim_src = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2, 'box'
                aim_px = (int((ax - ox) / r), int((ay - oy) / r))
                cv2.drawMarker(frame, aim_px, (0, 255, 255), cv2.MARKER_CROSS, 18, 2)
                cv2.putText(frame, 'AIM:' + aim_src, (aim_px[0] + 10, aim_px[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
                # 画面中心基准十字（对准点靠到它上面即「居中」）
                cv2.drawMarker(frame, (w // 2, h // 2), (255, 255, 0), cv2.MARKER_CROSS, 26, 1)
                # 允许带：对准点落在这个框里 = 已足够，机械臂完全不动
                cv2.rectangle(frame, (int(w / 2 - band_x), int(h / 2 - band_y)),
                              (int(w / 2 + band_x), int(h / 2 + band_y)), (0, 200, 0), 1)
                in_band = (abs((ax - ox) / r - w / 2) <= band_x and
                           abs((ay - oy) / r - h / 2) <= band_y)
                cv2.putText(frame, 'IN-BAND' if in_band else 'FOLLOW',
                            (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 220, 0) if in_band else (0, 140, 255), 2, cv2.LINE_AA)

                hit += 1; miss = 0; last_hit = time.time(); search_mode = False
                if hit >= HIT_TO_CONFIRM_TRACK and robot is not None:
                    cx = (ax - ox) / r
                    cy = (ay - oy) / r
                    ex = cx - w / 2
                    ey = cy - h / 2
                    ex_f = EMA_ALPHA * ex + (1 - EMA_ALPHA) * ex_f
                    ey_f = EMA_ALPHA_Y * ey + (1 - EMA_ALPHA_Y) * ey_f

                    # 基准：prop=以「此刻实际角度」为准（误差进带即停，不积分饱和）；
                    #       accum=在上一帧目标上累加（旧行为，容易过冲）
                    if opt.mode == 'prop':
                        try:
                            act = read_pose_vec(robot, names)
                            t_obs = time.time()
                        except Exception as e:
                            print('读实际角度失败(%s)，本帧退回累加基准' % e)
                            act = target_action.clone()
                    else:
                        act = target_action.clone()

                    pan_gain = PAN_GAIN * opt.gain
                    shoulder_gain = SHOULDER_GAIN * opt.gain * opt.v_gain
                    elbow_gain = ELBOW_GAIN * opt.gain * opt.v_gain * ELBOW_V_SHARE
                    sx = -1.0 if opt.invert_x else 1.0
                    sy = -1.0 if opt.invert_y else 1.0
                    d_pan, d_lift, d_elbow = follow_delta(
                        ex_f, ey_f, band_x, band_y, pan_gain, shoulder_gain, elbow_gain, sx, sy)
                    if d_pan:
                        act[0, idx['shoulder_pan.pos']] = float(np.clip(
                            act[0, idx['shoulder_pan.pos']] + d_pan, pan_center - 90, pan_center + 90))
                    if d_lift:
                        act[0, idx['shoulder_lift.pos']] = float(np.clip(
                            act[0, idx['shoulder_lift.pos']] + d_lift,
                            shoulder_center - 35, shoulder_center + 35))
                    if d_elbow:
                        act[0, idx['elbow_flex.pos']] = float(np.clip(
                            act[0, idx['elbow_flex.pos']] + d_elbow,
                            elbow_center - 35, elbow_center + 35))
                    moved = bool(d_pan or d_lift or d_elbow)
                    # 安全限速：限制每秒最大角度变化，防止机械臂猛地甩动
                    target_action = limiter.apply(act, time.time()) if limiter is not None else act
                    if moved:
                        send_action_to_robot(robot, target_action)
            else:
                hit = 0; miss += 1
                if miss > MISS_TO_SEARCH and time.time() - last_hit > LOST_TIMEOUT:
                    search_mode = True

            if search_mode and robot is not None:
                search_pan += search_dir * opt.search_step
                if search_pan > pan_center + SEARCH_PAN_RANGE:
                    search_pan = pan_center + SEARCH_PAN_RANGE; search_dir = -1.0
                elif search_pan < pan_center - SEARCH_PAN_RANGE:
                    search_pan = pan_center - SEARCH_PAN_RANGE; search_dir = 1.0
                target_action[0, idx['shoulder_pan.pos']] = float(search_pan)
                target_action = limiter.apply(target_action, time.time()) if limiter is not None else target_action
                send_action_to_robot(robot, target_action)

            if opt.show:
                cv2.imshow('pose follow', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            if srv is not None:
                srv.update(frame)
            if not opt.show and time.time() - last_debug > opt.debug_every:
                cv2.imwrite('/tmp/follow_debug.jpg', frame)
                last_debug = time.time()

            if opt.diag:
                now_d = time.time()
                obs_ms = (t_obs - t_npu) * 1000 if t_obs > t_npu else 0.0
                print('[diag] read=%.0fms npu=%.0fms 读角=%.0fms 视觉段=%.0fms 控制等其他=%.0fms 单轮=%.0fms dets=%d search=%s'
                      % ((t_rd - t0) * 1000, (t_npu - t_rd) * 1000, obs_ms,
                         (t_npu - t0) * 1000, (now_d - t_npu) * 1000,
                         (now_d - t0) * 1000, len(dets), search_mode))
                t_obs = 0.0

            fps_n += 1
            if time.time() - fps_t0 >= 5.0:
                print('FPS: %.1f  dets=%d  ex=%.1f ey=%.1f  带=(±%.0f, ±%.0f)  %s' %
                      (fps_n / (time.time() - fps_t0), len(dets), ex_f, ey_f,
                       band_x, band_y,
                       '带内(不动)' if (abs(ex_f) <= band_x and abs(ey_f) <= band_y)
                       else '带外(跟随中)'))
                fps_t0, fps_n = time.time(), 0

            dt = period - (time.time() - t0)
            if dt > 0:
                time.sleep(dt)
    except KeyboardInterrupt:
        print('\n收到 Ctrl+C，退出...')
    finally:
        if robot is not None:
            exit_routine(robot, names, park_pose, opt.exit_pose, hold_flag)
        cap.release()
        cv2.destroyAllWindows()
        rknn.release()
        print('资源已释放')


if __name__ == '__main__':
    main()
