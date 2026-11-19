#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""vision_interact.py — 视觉迎宾交互（到场问候 / 挥手告别 / 无人省电）

三个玩法
--------
1. **迎宾**：画面从「没人」变成「有人」并停留一小会儿
   → 灯环渐亮 + 流水跑一圈 + 机械臂在**基准位**上左右小幅旋转 + 让 AI 主动打招呼
2. **告别**：人在场时「把手举过肩并左右摆动」
   → 灯环闪两下 + 机械臂招手 + 目送
3. **省电**：人离开超过 N 秒
   → 灯环渐暗熄灭 + 机械臂回收纳位(PARK) + 释放力矩

基准位
------
「基准位」是 arm_pose_save.py 读下来的机械臂当前姿态，存在 /etc/greet-pose.json
的 `base_pose` 里。迎宾启动时先转到这个位置，有人靠近就在这个位置上左右小幅旋转。
文件不存在时回退到 HOME_POSE_DEG（老行为）。

让 AI 打招呼
------------
迎宾事件触发时，脚本会往 your_chat_bot 的本地 socket（/run/tuya_chatbot.sock）
写一行 {"cmd":"say","text":"打个招呼"}，app 内部调用 ai_agent_send_text()，
于是 AI 会正常思考 + TTS 说出招呼语（默认文案「打个招呼」，可用 --greet-say 改）。

分层
----
  感知  yolov8pose_head_follow_so101 的模型/解码（复用，17 点 COCO 人体姿态）
  决策  greet_logic.EventDetector（几何判据 + 时序去抖 + 状态机，纯逻辑可离线测）
  反馈  WS2812 灯环(led_client) + SO-101 机械臂动作(arm_actions) + AI 说话(notify_ai)

⚠️ 与「人脸跟随」互斥：两者都独占机械臂 /dev/ttyACM0 与摄像头，不能同时跑。

用法
----
    python3 vision_interact.py                    # 完整模式
    python3 vision_interact.py --dry-run          # 不连机械臂（只跑视觉 + 灯效）
    python3 vision_interact.py --calib            # 标定模式：只打印判据数值，不做反馈
    python3 vision_interact.py --debug-every 3    # 每 3 秒存标注图 /tmp/greet_debug.jpg
    python3 vision_interact.py --greet-say '你好呀'  # 自定义 AI 的招呼语
    python3 vision_interact.py --no-say           # 只动机械臂，不叫 AI 说话
    python3 vision_interact.py --demo-once 3      # 自检：3 秒后跑一次迎宾动作就退出
"""
import argparse
import collections
import json
import os
import time

import cv2
import numpy as np

import led_client as led
import notify_ai as ai_notify
import arm_actions as arm
from greet_logic import (EventDetector, DEFAULTS as LOGIC_DEFAULTS,
                         hand_raise_metric, count_reversals)  # noqa: F401 (便于调试导入)
from yolov8pose_head_follow_so101 import (
    ROT_MAP, HOME_POSE_DEG, PARK_POSE_DEG,
    open_camera, letterbox, decode, draw,
)
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.robots.so101_follower.so101_follower import SO101Follower
from rknnlite.api import RKNNLite as RKNN

CONF_THRESH = 0.5
CFG_PATH = '/etc/vision-greet.json'
GESTURE_MAP_PATH = '/etc/gesture-map.json'   # 「手势 -> 动作」映射（可选，改它不用改代码）
BASE_POSE_PATH = '/etc/greet-pose.json'      # 由 arm_pose_save.py 写入
# 自检钩子（见主循环注释）：假装镜头前有人，用来验证「有人」这条路径不崩
FAKE_PERSON = bool(os.environ.get('GREET_FAKE_PERSON'))
SAY_SOCK_PATH = '/run/tuya_chatbot.sock'     # your_chat_bot 的本地文本入口
CAM_WANT_FLAG = '/run/cam-want'             # your_chat_bot 拍照前 touch 它 -> 本脚本让出摄像头
MODEL_PATH = '/root/rk_lerobot_tools/model/yolov8_pose_fp.rknn'

DEFAULTS = dict(LOGIC_DEFAULTS)
DEFAULTS.update({
    'cam_id': 12,
    'rot': 90,
    'size': 320,
    'fps': 20,
    'idle_color': '#FFB45A',    # 在场常态色（暖白）
    'idle_bright': 35,
    'greet_color': '#FFD27F',
    'bye_color': '#66CCFF',
    'greet_bright': 60,
    'nod_amp': 13.0,            # 点头幅度（度，保留兼容；迎宾现在改用 pan_wiggle）
    'shake_amp': 10.0,          # 摇摇头幅度（度，只绕 shoulder_pan，其它关节不动）
    'wave_deg': 20.0,           # 招手幅度（度）
    'pan_amp': 16.0,            # 迎宾「左右小幅旋转」幅度（度）
    'pan_cycles': 2,            # 左右摆动来回次数
    'greet_say': '打个招呼',     # 迎宾时让 AI 说的话（走 ai_agent_send_text）
    # ---------- 挥手互动 ----------
    # showcase = 挥手 -> 拍张照并自我介绍（有人挥手打招呼时主动展示自己）
    # bye      = 挥手 -> 原来的「挥手告别」动作
    # both     = 两件事都做
    'wave_action': 'showcase',
    # showcase 时送给云端的完整引导语（走 app_vision_request_prompt，不套模板）。
    # 之所以要「完整指定」：默认的自动模板只会说「用一到两句自然口语回答他」，
    # 讲不出一段自我介绍。
    'wave_prompt': ('有人正在向你挥手打招呼。请你结合刚拍到的画面，'
                    '热情地回应一声，然后简短介绍一下你自己：你是谁、你能做什么。'),
    # ---------- 手势（默认映射：OK -> 点头 / 手掌 -> 横向跟随）----------
    # 映射表可外部覆盖：/etc/gesture-map.json（见 GESTURE_MAP_PATH）
    # 可用动作：none / follow / stop_follow / nod / shake_head / wave / look_up / back
    #          / pan_wiggle / gripper_wiggle / greet / photo
    # 可用手势名（hand_gesture.classify 的输出）：ok / palm / fist / point / unknown
    'gesture': True,             # 是否识别手势（关掉可以省 CPU 给人体姿态）
    'gesture_pan_range': 28.0,   # 手掌跟随：左右可转的角度范围（度）
    'gesture_invert': False,     # 左右反向（摄像头画面是镜像时开）
    'gesture_ok_cooldown': 6.0,  # OK 手势 -> 点头 的冷却（秒）
    'gesture_stable': 2,         # 连续几帧判成同一手势才认（防抖）
    'gesture_lost': 1.2,         # 连续多久看不到手掌就停止跟随（秒）
    'gesture_conf': 0.8,         # 关键点置信度下限
    # 跟随目标来源：palm=只跟手掌 / face=只跟人脸(鼻子) / auto=手掌优先，没手掌就跟人脸
    # 手掌检测不稳时就切 face —— 人脸从左到右移动比手掌更好判。
    'gesture_follow_mode': 'auto',
    'gesture_face_conf': 0.5,    # 人脸(鼻子/眼)关键点置信度下限
    # 手部检测是否「不管有没有人都跑」：
    #   人站在台灯前挥手时手离镜头很近，身体会被挡在画面外，
    #   yolov8-pose 检不到人（person=False）→ 若只在 person 时才跑手部模型，
    #   就会出现「手明明在镜头前却什么都认不出」。
    #   代价：约 20fps → 14~15fps；误检由 hand_big_size 卡住。
    'hand_always': True,
    'hand_big_size': 0.06,       # 手掌尺寸(腕→中指根, 归一化)下限才算「真有只手在镜头前」
    'hand_palm_score': 0.85,     # 手掌检测器置信度下限（21 点失败时用 bbox 兜底）
    # 挥手响应的动作形式：pan=转灯头(5号舵机 shoulder_pan) / wrist=摆腕 / both
    'wave_motion': 'both',
    'wave_pan_amp': 18.0,        # 挥手时灯头左右转的幅度（度）
    'wave_pan_cycles': 2,
    # 调试
    'gesture_debug': False,      # 逐帧写 /tmp/gesture_dbg.jsonl（用于事后分析）
    'preview_port': 0,           # >0 = 开 MJPEG 预览（浏览器实时看画面+识别结果）
    'say_enabled': True,        # 是否叫 AI 说话
    'say_sock': SAY_SOCK_PATH,  # your_chat_bot 的本地文本入口
    'debug_every': 0.0,
})


# ===========================================================================
# 机械臂控制（含"省电松手 → 下次自动使能"）
# ===========================================================================

class ControlServer(object):
    """本机控制口 `/run/vision-greet.sock` —— 让「已经在跑」的迎宾服务顺手做个动作

    为什么需要它：迎宾跑着的时候串口是它独占的，外面的脚本连不进去；
    只为了点个头就去 stop/start 服务（约 10 秒，而且灯头要先收起再抬起）太蠢。

    协议：一行一个词，回一行文字。
        nod / shake_head / wave / look_up / back / pan_wiggle / state

    动作会排进主循环的动作队列（ActionRunner），和跟随/迎宾共用同一条队，
    所以不会出现两个线程抢机械臂的情况。
    """

    def __init__(self, path, actions, submit, state_fn):
        self.path = path
        self.actions = dict(actions)
        self.submit = submit
        self.state_fn = state_fn
        self._sock = None

    def start(self):
        import socket
        import threading
        try:
            if os.path.exists(self.path):
                os.unlink(self.path)
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.bind(self.path)
            try:
                os.chmod(self.path, 0o666)
            except Exception:
                pass
            s.listen(4)
            self._sock = s
        except Exception as e:
            print('[conctl] 控制口启动失败: %r' % e, flush=True)
            self._sock = None
            return
        threading.Thread(target=self._loop, daemon=True).start()
        print('[conctl] 控制口就绪 %s（可用：%s）'
              % (self.path, '/'.join(sorted(self.actions))), flush=True)

    def close(self):
        try:
            if self._sock is not None:
                self._sock.close()
        except Exception:
            pass
        try:
            if os.path.exists(self.path):
                os.unlink(self.path)
        except Exception:
            pass

    def _dispatch(self, word):
        if word in ('state', 'status'):
            try:
                return self.state_fn()
            except Exception as e:
                return 'error:%r' % e
        fn = self.actions.get(word)
        if fn is None:
            return 'unknown:%s（可用：%s）' % (word, '/'.join(sorted(self.actions)))
        return ('ok:%s' % word) if self.submit(word, fn) else 'busy:机械臂正忙，等一下再试'

    def _loop(self):
        while True:
            try:
                conn, _ = self._sock.accept()
            except Exception:
                return
            try:
                raw = conn.recv(64).decode('utf-8', 'replace').strip()
                word = raw.split()[0] if raw.split() else ''
                conn.sendall((self._dispatch(word) + '\n').encode('utf-8'))
            except Exception as e:
                try:
                    conn.sendall(('error:%r\n' % e).encode('utf-8'))
                except Exception:
                    pass
            finally:
                try:
                    conn.close()
                except Exception:
                    pass


# ---------------------------------------------------------------- 机械臂串口
# CH343（1a86:55d3）每次断电重连都会重新枚举：ttyACM0 -> ttyACM1 -> ……，编号会漂移。
# 开机时表现为「Missing motor IDs ... found {}」，看着像舵机没供电，其实是找错了节点。
# 所以优先用 /dev/serial/by-id 的稳定路径，找不到才退回编号最小的 ttyACM。
ARM_BY_ID = '/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AB0179621-if00'


def resolve_arm_port(preferred='auto'):
    """把 --port 解析成真实设备路径（显式给了存在的路径就尊重它）"""
    import glob
    if preferred and preferred != 'auto' and os.path.exists(preferred):
        return preferred
    if os.path.exists(ARM_BY_ID):
        return ARM_BY_ID
    cands = []
    for p in glob.glob('/dev/ttyACM*'):
        try:
            cands.append((int(p.replace('/dev/ttyACM', '')), p))
        except ValueError:
            pass
    if cands:
        return sorted(cands)[0][1]
    return preferred if (preferred and preferred != 'auto') else '/dev/ttyACM0'


class ArmCtl:
    def __init__(self, port, robot_id, enabled=True):
        self.port = port
        self.robot_id = robot_id
        self.enabled = enabled
        self.robot = None
        self.names = []
        self.connected = False

    def connect(self):
        if not self.enabled:
            return
        cfg = SO101FollowerConfig(port=self.port, id=self.robot_id,
                                  cameras={}, use_degrees=True)
        self.robot = SO101Follower(cfg)
        self.robot.connect()
        self.names = list(self.robot.action_features.keys())
        self.connected = True
        print('[arm] 已连接 %s  joints=%s' % (self.port, self.names), flush=True)

    def release(self):
        """松力矩 + 关串口（省电：之后可用手自由摆动机械臂）"""
        if self.robot is not None and self.connected:
            try:
                self.robot.bus.disconnect(disable_torque=True)
                print('[arm] 已释放力矩', flush=True)
            except Exception as e:
                print('[arm] 释放力矩失败: %r' % e, flush=True)
            self.connected = False

    def ensure(self):
        """确保力矩开启（省电松手之后、下次动作前调用）"""
        if not self.enabled:
            return
        if self.robot is None:
            self.connect()
        elif not self.connected:
            try:
                self.robot.bus.connect()
                self.connected = True
                print('[arm] 已重新使能舵机', flush=True)
            except Exception as e:
                print('[arm] 重新使能失败(%r)，尝试整机重连' % e, flush=True)
                try:
                    self.robot.connect()
                    self.connected = True
                    print('[arm] 整机重连成功', flush=True)
                except Exception as e2:
                    print('[arm] 整机重连也失败: %r' % e2, flush=True)

    def finish(self, park=True):
        """进程退出：回收纳位 + 松力矩"""
        if self.robot is None or not self.enabled:
            return
        try:
            self.ensure()
            if park and self.connected:
                print('[arm] 退出：平滑回收纳位...', flush=True)
                arm.goto_park(self.robot, self.names, duration=3.0)
            self.release()
        except Exception as e:
            print('[arm] 退出处理失败: %r' % e, flush=True)


# ===========================================================================
# 动作执行线程（机械臂动作是阻塞的，放独立线程，不卡视觉循环）
# ===========================================================================

class ActionRunner(object):
    def __init__(self):
        import threading
        import queue
        self._threading = threading
        self.q = queue.Queue(maxsize=1)
        self.busy = threading.Event()
        self.stop_evt = threading.Event()
        self.count = 0
        self._t = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self._t.start()

    def submit(self, name, fn):
        if self.q.full():
            print('[act] 队列忙，丢弃事件 %s' % name, flush=True)
            return False
        self.q.put((name, fn))
        return True

    def _loop(self):
        import queue
        while not self.stop_evt.is_set():
            try:
                name, fn = self.q.get(timeout=0.2)
            except queue.Empty:
                continue
            self.busy.set()
            t0 = time.time()
            print('[act] 执行 %s' % name, flush=True)
            try:
                fn()
                print('[act] %s 完成 (%.2fs)' % (name, time.time() - t0), flush=True)
                self.count += 1
            except Exception as e:
                print('[act] %s 异常: %r' % (name, e), flush=True)
            finally:
                self.busy.clear()

    def shutdown(self):
        self.stop_evt.set()


# ===========================================================================
# 手势：OK -> 点头 / 手掌 -> 横向跟随（只转 shoulder_pan，不调高度）
#
# 模型是 opencv_zoo 的 palm_detection + handpose（MediaPipe 版，21 关键点），
# 用 OpenCV DNN 在 CPU 上跑。之所以不用 MediaPipe 官方包：它的 wheel 编译时
# 开了 LSE 原子指令（ARMv8.2+），RK3576 是 Cortex-A72/A53（ARMv8.0）跑不了，
# 一 import 就 SIGILL。
# ===========================================================================

GESTURE_DIR = '/root/gesture'


def load_gesture_models(enable=True):
    """加载手部关键点模型。失败返回 None -> 手势功能自动关闭，不影响迎宾。"""
    if not enable:
        return None
    try:
        import sys
        if GESTURE_DIR not in sys.path:
            sys.path.insert(0, GESTURE_DIR)
        from mp_palmdet import MPPalmDet
        from mp_handpose import MPHandPose
        import hand_gesture
        gd = {
            'palm': MPPalmDet(GESTURE_DIR + '/palm_detection_mediapipe_2023feb.onnx',
                              scoreThreshold=0.65),
            'pose': MPHandPose(GESTURE_DIR + '/handpose_estimation_mediapipe_2023feb.onnx',
                               confThreshold=0.8),
            'hg': hand_gesture,
        }
        print('[gesture] 手部关键点模型就绪', flush=True)
        return gd
    except Exception as e:
        print('[gesture] 手部模型加载失败，手势功能关闭: %r' % e, flush=True)
        return None


def face_x_from_kpts(kpts, conf_thr=0.5):
    """从人体姿态关键点里取「人脸中心的归一化 x」；取不到返回 None。

    用鼻子(0) 优先，鼻子不可信时退回双眼(1,2) 中点。x 是像素坐标，
    除以图片宽度归一化到 0~1（与手掌中心的归一化方式一致，可直接喂给跟随器）。

    ⚠️ 注意：yolov8pose 的关键点坐标是「letterbox 后 320/640 尺度」的像素值，
    所以这里要除以 opt.size 而不是原图宽。
    """
    if kpts is None:
        return None
    try:
        nose = kpts[0]
        if float(nose[2]) > conf_thr:
            return float(nose[0])
        e1, e2 = kpts[1], kpts[2]
        if float(e1[2]) > conf_thr and float(e2[2]) > conf_thr:
            return (float(e1[0]) + float(e2[0])) / 2.0
    except Exception:
        return None
    return None


def load_gesture_map(path=GESTURE_MAP_PATH):
    """读「手势 -> 动作」映射（可选文件，读不到就用默认）。

    文件格式（全部可选，缺的用默认）：
        {"ok": "nod", "palm": "follow", "fist": "none", "point": "none"}
    这样想加新玩法（比如 point -> look_up）只要改 JSON，不用改代码。
    """
    default = {
        'ok': 'nod',
        'palm': 'follow',
        'fist': 'none',
        'point': 'none',
        'unknown': 'none',
    }
    try:
        with open(path, 'r', encoding='utf-8') as f:
            raw = json.load(f)
    except FileNotFoundError:
        return default
    except Exception as e:
        print('[gesture] 读 %s 失败: %r（用默认映射）' % (path, e), flush=True)
        return default
    if not isinstance(raw, dict):
        return default
    for k, v in raw.items():
        if k in default and isinstance(v, str):
            default[k] = v
    return default


def detect_gesture(gd, frame):
    """识别画面里的手势。

    返回 (手势名, 归一化手掌中心 x, 21点置信, 详情 dict)

    ⚠️ 详情里刻意分成两组证据，因为它们**互相独立**：
      * `palm_*` 来自手掌检测器（bbox + score）—— 手贴着镜头（透视极端、
        甚至只露半只）时照样能出，但只能说"有只手、在哪、多大"；
      * `palm_size` / 手势名 来自 21 点关键点模型 —— 能说清是什么手势，
        但手怼太近时这个模型可能直接失败。
    手挡住身体时我们必须靠前者才能知道"手在镜头前"，靠后者才能判手势，
    所以两组都要往外传。

    手势名见 hand_gesture.classify()：palm / ok / fist / point / unknown
    """
    empty = {'palms': 0, 'palm_score': 0.0, 'palm_size_norm': 0.0,
             'palm_cx': None, 'palm_boxes': []}
    if gd is None or frame is None:
        return 'none', None, 0.0, dict(empty)
    try:
        palms = gd['palm'].infer(frame)
    except Exception:
        return 'none', None, 0.0, dict(empty)
    if palms is None or len(palms) == 0:
        return 'none', None, 0.0, dict(empty)

    H, W = frame.shape[:2]
    # ---- 手掌检测器自带证据：列布局 [bbox(4), 21点(14), score(1)] ----
    pscore, pbox, pcx = 0.0, 0.0, None
    boxes = []          # 所有手掌框（像素）：给预览画框用，让用户肉眼确认"看没看到手"
    for p in palms:
        s = float(p[18])
        x1, y1, x2, y2 = (float(p[0]), float(p[1]), float(p[2]), float(p[3]))
        bw = max(0.0, x2 - x1) / max(1, W)
        bh = max(0.0, y2 - y1) / max(1, H)
        boxes.append((int(x1), int(y1), int(x2), int(y2), s))
        if max(bw, bh) > pbox:
            pbox, pcx = max(bw, bh), ((x1 + x2) / 2.0) / max(1, W)
        pscore = max(pscore, s)
    base = {'palms': int(len(palms)), 'palm_score': pscore,
            'palm_size_norm': pbox, 'palm_cx': pcx, 'palm_boxes': boxes}

    best = None
    for p in palms:
        try:
            r = gd['pose'].infer(frame, p)
        except Exception:
            continue
        if r is None or len(r) < 132:
            continue
        conf = float(r[131])
        if conf < 0.8:
            continue
        # r[4:67] = 21 个 screen landmarks（像素坐标），这里归一化成 0~1
        lms = np.array(r[4:67], dtype=np.float32).reshape(21, 3)
        norm = np.stack([lms[:, 0] / max(1, W), lms[:, 1] / max(1, H)], axis=1)
        gname, detail = gd['hg'].classify(norm)
        detail = dict(detail)
        detail.update(base)
        detail['gconf'] = conf
        if best is None or conf > best[2]:
            best = (gname, float(gd['hg'].palm_center_x(norm)), conf, detail)
    if best is None:
        # 21 点全失败：手势是什么不知道，但"有只多大的手、在哪"仍然有效
        return 'none', None, 0.0, base
    return best


class GestureFollower(object):
    """手掌横向跟随：独立线程，只动 shoulder_pan（不调俯仰高度）

    主线程把「手掌中心的归一化 x」写进来，这里换算成 pan 角度并平滑转过去。
    机械臂动作是阻塞的，所以必须放在单独线程里，不能卡视觉循环。
    """

    def __init__(self, armctl, base_pose, pan_range=28.0, invert=False,
                 period=0.35, deadzone=0.04):
        import threading
        self.armctl = armctl
        self.base = dict(base_pose)
        self.pan_range = pan_range
        self.invert = invert
        self.period = period
        self.deadzone = deadzone
        self._lock = threading.Lock()
        self._target = None        # 归一化 x
        self._stop = threading.Event()
        self.active = False
        self._t = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self._t.start()

    def set_target(self, x):
        with self._lock:
            self._target = x

    def clear_target(self):
        with self._lock:
            self._target = None

    def stop(self):
        self._stop.set()

    def _loop(self):
        import time as _t
        while not self._stop.is_set():
            with self._lock:
                x = self._target
            if not self.active or x is None or not self.armctl.connected:
                _t.sleep(0.08)
                continue
            # 画面左右 -> pan 角度；invert 用来修正镜像（摄像头装反时）
            off = (x - 0.5) * 2.0 * self.pan_range
            if self.invert:
                off = -off
            try:
                self.armctl.ensure()
                arm.turn_to(self.armctl.robot, self.armctl.names,
                            base=self.base, pan=off, duration=self.period * 0.9)
            except Exception as e:
                print('[gesture] 跟随时转臂失败: %r' % e, flush=True)
            _t.sleep(0.02)


# ===========================================================================
# 实时预览（调试用）：把带标注的画面以 MJPEG 推给浏览器
#
# 目的：让「我做的手势到底被识别成什么」肉眼可见，不用猜。
# 只在 --preview-port / GREET_PREVIEW_PORT 打开时启动，生产默认关闭（不占 CPU）。
# ===========================================================================

class PreviewServer(object):
    def __init__(self, port, quality=75):
        self.port = port
        self.quality = quality
        self._jpg = None
        self._scene = b''          # 当前场景文字（HTML 里展示）
        self._lock = None
        self._t = None

    def start(self):
        import threading
        self._lock = threading.Lock()
        self._t = threading.Thread(target=self._serve, daemon=True)
        self._t.start()
        print('[preview] 浏览器打开 http://<板子IP>:%d 即可实时看画面' % self.port,
              flush=True)

    def push(self, frame, scene):
        """视觉循环每帧调用：压成 JPEG 存起来（供 HTTP 线程取）"""
        try:
            ok, buf = cv2.imencode('.jpg', frame,
                                   [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
        except Exception:
            return
        if not ok:
            return
        with self._lock:
            self._jpg = buf.tobytes()
            self._scene = scene

    PAGE = (b'<html><head><meta charset="utf-8"><title>greet gesture debug</title>'
            b'<style>body{background:#111;color:#eee;font-family:sans-serif;margin:14px}'
            b'img{max-width:100%;border:1px solid #444}pre{color:#8ef;font-size:13px}'
            b'</style></head><body><h3>Gesture / Wave live debug</h3>'
            b'<img src="/stream"><pre id="s"></pre>'
            b'<script>setInterval(function(){fetch("/scene").then(function(r){'
            b'return r.text()}).then(function(t){document.getElementById("s").textContent=t})'
            b'},400)</script></body></html>')

    def _serve(self):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                if self.path.startswith('/stream'):
                    self.send_response(200)
                    self.send_header('Cache-Control', 'no-cache, private')
                    self.send_header('Pragma', 'no-cache')
                    self.send_header('Content-Type',
                                     'multipart/x-mixed-replace; boundary=--jpgb')
                    self.end_headers()
                    try:
                        while True:
                            with outer._lock:
                                jpg = outer._jpg
                            if jpg:
                                self.wfile.write(b'--jpgb\r\n')
                                self.wfile.write(b'Content-Type: image/jpeg\r\n')
                                self.wfile.write(('Content-Length: %d\r\n\r\n'
                                                  % len(jpg)).encode())
                                self.wfile.write(jpg)
                                self.wfile.write(b'\r\n')
                            time.sleep(0.06)
                    except Exception:
                        return
                elif self.path.startswith('/scene'):
                    with outer._lock:
                        body = outer._scene
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/plain; charset=utf-8')
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/html; charset=utf-8')
                    self.end_headers()
                    self.wfile.write(outer.PAGE)

        try:
            srv = ThreadingHTTPServer(('0.0.0.0', self.port), H)
            srv.serve_forever()
        except Exception as e:
            print('[preview] 启动失败: %r' % e, flush=True)


# ===========================================================================
# 主程序
# ===========================================================================

def load_cfg_file(path=CFG_PATH):
    cfg = {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            cfg = {k: v for k, v in raw.items() if k in DEFAULTS}
    except FileNotFoundError:
        pass
    except Exception as e:
        print('[cfg] 读取 %s 失败: %r（用默认值）' % (path, e), flush=True)
    return cfg


def load_base_pose(path=BASE_POSE_PATH):
    """读取迎宾「基准位」（arm_pose_save.py 写的 base_pose）。

    返回 (pose_dict, 说明)；读不到就回退 HOME_POSE_DEG。
    缺失的关节用 HOME 的值补齐 —— build_pose() 对缺失键会填 0.0，那是很危险的角度。
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            raw = json.load(f)
    except FileNotFoundError:
        return dict(HOME_POSE_DEG), '%s 不存在，用 HOME 姿态' % path
    except Exception as e:
        return dict(HOME_POSE_DEG), '读取 %s 失败(%r)，用 HOME 姿态' % (path, e)

    pose = raw.get('base_pose') if isinstance(raw, dict) else None
    if not isinstance(pose, dict) or not pose:
        return dict(HOME_POSE_DEG), '%s 里没有 base_pose，用 HOME 姿态' % path

    clean = {k: float(v) for k, v in pose.items() if isinstance(v, (int, float))}
    if not clean:
        return dict(HOME_POSE_DEG), 'base_pose 没有有效数值，用 HOME 姿态'

    merged = dict(HOME_POSE_DEG)
    merged.update(clean)
    saved = raw.get('saved_at', '?') if isinstance(raw, dict) else '?'
    missing = [k for k in HOME_POSE_DEG if k not in clean]
    note = '已加载 %s 的 base_pose（保存于 %s）' % (path, saved)
    if missing:
        note += '，其中 %s 用 HOME 值补齐' % ','.join(missing)
    return merged, note


def parse_args():
    d = dict(DEFAULTS)
    d.update(load_cfg_file())
    # 环境变量覆盖（方便测试脚本不改配置文件、不改 systemd unit 就打开调试）
    if os.environ.get('GREET_GESTURE_DEBUG'):
        d['gesture_debug'] = True
    if os.environ.get('GREET_PREVIEW_PORT'):
        try:
            d['preview_port'] = int(os.environ['GREET_PREVIEW_PORT'])
        except ValueError:
            pass
    if os.environ.get('GREET_FOLLOW_MODE') in ('palm', 'face', 'auto'):
        d['gesture_follow_mode'] = os.environ['GREET_FOLLOW_MODE']
    ap = argparse.ArgumentParser(description='视觉迎宾交互（到场问候/挥手告别/无人省电）')
    ap.add_argument('--cam-id', type=int, default=d['cam_id'])
    ap.add_argument('--rot', type=int, default=d['rot'], choices=[0, 90, 180, 270])
    ap.add_argument('--size', type=int, default=d['size'])
    ap.add_argument('--fps', type=int, default=d['fps'])
    ap.add_argument('--port', default='auto',
                    help='机械臂串口；auto=优先 /dev/serial/by-id（编号漂移也不用管）')
    ap.add_argument('--robot-id', default='my_awesome_follower_arm')
    ap.add_argument('--dry-run', action='store_true', help='不连机械臂')
    ap.add_argument('--no-arm', action='store_true', help='不连机械臂（同 --dry-run）')
    ap.add_argument('--no-led', action='store_true', help='不控灯')
    ap.add_argument('--calib', action='store_true', help='标定模式：只打印判据，不做反馈')
    ap.add_argument('--present-frames', type=int, default=d['present_frames'])
    ap.add_argument('--present-hold', type=float, default=d['present_hold'])
    ap.add_argument('--absent-sec', type=float, default=d['absent_sec'])
    ap.add_argument('--greet-cooldown', type=float, default=d['greet_cooldown'])
    ap.add_argument('--wave-cooldown', type=float, default=d['wave_cooldown'])
    ap.add_argument('--wave-amp', type=float, default=d['wave_amp'])
    ap.add_argument('--wave-reversals', type=int, default=d['wave_reversals'])
    ap.add_argument('--wave-raise', type=float, default=d['wave_raise'])
    ap.add_argument('--min-box', type=float, default=d['min_box'])
    ap.add_argument('--idle-color', default=d['idle_color'])
    ap.add_argument('--idle-bright', type=int, default=d['idle_bright'])
    ap.add_argument('--greet-color', default=d['greet_color'])
    ap.add_argument('--bye-color', default=d['bye_color'])
    ap.add_argument('--greet-bright', type=int, default=d['greet_bright'])
    ap.add_argument('--nod-amp', type=float, default=d['nod_amp'])
    ap.add_argument('--shake-amp', type=float, default=d['shake_amp'],
                    help='摇摇头幅度（度，绕 shoulder_pan）')
    ap.add_argument('--wave-deg', type=float, default=d['wave_deg'])
    ap.add_argument('--wave-action', default=d['wave_action'],
                    choices=['showcase', 'bye', 'both'],
                    help='识别到挥手时做什么：showcase=拍照并自我介绍（默认）'
                         ' / bye=挥手告别动作 / both=都做')
    ap.add_argument('--wave-prompt', default=d['wave_prompt'],
                    help='showcase 时送给云端的完整引导语')
    ap.add_argument('--pan-amp', type=float, default=d['pan_amp'],
                    help='迎宾「左右小幅旋转」的幅度（度）')
    ap.add_argument('--pan-cycles', type=int, default=d['pan_cycles'],
                    help='左右摆动来回次数')
    ap.add_argument('--base-pose', default=BASE_POSE_PATH,
                    help='迎宾基准位文件（arm_pose_save.py 写的）')
    ap.add_argument('--greet-say', default=d['greet_say'],
                    help='迎宾时让 AI 说的话（走 ai_agent_send_text）')
    ap.add_argument('--say-sock', default=d['say_sock'],
                    help='your_chat_bot 本地文本入口 socket')
    ap.add_argument('--no-say', dest='say_enabled', action='store_false', default=True,
                    help='不叫 AI 说话（只做灯效 + 机械臂动作）')
    ap.add_argument('--demo-once', nargs='?', type=float, const=3.0, default=0.0,
                    help='自检用：启动 N 秒后执行一次完整迎宾动作（灯效 + 左右旋转 + AI 打招呼）'
                         '然后退出。不带值默认 3 秒。')
    ap.add_argument('--demo-wave', nargs='?', type=float, const=3.0, default=0.0,
                    help='自检用：不用真人挥手，启动 N 秒后触发一次「挥手响应」'
                         '（拍照 + 自我介绍 + 机械臂动作）然后退出。不带值默认 3 秒。')
    ap.add_argument('--debug-every', type=float, default=d['debug_every'])
    ap.add_argument('--wave-motion', default=d['wave_motion'],
                    choices=['pan', 'wrist', 'both'],
                    help='挥手时机械臂怎么做：pan=转灯头(5号舵机)（推荐）'
                         ' / wrist=摆腕 / both=都做')
    ap.add_argument('--wave-pan-amp', type=float, default=d['wave_pan_amp'],
                    help='挥手时灯头左右转的幅度（度）')
    ap.add_argument('--wave-pan-cycles', type=int, default=d['wave_pan_cycles'],
                    help='挥手时灯头左右摆动的来回次数')
    ap.add_argument('--follow-mode', default=d['gesture_follow_mode'],
                    choices=['palm', 'face', 'auto'],
                    help='横向跟随跟谁：palm=只跟手掌 / face=只跟人脸'
                         ' / auto=手掌优先，没手掌就跟人脸（默认）')
    ap.add_argument('--gesture-map', default=GESTURE_MAP_PATH,
                    help='「手势 -> 动作」映射 JSON（改它即可扩展新玩法，不用改代码）')
    ap.add_argument('--gesture-face-conf', type=float, default=d['gesture_face_conf'],
                    help='人脸(鼻子/眼)关键点置信度下限（follow-mode=face/auto 用）')
    ap.add_argument('--no-hand-always', dest='hand_always', action='store_false',
                    default=d['hand_always'],
                    help='手部检测只在检测到人时才跑（省 CPU，但手挡住身体时会失效）')
    ap.add_argument('--hand-big-size', type=float, default=d['hand_big_size'],
                    help='「真有只手在镜头前」的手掌尺寸下限（归一化，默认 0.06）')
    ap.add_argument('--hand-palm-score', type=float, default=d['hand_palm_score'],
                    help='手掌检测器置信度下限（21 点失败时用 bbox 兜底，默认 0.85）')
    ap.add_argument('--hand-jump', type=float, default=d['hand_jump'],
                    help='手掌中心单帧位移超过该值即视为不可信，清空往复历史（默认 0.35）')
    ap.add_argument('--gesture-debug', action='store_true', default=d['gesture_debug'],
                    help='逐帧把识别判据写 /tmp/gesture_dbg.jsonl（事后分析用）')
    ap.add_argument('--preview-port', type=int, default=d['preview_port'],
                    help='>0 时开 MJPEG 实时预览，浏览器看画面+识别结果（调试用）')
    ap.add_argument('--release-on-leave', action='store_true', default=True,
                    help='无人省电时释放舵机力矩（默认开）')
    ap.add_argument('--no-release', dest='release_on_leave', action='store_false')
    # ---- 手势 ----
    ap.add_argument('--no-gesture', dest='gesture', action='store_false',
                    default=d['gesture'], help='关闭手势识别（省 CPU）')
    ap.add_argument('--gesture-pan-range', type=float, default=d['gesture_pan_range'],
                    help='手掌跟随的左右角度范围（度）')
    ap.add_argument('--gesture-invert', action='store_true', default=d['gesture_invert'],
                    help='手掌跟随左右反向（画面是镜像时用）')
    ap.add_argument('--gesture-ok-cooldown', type=float, default=d['gesture_ok_cooldown'],
                    help='OK 手势触发点头的冷却（秒）')
    ap.add_argument('--gesture-stable', type=int, default=d['gesture_stable'],
                    help='连续几帧判成同一手势才认（防抖）')
    ap.add_argument('--gesture-lost', type=float, default=d['gesture_lost'],
                    help='连续多久看不到手掌就停止跟随（秒）')
    ap.add_argument('--gesture-conf', type=float, default=d['gesture_conf'],
                    help='手部关键点置信度下限（也用于 hand_big 判定，默认 0.8）')
    opt = ap.parse_args()

    cfg = {k: getattr(opt, k.replace('-', '_')) for k in
           ('present_frames', 'present_hold', 'absent_sec', 'greet_cooldown',
            'wave_cooldown', 'wave_amp', 'wave_reversals', 'wave_raise',
            'min_box', 'hand_jump')}
    cfg['wave_window'] = d['wave_window']
    return opt, cfg


def main():
    opt, cfg = parse_args()
    arm_enabled = (not opt.dry_run) and (not opt.no_arm)
    led_enabled = not opt.no_led

    print('=' * 62, flush=True)
    print('视觉迎宾交互  %s' % time.strftime('%Y-%m-%d %H:%M:%S'), flush=True)
    print('  机械臂: %s    灯环: %s    标定模式: %s'
          % (('启用 ' + opt.port) if arm_enabled else '禁用(不动臂)',
             '启用' if led_enabled else '禁用', opt.calib), flush=True)
    print('  在场判定: 连续 %d 帧 / 停留 %.1fs    无人 %.1fs 后告别'
          % (cfg['present_frames'], cfg['present_hold'], cfg['absent_sec']), flush=True)
    print('  挥手判定: 双通道 / %.1fs 窗口，方向反转 ≥%d 次即算'
          % (cfg['wave_window'], cfg['wave_reversals']), flush=True)
    print('    A 人体通道: 举手(腕高于肩 %.2f×肩宽) + 腕 x 往复   [需人完整入镜]'
          % cfg['wave_raise'], flush=True)
    print('    B 手部通道: 手掌尺寸 ≥%.3f + 手掌中心 x 往复      [手挡住身体时兜底]'
          % opt.hand_big_size, flush=True)
    print('  冷却: 迎宾 %.0fs / 挥手 %.0fs'
          % (cfg['greet_cooldown'], cfg['wave_cooldown']), flush=True)
    print('  挥手响应: %s%s'
          % ({'showcase': '拍照并自我介绍（主动展示自己）',
              'bye': '挥手告别动作',
              'both': '拍照自我介绍 + 告别动作'}.get(opt.wave_action, opt.wave_action),
             '' if opt.say_enabled else '   (AI 说话已关闭)'), flush=True)
    if opt.gesture:
        _fm = {'palm': '只看手掌', 'face': '只看人脸', 'auto': '手掌优先,没手掌跟人脸'}
        print('  手势: 冷却%.0fs  跟随来源=%s  灯头跟随范围 ±%.0f°%s'
              % (opt.gesture_ok_cooldown, _fm.get(opt.follow_mode, opt.follow_mode),
                 opt.gesture_pan_range,
                 ' (左右反向)' if opt.gesture_invert else ''), flush=True)
        print('        手部检测: %s   手掌尺寸下限 %.3f'
              % ('全程运行（推荐）' if opt.hand_always else '仅检测到人时',
                 opt.hand_big_size), flush=True)
    else:
        print('  手势: 已关闭（--no-gesture）', flush=True)
    print('  迎宾动作: 基准位左右小幅旋转 ±%.0f° x%d 次    AI 招呼语: %s'
          % (opt.pan_amp, opt.pan_cycles,
             repr(opt.greet_say) if opt.say_enabled else '(已关闭)'), flush=True)
    print('=' * 62, flush=True)

    # ---- 灯效自检 ----
    if led_enabled:
        r = led.state()
        if not r.get('ok'):
            print('[led] 警告: %s' % r.get('error'), flush=True)
            led_enabled = False
        else:
            print('[led] 守护进程正常', flush=True)

    # ---- NPU 模型 ----
    print('[npu] 加载模型 %s' % MODEL_PATH, flush=True)
    rknn = RKNN(verbose=False)
    assert rknn.load_rknn(MODEL_PATH) == 0, 'load_rknn failed'
    assert rknn.init_runtime() == 0, 'init_runtime failed'
    print('[npu] 模型就绪', flush=True)

    # ---- 摄像头 ----
    cap, cam_idx = open_camera(opt.cam_id)
    assert cap is not None, '找不到可用摄像头'
    opt.cam_id = cam_idx
    print('[cam] /dev/video%d 就绪' % cam_idx, flush=True)

    # ---- 迎宾基准位（arm_pose_save.py 读下来的机械臂当前姿态）----
    base_pose, pose_note = load_base_pose(opt.base_pose)
    print('[pose] %s' % pose_note, flush=True)
    print('[pose] 基准位: %s'
          % {k: round(v, 2) for k, v in base_pose.items()}, flush=True)

    # ---- 机械臂 ----
    opt.port = resolve_arm_port(opt.port)
    print('[arm] 串口: %s' % opt.port, flush=True)
    armctl = ArmCtl(opt.port, opt.robot_id, enabled=arm_enabled)
    if arm_enabled:
        try:
            armctl.connect()
            print('[arm] 转到基准位...', flush=True)
            arm.goto(armctl.robot, armctl.names, base_pose, duration=2.5)
        except Exception as e:
            print('[arm] 连接失败: %r —— 转为不动臂模式' % e, flush=True)
            arm_enabled = False
            armctl.enabled = False

    runner = ActionRunner()
    runner.start()

    def do_greet():
        # 1) 先叫 AI：文本进云端要 1~2 秒思考 + 出 TTS，先发出去，好让机械臂动作和说话重叠
        if opt.say_enabled:
            try:
                ok, resp = ai_notify.send_line(
                    opt.say_sock,
                    json.dumps({'cmd': 'say', 'text': opt.greet_say},
                               ensure_ascii=False),
                    timeout=1.5)
                print('[say] AI 招呼语 %r -> %s' % (opt.greet_say, resp or '(空)'), flush=True)
            except Exception as e:
                print('[say] 通知 AI 失败: %r（AI 这次不会说话，动作照做）' % e, flush=True)
        # 2) 灯效
        if led_enabled:
            led.fade_in(opt.greet_color, opt.greet_bright, steps=12, delay=0.05)
            led.sweep(opt.greet_color, 0.8, opt.greet_bright)
        # 3) 在基准位上左右小幅旋转（而不是点头）
        if arm_enabled:
            armctl.ensure()
            arm.pan_wiggle(armctl.robot, armctl.names, base=base_pose,
                           amp=opt.pan_amp, cycles=opt.pan_cycles)
        if led_enabled:
            led.idle(opt.idle_color, opt.idle_bright)

    def do_wave():
        # 1) 先叫 AI：云端要 1~2 秒思考 + 出 TTS，先发出去，好让拍照/说话和机械臂动作重叠
        #    这里用的是 {"cmd":"photo","prompt":...} —— 让 AI 先拍一张看到来人，
        #    再打招呼并自我介绍；prompt 完整指定，不走「一到两句回答他」的模板。
        if opt.say_enabled and opt.wave_action in ('showcase', 'both'):
            try:
                ok, resp = ai_notify.send_line(
                    opt.say_sock,
                    json.dumps({'cmd': 'photo', 'prompt': opt.wave_prompt},
                               ensure_ascii=False),
                    timeout=1.5)
                print('[say] 挥手 -> 拍照+自我介绍 %s: %s'
                      % ('ok' if ok else 'fail', resp or '(空)'), flush=True)
            except Exception as e:
                print('[say] 通知 AI 失败: %r（这次不拍照介绍，动作照做）' % e, flush=True)
        if led_enabled:
            led.flash(opt.bye_color, times=2, level=opt.greet_bright)
        if arm_enabled:
            armctl.ensure()
            # 主动作：转灯头（shoulder_pan = 5 号舵机）左右摆动 —— 最远、最显眼
            if opt.wave_motion in ('pan', 'both'):
                arm.pan_wiggle(armctl.robot, armctl.names, base=base_pose,
                               amp=opt.wave_pan_amp, cycles=opt.wave_pan_cycles)
            # 副动作：腕部摆手（wrist_roll），让"招手"更像招手
            if opt.wave_motion in ('wrist', 'both'):
                arm.wave(armctl.robot, armctl.names, base=base_pose,
                         amp=opt.wave_deg, cycles=3)
            arm.goto(armctl.robot, armctl.names, base_pose, duration=1.0)
        if led_enabled:
            led.idle(opt.idle_color, opt.idle_bright)

    def do_leave():
        if led_enabled:
            led.fade_out(opt.idle_color, opt.idle_bright, steps=12)
        if arm_enabled:
            armctl.ensure()
            arm.goto_park(armctl.robot, armctl.names, duration=3.0)
            if opt.release_on_leave:
                armctl.release()

    def do_nod():
        """OK 手势 -> 点头回应"""
        if arm_enabled:
            armctl.ensure()
            arm.nod(armctl.robot, armctl.names, base=base_pose,
                    amp=opt.nod_amp, cycles=2)

    def do_shake():
        """摇摇头（"不对 / 不知道"）：**只转灯头**

        只驱动 shoulder_pan（5 号舵机）左右各拧一下，其余五个关节全程写回
        迎宾基准位、一动不动 —— 所以看上去是「灯头在摇头」，不是整条臂在扫。
        """
        if arm_enabled:
            armctl.ensure()
            arm.shake_head(armctl.robot, armctl.names, base=base_pose,
                           amp=opt.shake_amp, cycles=2)

    def do_back():
        """停跟随 -> 回到迎宾基准位"""
        if arm_enabled:
            armctl.ensure()
            arm.goto(armctl.robot, armctl.names, base_pose, duration=1.0)

    def do_look_up():
        """抬头（表示"注意到你了"）"""
        if arm_enabled:
            armctl.ensure()
            arm.look_up(armctl.robot, armctl.names, base=base_pose)

    def do_pan_wiggle():
        """灯头左右摆两下（谁都能看懂的小回应）"""
        if arm_enabled:
            armctl.ensure()
            arm.pan_wiggle(armctl.robot, armctl.names, base=base_pose,
                           amp=opt.wave_pan_amp, cycles=opt.wave_pan_cycles)
            arm.goto(armctl.robot, armctl.names, base_pose, duration=1.0)

    def do_gripper_wiggle():
        """夹爪开合两下（俏皮）"""
        if arm_enabled:
            armctl.ensure()
            arm.gripper_wiggle(armctl.robot, armctl.names, base=base_pose)

    def do_photo():
        """让 AI 拍一张照并说点什么（走本机 photo 指令）"""
        if opt.say_enabled:
            try:
                ok, resp = ai_notify.send_line(
                    opt.say_sock,
                    json.dumps({'cmd': 'photo', 'prompt': opt.wave_prompt},
                               ensure_ascii=False), timeout=1.5)
                print('[say] 手势 -> 拍照 %s: %s'
                      % ('ok' if ok else 'fail', resp or '(空)'), flush=True)
            except Exception as e:
                print('[say] 通知 AI 失败: %r' % e, flush=True)

    # 「手势名 -> 动作」里能用的动作名 -> 实际函数（扩展玩法只需改 /etc/gesture-map.json）
    ACTION_FNS = {
        'none': None,
        'follow': None,            # 特殊：由跟随器逻辑处理，不在这里一次性执行
        'stop_follow': None,       # 特殊：同上
        'nod': do_nod,
        'shake_head': do_shake,
        'wave': do_wave,
        'look_up': do_look_up,
        'back': do_back,
        'pan_wiggle': do_pan_wiggle,
        'gripper_wiggle': do_gripper_wiggle,
        'greet': do_greet,
        'photo': do_photo,
    }

    # ---- 手势识别（OpenCV DNN / CPU）----
    gd = load_gesture_models(opt.gesture)
    gesture_map = load_gesture_map(opt.gesture_map)
    if gd is not None:
        print('[gesture] 手势->动作映射: %s（改 %s 即可扩展玩法）'
              % (json.dumps(gesture_map, ensure_ascii=False), opt.gesture_map),
              flush=True)
    follower = None
    g_state = {'last': 'none', 'n': 0, 'last_ok': -1e9,
               'last_seen': 0.0, 'following': False, 'src': '-',
               'no_follow_until': 0.0, 'cd': {}, 'dbg_t': 0.0, 'last_act': '-'}
    if gd is not None and arm_enabled:
        follower = GestureFollower(armctl, base_pose,
                                   pan_range=opt.gesture_pan_range,
                                   invert=opt.gesture_invert)
        follower.start()

    # ---- 可选：MJPEG 实时预览（调试时把「识别成了什么」直接画在画面上）----
    preview = PreviewServer(int(opt.preview_port)) if opt.preview_port > 0 else None
    if preview is not None:
        preview.start()
    dbg_fp = None
    if opt.gesture_debug:
        try:
            dbg_fp = open('/tmp/gesture_dbg.jsonl', 'a', encoding='utf-8')
            print('[gesture] 逐帧诊断日志 -> /tmp/gesture_dbg.jsonl', flush=True)
        except Exception as e:
            print('[gesture] 打不开诊断日志: %r' % e, flush=True)

    det = EventDetector(cfg)

    # ---- 本机控制口：MCP 工具(arm_do.py) -> 这里 -> 排进动作队列 ----
    # 注意用的是「已在跑」的机械臂连接，不重新连串口，所以不会和迎宾抢设备。
    conctl = ControlServer(
        '/run/vision-greet.sock',
        {'nod': do_nod, 'shake_head': do_shake, 'wave': do_wave,
         'look_up': do_look_up, 'back': do_back, 'pan_wiggle': do_pan_wiggle},
        lambda name, fn: runner.submit('ctrl_' + name, fn),
        lambda: 'state:present=%s following=%s queue=%d last=%s'
                % (det.present, g_state['following'], runner.q.qsize(),
                   g_state['last_act']),
    )
    conctl.start()

    period = 1.0 / max(1, opt.fps)
    cam_fail = 0
    last_debug = 0.0
    fps_t0, fps_n = time.time(), 0
    t_start = time.time()      # 诊断日志用相对时间，方便和时间线对齐
    stats = collections.Counter()
    demo_t0 = time.time()
    demo_state = 0          # 0=等触发 1=已提交，等动作结束
    demo_t1 = 0.0

    try:
        while True:
            t0 = time.time()
            now = t0

            # ---- CAM_WANT_FLAG: your_chat_bot 要拍照，先把摄像头交出去 ----
            if os.path.exists(CAM_WANT_FLAG):
                if cap is not None:
                    cap.release()
                    cap = None
                    print('[cam] 让出摄像头给 your_chat_bot', flush=True)
                time.sleep(0.1)
                continue
            if cap is None:
                cap, cam_idx = open_camera(opt.cam_id)
                if cap is None:
                    time.sleep(0.5)
                    continue
                print('[cam] 重新拿回摄像头 /dev/video%d' % cam_idx, flush=True)

            ok, frame = cap.read()
            if not ok:
                cam_fail += 1
                if cam_fail % 20 == 1:
                    print('[cam] 读帧失败 (连续 %d)' % cam_fail, flush=True)
                if cam_fail >= 60:
                    print('[cam] 连续失败，退出', flush=True)
                    break
                time.sleep(0.2)
                continue
            cam_fail = 0
            if opt.rot:
                frame = cv2.rotate(frame, ROT_MAP[opt.rot])

            lb, r, ox, oy = letterbox(frame, opt.size)
            inp = np.expand_dims(lb[..., ::-1], 0)      # BGR->RGB + batch 维（与跟随脚本一致）
            outs = rknn.inference(inputs=[inp], data_format='nhwc')
            if outs is None:
                # Ctrl+C 打断 NPU 推理时 rknnlite 会吞掉中断并返回空值 -> 主动退出走优雅收尾
                print('[npu] 推理被中断，退出...', flush=True)
                break
            dets = decode(outs, CONF_THRESH, opt.size)

            # 取面积最大的人（最靠近镜头），并要求足够大（滤掉远处人影）
            person = False
            kpts = None
            area_ratio = 0.0
            if dets:
                best = None
                for box, score, kp in dets:
                    bw = max(0.0, box[2] - box[0])
                    bh = max(0.0, box[3] - box[1])
                    ar = (bw * bh) / float(opt.size * opt.size)
                    if best is None or ar > best[0]:
                        best = (ar, box, score, kp)
                area_ratio, box, score, kp = best
                if area_ratio >= cfg['min_box']:
                    person = True
                    kpts = kp
                if opt.debug_every > 0:
                    draw(frame, box, score, kp, r, ox, oy)

            # ---- 自检钩子：GREET_FAKE_PERSON=1 时假装镜头前有一个人 ----
            # 用途：没人站在镜头前时，也能把「有人」这条代码路径走一遍，
            # 确认不崩、诊断日志/预览都正常。合成关键点（letterbox 的 size 尺度）。
            # ⚠️ 只做自检用，正常跑不要开（会真的驱动机械臂跟随）。
            if FAKE_PERSON:
                person, area_ratio = True, 0.25
                kp = np.zeros((17, 3), dtype=np.float32)
                kp[0] = (160, 90, 0.9)                     # 鼻子
                kp[1] = (145, 85, 0.9)
                kp[2] = (175, 85, 0.9)                     # 双眼
                kp[5] = (120, 150, 0.9)
                kp[6] = (200, 150, 0.9)                    # 双肩（肩宽 80）
                kp[9] = (105, 230, 0.9)
                kp[10] = (215, 230, 0.9)                   # 双腕（低于肩，不算举手）
                kpts = kp

            # ---- 手势互动：映射表驱动 ----
            # 默认映射（/etc/gesture-map.json 可覆盖、可扩展）：
            #   ok   -> nod        （点头回应）
            #   palm -> follow     （灯头横向跟随手掌 / 人脸）
            #
            # 门禁为什么不能只靠 person：
            #   人站在台灯前挥手时，手往往离镜头很近，身体被手挡在画面外，
            #   yolov8-pose 会直接检不到人（实测 person=False / area=0.000），
            #   于是整段手势代码被跳过 —— 手就在镜头前却什么都认不出来。
            #   → 所以只要「有人」**或**「有只够大够可信的手在画面里」就继续跑。
            #   （单独一只手的误检用 hand_big 卡：手掌尺寸 ≥ hand_big_size 才算数。）
            gname, gx, gconf, gdetail, fx = 'none', None, 0.0, {}, None
            gact, gsrc = '-', '-'
            hand_big, hand_x = False, None
            if gd is not None and (person or opt.hand_always):
                gname, gx, gconf, gdetail = detect_gesture(gd, frame)
                fx = face_x_from_kpts(kpts, opt.gesture_face_conf)
                # 「有只手确实在镜头前」：两组证据取并集 ——
                #   a) 21 点模型可信 且 掌尺寸够大
                #   b) 手掌检测器本身可信 且 bbox 够大（21 点失败时兜底）
                _ps21 = gdetail.get('palm_size', 0.0)
                _pbox = gdetail.get('palm_size_norm', 0.0)
                _pscore = gdetail.get('palm_score', 0.0)
                hand_big = (
                    (_ps21 >= opt.hand_big_size and gconf >= opt.gesture_conf)
                    or (_pbox >= opt.hand_big_size
                        and _pscore >= opt.hand_palm_score))
                # 手部通道的横向位置：优先 21 点的掌中心；21 点失败就用 bbox 中心
                hand_x = (gx if gx is not None else gdetail.get('palm_cx')) \
                    if hand_big else None
                if gname == g_state['last']:
                    g_state['n'] += 1
                else:
                    g_state['last'] = gname
                    g_state['n'] = 1
                    if gname != 'none':
                        print('[gesture] 识别到 %s (x=%s conf=%.2f 伸直%d指 捏合比%.2f '
                              '掌宽%.3f 手%d只%s)'
                              % (gname, ('%.3f' % gx) if gx is not None else '-',
                                 gconf, gdetail.get('n_extended', -1),
                                 gdetail.get('pinch_r', -1.0),
                                 gdetail.get('palm_size', 0.0),
                                 gdetail.get('palms', 0),
                                 ' big' if hand_big else ''), flush=True)
                stable = (g_state['n'] >= max(1, opt.gesture_stable))
                gact = gesture_map.get(gname, 'none')

                # ---- 跟随目标（palm 优先 / face 回退）----
                # 手掌通道必须 hand_big：否则地面纹理误检出的"手"会把灯头乱转。
                palm_ok = (gname == 'palm' and stable and gx is not None and hand_big)
                tgt = None
                if now >= g_state['no_follow_until']:
                    if opt.follow_mode == 'palm':
                        if palm_ok:
                            tgt, gsrc = gx, 'palm'
                    elif opt.follow_mode == 'face':
                        if fx is not None:
                            tgt, gsrc = fx / float(opt.size), 'face'
                    else:                                   # auto
                        if palm_ok:
                            tgt, gsrc = gx, 'palm'
                        elif gname != 'palm' and fx is not None:
                            tgt, gsrc = fx / float(opt.size), 'face'

                if follower is not None and 'follow' in gesture_map.values():
                    if tgt is not None and gact != 'stop_follow':
                        g_state['last_seen'] = now
                        if not g_state['following']:
                            g_state['following'] = True
                            follower.active = True
                            print('[gesture] 开始灯头跟随（来源=%s）' % gsrc, flush=True)
                        elif gsrc != g_state['src']:
                            print('[gesture] 跟随来源切换 -> %s' % gsrc, flush=True)
                        g_state['src'] = gsrc
                        follower.set_target(tgt)
                    if (g_state['following']
                            and (now - g_state['last_seen']) > opt.gesture_lost):
                        g_state['following'] = False
                        follower.active = False
                        follower.clear_target()
                        g_state['src'] = '-'
                        print('[gesture] 目标消失 -> 停止跟随并回基准位', flush=True)
                        runner.submit('gesture_back', do_back)

                # ---- 一次性动作（映射表里除 follow/stop_follow 之外的动作）----
                # 同样要求「人」或「够大够真的手」，防纹理误检触发机械臂动作。
                if (stable and (person or hand_big)
                        and gact not in ('none', 'follow') and gact in ACTION_FNS):
                    if (now - g_state['cd'].get(gname, -1e9)) >= opt.gesture_ok_cooldown:
                        g_state['cd'][gname] = now
                        # OK 之类的动作要打断跟随，否则手臂会被跟随线程抢走
                        if gact in ('nod', 'shake_head', 'back', 'stop_follow',
                                    'pan_wiggle', 'gripper_wiggle', 'look_up', 'wave'):
                            if g_state['following']:
                                g_state['following'] = False
                                follower.active = False
                                follower.clear_target()
                                g_state['src'] = '-'
                            g_state['no_follow_until'] = now + 2.5
                        fn = ACTION_FNS.get(gact)
                        if fn is not None:
                            print('[gesture] %s -> 动作 %s' % (gname, gact), flush=True)
                            runner.submit('gesture_' + gact, fn)
                            g_state['last_act'] = '%s@%.1f' % (gact, now)
                        else:
                            print('[gesture] %s -> stop_follow' % gname, flush=True)
                            runner.submit('gesture_back', do_back)
                elif gact not in ('none', 'follow'):
                    if gact not in ACTION_FNS:
                        if opt.debug_every > 0:
                            print('[gesture] 未知动作 %r（跳过）' % gact, flush=True)

            events = det.update(person, kpts, area_ratio, now,
                                hand_x=hand_x, hand_big=hand_big)
            for ev in events:
                stats[ev] += 1
                print('[evt] %-7s (area=%.3f wrist=%d)'
                      % (ev, area_ratio, det.calib.get('wrist_n', 0)), flush=True)
                if opt.calib:
                    continue
                if ev == 'greet':
                    # 正在手势互动（横向跟随）时别再抢机械臂，迎宾这次跳过
                    if g_state.get('following'):
                        print('[evt] greet 被手势跟随占用，跳过', flush=True)
                    else:
                        runner.submit('greet', do_greet)
                elif ev == 'wave':
                    runner.submit('wave', do_wave)
                elif ev == 'leave':
                    runner.submit('leave', do_leave)

            # ---- 逐帧诊断日志（事后分析用）----
            # 注意：① 这一段刻意放在「有人在」门禁**外面** —— 恰恰是「镜头里没人」
            #       这种情况最需要留下记录（否则只知道没反应、不知道为什么）；
            #       ② 必须放在 det.update() **之后**，否则 present/ev 是上一帧的、
            #       看不到「这一帧触发了什么事件」；
            #       ③ 平时每 0.1s 写一行，但**只要有事件就强制写**，
            #       否则事件那一帧刚好被节流掉、事后查不到是谁触发的。
            if dbg_fp is not None and ((now - g_state['dbg_t']) >= 0.1 or events):
                g_state['dbg_t'] = now
                c = det.calib
                try:
                    dbg_fp.write(json.dumps({
                        't': round(now - t_start, 2), 'person': bool(person),
                        'area': round(area_ratio, 4),
                        'present': bool(det.present),
                        'hit': c.get('hit', 0), 'miss': c.get('miss', 0),
                        'ev': ','.join(events),
                        'sw': round(c.get('sw', 0.0), 1),
                        'raised': bool(c.get('raised', False)),
                        'rise': round(c.get('rise', 0.0), 1),
                        'xn': (round(c['x_norm'], 3)
                               if c.get('x_norm') is not None else None),
                        'turns': c.get('turns', 0),
                        'turns_torso': c.get('turns_torso', 0),
                        'turns_hand': c.get('turns_hand', 0),
                        'wrist_n': c.get('wrist_n', 0),
                        'hand_n': c.get('hand_n', 0),
                        'hand_big': bool(hand_big),
                        'hand_x': (round(hand_x, 3) if hand_x is not None else None),
                        'gest': gname, 'gx': (round(gx, 3) if gx is not None else None),
                        'gconf': round(gconf, 3),
                        'palms': gdetail.get('palms', 0),
                        'palm_size': round(gdetail.get('palm_size', 0.0), 4),
                        'palm_box': round(gdetail.get('palm_size_norm', 0.0), 4),
                        'palm_score': round(gdetail.get('palm_score', 0.0), 3),
                        'palm_cx': (round(gdetail['palm_cx'], 3)
                                    if gdetail.get('palm_cx') is not None else None),
                        'fing': gdetail.get('n_extended', -1),
                        'pinch': gdetail.get('pinch_r', -1.0),
                        'face_x': (round(fx, 1) if fx is not None else None),
                        'act': gact, 'src': gsrc,
                        'following': bool(g_state['following']),
                        'queue': runner.q.qsize(),
                    }, ensure_ascii=False) + '\n')
                    dbg_fp.flush()
                except Exception:
                    pass

            if opt.calib:
                c = det.calib
                print('[calib] present=%d hit=%d miss=%d raised=%s rise=%.0f sw=%.0f '
                      'x=%s wrist=%d silent=%.1fs'
                      % (c['present'], c['hit'], c['miss'], c['raised'], c['rise'],
                         c['sw'],
                         ('%.3f' % c['x_norm']) if c['x_norm'] is not None else '-',
                         c['wrist_n'], c['silent']), flush=True)

            if opt.debug_every > 0 and now - last_debug > opt.debug_every:
                cv2.putText(frame, 'present=%s q=%d ev=%s'
                            % (det.present, runner.count, ','.join(events) or '-'),
                            (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imwrite('/tmp/greet_debug.jpg', frame)
                last_debug = now

            # ---- 实时预览：把「识别成了什么」画在画面上推给浏览器 ----
            if preview is not None:
                c = det.calib
                _fw = max(1, frame.shape[1])
                _fh = max(1, frame.shape[0])
                # 手掌/手部检测框：绿=够大（hand_big 认可，挥手/跟随都吃它）、
                # 灰=太小（会被 hand_big 挡掉，避免桌面小物件误检）。框旁边是 score。
                for (_bx1, _by1, _bx2, _by2, _bs) in gdetail.get('palm_boxes', []):
                    _sz = max(max(0, _bx2 - _bx1) / _fw, max(0, _by2 - _by1) / _fh)
                    _ok = (_sz >= opt.hand_big_size and _bs >= opt.hand_palm_score)
                    _col = (80, 255, 120) if _ok else (150, 150, 150)
                    cv2.rectangle(frame, (_bx1, _by1), (_bx2, _by2), _col, 2)
                    cv2.putText(frame, '%.2f %.3f' % (_bs, _sz),
                                (_bx1, max(12, _by1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3,
                                cv2.LINE_AA)
                    cv2.putText(frame, '%.2f %.3f' % (_bs, _sz),
                                (_bx1, max(12, _by1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, _col, 1,
                                cv2.LINE_AA)
                # hand_x 竖线：挥手判据之一就是它左右往复
                if hand_x is not None:
                    _hx = int(hand_x * _fw)
                    cv2.line(frame, (_hx, 0), (_hx, 26), (0, 220, 255), 2)

                def _txt(y, s, color):
                    cv2.putText(frame, s, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(frame, s, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                color, 1, cv2.LINE_AA)
                _txt(22, 'PERSON=%s AREA=%.3f (need>=%.2f)'
                     % (person, area_ratio, cfg['min_box']),
                     (80, 255, 120) if person else (120, 120, 255))
                _txt(44, 'HAND big=%s  box=%.3f score=%.2f  palm_size=%.3f  palms=%d'
                     % (hand_big, gdetail.get('palm_size_norm', 0.0),
                        gdetail.get('palm_score', 0.0),
                        gdetail.get('palm_size', 0.0), gdetail.get('palms', 0)),
                     (80, 255, 120) if hand_big else (210, 210, 210))
                _txt(66, 'GEST=%-8s FING=%s PINCH=%.2f'
                     % (gname, gdetail.get('n_extended', '-'),
                        gdetail.get('pinch_r', -1.0)),
                     (80, 255, 120) if gname in ('palm', 'ok') else (210, 210, 210))
                _txt(88, 'FOLLOW=%-5s src=%-4s act=%-14s'
                     % (g_state['following'], g_state['src'], gact),
                     (80, 255, 120) if g_state['following'] else (210, 210, 210))
                _txt(110, 'WAVE torso: RAISED=%s SW=%.0f TURNS=%d | hand: TURNS=%d x=%s'
                     % (c.get('raised'), c.get('sw', 0.0), c.get('turns_torso', 0),
                        c.get('turns_hand', 0),
                        ('%.2f' % hand_x) if hand_x is not None else '-'),
                     (80, 255, 120) if c.get('turns', 0) >= 2 else (210, 210, 210))
                _txt(132, 'FPS=%.1f  q=%d  ev=%s'
                     % (fps_n / max(0.001, now - fps_t0), runner.q.qsize(),
                        ','.join(events) or '-'), (170, 170, 170))
                if gname == 'palm' and hand_big:
                    _txt(162, '>> PALM: 灯头跟随', (80, 255, 120))
                elif gname == 'ok' and hand_big:
                    _txt(162, '>> OK: 点头', (80, 220, 255))
                if c.get('turns', 0) >= 2:
                    _txt(184, '>> WAVE 挥手', (80, 255, 120))
                scene = (
                    'person=%s area=%.3f   hand_big=%s\n'
                    'palm: n=%d box=%.3f size21=%.3f score=%.2f cx=%s\n'
                    'gest=%s conf=%.2f fingers=%s pinch=%.2f\n'
                    'follow=%s src=%s act=%s\n'
                    'wave torso: raised=%s rise=%.0f sw=%.0f turns=%d wrist_n=%d\n'
                    'wave hand : turns=%d hand_n=%d hand_x=%s\n'
                    'fps=%.1f queue=%d events=%s\n'
                    'step=%s rot=%s models=%s hand_always=%s'
                    % (person, area_ratio, hand_big,
                       gdetail.get('palms', 0), gdetail.get('palm_size_norm', 0.0),
                       gdetail.get('palm_size', 0.0), gdetail.get('palm_score', 0.0),
                       ('%.3f' % gdetail['palm_cx'])
                       if gdetail.get('palm_cx') is not None else '-',
                       gname, gconf, gdetail.get('n_extended', '-'),
                       gdetail.get('pinch_r', -1.0),
                       g_state['following'], g_state['src'], gact,
                       c.get('raised'), c.get('rise', 0.0), c.get('sw', 0.0),
                       c.get('turns_torso', 0), c.get('wrist_n', 0),
                       c.get('turns_hand', 0), c.get('hand_n', 0),
                       ('%.3f' % hand_x) if hand_x is not None else '-',
                       fps_n / max(0.001, now - fps_t0), runner.q.qsize(),
                       ','.join(events) or '-', opt.size, opt.rot,
                       'on' if gd is not None else 'off', opt.hand_always)
                ).encode('utf-8')
                preview.push(frame, scene)

            fps_n += 1
            if now - fps_t0 >= 5.0:
                print('[fps] %.1f  present=%s  events=%s'
                      % (fps_n / (now - fps_t0), det.present,
                         dict(stats) or '{}'), flush=True)
                fps_t0, fps_n = time.time(), 0

            dt = period - (time.time() - t0)
            if dt > 0:
                time.sleep(dt)

            # --demo-once / --demo-wave：不需要有人站在镜头前，跑一次动作做自检
            if opt.demo_once > 0 or opt.demo_wave > 0:
                if demo_state == 0 and (now - demo_t0) >= max(opt.demo_once, opt.demo_wave):
                    demo_state = 1
                    demo_t1 = now
                    if opt.demo_wave > 0:
                        print('[demo] --demo-wave：触发一次挥手响应（拍照 + 自我介绍）', flush=True)
                        runner.submit('wave', do_wave)
                    else:
                        print('[demo] --demo-once：触发一次完整迎宾动作', flush=True)
                        runner.submit('greet', do_greet)
                elif (demo_state == 1 and now - demo_t1 >= 1.0
                        and not runner.busy.is_set()):
                    print('[demo] 动作结束，退出', flush=True)
                    break
    except KeyboardInterrupt:
        print('\n收到 Ctrl+C，退出...', flush=True)
    finally:
        if follower is not None:
            follower.active = False
            follower.stop()
        runner.shutdown()
        print('[exit] 事件汇总: %s' % (dict(stats) or '无'), flush=True)
        if dbg_fp is not None:
            try:
                dbg_fp.close()
                print('[exit] 诊断日志已保存: /tmp/gesture_dbg.jsonl', flush=True)
            except Exception:
                pass
        armctl.finish(park=True)
        try:
            conctl.close()
        except Exception:
            pass
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        rknn.release()
        print('[exit] 资源已释放', flush=True)


if __name__ == '__main__':
    main()
