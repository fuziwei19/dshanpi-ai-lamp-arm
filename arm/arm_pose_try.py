#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""arm_pose_try.py — 在迎宾基准位上「试姿态」，用于目视微调（A/B 对比）

背景
----
迎宾基准位（/etc/greet-pose.json 的 base_pose）是从机械臂「实际姿态」里记住的。
挂上台灯之后，「灯头朝向」往往还需要人工微调 —— 比如让灯面水平朝上而不是朝地面。

本脚本让你把某个关节偏几度「试出来」：转过去 → 停住 → 你看一眼 → 再换下一组。
不用改配置文件、不用重启服务，几秒钟就能比较两三个候选值。

用法
----
    # 单步：基准位把 elbow 加 5 度，停 15 秒
    python3 arm_pose_try.py --step 'elbow+=5' --hold 15

    # A/B 对比：先 +5，自动回基准位，再 -14，各停 12 秒
    python3 arm_pose_try.py --step 'elbow+=5' --step 'elbow+=-14' --hold 12

    # 直接给绝对值
    python3 arm_pose_try.py --step 'elbow=44.47' --hold 15

    # 只看会发什么指令，不动臂
    python3 arm_pose_try.py --step 'elbow+=5' --dry-run

    # 试完停在最后一个姿态不回收纳位（力矩保持，Ctrl+C 才放）
    python3 arm_pose_try.py --step 'elbow+=-14' --hold 20 --keep

参数
----
    --step SPEC      一步姿态。SPEC 形如 关节[+]=值
                        'elbow+=5'    -> 基准位 + 5 度（增量）
                        'elbow+=-14'  -> 基准位 - 14 度（增量，负号照写）
                        'elbow=44.47' -> 绝对值 44.47（没有 + 就是绝对值）
                      可重复多次，按顺序依次演示（中间自动回基准位做参照）
    --label TEXT     给对应的那一步起个名字（可重复，与 --step 一一对应，可省）
    --hold SEC       每一步停留多少秒（默认 12）
    --ref-hold SEC   两步之间在基准位停留多少秒（默认取 min(hold, 5)，0 = 不停）
    --pose PATH      基准位文件（默认 /etc/greet-pose.json）
    --port PORT      串口（默认 /dev/ttyACM0）
    --robot-id ID    标定 id（默认 my_awesome_follower_arm）
    --dry-run        只打印计算出来的姿态，不动机械臂
    --keep           结束时保持最后一个姿态（不回收纳位、不释放力矩）

关节短名
--------
    pan    -> shoulder_pan.pos      水平旋转
    lift   -> shoulder_lift.pos     大臂抬降
    elbow  -> elbow_flex.pos        肘
    wf     -> wrist_flex.pos        腕俯仰
    wroll  -> wrist_roll.pos        腕自转
    grip   -> gripper.pos           夹爪
    （全名 / 全名.pos 也认）

⚠️ /dev/ttyACM0 是独占的
    人脸跟随 / 迎宾服务在跑的时候连不上（会报 Device or resource busy）：
        systemctl stop face-follow vision-greet
    除 --keep 外，结束时都会平滑回「收纳位」并释放力矩。

⚠️ 朝向小抄（重要）
    shoulder_lift / elbow_flex / wrist_flex 三个轴互相平行，都是俯仰轴。
    世界俯仰角 ≈ lift + elbow + wf（同向累加，大 = 越朝下）：
        HOME  :  30 - 65 + 50 = 15
        基准位: -77.05 + 39.47 + 51.16 = 13.58   <- 略朝地面
        PARK  : -101.76 + 93.89 + 73.41 = 65.54  <- 明显朝下（收纳位）
    所以想让灯面「抬起来、水平朝上」，要【减小】这个和 ——
    即减小 elbow（或 wf），而不是加大。若实测方向与此相反，以实测为准。
"""
import argparse
import json
import os
import sys
import time

from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.robots.so101_follower.so101_follower import SO101Follower

import arm_actions as arm

DEFAULT_POSE = '/etc/greet-pose.json'

JOINT_ALIAS = {
    'pan': 'shoulder_pan.pos',
    'shoulder_pan': 'shoulder_pan.pos',
    'shoulder_pan.pos': 'shoulder_pan.pos',
    'lift': 'shoulder_lift.pos',
    'shoulder_lift': 'shoulder_lift.pos',
    'shoulder_lift.pos': 'shoulder_lift.pos',
    'elbow': 'elbow_flex.pos',
    'elbow_flex': 'elbow_flex.pos',
    'elbow_flex.pos': 'elbow_flex.pos',
    'wf': 'wrist_flex.pos',
    'wrist': 'wrist_flex.pos',
    'wrist_flex': 'wrist_flex.pos',
    'wrist_flex.pos': 'wrist_flex.pos',
    'wroll': 'wrist_roll.pos',
    'roll': 'wrist_roll.pos',
    'wrist_roll': 'wrist_roll.pos',
    'wrist_roll.pos': 'wrist_roll.pos',
    'grip': 'gripper.pos',
    'gripper': 'gripper.pos',
    'gripper.pos': 'gripper.pos',
}

ORDER = ['shoulder_pan.pos', 'shoulder_lift.pos', 'elbow_flex.pos',
         'wrist_flex.pos', 'wrist_roll.pos', 'gripper.pos']


# ---------------------------------------------------------------------------
# 基准位读取 / 姿态计算
# ---------------------------------------------------------------------------
def load_base_pose(path):
    """读 base_pose；缺失关节用 HOME 补齐（build_pose 缺键会填 0.0，是很危险的角度）"""
    pose = None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            bp = raw.get('base_pose')
            if isinstance(bp, dict) and bp:
                pose = {k: float(v) for k, v in bp.items()}
                print('[pose] 已加载 %s 的 base_pose（保存于 %s）'
                      % (path, raw.get('saved_at', '未知')), flush=True)
    except FileNotFoundError:
        print('[pose] %s 不存在' % path, flush=True)
    except Exception as e:
        print('[pose] 读取 %s 失败(%r)' % (path, e), flush=True)

    if pose is None:
        print('[pose] 回退到内置 HOME_POSE_DEG', flush=True)
        pose = dict(arm.HOME_POSE_DEG)

    filled = []
    for n in ORDER:
        if n not in pose:
            pose[n] = float(arm.HOME_POSE_DEG.get(n, 0.0))
            filled.append(n)
    if filled:
        print('[pose] 以下关节缺失，已用 HOME 补齐: %s' % ', '.join(filled), flush=True)
    return pose


def parse_step(spec):
    """'elbow+=5' / 'elbow+=-14' / 'elbow=44.47' -> (joint, mode, value)"""
    if '+=' in spec:
        name, _, val = spec.partition('+=')
        mode = 'delta'
    elif '=' in spec:
        name, _, val = spec.partition('=')
        mode = 'abs'
    else:
        raise ValueError('缺少 = 号: %r' % spec)
    key = name.strip().lower()
    if key not in JOINT_ALIAS:
        raise ValueError('不认识的关节 %r（可用: %s）'
                         % (name.strip(), ', '.join(sorted(set(JOINT_ALIAS)))))
    try:
        value = float(val)
    except ValueError:
        raise ValueError('数值解析失败: %r' % val)
    return JOINT_ALIAS[key], mode, value


def apply_step(base, joint, mode, value):
    pose = dict(base)
    if mode == 'delta':
        pose[joint] = float(pose.get(joint, 0.0)) + value
    else:
        pose[joint] = value
    return pose


def fmt_pose(pose):
    return '  '.join('%s=%.2f' % (n.split('.')[0], pose.get(n, 0.0)) for n in ORDER)


def pitch_sum(pose):
    """三个平行俯仰角之和 —— 越大越朝下（详见文件头「朝向小抄」）"""
    return (pose.get('shoulder_lift.pos', 0.0)
            + pose.get('elbow_flex.pos', 0.0)
            + pose.get('wrist_flex.pos', 0.0))


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description='在迎宾基准位上试姿态（目视 A/B 微调）',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--step', action='append', default=[], metavar='SPEC',
                    help="关节[+]=值，如 'elbow+=5' / 'elbow+=-14' / 'elbow=44.47'，可重复")
    ap.add_argument('--label', action='append', default=[], metavar='TEXT',
                    help='给对应那一步起名字，可重复')
    ap.add_argument('--hold', type=float, default=12.0, metavar='SEC',
                    help='每步停留秒数（默认 12）')
    ap.add_argument('--ref-hold', type=float, default=None, metavar='SEC',
                    help='两步之间在基准位停留秒数（默认 min(hold,5)，0=不停）')
    ap.add_argument('--pose', default=DEFAULT_POSE, help='基准位文件')
    ap.add_argument('--port', default='/dev/ttyACM0')
    ap.add_argument('--robot-id', default='my_awesome_follower_arm')
    ap.add_argument('--dry-run', action='store_true', help='只打印不动臂')
    ap.add_argument('--keep', action='store_true',
                    help='结束时保持最后姿态，不回收纳位、不释放力矩')
    opt = ap.parse_args()

    if not opt.step:
        ap.error('至少给一个 --step，例如 --step "elbow+=5"')

    base = load_base_pose(opt.pose)

    # 解析每一步
    steps = []
    for i, spec in enumerate(opt.step):
        joint, mode, value = parse_step(spec)
        label = (opt.label[i] if i < len(opt.label)
                 else '%s %s %+.2f' % (joint.split('.')[0], '增量' if mode == 'delta' else '绝对', value))
        steps.append((label, apply_step(base, joint, mode, value)))

    ref_hold = (min(opt.hold, 5.0) if opt.ref_hold is None else max(0.0, opt.ref_hold))

    # 播放序列：基准位 -> 第1步 -> 基准位 -> 第2步 -> ...
    seq = [('基准位(base)', dict(base), ref_hold if ref_hold > 0 else 0.0)]
    for i, (label, pose) in enumerate(steps):
        seq.append((label, pose, opt.hold))
        if ref_hold > 0 and i != len(steps) - 1:
            seq.append(('基准位(base)', dict(base), ref_hold))

    # 先把要演的姿态全部打印出来
    print()
    print('=== 演示序列（共 %d 段，约 %.0f 秒动作 + %.0f 秒停留）==='
          % (len(seq), len(seq) * 1.5, sum(s[2] for s in seq)))
    for i, (label, pose, hold) in enumerate(seq, 1):
        print('  %d) %-28s 停 %4.1fs  俯仰和 %+7.2f  |  %s'
              % (i, label, hold, pitch_sum(pose), fmt_pose(pose)))
    print()
    print('  基线基准位俯仰和 = %+.2f（越大越朝下；想抬起来就往小改）' % pitch_sum(base))
    print()

    if opt.dry_run:
        print('[dry-run] 不动机械臂，退出。')
        return 0

    # 连接
    print('[arm] 连接 %s ...' % opt.port, flush=True)
    cfg = SO101FollowerConfig(port=opt.port, id=opt.robot_id,
                              cameras={}, use_degrees=True)
    robot = SO101Follower(cfg)
    try:
        robot.connect()
    except Exception as e:
        msg = str(e)
        print('[error] 连接失败: %s' % msg, file=sys.stderr)
        if 'busy' in msg.lower() or 'Resource temporarily unavailable' in msg:
            print('        串口被占用，先停掉占用它的服务：', file=sys.stderr)
            print('        systemctl stop face-follow vision-greet', file=sys.stderr)
        return 1

    names = list(robot.action_features.keys())
    print('[arm] 已连接 joints=%s' % names, flush=True)

    rc = 0
    try:
        for i, (label, pose, hold) in enumerate(seq, 1):
            print('[%d/%d] -> %s' % (i, len(seq), label), flush=True)
            arm.goto(robot, names, pose, duration=1.5)
            if hold > 0:
                time.sleep(hold)
    except KeyboardInterrupt:
        print('\n[arm] 收到 Ctrl+C，提前结束', flush=True)
    except Exception as e:
        print('[error] 动作异常: %r' % e, file=sys.stderr)
        rc = 1
    finally:
        try:
            if opt.keep:
                print('[arm] --keep：保持当前姿态，仅断开串口（力矩保持）', flush=True)
                robot.bus.disconnect(disable_torque=False)
            else:
                print('[arm] 平滑回收纳位...', flush=True)
                arm.goto_park(robot, names, duration=3.0)
                robot.bus.disconnect(disable_torque=True)
                print('[arm] 已释放力矩', flush=True)
        except Exception as e:
            print('[warn] 收尾失败: %r' % e, file=sys.stderr)

    print()
    print('=== 参考：各候选姿态的关节角度（可直接让我写进 /etc/greet-pose.json）===')
    for label, pose, _ in seq:
        print('  %-28s %s' % (label, json.dumps(pose, ensure_ascii=False)))
    return rc


if __name__ == '__main__':
    sys.exit(main())
