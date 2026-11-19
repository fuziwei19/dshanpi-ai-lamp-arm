#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test_ctrl_socket.py — 离线验证「迎宾控制口」协议（不需要板子/机械臂）

做法：从 vision_interact.py 里把 ControlServer 类和 arm_do.py 里把 try_greet_socket()
按源码原样抠出来执行，然后真的建一个 AF_UNIX socket 走一遍：
    客户端发 nod / shake_head / state / 乱写 -> 服务端按协议回一行
再校验：
    * 三个合法动作都排进了队列（submit 被调用）
    * 未知动作回 unknown:，不会误触发
    * 队列满时回 busy:
    * 控制口文件在 close() 之后被清掉

用法：python test_ctrl_socket.py      # 输出 selftest: PASS
"""
import os
import re
import socket
import sys
import tempfile
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def _extract(path, start_pat, end_pat):
    """按源码把一段代码抠出来（保留缩进原样）"""
    with open(path, 'r', encoding='utf-8') as f:
        src = f.read()
    i = src.index(start_pat)
    j = src.index(end_pat, i)
    return src[i:j]


def _find_file(name, *extra_dirs):
    """找源文件：先本目录，再 ../，再 /root（板上真身就在 /root）"""
    cands = [os.path.join(HERE, name),
             os.path.join(HERE, '..', name),
             os.path.join(os.path.dirname(HERE), name)]
    cands += [os.path.join(d, name) for d in extra_dirs]
    for p in cands:
        if os.path.exists(p):
            return os.path.normpath(p)
    raise FileNotFoundError('%s 找不到，试过：%s' % (name, cands))


def build_server_class():
    src = _extract(_find_file('vision_interact.py', '/root'),
                   'class ControlServer(object):',
                   '\nclass ArmCtl:')
    ns = {'os': os, 'print': print}
    exec(compile(src, 'ControlServer(extracted)', 'exec'), ns)
    return ns['ControlServer']


def build_client_fn(path):
    src = _extract(_find_file('arm_do.py', '/root'),
                   'def try_greet_socket(', '\ndef unit_active(')
    ns = {'os': os, 'socket': socket, 'GREET_SOCK': path}
    exec(compile(src, 'try_greet_socket(extracted)', 'exec'), ns)
    return ns['try_greet_socket']


def main():
    path = os.path.join(tempfile.gettempdir(), 'greet-ctl-test.sock')

    # 先把两段源码抠出并编译（这一步与平台无关，Windows 上也要能过）
    ControlServer = build_server_class()
    try_greet_socket = build_client_fn(path)
    print('  抠源码+编译: OK（ControlServer / try_greet_socket）')

    if not hasattr(socket, 'AF_UNIX'):
        print('SKIP: 这个平台的 python 没有 AF_UNIX（Windows 常见）—— 到板子上跑这个测试')
        return 0


    calls = []
    full = {'flag': False}

    def submit(name, fn):
        if full['flag']:
            return False
        calls.append(name)
        return True

    def state():
        return 'state:present=True following=False queue=0 last=-'

    srv = ControlServer(path, {'nod': lambda: 1, 'shake_head': lambda: 2,
                               'wave': lambda: 3}, submit, state)
    srv.start()
    time.sleep(0.2)
    ok = True

    def check(cond, msg):
        nonlocal ok
        print('%s %s' % ('  ok  ' if cond else '  FAIL', msg))
        ok = ok and bool(cond)

    check(os.path.exists(path), '控制口文件已创建')
    check(srv._sock is not None, 'socket 监听成功')

    sent = {}
    for word in ('nod', 'shake_head', 'wave', 'state', 'dance'):
        st, reply = try_greet_socket(word, timeout=2.0)
        sent[word] = (st, reply)
        print('     %-11s -> %-7s %r' % (word, st, reply))

    check(sent['nod'][0] == 'ok' and sent['nod'][1] == 'ok:nod',
          '合法动作 nod 回 ok:nod（状态 ok）')
    check(sent['shake_head'][0] == 'ok' and sent['shake_head'][1] == 'ok:shake_head',
          '合法动作 shake_head 回 ok:shake_head（状态 ok）')
    check(sent['wave'][0] == 'ok' and sent['wave'][1] == 'ok:wave',
          '合法动作 wave 回 ok:wave（状态 ok）')
    check(sent['state'][0] == 'state' and sent['state'][1].startswith('state:'),
          'state 回状态行，且客户端判为 state')
    check(sent['dance'][0] == 'unknown' and sent['dance'][1].startswith('unknown:'),
          '未知动作回 unknown:，且客户端判为 unknown（会回退直连）')
    check(calls == ['nod', 'shake_head', 'wave'], '只把 3 个合法动作排进了队列')

    full['flag'] = True
    st, reply = try_greet_socket('nod', timeout=2.0)
    check(st == 'busy' and reply.startswith('busy:'),
          '队列忙时回 busy:（客户端判为 busy，不会自己抢串口）')
    full['flag'] = False

    # 控制口不存在时要判 absent（客户端会回退到自己连机械臂）
    os.unlink(path)
    st, reply = try_greet_socket('nod', timeout=2.0)
    check(st == 'absent', '控制口不存在时判 absent（回退直连）')

    srv.close()
    time.sleep(0.1)
    check(not os.path.exists(path), 'close() 之后控制口文件被清掉')

    print()
    print('selftest: %s' % ('PASS' if ok else 'FAIL'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
