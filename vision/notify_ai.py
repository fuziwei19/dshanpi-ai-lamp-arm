#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""notify_ai.py — 把一段文本塞给 your_chat_bot，让 AI 说话（端侧文本对话入口）

原理
----
your_chat_bot 里跑着一个本地 UNIX socket 服务（/run/tuya_chatbot.sock），
它收到文本后会调用 TuyaOpen 的 `ai_agent_send_text()`，等价于「用户说了这句话」：
AI 会正常思考 + TTS 回复。视觉迎宾脚本就是靠它让 AI 主动开口打招呼的。

用法
----
    python3 notify_ai.py 打个招呼          # -> {"cmd":"say","text":"打个招呼"}
    python3 notify_ai.py --status          # 查询 AI 是否就绪
    python3 notify_ai.py --retry 5 打个招呼 # 未就绪时最多重试 5 次
    echo '打个招呼' | python3 notify_ai.py -   # 从 stdin 读

返回值
------
    0 = 已被 AI 接受    1 = 失败（socket 不可用 / AI 未就绪）
"""
import argparse
import json
import socket
import sys
import time

DEFAULT_SOCK = '/run/tuya_chatbot.sock'


def send_line(sock_path, payload, timeout=3.0):
    """连上 socket 发一行，等一行回复。返回 (ok, 回复原文)。"""
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect(sock_path)
        s.sendall(payload.encode('utf-8') + b'\n')
        buf = b''
        while not buf.endswith(b'\n'):
            chunk = s.recv(256)
            if not chunk:
                break
            buf += chunk
        return True, buf.decode('utf-8', 'replace').strip()
    finally:
        s.close()


def main():
    ap = argparse.ArgumentParser(description='让 your_chat_bot 里的 AI 说一句话')
    ap.add_argument('text', nargs='?', default=None, help='要说的话；传 - 表示从 stdin 读')
    ap.add_argument('--sock', default=DEFAULT_SOCK)
    ap.add_argument('--status', action='store_true', help='只查询 AI 是否就绪')
    ap.add_argument('--retry', type=int, default=1, help='AI 未就绪时的重试次数（默认 1）')
    ap.add_argument('--retry-delay', type=float, default=1.0)
    ap.add_argument('--json', action='store_true', help='直接发原始 JSON 行')
    opt = ap.parse_args()

    if opt.status:
        payload = '{"cmd":"status"}'
    else:
        text = opt.text
        if text == '-':
            text = sys.stdin.read().strip()
        if not text:
            ap.error('要么给一句话，要么用 --status')
        payload = text if opt.json else json.dumps({'cmd': 'say', 'text': text},
                                                   ensure_ascii=False)

    last = ''
    for attempt in range(max(1, opt.retry)):
        try:
            ok, resp = send_line(opt.sock, payload)
        except FileNotFoundError:
            print('[error] socket 不存在: %s（your_chat_bot 没在跑？）' % opt.sock,
                  file=sys.stderr)
            return 1
        except ConnectionRefusedError:
            print('[error] socket 拒绝连接: %s（服务还没起来？）' % opt.sock, file=sys.stderr)
            return 1
        except Exception as e:
            print('[error] 连接 %s 失败: %r' % (opt.sock, e), file=sys.stderr)
            return 1

        last = resp
        try:
            obj = json.loads(resp) if resp else {}
        except Exception:
            obj = {}
        if obj.get('ok'):
            print(resp)
            return 0
        if attempt + 1 < max(1, opt.retry):
            time.sleep(opt.retry_delay)

    print('[warn] 未被接受: %s' % (last or '(空回复)'), file=sys.stderr)
    return 1


if __name__ == '__main__':
    sys.exit(main())
