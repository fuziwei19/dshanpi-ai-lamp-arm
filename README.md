# DshanPi-A1 AI 台灯机械臂

在 **DshanPi-A1（RK3576）** 上把三件事拼成一个会看、会说、会动的桌面搭档：

1. **涂鸦 AI 语音对话**（TuyaOpen `your_chat_bot`，麦克风唤醒 + ASR + LLM + TTS）
2. **WS2812 情绪灯环**（自写内核驱动 + 用户态守护进程，随对话情绪变色）
3. **Lerobot SO-101 机械臂 + 摄像头视觉**（人脸/手势识别 → 迎宾、点头、摇头、招手、人脸跟随）

AI 大模型可以通过**端侧 MCP 工具**直接控制灯和机械臂——「把灯调成暖黄色」「跟我打个招呼」，它会自己去调。

---

## 硬件清单

| 部件 | 型号 / 规格 | 说明 |
|---|---|---|
| 主控板 | DshanPi-A1（RK3576，4×A72 + 4×A53，6 TOPS NPU） | Linux 侧跑 TuyaOpen + Python 服务 |
| 机械臂 | Lerobot **SO-101**（6 自由度，串口总线舵机） | 走 `/dev/ttyACM*`（Feetech 总线舵机） |
| 灯环 | WS2812B × 24（SPI2 驱动，DMA 时序） | 走 SPI2 + 自定义内核模块 |
| 摄像头 | USB UVC 摄像头 | OpenCV 采集，NPU 跑 YOLOv8-pose |
| 音频 | I2S/SAI 单声道麦克风 + 喇叭 | ALSA `plughw:0,0` |

> 灯环的像素数在 `driver/ws2812.dts`（`leds,number`）和 `systemd/ws2812d.service`（`--count 24`）两处，接线不同时都要改。

## 软件栈

| 层 | 组件 | 版本 / 备注 |
|---|---|---|
| AI 框架 | [TuyaOpen](https://github.com/tuya/TuyaOpen) | v1.9.0 系列，Linux target（`boards/LINUX/DshanPi_A1`） |
| 系统 | Armbian（Ubuntu noble）/ Linux 6.1.115-vendor-rk35xx | 内核需带 headers 才能编 WS2812 模块 |
| 机械臂 | `lerobot`（`SO101Follower`） | Python 3.10 venv |
| 视觉 | OpenCV + `rknnlite`（RKNN Toolkit Lite2 2.3.2） | NPU 推理 |
| 手势 | MediaPipe palm detection / hand pose（ONNX 转 RKNN） | 模型需自行下载，见下文 |
| 灯环 | 自写 out-of-tree 内核模块 + `ws2812d.py` | 用户态 socket 控制 |
| 语音 | TuyaOpen 内置 KWS（mdtc_chunk_300ms.mnn）+ 云端 ASR/TTS | 模型放 `~/tuyaopen_models/` |

## 系统架构

```
                        ┌──────────────────────────────────────────┐
   麦克风 ──KWS──►       │  your_chat_bot (TuyaOpen, C)             │
   喇叭   ◄──TTS──       │  ├─ ai_chat: 唤醒词 / 按键 / 打断         │
                        │  ├─ app_mcp_tools: 14 个端侧 MCP 工具     │
   屏幕   ◄──LVGL──      │  ├─ app_emo_lamp: 情绪 → 灯效映射         │
                        │  ├─ app_vision: 视觉事件 → AI 主动说话     │
                        │  └─ im/: 飞书 IM 频道（可选）             │
                        └───────┬───────────────┬──────────────────┘
                                │ /run/tuya_chatbot.sock            │
                                │ (文本注入 AI，让它主动开口)         │
        ┌───────────────────────┴─────────┐     │
        │ vision_interact.py (迎宾大脑)     │     │ /run/vision-greet.sock
        │ ├─ 人脸/手势检测 (rknnlite)       │◄────┘ (机械臂控制口，串行化动作)
        │ ├─ greet_logic.py 状态机          │
        │ ├─ led_client.py ──► ws2812d.py  │──► /dev/ws2812 (SPI2)
        │ └─ arm_actions.py ──► SO-101      │──► /dev/ttyACM0
        └──────────────────────────────────┘
```

关键点：**`/dev/ttyACM0` 是独占的**，迎宾、人脸跟随、单次动作三条路径不能同时抓串口，
所以统一由 `vision_interact.py` 的控制口排队执行（`arm_do.py` 就是这么做的）。

## 目录结构

```
.
├── app/              # TuyaOpen 应用 your_chat_bot（整份，含自研 src/include/im）
│   ├── src/          # app_chat_bot / app_mcp_tools / app_emo_lamp / app_vision / app_feishu …
│   ├── im/           # 飞书 IM 频道（自研）：channels/feishu_bot.c、bus/message_bus.c
│   ├── config/       # 板级配置（DshanPi_A1.config 等）
│   ├── app_default.config
│   └── run_chatbot.sh
├── driver/           # WS2812 内核模块：ws2812_drv.c + dts overlay + Makefile + 构建脚本
├── led/              # ws2812d.py 灯环守护进程 + 设备就绪等待脚本
├── arm/              # 机械臂：动作库、单次动作执行、示教位姿采集
├── vision/           # 迎宾/跟随：vision_interact.py、greet_logic.py、YOLOv8-pose、gesture/
├── systemd/          # chatbot / ws2812d / vision-greet / face-follow 四个 unit
├── config/           # greet-pose.json、gesture-map.json、feishu.json.example（脱敏）
├── tools/            # 运维脚本：启动停止、部署、控制口客户端、MCP 灯控桥
├── patches/          # 对 TuyaOpen 官方代码的本地改动（unified diff）
└── docs/             # 技术文档 + 收集清单
```

## 快速开始

### 0. 前置

```bash
# 板子上需要：git、python3.10 venv、内核 headers、OpenCV
sudo apt install -y git python3-venv linux-headers-$(uname -r)
```

### 1. 拿 TuyaOpen 并打补丁

```bash
git clone https://github.com/tuya/TuyaOpen.git ~/TuyaOpen
cd ~/TuyaOpen && git checkout <本仓库 patches 对应的基线 commit，见 patches/tuyaopen-git-status.txt>
git apply --stat ../patches/tuyaopen-local-changes.patch   # 先看改了什么
git apply      ../patches/tuyaopen-local-changes.patch
```

补丁内容（详见 `patches/tuyaopen-local-changes.patch`，5 个文件 / 约 270 行）：

| 文件 | 作用 |
|---|---|
| `boards/LINUX/DshanPi_A1/board_com_api.c` | 按键注册名改为 `ai_chat_button`，与 `ai_chat_main.c` 对齐 |
| `boards/LINUX/DshanPi_A1/tdd_button_keyboard.c/.h` | 键盘按键增加 `key_char` 字段，支持按字符匹配 |
| `src/ai_components/ai_mode/src/ai_mode_free.c` | 免费/自建模型通路适配 |
| `src/peripherals/audio_codecs/tdd_audio/src/tdd_audio_alsa.c` | ALSA 单声道 SAI 录音/播放修复（解决喇叭无声） |

### 2. 放应用并编译

```bash
cp -r app ~/TuyaOpen/apps/tuya.ai/your_chat_bot
cd ~/TuyaOpen/apps/tuya.ai/your_chat_bot

# 改 PID！（app_default.config 里是占位符，PID 与设备授权绑定）
vi app_default.config          # CONFIG_TUYA_PRODUCT_ID="你的 PID"

tos.py config choice           # 选 DshanPi_A1
tos.py build
```

KWS 模型放 `~/tuyaopen_models/mdtc_chunk_300ms.mnn` 与 `tokens.txt`（路径在 `app_default.config` 的 `CONFIG_KWS_MODEL_PATH`）。

### 3. 编内核模块（灯环）

```bash
cd driver && make          # 产出 ws2812_drv.ko
# dts overlay：用 armbian-config 或 /boot/overlay-user 挂 ws2812.dts，重启后出现 /dev/ws2812
sudo insmod ws2812_drv.ko
```

### 4. 部署用户态服务

```bash
sudo install -m755 led/ws2812d.py led/ws2812_wait_dev.sh /usr/local/bin/
sudo install -m644 systemd/*.service /etc/systemd/system/
sudo mkdir -p /etc && sudo cp config/greet-pose.json config/gesture-map.json /etc/
sudo cp config/feishu.json.example /etc/feishu.json && sudo vi /etc/feishu.json   # 填自己的飞书应用

# 机械臂第一次用要先标定
python3 arm/arm_pose_save.py    # 采集迎宾位姿 → /etc/greet-pose.json

sudo systemctl daemon-reload
sudo systemctl enable --now ws2812d chatbot vision-greet
```

### 5. 模型文件（不入库）

| 文件 | 用途 | 放置位置 |
|---|---|---|
| `palm_detection_mediapipe_2023feb.onnx` | 手掌检测 | `vision/gesture/` |
| `hand_landmarker.task` / `handpose_estimation_mediapipe_2023feb.onnx` | 手势关键点 | `vision/gesture/` |
| YOLOv8-pose RKNN 模型 | 人脸/头部跟随 | `vision/`（路径见 `yolov8pose_head_follow_so101.py`） |
| `mdtc_chunk_300ms.mnn` + `tokens.txt` | 唤醒词 | `~/tuyaopen_models/` |

## 端侧 MCP 工具

`app/src/app_mcp_tools.c` 注册了 14 个工具，AI 可直接调用：

| 工具 | 作用 |
|---|---|
| `device_light_power_set` / `device_light_control` / `device_light_state_get` | 灯环开关、颜色/亮度/动效、状态查询 |
| `device_arm_nod` / `device_arm_shake_head` | 点头 / 摇头 |
| `device_greeting_start` / `device_greeting_stop` / `device_greeting_status_get` | 迎宾模式起停与状态 |
| `device_greeting_pose_get` / `device_greeting_pose_save` | 迎宾基准位姿读写 |
| `device_face_follow_start` / `device_face_follow_stop` / `device_face_follow_status_get` | 人脸跟随 |
| `device_take_photo` | 拍照（配合多模态理解） |

## 运行时接口

| 接口 | 方向 | 说明 |
|---|---|---|
| `/run/tuya_chatbot.sock` | Python → AI | 注入一段文本让 AI 主动说话（`vision/notify_ai.py`） |
| `/run/vision-greet.sock` | Python → 迎宾进程 | 机械臂动作排队执行（`tools/led_client.py`、`arm/arm_do.py`） |
| `/run/ws*` | Python → ws2812d | 灯环控制 |
| `/dev/ws2812` | 内核模块 | SPI2 灯环设备节点 |
| `/dev/ttyACM0` | Lerobot SO-101 | 总线舵机串口，独占 |

## 注意事项

- **`app_default.config` 的 PID 是占位符**。PID 与设备授权绑定，clone 后必须换成自己的，否则设备激活失败。
- **`tuyadb/`（设备激活 blob）不入库**，`run_chatbot.sh` 必须在 app 目录下运行（KV 库是相对 cwd 的）。
- **`config/feishu.json` 只提供 `.example`**，真实 `app_id` / `app_secret` 请自行填写，不要提交。
- `app/src/display2/` 下的字体、GIF 是 LVGL 生成的 C 数组（约 9 MB），体积大但编译需要，未做裁剪。
- `app/src/app_*.c.bak*`、`*.pre*` 这类历史备份在收集时已全部排除。

## 文档

`docs/` 下另有几份排障/集成记录（唤醒修复、喇叭无声、情绪灯效与图片理解、端侧 MCP 工具移植、飞书 IM 集成、手势互动调试），
见 `docs/README.md` 的清单。

## License

[Apache-2.0](LICENSE)。`app/` 与 `patches/` 是对 [TuyaOpen](https://github.com/tuya/TuyaOpen)（Apache-2.0）的衍生，
`app/im/` 目录保留其原始 LICENSE。
