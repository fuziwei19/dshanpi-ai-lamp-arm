# docs — 技术文档

本目录用于放**排障记录与集成说明**。这些文档是在板子上边调边写的，属于「为什么这么改」的现场证据，
代码本身在仓库其余目录。

## 已随代码收录

| 文件 | 内容 |
|---|---|
| `collection-manifest.txt` | 从板子收集源码时的逐项清单（OK / MISS / WARN），用于核对有没有漏文件 |

## 待补（在另一台开发机上，尚未同步）

| 文件 | 内容 |
|---|---|
| `TuyaOpen-your_chat_bot_唤醒修复说明.md` | 唤醒词不触发、按键与 KWS 共存的排查过程 |
| `TuyaOpen-喇叭无声修复-单声道SAI问题.md` | ALSA 单声道 SAI 导致的无声问题（对应 `patches/` 里 `tdd_audio_alsa.c` 的改动） |
| `TuyaOpen-情绪灯效与图片理解.md` | 情绪 → 灯效映射，以及 `device_take_photo` + 多模态理解链路 |
| `TuyaOpen-端侧MCP工具移植-灯环与人脸跟随.md` | 14 个 MCP 工具是怎么注册到 `ai_mcp` 的 |
| `TuyaOpen-飞书IM频道集成-使用说明.md` | `app/im/` 的设计与飞书应用配置步骤 |
| `TuyaOpen-手势互动测试与调试说明.md` | MediaPipe 手势模型在 NPU 上的移植与调试 |
| `使用说明书.md` | 面向使用者的操作说明（开机、配网、常用指令） |
