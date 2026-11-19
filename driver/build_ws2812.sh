#!/bin/bash
# 手动完成内核头文件包 postinst 未执行完的步骤（编译 host 工具），然后编译 WS2812 驱动
# 关键：内核未开启 CONFIG_MODULE_SIG，所以不需要 libssl-dev
KD=/usr/src/linux-headers-6.1.115-vendor-rk35xx
NCPU=$(nproc)

cd "$KD" || exit 1

echo "===== 1) olddefconfig ====="
make ARCH=arm64 olddefconfig 2>&1 | tail -3

echo "===== 2) make scripts (编译 fixdep/modpost 等 host 工具) ====="
make ARCH=arm64 -j"$NCPU" scripts 2>&1 | grep -vE '^\s*(CC|HOSTCC|LD|AR)\s' | tail -25
echo "scripts_rc=$?"

echo "===== 3) make M=scripts/mod/ ====="
make ARCH=arm64 -j"$NCPU" M=scripts/mod/ 2>&1 | tail -8
echo "mod_rc=$?"

echo "===== 4) 关键产物 ====="
ls -l scripts/basic/fixdep scripts/mod/modpost Module.symvers 2>&1

echo "===== 5) 编译 WS2812 驱动与测试程序 ====="
cd /root/ws2812/ws2812 || exit 1
make 2>&1 | tail -15
echo "make_rc=$?"
ls -l ws2812_drv.ko ws2812_app 2>&1
echo "===== 模块信息 ====="
modinfo ws2812_drv.ko 2>&1 | head -6
echo DONE
