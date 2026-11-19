# Makefile for hello_drv kernel module

# 模块名，不需要写 .c 后缀
obj-m := ws2812_drv.o

# 内核构建目录（自动指向当前运行内核的 headers）
KDIR := /lib/modules/$(shell uname -r)/build

# 当前路径
PWD := $(shell pwd)

# 默认目标：编译驱动模块 + 应用
all: module app

# 编译内核模块
module:
	$(MAKE) -C $(KDIR) M=$(PWD) modules

# 编译应用程序
app:
	$(CC) ws2812_app.c -o ws2812_app

# 清理目标文件
clean:
	$(MAKE) -C $(KDIR) M=$(PWD) clean
	rm -f ws2812_app
