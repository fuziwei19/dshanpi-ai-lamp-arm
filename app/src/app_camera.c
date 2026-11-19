/**
 * @file app_camera.c
 * @brief 摄像头拍照实现（自包含 V4L2 / UVC，MJPEG 单帧抓取）
 *
 * 为什么不直接用平台层的 tkl_camera_v4l2_*
 * ----------------------------------------
 * platform/LINUX/tuyaos_adapter/tuyaopen_adapter.cmake 里，摄像头源文件是条件编译的：
 *     if(CONFIG_ENABLE_CAMERA STREQUAL "y" AND CONFIG_ENABLE_CAMERA_V4L2 STREQUAL "y")
 *         list(APPEND SOURCES .../tkl_camera/tkl_camera_v4l2.c)
 *         list(APPEND SOURCES .../tkl_jpeg_codec/tkl_jpeg_codec.c)   # 顺带拉进预编译 libjpeg.a
 * 本板（DshanPi_A1）这两个 Kconfig 都是关的，链接期直接 undefined reference。
 * 与其去动 SDK 的 Kconfig / cmake（还会被 tos.py update 覆盖），不如在 app 内自带
 * 一份等价的抓帧实现。流程与 tkl_camera_v4l2.c 完全一致：
 *
 *     open(O_RDWR|O_NONBLOCK) -> VIDIOC_QUERYCAP -> VIDIOC_S_FMT
 *       -> VIDIOC_REQBUFS(MMAP,4) -> 每个 buf: QUERYBUF + mmap + QBUF
 *       -> VIDIOC_STREAMON -> select + VIDIOC_DQBUF -> 拷贝 -> VIDIOC_QBUF
 *       -> VIDIOC_STREAMOFF -> munmap -> close
 *
 * 设备节点：自动探测，不硬编码（重要！）
 * --------------------------------------
 * 一个 UVC 摄像头会注册两个 video 节点：capture + metadata。两者的编号
 * **会随 USB 重新枚举而漂移** —— 实测同一块板子上，capture 节点从 /dev/video12
 * 变成了 /dev/video11，而硬编码 video12 之后就必然 VIDIOC_S_FMT 失败(EINVAL)，
 * 因为 video12 变成了 metadata 节点。所以这里改为：
 *   1) /etc/ai-camera.dev 第一行若以 '/' 开头 -> 强制使用该节点
 *   2) 否则扫描 /dev/video0..N，用 QUERYCAP 找真正的采集节点，优先 USB 设备
 *   3) 结果缓存；若缓存节点失效则自动重新探测
 *
 * 与板载视觉服务互斥：让路协议 /run/cam-want
 * -----------------------------------------
 * 一个 UVC 摄像头同一时刻只能被一个进程 STREAMON，而板上的
 * vision-greet.service（迎宾）/ face-follow.service（人脸跟随）会长期占着它。
 * 所以拍照前先创建 /run/cam-want，对端（已打补丁的 vision_interact.py）看到后
 * 会 cap.release() 并等待；本模块随后重试抓帧，拍完删除该文件，对端自动拿回。
 * 若对端没装补丁，重试若干次后仍会失败，此时给出「被占用」的明确提示。
 *
 * 错误归因
 * --------
 * 「节点压根不是 capture」（EINVAL）和「节点被别人占着」（EBUSY）要分清楚，
 * 否则日志会把「节点选错」误报成「摄像头不支持 MJPG」。
 *
 * @copyright Copyright (c) 2026
 */

#include "app_camera.h"

#include <errno.h>
#include <fcntl.h>
#include <linux/videodev2.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/select.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "tal_api.h"

#ifndef O_CLOEXEC
#define O_CLOEXEC 0
#endif

/***********************************************************
************************macro define************************
***********************************************************/
#define APP_CAMERA_DEV_CONF      "/etc/ai-camera.dev"  /* 第一行可强制指定节点 */
#define APP_CAMERA_DEV_SCAN_MAX  (32)                  /* 扫描 /dev/video0..31 */
#define APP_CAMERA_WANT_FLAG     "/run/cam-want"       /* 拍照前通知板上视觉服务让路 */
#define APP_CAMERA_ACQUIRE_TRY   (12)                  /* 等对端让出的最大重试次数 */
#define APP_CAMERA_ACQUIRE_WAIT  (150)                 /* 每次重试间隔(ms) → 最多约 1.8s */
#define APP_CAMERA_WIDTH         (640)
#define APP_CAMERA_HEIGHT        (480)
#define APP_CAMERA_FPS           (15)
#define APP_CAMERA_BUFS          (4)
#define APP_CAMERA_WARMUP        (2)      /* 丢弃前 N 帧，等自动曝光/白平衡稳定 */
#define APP_CAMERA_MAX_TRY       (8)      /* 最多等帧次数，防止死等 */
#define APP_CAMERA_MIN_JPEG      (256)    /* 小于这个长度视为无效帧 */
#define APP_CAMERA_WAIT_MS       (2000)   /* 单次等帧超时 */

/* __probe_node 的返回值 */
#define APP_CAM_NODE_DEAD        (0)   /* 不是采集节点 / 打不开 / 不支持 MJPG */
#define APP_CAM_NODE_OK          (1)   /* 是采集节点，且能配成 MJPG */
#define APP_CAM_NODE_BUSY        (2)   /* 是采集节点，但被别的进程占着 */

/***********************************************************
***********************typedef define***********************
***********************************************************/
typedef struct {
    void  *start;
    size_t length;
} CAM_BUF_T;

/***********************************************************
***********************variable define**********************
***********************************************************/
static char sg_dev[64]        = {0};   /* 当前使用的节点，空 = 尚未探测 */
static char sg_dev_forced[64] = {0};   /* /etc/ai-camera.dev 强制指定的节点 */
static char sg_err[256]       = {0};
/* 同一时刻只允许一次抓帧：MCP 工具回调与 app_vision 工作线程都可能调进来 */
static MUTEX_HANDLE sg_lock = NULL;

/***********************************************************
***********************function define**********************
***********************************************************/
static void __set_err(const char *fmt, ...)
{
    va_list ap;

    va_start(ap, fmt);
    vsnprintf(sg_err, sizeof(sg_err), fmt, ap);
    va_end(ap);
}

static int __xioctl(int fd, unsigned long request, void *arg)
{
    int r;

    do {
        r = ioctl(fd, request, arg);
    } while (r == -1 && errno == EINTR);
    return r;
}

static void __msleep(unsigned ms)
{
    struct timespec ts;

    ts.tv_sec  = ms / 1000;
    ts.tv_nsec = (long)(ms % 1000) * 1000000L;
    (void)nanosleep(&ts, NULL);
}

/* 通知板上视觉服务（迎宾/跟随）把摄像头让出来 */
static void __want_camera(void)
{
    FILE *fp = fopen(APP_CAMERA_WANT_FLAG, "w");

    if (NULL != fp) {
        (void)fprintf(fp, "%d\n", (int)getpid());
        (void)fclose(fp);
    }
}

/* 拍完了，撤掉请求，对端会自动把摄像头拿回去 */
static void __yield_camera(void)
{
    (void)unlink(APP_CAMERA_WANT_FLAG);
}

/* 读 /etc/ai-camera.dev 第一行作为强制设备节点；文件不存在则走自动探测 */
static void __load_conf(void)
{
    FILE *fp = fopen(APP_CAMERA_DEV_CONF, "r");
    char  line[64];

    if (NULL == fp) {
        return;
    }
    if (NULL != fgets(line, sizeof(line), fp)) {
        size_t n = strlen(line);
        while (n > 0 && (line[n - 1] == '\n' || line[n - 1] == '\r' || line[n - 1] == ' ')) {
            line[--n] = '\0';
        }
        if (n > 0 && line[0] == '/') {
            snprintf(sg_dev_forced, sizeof(sg_dev_forced), "%s", line);
        }
    }
    fclose(fp);
}

/**
 * 试一个节点是否可用作「MJPG 单帧采集」
 * 返回 APP_CAM_NODE_xxx
 */
static int __probe_node(const char *path)
{
    int                     fd;
    struct v4l2_capability  cap;
    struct v4l2_format      fmt;
    int                     ret = APP_CAM_NODE_DEAD;

    fd = open(path, O_RDWR | O_NONBLOCK | O_CLOEXEC);
    if (fd < 0) {
        return APP_CAM_NODE_DEAD;
    }

    memset(&cap, 0, sizeof(cap));
    if (__xioctl(fd, VIDIOC_QUERYCAP, &cap) == -1) {
        (void)close(fd);
        return APP_CAM_NODE_DEAD;
    }

    /* metadata / ISP 节点没有 VIDEO_CAPTURE，直接排除 */
    if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE) ||
        !(cap.capabilities & V4L2_CAP_STREAMING)) {
        (void)close(fd);
        return APP_CAM_NODE_DEAD;
    }

    memset(&fmt, 0, sizeof(fmt));
    fmt.type                = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width       = APP_CAMERA_WIDTH;
    fmt.fmt.pix.height      = APP_CAMERA_HEIGHT;
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_MJPEG;
    fmt.fmt.pix.field       = V4L2_FIELD_ANY;

    if (__xioctl(fd, VIDIOC_S_FMT, &fmt) == 0) {
        ret = APP_CAM_NODE_OK;
    } else if (EBUSY == errno) {
        ret = APP_CAM_NODE_BUSY;   /* 节点是对的，只是正被别人用 */
    } else {
        ret = APP_CAM_NODE_DEAD;   /* 比如 metadata 节点：S_FMT 返回 EINVAL */
    }

    (void)close(fd);
    return ret;
}

/* 判断节点是不是 USB 摄像头（bus_info 里带 usb） */
static int __is_usb_node(const char *path)
{
    int                    fd;
    struct v4l2_capability cap;
    int                    usb = 0;

    fd = open(path, O_RDWR | O_NONBLOCK | O_CLOEXEC);
    if (fd < 0) {
        return 0;
    }
    memset(&cap, 0, sizeof(cap));
    if (__xioctl(fd, VIDIOC_QUERYCAP, &cap) == 0) {
        if (NULL != strstr((const char *)cap.bus_info, "usb")) {
            usb = 1;
        }
    }
    (void)close(fd);
    return usb;
}

/**
 * 挑一个可用的采集节点（结果缓存到 sg_dev）
 * 优先级：/etc/ai-camera.dev 指定 > 缓存仍可用 > 扫描(USB 优先) > 被占用的备选
 */
static const char *__pick_device(void)
{
    int  pass;
    char path[64];
    char busy_path[64] = {0};

    /* 1) conf 强制指定 */
    if ('\0' != sg_dev_forced[0]) {
        snprintf(sg_dev, sizeof(sg_dev), "%s", sg_dev_forced);
        return sg_dev;
    }

    /* 2) 缓存的节点只要还是采集节点就直接复用；busy 也交给上层重试 ——
          否则每次拍照都要全量扫描，还会重复刷 "all capture nodes busy" 的日志 */
    if ('\0' != sg_dev[0]) {
        int cached = __probe_node(sg_dev);

        if (APP_CAM_NODE_OK == cached || APP_CAM_NODE_BUSY == cached) {
            return sg_dev;
        }
    }

    /* 3) 扫描两轮：先只认 USB 设备，再退而求其次看所有采集节点 */
    for (pass = 0; pass < 2; pass++) {
        int i;

        for (i = 0; i < APP_CAMERA_DEV_SCAN_MAX; i++) {
            int r;

            snprintf(path, sizeof(path), "/dev/video%d", i);
            if (0 != access(path, F_OK)) {
                continue;
            }
            if (0 == pass && !__is_usb_node(path)) {
                continue;
            }
            r = __probe_node(path);
            if (APP_CAM_NODE_OK == r) {
                snprintf(sg_dev, sizeof(sg_dev), "%s", path);
                PR_NOTICE("app_camera: auto-detected capture node %s", sg_dev);
                return sg_dev;
            }
            if (APP_CAM_NODE_BUSY == r && '\0' == busy_path[0]) {
                snprintf(busy_path, sizeof(busy_path), "%s", path);
            }
        }
    }

    /* 4) 没有空闲的：退用「被占用」的那个，好让上层给出准确报错 */
    if ('\0' != busy_path[0]) {
        snprintf(sg_dev, sizeof(sg_dev), "%s", busy_path);
        PR_WARN("app_camera: all capture nodes busy, fallback to %s", sg_dev);
        return sg_dev;
    }

    sg_dev[0] = '\0';
    return NULL;
}

/* munmap 所有已映射的缓冲并释放数组 */
static void __free_bufs(CAM_BUF_T *bufs, uint32_t num)
{
    uint32_t i;

    if (NULL == bufs) {
        return;
    }
    for (i = 0; i < num; i++) {
        if (NULL != bufs[i].start && bufs[i].length > 0) {
            (void)munmap(bufs[i].start, bufs[i].length);
        }
    }
    tal_free(bufs);
}

/**
 * 单次抓帧尝试（节点已选定）
 * @param[out] pbusy 置 1 表示失败原因是「设备被别的进程占着」，值得重试
 */
static OPERATE_RET __snapshot_once(const char *dev, uint8_t **jpeg, uint32_t *len, int *pbusy)
{
    int             fd = -1;
    CAM_BUF_T      *bufs = NULL;
    uint32_t        bufs_num = 0;
    struct v4l2_format          fmt;
    struct v4l2_requestbuffers  req;
    enum v4l2_buf_type          type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    int             i;
    int             got = 0;

    *pbusy = 0;

    /* ---- 1) 打开 ---- */
    fd = open(dev, O_RDWR | O_NONBLOCK | O_CLOEXEC);
    if (fd < 0) {
        if (EBUSY == errno) {
            *pbusy = 1;
            __set_err("摄像头 %s 被其它程序占用，正在等它让出…", dev);
        } else {
            __set_err("打开摄像头 %s 失败（%s）", dev, strerror(errno));
        }
        PR_ERR("app_camera: open %s failed: %s", dev, strerror(errno));
        return OPRT_COM_ERROR;
    }

    /* ---- 2) 能力检查 ---- */
    {
        struct v4l2_capability cap;
        memset(&cap, 0, sizeof(cap));
        if (__xioctl(fd, VIDIOC_QUERYCAP, &cap) == -1) {
            __set_err("摄像头 %s 不支持 V4L2（VIDIOC_QUERYCAP 失败：%s）", dev, strerror(errno));
            goto out;
        }
        if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE) ||
            !(cap.capabilities & V4L2_CAP_STREAMING)) {
            __set_err("摄像头 %s 不是可用的采集节点（缺 VIDEO_CAPTURE / STREAMING 能力）", dev);
            goto out;
        }
    }

    /* ---- 3) 设置格式：MJPG ---- */
    memset(&fmt, 0, sizeof(fmt));
    fmt.type                = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width       = APP_CAMERA_WIDTH;
    fmt.fmt.pix.height      = APP_CAMERA_HEIGHT;
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_MJPEG;
    fmt.fmt.pix.field       = V4L2_FIELD_ANY;
    if (__xioctl(fd, VIDIOC_S_FMT, &fmt) == -1) {
        if (EBUSY == errno) {
            *pbusy = 1;
            __set_err("摄像头 %s 被其它程序占用（迎宾 / 人脸跟随服务），正在等它让出…", dev);
        } else {
            __set_err("摄像头 %s 不接受 %dx%d MJPG 格式（VIDIOC_S_FMT：%s）",
                      dev, APP_CAMERA_WIDTH, APP_CAMERA_HEIGHT, strerror(errno));
        }
        goto out;
    }

    {
        struct v4l2_streamparm parm;
        memset(&parm, 0, sizeof(parm));
        parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        parm.parm.capture.timeperframe.numerator   = 1;
        parm.parm.capture.timeperframe.denominator = APP_CAMERA_FPS;
        (void)__xioctl(fd, VIDIOC_S_PARM, &parm);
    }

    /* ---- 4) 申请缓冲并 mmap ---- */
    memset(&req, 0, sizeof(req));
    req.count  = APP_CAMERA_BUFS;
    req.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
    if (__xioctl(fd, VIDIOC_REQBUFS, &req) == -1 || req.count < 2) {
        if (EBUSY == errno) {
            *pbusy = 1;
        }
        __set_err("摄像头 %s 申请缓冲失败（VIDIOC_REQBUFS：%s）", dev, strerror(errno));
        goto out;
    }

    bufs = (CAM_BUF_T *)tal_malloc(sizeof(CAM_BUF_T) * req.count);
    if (NULL == bufs) {
        __set_err("内存不足，无法为摄像头分配缓冲描述符");
        goto out;
    }
    memset(bufs, 0, sizeof(CAM_BUF_T) * req.count);
    bufs_num = req.count;

    for (i = 0; i < (int)bufs_num; i++) {
        struct v4l2_buffer buf;
        memset(&buf, 0, sizeof(buf));
        buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index  = (uint32_t)i;

        if (__xioctl(fd, VIDIOC_QUERYBUF, &buf) == -1) {
            __set_err("摄像头 %s 查询缓冲失败（VIDIOC_QUERYBUF：%s）", dev, strerror(errno));
            goto out;
        }
        bufs[i].length = buf.length;
        bufs[i].start  = mmap(NULL, buf.length, PROT_READ | PROT_WRITE,
                              MAP_SHARED, fd, buf.m.offset);
        if (MAP_FAILED == bufs[i].start) {
            bufs[i].start = NULL;
            __set_err("摄像头 %s mmap 失败（%s）", dev, strerror(errno));
            goto out;
        }
        if (__xioctl(fd, VIDIOC_QBUF, &buf) == -1) {
            __set_err("摄像头 %s 入队缓冲失败（VIDIOC_QBUF：%s）", dev, strerror(errno));
            goto out;
        }
    }

    /* ---- 5) 开始流 ---- */
    if (__xioctl(fd, VIDIOC_STREAMON, &type) == -1) {
        if (EBUSY == errno) {
            *pbusy = 1;
        }
        __set_err("摄像头 %s 启动视频流失败（VIDIOC_STREAMON：%s），设备可能被其它程序占用",
                  dev, strerror(errno));
        goto out;
    }

    /* ---- 6) 等帧：丢掉前 WARMUP 帧，取一帧拷出来 ---- */
    for (i = 0; i < APP_CAMERA_MAX_TRY; i++) {
        fd_set         fds;
        struct timeval tv;
        struct v4l2_buffer buf;
        int            r;

        FD_ZERO(&fds);
        FD_SET(fd, &fds);
        tv.tv_sec  = APP_CAMERA_WAIT_MS / 1000;
        tv.tv_usec = (APP_CAMERA_WAIT_MS % 1000) * 1000;

        r = select(fd + 1, &fds, NULL, NULL, &tv);
        if (r == -1) {
            if (EINTR == errno) {
                continue;
            }
            __set_err("等待摄像头帧出错（select：%s）", strerror(errno));
            break;
        }
        if (r == 0) {
            continue; /* 超时，再来一次 */
        }

        memset(&buf, 0, sizeof(buf));
        buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        if (__xioctl(fd, VIDIOC_DQBUF, &buf) == -1) {
            if (EAGAIN == errno) {
                continue;
            }
            __set_err("摄像头取帧失败（VIDIOC_DQBUF：%s）", strerror(errno));
            break;
        }

        if (buf.index < bufs_num && i >= APP_CAMERA_WARMUP && buf.bytesused >= APP_CAMERA_MIN_JPEG) {
            uint8_t *copy = (uint8_t *)tal_malloc(buf.bytesused);
            if (NULL != copy) {
                memcpy(copy, bufs[buf.index].start, buf.bytesused);
                *jpeg = copy;
                *len  = buf.bytesused;
                got   = 1;
            } else {
                __set_err("内存不足，无法保存照片（需要 %u 字节）", (unsigned)buf.bytesused);
            }
        }
        (void)__xioctl(fd, VIDIOC_QBUF, &buf);
        if (got) {
            break;
        }
    }

    if (!got && '\0' == sg_err[0]) {
        __set_err("摄像头 %s 连续等帧超时，没有拿到有效画面", dev);
    }

out:
    if (fd >= 0) {
        (void)__xioctl(fd, VIDIOC_STREAMOFF, &type);
    }
    __free_bufs(bufs, bufs_num);
    if (fd >= 0) {
        (void)close(fd);
    }

    if (NULL != *jpeg) {
        PR_NOTICE("app_camera: snapshot ok, %u bytes (%s)", (unsigned)*len, dev);
        return OPRT_OK;
    }

    PR_ERR("app_camera: snapshot failed: %s", sg_err);
    return OPRT_COM_ERROR;
}

OPERATE_RET app_camera_init(void)
{
    const char *dev;

    if (NULL == sg_lock) {
        if (OPRT_OK != tal_mutex_create_init(&sg_lock)) {
            PR_ERR("app_camera: create mutex failed");
            return OPRT_COM_ERROR;
        }
    }
    __load_conf();

    dev = __pick_device();
    if (NULL == dev) {
        PR_WARN("app_camera: ready, but NO usable capture node found "
                "(plug the USB camera, or pin one via %s)", APP_CAMERA_DEV_CONF);
    } else {
        PR_NOTICE("app_camera: ready, dev=%s (%dx%d MJPEG)%s",
                  dev, APP_CAMERA_WIDTH, APP_CAMERA_HEIGHT,
                  ('\0' != sg_dev_forced[0]) ? " [pinned by conf]" : "");
    }
    return OPRT_OK;
}

const char *app_camera_last_error(void)
{
    return sg_err;
}

const char *app_camera_device(void)
{
    return sg_dev;
}

void app_camera_free(uint8_t *jpeg)
{
    if (NULL != jpeg) {
        tal_free(jpeg);
    }
}

OPERATE_RET app_camera_snapshot(uint8_t **jpeg, uint32_t *len)
{
    int         attempt;
    int         busy = 0;
    OPERATE_RET rt   = OPRT_COM_ERROR;

    if (NULL == jpeg || NULL == len) {
        return OPRT_INVALID_PARM;
    }
    *jpeg = NULL;
    *len  = 0;
    sg_err[0] = '\0';

    if (NULL == sg_lock) {
        (void)tal_mutex_create_init(&sg_lock); /* init 没被调过的兜底 */
    }
    if (NULL != sg_lock) {
        tal_mutex_lock(sg_lock);
    }

    /* 请板上视觉服务让出摄像头（对端没装补丁时，下面照样会给出准确报错） */
    __want_camera();

    for (attempt = 0; attempt < APP_CAMERA_ACQUIRE_TRY; attempt++) {
        const char *dev;

        dev = __pick_device();
        if (NULL == dev) {
            __set_err("找不到可用的摄像头采集节点。请确认 USB 摄像头已插好"
                      "（可运行 ls /dev/video* 查看），或用 %s 指定节点", APP_CAMERA_DEV_CONF);
            PR_ERR("app_camera: no capture node available");
            break;
        }

        busy = 0;
        rt   = __snapshot_once(dev, jpeg, len, &busy);
        if (OPRT_OK == rt) {
            break;
        }

        if (!busy) {
            /* 不是「被占用」（节点不对 / 没画面）→ 清掉缓存节点，下次重新探测 */
            if ('\0' == sg_dev_forced[0]) {
                sg_dev[0] = '\0';
            }
            break;
        }

        if (attempt + 1 < APP_CAMERA_ACQUIRE_TRY) {
            PR_NOTICE("app_camera: %s busy, waiting for the vision service to yield (%d/%d)...",
                      dev, attempt + 1, APP_CAMERA_ACQUIRE_TRY);
            __msleep(APP_CAMERA_ACQUIRE_WAIT);
        }
    }

    if (NULL == *jpeg && busy) {
        __set_err("摄像头被其它程序占用（迎宾 / 人脸跟随服务正在运行），"
                  "已等待约 %d ms 仍未让出，请先停掉它再拍照",
                  APP_CAMERA_ACQUIRE_TRY * APP_CAMERA_ACQUIRE_WAIT);
    }

    /* 拍完（无论成败）都撤掉请求，让对端把摄像头拿回去 */
    __yield_camera();

    if (NULL != sg_lock) {
        tal_mutex_unlock(sg_lock);
    }

    return (NULL != *jpeg) ? OPRT_OK : OPRT_COM_ERROR;
}
