#include <linux/types.h>
#include <linux/kernel.h>
#include <linux/delay.h>
#include <linux/init.h>
#include <linux/module.h>
#include <linux/errno.h>
#include <linux/gpio.h>
#include <linux/cdev.h>
#include <linux/device.h>
#include <linux/of_gpio.h>
#include <linux/semaphore.h>
#include <linux/timer.h>
#include <linux/i2c.h>
#include <linux/spi/spi.h>
#include <linux/of.h>
#include <linux/of_address.h>
#include <linux/of_gpio.h>
#include <linux/platform_device.h>
#include <linux/slab.h>

#define RGB_RST_BYTE (50) 

struct ws2812_dev
{
	struct device *device;		/* 设备 	 */
	struct spi_device *spi_dev;			/* 私有数据 		*/
    int led_num;
    u8 *rgb_data;
    u8 *rgb_cvt_bit_data;
    u8 rst_buf[RGB_RST_BYTE];
    struct file_operations *fops;

    struct cdev cdev;
    dev_t devno;
    struct class *class;
    struct device *cdevice;
    
    // 工作队列相关
    struct workqueue_struct *wq;
    struct work_struct work;
      
    // 同步机制
    struct mutex lock;
    bool work_pending;

};

static struct ws2812_dev ws2812_dev = {
    .led_num = 8,
    .rgb_data = NULL,
};


static int __maybe_unused send_rgb_data(struct ws2812_dev *dev);


void __maybe_unused dump_hex(char *buf,int count)
{
    int i;
    pr_info("hex:");
    for(i = 0; i < count;i++)
    {
        printk("%02x ",buf[i]);
    }
    printk("\n");
}

static ssize_t ws2812_write(struct file *filp, const char __user *buf,
                           size_t count, loff_t *f_pos)
{
    struct  ws2812_dev *dev  = filp->private_data;
    ssize_t retval = 0;
    int new_led_num;
    u8 *new_rgb_data, *new_rgb_cvt_bit_data;
    
    if (count % 3 != 0)
        return -EINVAL;

    new_led_num = count / 3;

    mutex_lock(&dev->lock);
    
    if (new_led_num != dev->led_num) {
        new_rgb_data = kzalloc(new_led_num * 3, GFP_KERNEL);
        if (!new_rgb_data) {
            retval = -ENOMEM;
            goto out;
        }

        new_rgb_cvt_bit_data = kzalloc(new_led_num * 24, GFP_KERNEL);
        if (!new_rgb_cvt_bit_data) {
            kfree(new_rgb_data);
            retval = -ENOMEM;
            goto out;
        }

        kfree(dev->rgb_data);
        kfree(dev->rgb_cvt_bit_data);

        dev->rgb_data = new_rgb_data;
        dev->rgb_cvt_bit_data = new_rgb_cvt_bit_data;
        dev->led_num = new_led_num;
    }
    
    if (copy_from_user(dev->rgb_data, buf, count)) {
        retval = -EFAULT;
        goto out;
    }
    
    // 触发工作队列
    if (!dev->work_pending) {
        dev->work_pending = true;
        queue_work(dev->wq, &dev->work);
    }
    
    retval = count;
    
 out:
    mutex_unlock(&dev->lock);
    return retval;
}

static int ws2812_open(struct inode *inode, struct file *filp)
{
    struct  ws2812_dev *dev  = container_of(inode->i_cdev, 
                                            struct ws2812_dev, cdev);
    filp->private_data = dev;
    return 0;
}

static int ws2812_release(struct inode *inode, struct file *filp)
{
    struct ws2812_dev *dev = filp->private_data;

    /* 关闭所有灯 */
    mutex_lock(&dev->lock);
    if (dev->rgb_data) {
        memset(dev->rgb_data, 0, dev->led_num * 3);
        if (!dev->work_pending) {
            dev->work_pending = true;
            queue_work(dev->wq, &dev->work);
        }
    }
    mutex_unlock(&dev->lock);

    return 0;
}

static struct file_operations ws2812_fops = {
    .owner = THIS_MODULE,
    .write = ws2812_write,
    .open = ws2812_open,
    .release = ws2812_release,
};


static void ws2812_work_handler(struct work_struct *work)
{
    struct  ws2812_dev *dev = container_of(work, struct ws2812_dev, work);
    
    mutex_lock(&dev->lock);
    
    if(dev->work_pending)
        send_rgb_data(dev); 
    dev->work_pending = false;
    mutex_unlock(&dev->lock);

}

static __maybe_unused void calcu_send_rgb_data_byte(u8 data,u8 *output_data)
{
    //1byte is 1bit
    int i;
    for(i = 0;i<8;i++)
    {
        if(data&0x80)
        {
            output_data[i] = 0xfc;
        }
        else
        {
            output_data[i] = 0x30;
        }
        data = data << 1;
    }
}


static int __maybe_unused send_rgb_data(struct ws2812_dev *dev) 
{
    int ret;
    int i;
    struct spi_transfer rst_t;
    struct spi_message rst_m;
    struct spi_transfer rgb_t;
    struct spi_message rgb_m;
    
    // 转换RGB数据
    for(i = 0; i < dev->led_num*3; i++) {
        calcu_send_rgb_data_byte(dev->rgb_data[i], dev->rgb_cvt_bit_data + i * 8);
    }
    
    // 发送复位信号
    memset(&rst_t, 0, sizeof(rst_t));
    rst_t.tx_buf = dev->rst_buf;
    rst_t.len = RGB_RST_BYTE;
    rst_t.bits_per_word = 8;
    
    spi_message_init(&rst_m);
    spi_message_add_tail(&rst_t, &rst_m);
    ret = spi_sync(dev->spi_dev, &rst_m);
    if (ret) {
        pr_err("%s: rst spi_sync failed: %d\n", __func__, ret);
        return ret;
    }
    if (rst_m.actual_length && rst_m.actual_length != RGB_RST_BYTE) {
        pr_err("%s rst len=40 actual_length=%d\n", __func__, rst_m.actual_length);
        return -EIO;
    }
    
    // 发送RGB数据
    memset(&rgb_t, 0, sizeof(rgb_t));
    rgb_t.tx_buf = dev->rgb_cvt_bit_data;
    rgb_t.len = dev->led_num * 24;
    rgb_t.bits_per_word = 8;
    
    spi_message_init(&rgb_m);
    spi_message_add_tail(&rgb_t, &rgb_m);
    ret = spi_sync(dev->spi_dev, &rgb_m);
    if (ret) {
        pr_err("%s: rgb spi_sync failed: %d\n", __func__, ret);
        return ret;
    }
    if (rgb_m.actual_length && rgb_m.actual_length != dev->led_num * 24) {
        pr_err("%s rgb len=%d actual_length=%d\n", __func__, 
               dev->led_num * 24, rgb_m.actual_length);
        return -EIO;
    }
    return 0;
}

int ws2812_register(struct ws2812_dev *dev)
{
    int ret;
    // 初始化互斥锁
    mutex_init(&dev->lock);
    // 分配设备号
    ret = alloc_chrdev_region(&dev->devno, 0, 1, "ws2812-leds");
    if (ret < 0) {
        pr_err("无法分配设备号\n");
        goto err_alloc;
    }
    
    // 初始化字符设备
    cdev_init(&dev->cdev, &ws2812_fops);
    dev->cdev.owner = THIS_MODULE;
    
    ret = cdev_add(&dev->cdev,dev->devno, 1);
    if (ret < 0) {
        pr_err("无法添加字符设备\n");
        goto err_cdev;
    }
    
    // 创建设备类
    dev->class = class_create(THIS_MODULE, "ws2812_class");
    if (IS_ERR(dev->class)) {
        ret = PTR_ERR(dev->class);
        goto err_class;
    }
    
    // 创建设备节点
    dev->cdevice = device_create(dev->class, NULL, dev->devno,
                                  NULL, "ws2812-leds");
    if (IS_ERR(dev->cdevice)) {
        ret = PTR_ERR(dev->cdevice);
        goto err_device;
    }
    
    // 创建工作队列
    dev->wq = create_singlethread_workqueue("ws2812_workqueue");
    if (!dev->wq) {
        ret = -ENOMEM;
        goto err_wq;
    }
    
    // 初始化工作
    INIT_WORK(&dev->work, ws2812_work_handler);
    dev->work_pending = false;
    
    pr_info("设备初始化成功，设备号: %d:%d\n", 
            MAJOR(dev->devno), MINOR(dev->devno));
    return send_rgb_data(dev);
    
err_wq:
    device_destroy(dev->class, dev->devno);
err_device:
    class_destroy(dev->class);
err_class:
    cdev_del(&dev->cdev);
err_cdev:
    unregister_chrdev_region(dev->devno, 1);
err_alloc:
    return ret;
}

/* probe 函数 */ 
static int ws2812_probe(struct spi_device *spi) 
{ 
    /* 具体函数内容 */ 
    int ret;
    spi->mode = SPI_MODE_0;
    spi->bits_per_word = 8;
    ret = spi_setup(spi);
    if(ret)
    {
        dev_err(&spi->dev, "ERR: fail to setup spi\n");
		return -1;
    }
    ws2812_dev.spi_dev = spi;
    ws2812_dev.device = &spi->dev;
 
    ret = device_property_read_u32(ws2812_dev.device,"leds,number",&ws2812_dev.led_num);
    if(ret)
    {
        dev_err(ws2812_dev.device,"fail to read leds,number using default num 8\n");
    }
    ws2812_dev.rgb_data = (u8 *)kzalloc(ws2812_dev.led_num*3, GFP_KERNEL);
    if(!ws2812_dev.rgb_data)
    {
        dev_err(&spi->dev, "ERR: no memory for spi_test_data\n");
		return -ENOMEM;
    }
 
    ws2812_dev.rgb_cvt_bit_data = (u8 *)kzalloc(ws2812_dev.led_num*24, GFP_KERNEL);
    if(!ws2812_dev.rgb_cvt_bit_data)
    {
        dev_err(&spi->dev, "ERR: no memory for spi_test_data\n");
		return -ENOMEM;
    }
    dev_info(&spi->dev,"led-num %d max-freq %d",ws2812_dev.led_num,spi->max_speed_hz);
    return ws2812_register(&ws2812_dev); 
} 
 
/* remove 函数 */ 
static void ws2812_remove(struct spi_device *spi) 
{ 
    struct ws2812_dev *dev = &ws2812_dev;

    cancel_work_sync(&dev->work);
    if (dev->wq)
        destroy_workqueue(dev->wq);
        
    device_destroy(dev->class, dev->devno);
    class_destroy(dev->class);
    cdev_del(&dev->cdev);
    unregister_chrdev_region(dev->devno, 1);
    
    if (dev->rgb_data)
        kfree(dev->rgb_data);
    if (dev->rgb_cvt_bit_data)
        kfree(dev->rgb_cvt_bit_data);
} 
/* 传统匹配方式 ID 列表 */ 
static const struct spi_device_id ws2812_id[] = { 
    {"ws2812", 0},   
    {} 
}; 
 
/* 设备树匹配列表 */ 
static const struct of_device_id ws2812_of_match[] = { 
    { .compatible = "dshanpi-a1,ws2812"}, 
    { .compatible = "led,ws2812"}, 
    { /* Sentinel */ } 
}; 
MODULE_DEVICE_TABLE(of, ws2812_of_match);

/* SPI 驱动结构体 */  
static struct spi_driver ws2812_driver = { 
    .probe = ws2812_probe, 
    .remove = ws2812_remove, 
    .driver = { 
            .owner = THIS_MODULE, 
            .name = "ws2812", 
            .of_match_table = ws2812_of_match,  
          }, 
    .id_table = ws2812_id, 
}; 
          
/* 驱动入口函数 */ 
static int __init ws2812_init(void) 
{ 
    pr_info("WS2812 Driver Loading...\n");
    return spi_register_driver(&ws2812_driver); 
} 
 
/* 驱动出口函数 */ 
static void __exit ws2812_exit(void) 
{ 
    spi_unregister_driver(&ws2812_driver); 
} 
module_init(ws2812_init); 
module_exit(ws2812_exit); 

MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("WS2812 RGB LED Driver"); 
