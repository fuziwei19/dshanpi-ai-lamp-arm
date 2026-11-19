#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <signal.h>

#define DEFAULT_LED_NUM 15
#define DEV_PATH "/dev/ws2812-leds"

static int g_fd = -1;

// Signal handler to ensure file is closed (which triggers driver release)
void signal_handler(int signum) {
    if (g_fd >= 0) {
        printf("\nExiting... Turning off LEDs.\n");
        close(g_fd);
        g_fd = -1;
    }
    exit(0);
}

// Color structure
typedef struct {
    uint8_t g;
    uint8_t r;
    uint8_t b;
} rgb_t;

void msleep(unsigned int ms) {
    usleep(ms * 1000);
}

// Simple function to set all LEDs to a specific color
void set_all_color(rgb_t *leds, int num, uint8_t r, uint8_t g, uint8_t b) {
    for (int i = 0; i < num; i++) {
        leds[i].r = r;
        leds[i].g = g;
        leds[i].b = b;
    }
}

int main(int argc, char *argv[]) {
    int fd;
    int led_num = DEFAULT_LED_NUM;
    rgb_t *led_buffer;
    int ret;

    // 1. Parse command line argument for LED number
    if (argc > 1) {
        led_num = atoi(argv[1]);
        if (led_num <= 0) {
            fprintf(stderr, "Invalid LED number: %d\n", led_num);
            return 1;
        }
    }

    printf("WS2812 Application Started\n");
    printf("Target Device: %s\n", DEV_PATH);
    printf("LED Number: %d\n", led_num);

    // 2. Allocate memory for RGB data (3 bytes per LED)
    led_buffer = (rgb_t *)malloc(led_num * sizeof(rgb_t));
    if (!led_buffer) {
        perror("Failed to allocate memory");
        return 1;
    }
    memset(led_buffer, 0, led_num * sizeof(rgb_t));

    // 3. Open the device
    fd = open(DEV_PATH, O_RDWR);
    if (fd < 0) {
        perror("Failed to open device");
        printf("Make sure the driver is loaded and device node exists.\n");
        free(led_buffer);
        return 1;
    }

    // 4. Register signal handler
    signal(SIGINT, signal_handler);
    
    // 5. Main loop: Demonstrate effects
    printf("Starting color cycle... Press Ctrl+C to exit.\n");
    g_fd = fd;
    
    while (1) {
        // Red
        printf("Color: Red\n");
        set_all_color(led_buffer, led_num, 255, 0, 0);
        write(fd, led_buffer, led_num * sizeof(rgb_t));
        msleep(500);

        // Green
        printf("Color: Green\n");
        set_all_color(led_buffer, led_num, 0, 255, 0);
        write(fd, led_buffer, led_num * sizeof(rgb_t));
        msleep(500);

        // Blue
        printf("Color: Blue\n");
        set_all_color(led_buffer, led_num, 0, 0, 255);
        write(fd, led_buffer, led_num * sizeof(rgb_t));
        msleep(500);

        // White
        printf("Color: White\n");
        set_all_color(led_buffer, led_num, 255, 255, 255);
        write(fd, led_buffer, led_num * sizeof(rgb_t));
        msleep(500);
        
        // Running Light Effect
        printf("Effect: Running Light\n");
        memset(led_buffer, 0, led_num * sizeof(rgb_t)); // Clear
        for (int i = 0; i < led_num; i++) {
            // Set current LED to a color (e.g., Purple)
            led_buffer[i].r = 200;
            led_buffer[i].g = 0;
            led_buffer[i].b = 200;
            
            // Write to device
            ret = write(fd, led_buffer, led_num * sizeof(rgb_t));
            if (ret < 0) {
                perror("Write failed");
                break;
            }
            
            msleep(50); // Speed of running light
            
            // Turn off current LED for next iteration (simple dot effect)
            led_buffer[i].r = 0;
            led_buffer[i].g = 0;
            led_buffer[i].b = 0;
        }
    }

    // Cleanup (unreachable in this loop, but good practice)
    close(fd);
    free(led_buffer);
    return 0;
}
