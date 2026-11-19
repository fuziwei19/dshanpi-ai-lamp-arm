#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate lerobot
cd /root/lerobot
lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=my_awesome_follower_arm
echo '[DONE-EXIT]'
sleep 1800
