#!/bin/bash

source {$HOME}/ros_workspace/devel/setup.bash && roslaunch spinnaker_sdk_camera_driver acquisition.launch config_file:=src/spinnaker_sdk_camera_driver/params/multi_camera_rostool.yaml fps:=30 color:=false binning:=2 &
sleep 5
source {$HOME}/ros_workspace/devel/setup.bash && rosrun optical_flow optical_flow.py _input_topic:=/flir/cam0/image_raw/compressed _baudrate:=921600 &> /storage/of_logs/lastlog
