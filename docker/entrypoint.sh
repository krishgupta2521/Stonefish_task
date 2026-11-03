#!/bin/bash
# Source ROS 2 and the project's workspace, then run the given command.
set -e
source /opt/ros/jazzy/setup.bash
source /root/kyushu_project/camera_rov_project/ros2_ws/install/setup.bash
exec "$@"
