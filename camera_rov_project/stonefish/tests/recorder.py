"""Record /usv/control_debug + /usv/odom to CSV and save detection snapshots.

Usage: python3 recorder.py <output_dir>
"""
import csv
import math
import os
import sys

import cv2
import rclpy
from cv_bridge import CvBridge
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray

STATE_NAMES = [
    'SEARCH', 'YAW_ALIGN', 'YAW_SETTLE', 'VERTICAL_ALIGN', 'VERIFY_CENTER',
    'APPROACH', 'TARGET_LOST_HOLD', 'CALIBRATE', 'STOPPED',
    'TARGET_REACHED', 'BACK_OFF', 'SURFACING', 'SURFACED', 'RETREAT']

DEBUG_FIELDS = [
    'state', 't', 'x', 'y', 'depth', 'yaw', 'yaw_ref', 'yaw_rate',
    'depth_ref', 'heave_vel', 'yaw_cmd', 'heave_cmd', 'surge_cmd',
    'fresh', 'conf', 'e_x', 'e_y', 'h_err', 'v_err',
    'cam_h_is_x', 'cam_h_sign', 'cam_v_sign', 'cam_h_ver', 'cam_v_ver',
    'left_sign', 'right_sign', 'vert_sign', 'max_overshoot', 'align_time',
    'range', 'front_gap', 'bbox_w', 'speed_ref', 'body_vx',
    'aborted', 'detector_alive', 'have_range',
    'truth_x', 'truth_y', 'truth_z', 'bbox_h', 'terminal']


class Recorder(Node):

    def __init__(self, out_dir):
        super().__init__('task2_test_recorder')
        self.out_dir = out_dir
        self.bridge = CvBridge()
        self.last_image = None
        self.last_state = None
        self.roll = 0.0
        self.pitch = 0.0
        self.file = open(os.path.join(out_dir, 'telemetry.csv'), 'w', newline='')
        self.writer = csv.writer(self.file)
        self.writer.writerow(DEBUG_FIELDS + ['roll', 'pitch'])
        self.create_subscription(
            Float64MultiArray, '/usv/control_debug', self.on_debug, 50)
        self.create_subscription(Odometry, '/usv/odom', self.on_odom, 10)
        self.create_subscription(
            Image, '/stonefish/detection_image', self.on_image, 2)

    def on_odom(self, msg):
        q = msg.pose.pose.orientation
        self.roll = math.atan2(2 * (q.w * q.x + q.y * q.z),
                               1 - 2 * (q.x * q.x + q.y * q.y))
        self.pitch = math.asin(max(-1.0, min(1.0, 2 * (q.w * q.y - q.z * q.x))))

    def on_image(self, msg):
        self.last_image = msg

    def on_debug(self, msg):
        data = list(msg.data)
        data += [float('nan')] * (len(DEBUG_FIELDS) - len(data))
        self.writer.writerow(data[:len(DEBUG_FIELDS)] + [self.roll, self.pitch])
        state = int(data[0])
        if state != self.last_state:
            self.last_state = state
            self.save_snapshot(f'{data[1]:07.2f}_{STATE_NAMES[state]}')

    def save_snapshot(self, name):
        if self.last_image is None:
            return
        frame = self.bridge.imgmsg_to_cv2(self.last_image, 'bgr8')
        cv2.imwrite(os.path.join(self.out_dir, name + '.png'), frame)

    def close(self):
        self.file.close()


def main():
    out_dir = sys.argv[1]
    os.makedirs(out_dir, exist_ok=True)
    rclpy.init()
    node = Recorder(out_dir)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
