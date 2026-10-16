#!/usr/bin/env python3

import argparse
import os
import sys

import cv2
import rclpy

from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


# /dev/videoN numbers change between reboots and USB ports; the by-id path
# always points at the Innomaker USB camera. /dev/video0 is usually the
# laptop's built-in webcam.
EXTERNAL_CAMERA = (
    '/dev/v4l/by-id/'
    'usb-Innomaker_Innomaker-U20CAM-1080p-S1_SN0001-video-index0'
)
FALLBACK_CAMERA = '/dev/video0'


class ExternalCameraPublisher(Node):

    def __init__(self, device):
        super().__init__('external_camera_publisher')

        self.bridge = CvBridge()
        self.device = device

        # Open external USB camera
        self.cap = cv2.VideoCapture(device, cv2.CAP_V4L2)

        if not self.cap.isOpened():
            self.get_logger().error(
                f'Could not open external camera {device}'
            )
            raise RuntimeError(f'Could not open {device}')

        # Camera configuration
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        self.image_publisher = self.create_publisher(
            Image,
            '/external_camera/image_raw',
            10
        )

        # Publish at approximately 30 Hz
        self.timer = self.create_timer(
            1.0 / 30.0,
            self.publish_frame
        )

        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.get_logger().info(
            f'External camera opened successfully: {device} '
            f'({os.path.realpath(device)}), {width}x{height}'
        )

        self.get_logger().info(
            'Publishing: /external_camera/image_raw'
        )

    def publish_frame(self):

        ret, frame = self.cap.read()

        if not ret:
            self.get_logger().warning(
                f'Failed to read frame from {self.device}'
            )
            return

        try:
            # OpenCV BGR -> ROS Image
            msg = self.bridge.cv2_to_imgmsg(
                frame,
                encoding='bgr8'
            )

            msg.header.stamp = self.get_clock().now().to_msg()

            self.image_publisher.publish(msg)

        except Exception as e:
            self.get_logger().error(
                f'Failed to publish image: {e}'
            )

    def destroy_node(self):

        if self.cap.isOpened():
            self.cap.release()

        super().destroy_node()


def main(args=None):

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--device',
        default=None,
        help='camera device (default: Innomaker USB camera, '
             'else /dev/video0)'
    )
    parsed_args, ros_args = parser.parse_known_args()

    device = parsed_args.device
    if device is None:
        device = (EXTERNAL_CAMERA if os.path.exists(EXTERNAL_CAMERA)
                  else FALLBACK_CAMERA)

    rclpy.init(args=[sys.argv[0]] + ros_args)

    node = None

    try:
        node = ExternalCameraPublisher(device)
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    except Exception as e:
        print(f'Camera publisher error: {e}')

    finally:
        if node is not None:
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

