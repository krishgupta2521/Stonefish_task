#!/usr/bin/env python3
"""Publish the real camera, or a recorded video, on /external_camera/image_raw.

  external_camera_publisher.py                      Innomaker USB camera (else /dev/video0)
  external_camera_publisher.py --device /dev/video2 a specific camera
  external_camera_publisher.py --video pool.MP4     a video file instead of the camera

Every frame goes out as 640 x 480 (the camera's own mode), so the detector and
the simulator (case 6 maps these pixels onto the target) see the same thing
from a camera or a video. Video frames are scaled and padded top/bottom
(letterbox): the buoy keeps its shape.
"""

import argparse
import os
import sys
import time

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

OUTPUT_WIDTH = 640
OUTPUT_HEIGHT = 480
CAMERA_FPS = 30.0


def letterbox(frame, width, height):
    """Scale to fit width x height keeping the aspect ratio, pad with black."""
    h, w = frame.shape[:2]
    if (w, h) == (width, height):
        return frame
    scale = min(width / w, height / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    top = (height - new_h) // 2
    left = (width - new_w) // 2
    return cv2.copyMakeBorder(resized, top, height - new_h - top,
                              left, width - new_w - left,
                              cv2.BORDER_CONSTANT, value=(0, 0, 0))


class ExternalCameraPublisher(Node):

    def __init__(self, device=None, video=None, loop=True, fps=None):
        super().__init__('external_camera_publisher')

        self.bridge = CvBridge()
        self.video = video
        self.loop = loop
        self.source = video if video is not None else device

        if video is not None:
            # Recorded video (any format FFmpeg reads: mp4, webm, avi, ...)
            if not os.path.isfile(video):
                raise RuntimeError(f'Video file not found: {video}')
            self.cap = cv2.VideoCapture(video)
            if not self.cap.isOpened():
                raise RuntimeError(f'Could not open video {video}')
            video_fps = self.cap.get(cv2.CAP_PROP_FPS) or CAMERA_FPS
            self.publish_fps = fps if fps else min(video_fps, CAMERA_FPS)
            # Play in real time: skip frames when the video is faster than
            # the publish rate (a 60 fps GoPro clip -> every 2nd frame).
            self.frame_step = max(1, int(round(video_fps / self.publish_fps)))
            frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.get_logger().info(
                f'Video opened: {video} '
                f'({int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x'
                f'{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}, '
                f'{video_fps:.1f} fps, {frames / video_fps:.0f} s), '
                f'publishing every {self.frame_step}. frame at '
                f'{self.publish_fps:.0f} Hz, {"looping" if loop else "once"}'
            )
        else:
            # Open external USB camera
            self.cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
            if not self.cap.isOpened():
                raise RuntimeError(f'Could not open camera {device}')

            # Camera configuration
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, OUTPUT_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, OUTPUT_HEIGHT)
            self.cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
            self.publish_fps = fps if fps else CAMERA_FPS
            self.frame_step = 1

            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.get_logger().info(
                f'External camera opened successfully: {device} '
                f'({os.path.realpath(device)}), {width}x{height}'
            )

        self.image_publisher = self.create_publisher(
            Image,
            '/external_camera/image_raw',
            10
        )

        self.timer = self.create_timer(
            1.0 / self.publish_fps,
            self.publish_frame
        )

        self.published = 0
        self.rate_start = time.monotonic()

        self.get_logger().info(
            f'Publishing: /external_camera/image_raw '
            f'({OUTPUT_WIDTH}x{OUTPUT_HEIGHT})'
        )

    def read_frame(self):
        for _ in range(self.frame_step - 1):
            self.cap.grab()
        ret, frame = self.cap.read()
        if not ret and self.video is not None and self.loop:
            self.get_logger().info('End of video, restarting')
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
        return ret, frame

    def publish_frame(self):

        ret, frame = self.read_frame()

        if not ret:
            if self.video is not None:
                self.get_logger().info('End of video')
                self.timer.cancel()
                return
            self.get_logger().warning(
                f'Failed to read frame from {self.source}'
            )
            return

        frame = letterbox(frame, OUTPUT_WIDTH, OUTPUT_HEIGHT)

        try:
            # OpenCV BGR -> ROS Image
            msg = self.bridge.cv2_to_imgmsg(
                frame,
                encoding='bgr8'
            )

            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'external_camera'

            self.image_publisher.publish(msg)

        except Exception as e:
            self.get_logger().error(
                f'Failed to publish image: {e}'
            )
            return

        # Report the real rate now and then (decoding 4K video can be slower
        # than the requested rate; the video then plays in slow motion).
        self.published += 1
        elapsed = time.monotonic() - self.rate_start
        if elapsed >= 10.0:
            self.get_logger().info(
                f'publishing at {self.published / elapsed:.1f} Hz'
            )
            self.published = 0
            self.rate_start = time.monotonic()

    def destroy_node(self):

        if self.cap.isOpened():
            self.cap.release()

        super().destroy_node()


def main(args=None):

    parser = argparse.ArgumentParser(
        description='Publish a camera or a video file on '
                    '/external_camera/image_raw')
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        '--device',
        default=None,
        help='camera device (default: Innomaker USB camera, '
             'else /dev/video0)'
    )
    source.add_argument(
        '--video',
        default=None,
        help='play this video file instead of a camera'
    )
    parser.add_argument(
        '--no-loop',
        action='store_true',
        help='with --video: stop at the end instead of restarting'
    )
    parser.add_argument(
        '--fps',
        type=float,
        default=None,
        help='publish rate (default: 30, or the video rate if lower)'
    )
    parsed_args, ros_args = parser.parse_known_args()

    device = parsed_args.device
    if parsed_args.video is None and device is None:
        device = (EXTERNAL_CAMERA if os.path.exists(EXTERNAL_CAMERA)
                  else FALLBACK_CAMERA)

    rclpy.init(args=[sys.argv[0]] + ros_args)

    node = None

    try:
        node = ExternalCameraPublisher(
            device=device,
            video=parsed_args.video,
            loop=not parsed_args.no_loop,
            fps=parsed_args.fps
        )
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
