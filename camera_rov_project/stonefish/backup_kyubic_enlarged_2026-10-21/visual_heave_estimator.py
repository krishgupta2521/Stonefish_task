#!/usr/bin/env python3
"""Vertical motion of the robot from its forward camera (image data only).

When the robot moves down, everything it sees slides UP in the image; when it
drives forward, the scene spreads out from the image centre. Features are
tracked between consecutive frames (Lucas-Kanade) and their motion is fitted as

    flow_x = d * (x - cx) + c        flow_y = d * (y - cy) + b

    d = divergence (forward motion), c = horizontal shift (yaw / sway),
    b = vertical shift  ->  vertical motion (heave) and pitch rotation.

Turning also moves points off the image centre up or down (rotational flow,
flow_y = c * x * y / f^2 for a pan; f = focal length): with the 90 deg camera a
0.18 rad/s turn put up to 27 px/s into b, i.e. ~0.1 m/s of phantom heave. So b
comes from a second fit with that term removed (derotation):

    flow_x = d * x + c' * (1 + x^2 / f^2)      flow_y = d * y + b + c' * x * y / f^2

c (published) stays the plain fit's, so the controller's turn-rate gain holds.

b is published in pixels per second. The controller turns it into a vertical
speed with the camera focal length, an effective scene distance and the IMU
pitch rate, and integrates it into the robot's depth.

Publishes /vision/heave_flow (Float64MultiArray):
  [0] image stamp [s]  [1] b vertical shift [px/s]  [2] d divergence [1/s]
  [3] c horizontal shift [px/s]  [4] inlier features  [5] fit residual [px]
  [6] frame interval [s]
With too few trackable features (empty water) [1..3] are NaN and [4] is 0:
the node is alive but has no measurement for this frame.

usage: visual_heave_estimator.py [--image-topic /Stonefish/camera/image_raw] [--hfov-deg 90]
"""

import argparse
import sys

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray

MAX_FEATURES = 400
MIN_FEATURES = 12          # fewer inliers than this: no estimate this frame
MAX_FRAME_GAP_SEC = 0.5    # older previous frame: start over


def design(x, y, focal):
    """Rows for flow_x then flow_y; unknowns [d, c, b]. With a focal length
    the pan's rotational flow is included (c is then the pan's shift)."""
    a = np.zeros((2 * len(x), 3))
    a[:len(x), 0] = x
    a[:len(x), 1] = 1.0 if focal is None else 1.0 + (x / focal) ** 2
    a[len(x):, 0] = y
    a[len(x):, 2] = 1.0
    if focal is not None:
        a[len(x):, 1] = x * y / focal ** 2
    return a


def robust_fit(p0, flow, cx, cy, focal=None, iterations=3):
    """Least squares for (d, c, b) with outlier rejection (3 x MAD)."""
    keep = np.ones(len(p0), dtype=bool)
    xa, ya = p0[:, 0] - cx, p0[:, 1] - cy
    for _ in range(iterations):
        a = design(xa[keep], ya[keep], focal)
        rhs = np.concatenate([flow[keep, 0], flow[keep, 1]])
        sol, *_ = np.linalg.lstsq(a, rhs, rcond=None)
        # residual per feature over all features
        pred = design(xa, ya, focal) @ sol
        pred_x, pred_y = pred[:len(xa)], pred[len(xa):]
        res = np.hypot(flow[:, 0] - pred_x, flow[:, 1] - pred_y)
        mad = np.median(res[keep]) + 1e-3
        new_keep = res < 3.0 * 1.4826 * mad + 0.3
        if new_keep.sum() < MIN_FEATURES or np.array_equal(new_keep, keep):
            break
        keep = new_keep
    return sol, keep, float(np.median(res[keep])) if keep.any() else 0.0


class VisualHeaveEstimator(Node):

    def __init__(self, image_topic, output_topic, hfov_deg):
        super().__init__('visual_heave_estimator')
        self.hfov = np.radians(hfov_deg)
        self.bridge = CvBridge()
        self.prev_gray = None
        self.prev_stamp = None
        self.pub = self.create_publisher(Float64MultiArray, output_topic, 10)
        self.create_subscription(Image, image_topic, self.on_image, 5)
        self.frames = 0
        self.get_logger().info(f'{image_topic} -> {output_topic}')

    def publish_no_measurement(self, stamp, dt):
        """Too few features (e.g. facing empty water): say so, so the
        controller knows this node is alive and keeps its own prediction."""
        out = Float64MultiArray()
        out.data = [stamp, float('nan'), float('nan'), float('nan'), 0.0, 0.0, dt]
        self.pub.publish(out)

    def on_image(self, msg):
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        gray = cv2.cvtColor(self.bridge.imgmsg_to_cv2(msg, 'bgr8'), cv2.COLOR_BGR2GRAY)
        prev, prev_stamp = self.prev_gray, self.prev_stamp
        self.prev_gray, self.prev_stamp = gray, stamp
        if prev is None or prev.shape != gray.shape:
            return
        dt = stamp - prev_stamp
        if dt <= 0.0 or dt > MAX_FRAME_GAP_SEC:
            return

        p0 = cv2.goodFeaturesToTrack(prev, MAX_FEATURES, 0.01, 7)
        if p0 is None or len(p0) < MIN_FEATURES:
            self.publish_no_measurement(stamp, dt)
            return
        p1, status, _ = cv2.calcOpticalFlowPyrLK(prev, gray, p0, None,
                                                 winSize=(21, 21), maxLevel=3)
        ok = status.ravel() == 1
        p0 = p0.reshape(-1, 2)[ok]
        flow = (p1.reshape(-1, 2)[ok] - p0)
        if len(p0) < MIN_FEATURES:
            self.publish_no_measurement(stamp, dt)
            return

        h, w = gray.shape
        (d, c, _), keep, residual = robust_fit(p0, flow, w / 2.0, h / 2.0)
        if keep.sum() < MIN_FEATURES:
            self.publish_no_measurement(stamp, dt)
            return
        # Vertical shift with the turn's rotational flow removed.
        focal = (w / 2.0) / np.tan(self.hfov / 2.0)
        (_, _, b_derot), keep_r, _ = robust_fit(p0, flow, w / 2.0, h / 2.0, focal)
        if keep_r.sum() >= MIN_FEATURES:
            b = b_derot
        else:
            (_, _, b), _, _ = robust_fit(p0, flow, w / 2.0, h / 2.0)

        out = Float64MultiArray()
        out.data = [stamp, b / dt, d / dt, c / dt, float(keep.sum()), residual, dt]
        self.pub.publish(out)
        self.frames += 1
        if self.frames % 300 == 1:
            self.get_logger().info(
                f'vertical shift {b / dt:+.1f} px/s, {int(keep.sum())} features')


def main():
    parser = argparse.ArgumentParser(description='Vertical motion from the camera')
    parser.add_argument('--image-topic', default='/Stonefish/camera/image_raw')
    parser.add_argument('--output-topic', default='/vision/heave_flow')
    parser.add_argument('--hfov-deg', type=float, default=90.0,
                        help='camera horizontal field of view (ZED Mini / Stonefish camera: 90)')
    args, ros_args = parser.parse_known_args()
    rclpy.init(args=[sys.argv[0]] + ros_args)
    node = VisualHeaveEstimator(args.image_topic, args.output_topic, args.hfov_deg)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
