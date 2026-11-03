"""Real camera YOLO detector: /external_camera/image_raw -> /object/position.

The image comes from external_camera_publisher.py: the USB camera or a video
file (--video), so this node is the same for both.

  real_camera_yolo_detector.py [--model best.pt] [--conf 0.35] [--imgsz 640] [--rotate 90]

/object/position: x, y = buoy centre in pixels, z = confidence.

--rotate N: the camera is mounted turned; N = degrees CLOCKWISE that make its
picture upright (0, 90, 180, 270). Detection runs on the picture as the camera
gives it (the model was trained on that); the position sent to the simulator
and the image shown in rqt are turned upright, so moving the object left/right
moves it left/right in the simulator.
"""
import argparse
import os
import sys
import time

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Float64MultiArray

from cv_bridge import CvBridge
from ultralytics import YOLO

import cv2
import numpy as np


# (x, y) in -1..+1 (right, up) of the camera picture -> of the upright picture,
# for a clockwise turn of 0 / 90 / 180 / 270 degrees.
UPRIGHT = {
    0: lambda x, y: (x, y),
    90: lambda x, y: (y, -x),
    180: lambda x, y: (-x, -y),
    270: lambda x, y: (-y, x),
}
CV_ROTATE = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def rotate_point(px, py, width, height, rotate):
    """A pixel of the camera picture -> the same pixel of the turned picture."""
    if rotate == 90:
        return height - 1 - py, px
    if rotate == 180:
        return width - 1 - px, height - 1 - py
    if rotate == 270:
        return py, width - 1 - px
    return px, py


# Buoy model trained on real pool images (yolo26n, 300 epochs, 640 px).
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(REPO_ROOT, "runs/detect/buoy_yolo26n_300_epochs/weights/best.pt")


class RealCameraYOLODetector(Node):

    def __init__(self, model_path=MODEL_PATH, conf=0.35, imgsz=640, rotate=0):
        super().__init__("real_camera_yolo_detector")

        self.bridge = CvBridge()
        self.conf = conf
        self.imgsz = imgsz
        self.rotate = rotate

        # Load trained YOLO model (GPU if there is one)
        self.model = YOLO(model_path)
        try:
            import torch
            self.device = 0 if torch.cuda.is_available() else "cpu"
        except ImportError:
            self.device = "cpu"

        self.get_logger().info(
            f"Loaded model: {model_path} classes {self.model.names}, "
            f"conf {conf}, imgsz {imgsz}, device {self.device}, "
            f"camera picture turned {rotate} deg clockwise to be upright"
        )

        # Subscribe to external camera
        self.image_sub = self.create_subscription(
            Image,
            "/external_camera/image_raw",
            self.image_callback,
            10
        )

        # Publish annotated image
        self.image_pub = self.create_publisher(
            Image,
            "/external_camera/detection_image",
            10
        )

        # Publish object position
        self.position_pub = self.create_publisher(
            PointStamped,
            "/object/position",
            10
        )

        # Per-frame timing for system_logger.py:
        # [image_stamp, detector_rx, detection_publish (epoch s),
        #  detected, confidence, cx, cy, bbox_w, bbox_h, frame_w, frame_h]
        self.frame_info_pub = self.create_publisher(
            Float64MultiArray,
            "/external_camera/frame_info",
            10
        )

        self.get_logger().info(
            "Real Camera YOLO detector started."
        )

    def image_callback(self, msg):

        rx_time = time.time()

        try:
            frame = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding="bgr8"
            )
        except Exception as e:
            self.get_logger().error(
                f"CV Bridge error: {e}"
            )
            return

        # Run YOLO
        results = self.model.predict(
            source=frame,
            imgsz=self.imgsz,
            conf=self.conf,
            device=self.device,
            verbose=False
        )

        annotated = frame.copy()

        height, width = frame.shape[:2]

        # Picture area: a video is letterboxed into 640x480 (black bars), so
        # -1..+1 is measured over the real picture, not the padded frame.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rows = np.flatnonzero(gray.max(axis=1) > 8)
        cols = np.flatnonzero(gray.max(axis=0) > 8)
        top, bottom = (rows[0], rows[-1] + 1) if len(rows) else (0, height)
        left, right = (cols[0], cols[-1] + 1) if len(cols) else (0, width)

        # Camera center (centre of the picture area)
        camera_cx = (left + right) / 2
        camera_cy = (top + bottom) / 2
        half_w = max((right - left) / 2, 1.0)
        half_h = max((bottom - top) / 2, 1.0)

        best_detection = None
        best_confidence = 0.0
        best_label = ""
        texts = []   # (text, position, scale, colour) on the upright picture

        for result in results:

            if result.boxes is None:
                continue

            for box in result.boxes:

                confidence = float(box.conf[0])

                if confidence < best_confidence:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)

                label = self.model.names.get(int(box.cls[0]), "object")

                target_cx = int((x1 + x2) / 2)
                target_cy = int((y1 + y2) / 2)

                best_detection = (
                    x1,
                    y1,
                    x2,
                    y2,
                    target_cx,
                    target_cy
                )
                best_label = label

                best_confidence = confidence

        # Draw camera center
        cv2.circle(
            annotated,
            (int(camera_cx), int(camera_cy)),
            5,
            (255, 0, 0),
            -1
        )

        if best_detection is not None:

            (
                x1,
                y1,
                x2,
                y2,
                target_cx,
                target_cy
            ) = best_detection

            # Draw bounding box
            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Draw object center
            cv2.circle(
                annotated,
                (target_cx, target_cy),
                6,
                (0, 0, 255),
                -1
            )

            # Calculate image error
            # Normalised over the range the object can move in the picture:
            # x -1 = box touching the left edge .. +1 = touching the right edge,
            # y -1 = touching the bottom .. +1 = touching the top, 0 = centre.
            range_x = max(half_w - (x2 - x1) / 2, 1.0)
            range_y = max(half_h - (y2 - y1) / 2, 1.0)
            norm_x = round(float(np.clip((target_cx - camera_cx) / range_x, -1.0, 1.0)), 3) + 0.0
            norm_y = round(float(np.clip(-(target_cy - camera_cy) / range_y, -1.0, 1.0)), 3) + 0.0
            # Upright: the camera is mounted turned (--rotate).
            norm_x, norm_y = (v + 0.0 for v in UPRIGHT[self.rotate](norm_x, norm_y))
            # The same point as a pixel of the full 640x480 frame (y down), so
            # the simulator (which maps 0..640 / 0..480) gets the full range too.
            full_px = (norm_x + 1.0) * 0.5 * width
            full_py = (1.0 - norm_y) * 0.5 * height

            # Text is drawn after the picture is turned upright (below).
            corners = [rotate_point(cx, cy, width, height, self.rotate)
                       for cx, cy in ((x1, y1), (x2, y1), (x1, y2), (x2, y2))]
            label_x = int(min(c[0] for c in corners))
            label_y = int(min(c[1] for c in corners))
            texts.append((f"{best_label} {best_confidence:.2f}", (label_x, max(label_y - 10, 20)),
                          0.6, (0, 255, 0)))
            texts.append((f"X: {norm_x:+.2f}  Y: {norm_y:+.2f}", (20, 30), 0.7, (255, 255, 255)))

            # Publish position
            position_msg = PointStamped()

            position_msg.header = msg.header

            position_msg.point.x = float(full_px)
            position_msg.point.y = float(full_py)

            # Store confidence for now
            position_msg.point.z = float(best_confidence)

            self.position_pub.publish(position_msg)

        else:

            texts.append(("NO OBJECT DETECTED", (20, 40), 0.8, (0, 0, 255)))

        # The picture as the operator sees it: upright, with the text on top.
        if self.rotate in CV_ROTATE:
            annotated = cv2.rotate(annotated, CV_ROTATE[self.rotate])
        for text, org, scale, color in texts:
            cv2.putText(annotated, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)

        # Per-frame timing for system_logger.py (every frame, detected or not)
        publish_time = time.time()
        info = Float64MultiArray()
        if best_detection is not None:
            x1, y1, x2, y2, target_cx, target_cy = best_detection
            box = [1.0, best_confidence, float(full_px), float(full_py),
                   float(x2 - x1), float(y2 - y1)]
        else:
            box = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        info.data = [msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                     rx_time, publish_time] + box + [float(width), float(height)]
        self.frame_info_pub.publish(info)

        # Publish annotated image
        try:
            output_msg = self.bridge.cv2_to_imgmsg(
                annotated,
                encoding="bgr8"
            )

            output_msg.header = msg.header

            self.image_pub.publish(output_msg)

        except Exception as e:
            self.get_logger().error(
                f"Image publish error: {e}"
            )


def main(args=None):

    parser = argparse.ArgumentParser(description="Real camera YOLO detector")
    parser.add_argument("--model", default=MODEL_PATH, help="YOLO weights (.pt)")
    parser.add_argument("--conf", type=float, default=0.35, help="minimum confidence")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="YOLO input size (the model was trained at 640)")
    parser.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270],
                        help="degrees CLOCKWISE that make the camera picture upright "
                             "(camera mounted turned)")
    parsed_args, ros_args = parser.parse_known_args()

    rclpy.init(args=[sys.argv[0]] + ros_args)

    node = RealCameraYOLODetector(parsed_args.model, parsed_args.conf, parsed_args.imgsz, parsed_args.rotate)

    try:
        rclpy.spin(node)

    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass

    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
