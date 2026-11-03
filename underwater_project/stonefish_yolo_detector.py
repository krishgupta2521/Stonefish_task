#!/usr/bin/env python3

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


# Stonefish buoy model: detects the BULB only, yolo26n at 960 px, trained on
# the team's buoy in its real colours and size (real-size KYUBIC world) with
# exact labels from the simulator (stonefish/tools/make_dataset.sh,
# sim_buoy_dataset/). Earlier models (use --model): the bulb-only model for the
# enlarged demo world runs/stonefish/buoy_yolo26n_300_epochs/weights/best.pt
# (640 px), the full-stick model runs/detect/training/stonefish_v1/weights/best.pt.
# Paths are relative to the repository root (this file is in underwater_project/).
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(REPO_ROOT, "runs/stonefish/real_buoy_sim_960/weights/best.pt")


class YOLODetector(Node):

    def __init__(
        self,
        image_topic,
        detection_topic,
        node_name,
        output_image_topic,
        mode,
        bbox_topic,
        show_window,
        imgsz,
        frame_info_topic="",
        model_path=MODEL_PATH,
        fallback_imgsz=0,
        conf=0.30
    ):

        super().__init__(node_name)

        self.image_topic = image_topic
        self.detection_topic = detection_topic
        self.output_image_topic = output_image_topic
        self.mode = mode
        self.show_window = show_window
        self.imgsz = imgsz
        self.fallback_imgsz = fallback_imgsz
        self.conf = conf

        self.bridge = CvBridge()

        # =====================================================
        # LOAD YOLO MODEL
        # =====================================================

        self.model = YOLO(model_path)
        # GPU if there is one, else CPU (slower, but runs on any laptop)
        try:
            import torch
            self.device = 0 if torch.cuda.is_available() else "cpu"
        except ImportError:
            self.device = "cpu"

        self.get_logger().info("YOLO model loaded.")
        self.get_logger().info(f"Model: {model_path}  classes {self.model.names}  imgsz {imgsz}")

        # =====================================================
        # IMAGE SUBSCRIBER
        # =====================================================

        self.subscription = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            10
        )

        # =====================================================
        # DETECTION PUBLISHER
        # =====================================================

        self.target_pub = self.create_publisher(
            PointStamped,
            self.detection_topic,
            10
        )

        # Same detection in normalised image coordinates:
        # x -1 (left) .. +1 (right), y -1 (bottom) .. +1 (top), z = confidence.
        self.normalized_pub = self.create_publisher(
            PointStamped,
            "/stonefish/target/normalized",
            10
        )

        # =====================================================
        # BOUNDING BOX PUBLISHER
        # Task2Manager estimates the distance to the target from the
        # box size. Without it the robot will not drive forward.
        # [x_min, y_min, x_max, y_max, area, image_width, image_height]
        # =====================================================

        self.bbox_pub = None

        if bbox_topic:
            self.bbox_pub = self.create_publisher(
                Float64MultiArray,
                bbox_topic,
                10
            )

        # =====================================================
        # PER-FRAME TIMING (system_logger.py)
        # [image_stamp, detector_rx, detection_publish (epoch s),
        #  detected, confidence, cx, cy, bbox_w, bbox_h, frame_w, frame_h]
        # =====================================================

        self.frame_info_pub = None

        if frame_info_topic:
            self.frame_info_pub = self.create_publisher(
                Float64MultiArray,
                frame_info_topic,
                10
            )

        # =====================================================
        # ANNOTATED IMAGE PUBLISHER
        # =====================================================

        self.detection_image_pub = self.create_publisher(
            Image,
            self.output_image_topic,
            10
        )

        # =====================================================
        # STARTUP INFORMATION
        # =====================================================

        self.get_logger().info(
            "========================================"
        )

        self.get_logger().info(
            f"Mode: {self.mode}"
        )

        self.get_logger().info(
            f"Subscribed to: {self.image_topic}"
        )

        self.get_logger().info(
            f"Publishing detection to: {self.detection_topic}"
        )

        self.get_logger().info(
            f"Publishing annotated image to: {self.output_image_topic}"
        )

        if self.bbox_pub is not None:
            self.get_logger().info(
                f"Publishing bounding box to: {bbox_topic}"
            )

        self.get_logger().info(
            "========================================"
        )

    # =========================================================
    # IMAGE CALLBACK
    # =========================================================

    def image_callback(self, msg):

        rx_time = time.time()

        # =====================================================
        # ROS IMAGE -> OPENCV
        # =====================================================

        try:

            frame = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding="bgr8"
            )

        except Exception as e:

            self.get_logger().error(
                f"cv_bridge conversion failed: {e}"
            )

            return

        # =====================================================
        # YOLO INFERENCE
        # =====================================================

        results = self.model.predict(
            source=frame,
            imgsz=self.imgsz,
            conf=self.conf,
            verbose=False,
            device=self.device
        )

        # Nothing found: retry the same frame at a larger input size. The
        # bulb is small far away (640 px: 4-34 % at 7.5 m seen from behind;
        # 960 px finds it), while 640 is better close up.
        if (self.fallback_imgsz
                and not any(r.boxes is not None and len(r.boxes) for r in results)):
            results = self.model.predict(
                source=frame,
                imgsz=self.fallback_imgsz,
                conf=self.conf,
                verbose=False,
                device=self.device
            )

        # =====================================================
        # CAMERA INFORMATION
        # =====================================================

        height, width = frame.shape[:2]

        camera_cx = width // 2
        camera_cy = height // 2

        # =====================================================
        # DRAW CAMERA CENTER
        # =====================================================

        # Image centre cross (50 px across, easier to judge the error by eye)
        cv2.drawMarker(
            frame,
            (camera_cx, camera_cy),
            (255, 255, 255),
            cv2.MARKER_CROSS,
            50,
            2
        )

        # =====================================================
        # BEST DETECTION
        # =====================================================

        target_found = False

        best_confidence = 0.0

        best_target_cx = camera_cx
        best_target_cy = camera_cy

        best_box = None

        # =====================================================
        # PROCESS YOLO DETECTIONS
        # =====================================================

        for result in results:

            boxes = result.boxes

            if boxes is None:
                continue

            for box in boxes:

                confidence = float(box.conf[0])

                # Only keep highest-confidence detection
                if confidence < best_confidence:
                    continue

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0].tolist()
                )

                target_cx = int(
                    (x1 + x2) / 2
                )

                target_cy = int(
                    (y1 + y2) / 2
                )

                best_confidence = confidence

                best_target_cx = target_cx
                best_target_cy = target_cy

                best_box = (
                    x1,
                    y1,
                    x2,
                    y2
                )

                target_found = True

        # =====================================================
        # DRAW BEST DETECTION
        # =====================================================

        if target_found and self.bbox_pub is not None:

            # Box BEFORE drawing on the frame
            x1, y1, x2, y2 = best_box

            bbox_msg = Float64MultiArray()
            bbox_msg.data = [
                float(x1),
                float(y1),
                float(x2),
                float(y2),
                float((x2 - x1) * (y2 - y1)),
                float(width),
                float(height)
            ]

            self.bbox_pub.publish(
                bbox_msg
            )

        if target_found:

            x1, y1, x2, y2 = best_box

            error_x = (
                best_target_cx - camera_cx
            )

            error_y = (
                best_target_cy - camera_cy
            )

            # Normalised like the real camera view: over the range the box can
            # move, x -1 = touching the left edge .. +1 = right edge,
            # y -1 = bottom .. +1 = top, (0, 0) = image centre.
            range_x = max(width / 2.0 - (x2 - x1) / 2.0, 1.0)
            range_y = max(height / 2.0 - (y2 - y1) / 2.0, 1.0)
            norm_x = max(-1.0, min(1.0, (best_target_cx - width / 2.0) / range_x))
            norm_y = max(-1.0, min(1.0, -(best_target_cy - height / 2.0) / range_y))
            # (no "-0.00" in the overlay)
            norm_x = round(norm_x, 3) + 0.0
            norm_y = round(norm_y, 3) + 0.0

            norm_msg = PointStamped()
            norm_msg.header = msg.header
            norm_msg.point.x = float(norm_x)
            norm_msg.point.y = float(norm_y)
            norm_msg.point.z = float(best_confidence)
            self.normalized_pub.publish(norm_msg)

            # Bounding box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Target center
            cv2.circle(
                frame,
                (
                    best_target_cx,
                    best_target_cy
                ),
                6,
                (0, 0, 255),
                -1
            )

            # Line from camera center
            cv2.line(
                frame,
                (
                    camera_cx,
                    camera_cy
                ),
                (
                    best_target_cx,
                    best_target_cy
                ),
                (255, 0, 0),
                2
            )

            # Confidence
            cv2.putText(
                frame,
                f"Target: {best_confidence:.2f}",
                (
                    x1,
                    max(y1 - 35, 20)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            # Center
            cv2.putText(
                frame,
                f"Center: ({norm_x:+.2f}, {norm_y:+.2f})",
                (
                    x1,
                    max(y1 - 15, 20)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )

            # Error
            cv2.putText(
                frame,
                f"Error: X={norm_x:+.2f} Y={norm_y:+.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

        # =====================================================
        # CREATE ROS DETECTION MESSAGE
        # =====================================================

        target_msg = PointStamped()

        target_msg.header = msg.header

        if target_found:

            # =================================================
            # MODE A:
            # EXTERNAL CAMERA
            #
            # Publish ABSOLUTE PIXEL POSITION
            # =================================================

            if self.mode == "external":

                target_msg.point.x = float(
                    best_target_cx
                )

                target_msg.point.y = float(
                    best_target_cy
                )

                target_msg.point.z = float(
                    best_confidence
                )

            # =================================================
            # MODE B:
            # STONEFISH CAMERA
            #
            # Publish CENTER-RELATIVE ERROR
            # =================================================

            else:

                target_msg.point.x = float(
                    best_target_cx - camera_cx
                )

                target_msg.point.y = float(
                    best_target_cy - camera_cy
                )

                target_msg.point.z = float(
                    best_confidence
                )

            self.target_pub.publish(
                target_msg
            )

            # Status
            cv2.putText(
                frame,
                "TARGET DETECTED",
                (
                    10,
                    height - 20
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        else:

            # =================================================
            # TARGET LOST
            # =================================================

            target_msg.point.x = 0.0
            target_msg.point.y = 0.0
            target_msg.point.z = 0.0

            self.target_pub.publish(
                target_msg
            )

            cv2.putText(
                frame,
                "TARGET NOT DETECTED",
                (
                    10,
                    height - 20
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

        # =====================================================
        # PER-FRAME TIMING
        # =====================================================

        if self.frame_info_pub is not None:

            bw = bh = 0.0
            if target_found:
                bw = float(best_box[2] - best_box[0])
                bh = float(best_box[3] - best_box[1])

            info = Float64MultiArray()
            info.data = [
                msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                rx_time,
                time.time(),
                1.0 if target_found else 0.0,
                float(best_confidence),
                float(best_target_cx) if target_found else 0.0,
                float(best_target_cy) if target_found else 0.0,
                bw,
                bh,
                float(width),
                float(height)
            ]
            self.frame_info_pub.publish(info)

        # =====================================================
        # PUBLISH ANNOTATED IMAGE
        # =====================================================

        output_msg = self.bridge.cv2_to_imgmsg(
            frame,
            encoding="bgr8"
        )

        output_msg.header = msg.header

        self.detection_image_pub.publish(
            output_msg
        )

        # =====================================================
        # DISPLAY
        # =====================================================

        if self.show_window:

            window_name = (
                "External Camera YOLO26"
                if self.mode == "external"
                else "Stonefish YOLO26"
            )

            cv2.imshow(
                window_name,
                frame
            )

            cv2.waitKey(1)


# =============================================================
# MAIN
# =============================================================

def main(args=None):

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image-topic",
        required=True
    )

    parser.add_argument(
        "--detection-topic",
        required=True
    )

    parser.add_argument(
        "--node-name",
        required=True
    )

    parser.add_argument(
        "--output-image-topic",
        required=True
    )

    parser.add_argument(
        "--mode",
        choices=[
            "external",
            "stonefish"
        ],
        required=True
    )

    parser.add_argument(
        "--bbox-topic",
        default="/stonefish/target/bbox",
        help="bounding box output for range estimation ('' to disable)"
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=960,
        help="YOLO input size. The default model was trained at 960 (the real "
             "bulb is ~13 px wide at the far start). Use 640 with the older "
             "enlarged-world model."
    )

    parser.add_argument(
        "--fallback-imgsz",
        type=int,
        default=0,
        help="if nothing is detected at --imgsz, retry at this size (0 = off; "
             "960 suited the older 640 px model)"
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.30,
        help="minimum confidence. The default model: 100%% of held-out buoy "
             "frames and no false alarms at 0.30 (it scores small far bulbs "
             "~0.45); use 0.5 with the older models"
    )

    parser.add_argument(
        "--model",
        default=MODEL_PATH,
        help="YOLO weights (default: bulb-only Stonefish buoy model)"
    )

    parser.add_argument(
        "--frame-info-topic",
        default="/stonefish/frame_info",
        help="per-frame timing for system_logger.py ('' to disable)"
    )

    parser.add_argument(
        "--no-window",
        action="store_true",
        help="do not open an OpenCV window (view it in rqt_image_view)"
    )

    parsed_args, ros_args = parser.parse_known_args()

    rclpy.init(args=[sys.argv[0]] + ros_args)

    node = YOLODetector(
        image_topic=parsed_args.image_topic,
        detection_topic=parsed_args.detection_topic,
        node_name=parsed_args.node_name,
        output_image_topic=parsed_args.output_image_topic,
        mode=parsed_args.mode,
        bbox_topic=parsed_args.bbox_topic,
        show_window=not parsed_args.no_window,
        imgsz=parsed_args.imgsz,
        frame_info_topic=parsed_args.frame_info_topic,
        model_path=parsed_args.model,
        fallback_imgsz=parsed_args.fallback_imgsz,
        conf=parsed_args.conf
    )

    try:

        rclpy.spin(node)

    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):

        pass

    finally:

        cv2.destroyAllWindows()

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":

    main()