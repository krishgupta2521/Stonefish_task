#!/usr/bin/env python3

import argparse
import sys

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Float64MultiArray
from cv_bridge import CvBridge

from ultralytics import YOLO

import cv2


MODEL_PATH = "/home/krish/Documents/Development/kyushu_project/runs/detect/training/stonefish_v1/weights/best.pt"


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
        imgsz
    ):

        super().__init__(node_name)

        self.image_topic = image_topic
        self.detection_topic = detection_topic
        self.output_image_topic = output_image_topic
        self.mode = mode
        self.show_window = show_window
        self.imgsz = imgsz

        self.bridge = CvBridge()

        # =====================================================
        # LOAD YOLO MODEL
        # =====================================================

        self.model = YOLO(MODEL_PATH)

        self.get_logger().info("YOLO model loaded.")
        self.get_logger().info(f"Model: {MODEL_PATH}")

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
            conf=0.5,
            verbose=False,
            device=0
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

        cv2.drawMarker(
            frame,
            (camera_cx, camera_cy),
            (255, 255, 255),
            cv2.MARKER_CROSS,
            20,
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
                f"Center: ({best_target_cx}, {best_target_cy})",
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
                f"Error: X={error_x} Y={error_y}",
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
        help="YOLO input size. 960 suits the ZED Mini's 90 deg view: the "
             "target is small and 512 misses most frames beyond ~5 m"
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
        imgsz=parsed_args.imgsz
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