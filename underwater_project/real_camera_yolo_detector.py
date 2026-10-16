import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped

from cv_bridge import CvBridge
from ultralytics import YOLO

import cv2


MODEL_PATH = "/home/krish/Documents/Development/kyushu_project/runs/detect/runs/detect/real_camera_v1/weights/best.pt"


class RealCameraYOLODetector(Node):

    def __init__(self):
        super().__init__("real_camera_yolo_detector")

        self.bridge = CvBridge()

        # Load trained YOLO model
        self.model = YOLO(MODEL_PATH)

        self.get_logger().info(f"Loaded model: {MODEL_PATH}")

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

        self.get_logger().info(
            "Real Camera YOLO detector started."
        )

    def image_callback(self, msg):

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
            imgsz=640,
            conf=0.35,
            device=0,
            verbose=False
        )

        annotated = frame.copy()

        height, width = frame.shape[:2]

        # Camera center
        camera_cx = width / 2
        camera_cy = height / 2

        best_detection = None
        best_confidence = 0.0

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
            error_x = target_cx - camera_cx
            error_y = target_cy - camera_cy

            # Draw information
            cv2.putText(
                annotated,
                f"our_object {best_confidence:.2f}",
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            cv2.putText(
                annotated,
                f"Error X: {error_x:.0f}  Y: {error_y:.0f}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            # Publish position
            position_msg = PointStamped()

            position_msg.header = msg.header

            position_msg.point.x = float(target_cx)
            position_msg.point.y = float(target_cy)

            # Store confidence for now
            position_msg.point.z = float(best_confidence)

            self.position_pub.publish(position_msg)

        else:

            cv2.putText(
                annotated,
                "NO OBJECT DETECTED",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

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

    rclpy.init(args=args)

    node = RealCameraYOLODetector()

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
