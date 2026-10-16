import cv2
import numpy as np
import rclpy

from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Float64MultiArray
from cv_bridge import CvBridge


# Below this contour area (px) a blob is treated as noise.
MIN_AREA_PX = 50.0
# Contour area (px) at which confidence reaches 1.0.
FULL_CONFIDENCE_AREA_PX = 400.0


class ObjectDetector(Node):

    def __init__(self):
        super().__init__('stonefish_object_detector')

        self.bridge = CvBridge()

        # Subscribe to the Stonefish front camera
        self.subscription = self.create_subscription(
            Image,
            '/Stonefish/camera/image_raw',
            self.image_callback,
            10
        )

        # Publish annotated camera feed
        self.image_pub = self.create_publisher(
            Image,
            '/stonefish/detection_image',
            10
        )

        # Contract with Task2Manager (one message per camera frame):
        #   x = e_x = cx - width/2   (+ = target right of centre)
        #   y = e_y = cy - height/2  (+ = target below centre)
        #   (cx, cy) = centre of the bounding box = geometric centre of the
        #   target; the pixel centroid is pulled toward its heavier parts.
        #   z = confidence in [0, 1] (0 = nothing detected)
        self.detection_pub = self.create_publisher(
            PointStamped,
            '/stonefish/target/detection',
            10
        )

        # Bounding box of the same detection, used for range estimation:
        #   [x_min, y_min, x_max, y_max, area, image_width, image_height]
        self.bbox_pub = self.create_publisher(
            Float64MultiArray,
            '/stonefish/target/bbox',
            10
        )

        # Raw pixel centroid, kept for existing tools: x = cx, y = cy, z = area
        self.position_pub = self.create_publisher(
            PointStamped,
            '/stonefish/object/position',
            10
        )

        # HSV range for the yellow target. Water absorbs red first, so the
        # target is yellow up close (hue ~25) and turns green with distance
        # (core hue ~73-80 at 8 m). Water is hue >= 90, the seabed seen from
        # 3 m depth is hue 86-87 and brown seabed up close is hue <= 18.
        self.lower_yellow = np.array([20, 90, 35])
        self.upper_yellow = np.array([84, 255, 255])

        self.get_logger().info('Object detector started')
        self.get_logger().info('Listening to /Stonefish/camera/image_raw')
        self.get_logger().info('Publishing /stonefish/target/detection')

    def image_callback(self, msg):

        frame = self.bridge.imgmsg_to_cv2(
            msg,
            desired_encoding='bgr8'
        )
        height, width = frame.shape[:2]

        # Convert BGR image to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Detect yellow (underwater: yellow-to-green) regions
        mask = cv2.inRange(
            hsv,
            self.lower_yellow,
            self.upper_yellow
        )

        # Remove small noise
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel
        )
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel
        )

        # Find contours
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # Annotate a copy of the camera image
        output = frame.copy()

        detection = PointStamped()
        detection.header = msg.header
        detection.point.z = 0.0

        # The target is an upright object: reject wide, flat regions and
        # regions spanning the full image width (seabed, surface glare).
        candidates = []
        for contour in contours:
            bx, by, bw, bh = cv2.boundingRect(contour)
            spans_width = bx <= 1 and bx + bw >= width - 1
            if bw > 2.0 * bh or spans_width:
                continue
            candidates.append(contour)

        if candidates:
            # Select the largest yellow region
            largest = max(candidates, key=cv2.contourArea)
            area = cv2.contourArea(largest)

            # Ignore very small yellow regions
            if area > MIN_AREA_PX:

                x, y, w, h = cv2.boundingRect(largest)

                # Geometric centre of the target (bounding box centre)
                cx = x + w / 2.0
                cy = y + h / 2.0
                confidence = min(1.0, area / FULL_CONFIDENCE_AREA_PX)

                detection.point.x = cx - width / 2.0
                detection.point.y = cy - height / 2.0
                detection.point.z = confidence

                bbox = Float64MultiArray()
                bbox.data = [
                    float(x), float(y), float(x + w), float(y + h),
                    float(area), float(width), float(height)]
                self.bbox_pub.publish(bbox)

                position = PointStamped()
                position.header = msg.header
                position.point.x = cx
                position.point.y = cy
                position.point.z = float(area)
                self.position_pub.publish(position)

                # Draw bounding box and centroid
                cv2.rectangle(
                    output,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2
                )

                cv2.circle(
                    output,
                    (int(cx), int(cy)),
                    7,
                    (0, 0, 255),
                    -1
                )

                cv2.putText(
                    output,
                    f'e=({detection.point.x:.0f}, {detection.point.y:.0f}) '
                    f'conf={confidence:.2f}',
                    (x, max(y - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

        # Always publish, so the controller can tell "nothing seen"
        # apart from "detector not running".
        self.detection_pub.publish(detection)

        # Image centre cross
        cv2.drawMarker(
            output,
            (width // 2, height // 2),
            (255, 255, 255),
            cv2.MARKER_CROSS,
            20,
            1
        )

        # Publish annotated image
        output_msg = self.bridge.cv2_to_imgmsg(
            output,
            encoding='bgr8'
        )
        output_msg.header = msg.header
        self.image_pub.publish(output_msg)


def main(args=None):

    rclpy.init(args=args)
    node = ObjectDetector()

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
