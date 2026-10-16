import cv2
import numpy as np
import rclpy

from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge


class ObjectDetector(Node):

    def __init__(self):
        super().__init__('object_detector')

        self.bridge = CvBridge()

        # Subscribe to the real USB camera feed
        self.subscription = self.create_subscription(
            Image,
            '/external_camera/image_raw',
            self.image_callback,
            10
        )

        # Publish annotated camera feed
        self.image_pub = self.create_publisher(
            Image,
            '/external_camera/detection_image',
            10
        )

        # Publish detected object position
        self.position_pub = self.create_publisher(
            PointStamped,
            '/object/position',
            10
        )

        # HSV range for yellow
        self.lower_yellow = np.array([15, 80, 80])
        self.upper_yellow = np.array([40, 255, 255])

        self.get_logger().info('Object detector started')
        self.get_logger().info('Listening to /camera/image_raw')

    def image_callback(self, msg):

        frame = self.bridge.imgmsg_to_cv2(
            msg,
            desired_encoding='bgr8'
        )

        # Convert BGR image to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Detect yellow regions
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

        if contours:
            # Select the largest yellow region
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)

            # Ignore very small yellow regions
            if area > 500:

                x, y, w, h = cv2.boundingRect(largest)

                # Calculate centroid using image moments
                moments = cv2.moments(largest)

                if moments['m00'] != 0:
                    cx = int(moments['m10'] / moments['m00'])
                    cy = int(moments['m01'] / moments['m00'])

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
                        (cx, cy),
                        7,
                        (0, 0, 255),
                        -1
                    )

                    cv2.putText(
                        output,
                        f'Object: ({cx}, {cy})',
                        (x, max(y - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )

                    # Publish pixel coordinates
                    position = PointStamped()
                    position.header = msg.header
                    position.point.x = float(cx)
                    position.point.y = float(cy)
                    position.point.z = float(area)

                    self.position_pub.publish(position)

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
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
