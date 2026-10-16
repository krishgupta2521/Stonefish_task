import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2


class CameraNode(Node):

    def __init__(self):
        super().__init__('camera_node')

        # Open USB camera
        self.camera = cv2.VideoCapture(0)

        if not self.camera.isOpened():
            self.get_logger().error('Could not open camera /dev/video0')
            return

        # ROS Image publisher
        self.publisher = self.create_publisher(
            Image,
            '/camera/image_raw',
            10
        )

        # OpenCV <-> ROS converter
        self.bridge = CvBridge()

        # Publish at approximately 30 FPS
        self.timer = self.create_timer(
            1.0 / 30.0,
            self.publish_frame
        )

        self.get_logger().info('USB camera started: /dev/video0')
        self.get_logger().info('Publishing: /camera/image_raw')

    def publish_frame(self):

        ret, frame = self.camera.read()

        if not ret:
            self.get_logger().warning('Could not read frame')
            return

        # Convert OpenCV image to ROS Image message
        msg = self.bridge.cv2_to_imgmsg(
            frame,
            encoding='bgr8'
        )

        # Publish
        self.publisher.publish(msg)

    def destroy_node(self):

        self.camera.release()

        super().destroy_node()


def main(args=None):

    rclpy.init(args=args)

    node = CameraNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
