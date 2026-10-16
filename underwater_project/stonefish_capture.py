import os
import time

import cv2
import rclpy

from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class StonefishImageCollector(Node):

    def __init__(self):
        super().__init__('stonefish_image_collector')

        self.bridge = CvBridge()

        self.output_dir = (
            '/home/krish/Documents/Development/kyushu_project/'
            'underwater_project/raw_data/stonefish'
        )

        os.makedirs(self.output_dir, exist_ok=True)

        self.frame_count = 0
        self.last_saved_time = 0.0

        # Save one image every 0.5 seconds
        self.save_interval = 0.5

        self.subscription = self.create_subscription(
            Image,
            '/Stonefish/camera/image_raw',
            self.image_callback,
            10
        )

        self.get_logger().info(
            'Listening to /Stonefish/camera/image_raw'
        )

        self.get_logger().info(
            f'Saving images to: {self.output_dir}'
        )

    def image_callback(self, msg):

        current_time = time.time()

        # Don't save every camera frame
        if current_time - self.last_saved_time < self.save_interval:
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding='bgr8'
            )

            filename = os.path.join(
                self.output_dir,
                f'stonefish_{self.frame_count:06d}.jpg'
            )

            cv2.imwrite(filename, frame)

            self.get_logger().info(
                f'Saved: {filename}'
            )

            self.frame_count += 1
            self.last_saved_time = current_time

        except Exception as e:
            self.get_logger().error(
                f'Failed to process image: {e}'
            )


def main(args=None):

    rclpy.init(args=args)

    node = StonefishImageCollector()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
