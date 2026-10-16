import os
import cv2
import rclpy

from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


SAVE_DIR = "/home/krish/Documents/Development/kyushu_project/real_camera_dataset"


class RealCameraCapture(Node):

    def __init__(self):

        super().__init__('real_camera_capture')

        self.bridge = CvBridge()

        os.makedirs(SAVE_DIR, exist_ok=True)

        self.image_count = len([
            f for f in os.listdir(SAVE_DIR)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])

        self.subscription = self.create_subscription(
            Image,
            '/external_camera/image_raw',
            self.image_callback,
            10
        )

        self.get_logger().info(
            "=========================================="
        )
        self.get_logger().info(
            "Real Camera Dataset Capture Started"
        )
        self.get_logger().info(
            f"Saving images to: {SAVE_DIR}"
        )
        self.get_logger().info(
            "Press SPACE to capture an image"
        )
        self.get_logger().info(
            "Press Q to quit"
        )
        self.get_logger().info(
            "=========================================="
        )

    def image_callback(self, msg):

        try:

            frame = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding='bgr8'
            )

        except Exception as e:

            self.get_logger().error(
                f"CvBridge error: {e}"
            )

            return

        display = frame.copy()

        cv2.putText(
            display,
            f"Images: {self.image_count}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

        cv2.putText(
            display,
            "SPACE = capture | Q = quit",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "Real Camera Dataset Capture",
            display
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):

            filename = os.path.join(
                SAVE_DIR,
                f"image_{self.image_count:04d}.jpg"
            )

            cv2.imwrite(
                filename,
                frame
            )

            self.image_count += 1

            self.get_logger().info(
                f"Saved: {filename}"
            )

        elif key == ord('q'):

            self.get_logger().info(
                "Stopping dataset capture..."
            )

            rclpy.shutdown()


def main(args=None):

    rclpy.init(args=args)

    node = RealCameraCapture()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:

        cv2.destroyAllWindows()

        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()

