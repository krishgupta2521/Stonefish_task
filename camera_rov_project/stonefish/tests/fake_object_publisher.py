"""Stand-in for the real camera in case 6: publishes a slowly moving pixel
centroid on /object/position so the simulated target moves gently.

Usage: python3 fake_object_publisher.py [amplitude_px] [period_s]
"""
import math
import sys

import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.node import Node


class FakeObject(Node):

    def __init__(self, amplitude, period):
        super().__init__('fake_object_publisher')
        self.amplitude = amplitude
        self.period = period
        self.pub = self.create_publisher(PointStamped, '/object/position', 10)
        self.t0 = self.get_clock().now()
        self.create_timer(0.1, self.tick)

    def tick(self):
        t = (self.get_clock().now() - self.t0).nanoseconds * 1e-9
        phase = 2.0 * math.pi * t / self.period
        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        # horizontal pixel -> target depth, vertical pixel -> target Y (see UpdateTrackedObject)
        msg.point.x = 320.0 + self.amplitude * math.sin(phase)
        msg.point.y = 240.0 + 0.8 * self.amplitude * math.sin(0.5 * phase)
        msg.point.z = 1000.0   # area, as the real object_detector publishes
        self.pub.publish(msg)


def main():
    amplitude = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0
    period = float(sys.argv[2]) if len(sys.argv) > 2 else 60.0
    rclpy.init()
    node = FakeObject(amplitude, period)
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
