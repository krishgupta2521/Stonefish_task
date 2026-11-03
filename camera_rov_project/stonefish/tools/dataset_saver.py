#!/usr/bin/env python3
"""Write a YOLO dataset from the simulator's dataset mode (TASK2_DATASET=1).

Pairs every camera frame (/Stonefish/camera/image_raw) with its exact bulb
box (/dataset/label, same stamp; projected from the bulb mesh with the true
camera pose) and saves at most one frame per buoy placement:

    <out>/{train,valid}/images/<prefix>_NNNNN.jpg
    <out>/{train,valid}/labels/<prefix>_NNNNN.txt   (class 0 = buoy, YOLO format)

Frames are skipped while the buoy is moving, and when less than 35 % of its
box is inside the image or it is under 3 px. Frames with no buoy in view get
an empty label file. The first frames are also saved with their box drawn in
<out>/preview/ to check the labels by eye.

usage: dataset_saver.py <out_dir> [--count 1500] [--prefix run1] [--val-every 7]
"""
import argparse
import os
import sys

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray

MIN_VISIBLE = 0.35
MIN_SIZE_PX = 3.0
MIN_SPACING_SEC = 0.25     # one frame per placement (the buoy moves every 0.3 s)
PREVIEWS = 12


class DatasetSaver(Node):

    def __init__(self, args):
        super().__init__('dataset_saver')
        self.args = args
        for split in ('train', 'valid'):
            for kind in ('images', 'labels'):
                os.makedirs(os.path.join(args.out, split, kind), exist_ok=True)
        os.makedirs(os.path.join(args.out, 'preview'), exist_ok=True)
        self.images, self.labels = {}, {}
        self.saved = self.negatives = self.previews = 0
        self.last_saved = -1e9
        self.create_subscription(Image, '/Stonefish/camera/image_raw', self.on_image, 20)
        self.create_subscription(Float64MultiArray, '/dataset/label', self.on_label, 50)

    @staticmethod
    def key(t):
        return round(t, 6)

    def on_image(self, msg):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.images[self.key(t)] = msg
        self.match(self.key(t))

    def on_label(self, msg):
        self.labels[self.key(msg.data[0])] = list(msg.data)
        self.match(self.key(msg.data[0]))

    def match(self, k):
        if k not in self.images or k not in self.labels:
            return
        img, lab = self.images.pop(k), self.labels.pop(k)
        # drop stale unmatched entries
        for d in (self.images, self.labels):
            for old in [x for x in d if x < k - 2.0]:
                del d[old]
        self.save(k, img, lab)

    def save(self, t, msg, lab):
        _, usable, u0, v0, u1, v1 = lab
        if usable < 0.5 or t - self.last_saved < MIN_SPACING_SEC:
            return
        w, h = msg.width, msg.height
        line = ''
        if u0 >= 0.0 or u1 >= 0.0:
            area = max(u1 - u0, 1e-6) * max(v1 - v0, 1e-6)
            cu0, cv0 = max(u0, 0.0), max(v0, 0.0)
            cu1, cv1 = min(u1, float(w)), min(v1, float(h))
            inside = max(cu1 - cu0, 0.0) * max(cv1 - cv0, 0.0)
            if inside > 0.0:
                if inside / area < MIN_VISIBLE or cu1 - cu0 < MIN_SIZE_PX or cv1 - cv0 < MIN_SIZE_PX:
                    return
                line = (f'0 {(cu0 + cu1) / 2 / w:.6f} {(cv0 + cv1) / 2 / h:.6f} '
                        f'{(cu1 - cu0) / w:.6f} {(cv1 - cv0) / h:.6f}\n')
        frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, 3)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        split = 'valid' if self.saved % self.args.val_every == self.args.val_every - 1 else 'train'
        name = f'{self.args.prefix}_{self.saved:05d}'
        cv2.imwrite(os.path.join(self.args.out, split, 'images', name + '.jpg'), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        with open(os.path.join(self.args.out, split, 'labels', name + '.txt'), 'w') as f:
            f.write(line)
        if self.previews < PREVIEWS and line:
            p = frame.copy()
            cv2.rectangle(p, (int(cu0), int(cv0)), (int(cu1), int(cv1)), (0, 255, 0), 1)
            cv2.imwrite(os.path.join(self.args.out, 'preview', name + '.png'), p)
            self.previews += 1
        self.negatives += 0 if line else 1
        self.saved += 1
        self.last_saved = t
        if self.saved % 100 == 0:
            self.get_logger().info(f'{self.saved} frames ({self.negatives} without buoy)')
        if self.saved >= self.args.count:
            self.get_logger().info(f'done: {self.saved} frames in {self.args.out}')
            raise SystemExit(0)


def main():
    parser = argparse.ArgumentParser(description='YOLO dataset from the simulator')
    parser.add_argument('out')
    parser.add_argument('--count', type=int, default=1500)
    parser.add_argument('--prefix', default='sim')
    parser.add_argument('--val-every', type=int, default=7)
    args, ros_args = parser.parse_known_args()
    rclpy.init(args=[sys.argv[0]] + ros_args)
    node = DatasetSaver(args)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
