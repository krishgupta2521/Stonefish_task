#!/usr/bin/env python3
"""System evaluation log for AUV and ROV mode: one Excel workbook per run.

  system_evaluation_<time>.xlsx   (written when the logger stops)
    stonefish_camera   one row per Stonefish camera frame:
                       camera -> Stonefish YOLO -> controller -> T200s
    real_camera        one row per real camera / video frame:
                       camera -> real YOLO -> simulator -> simulated target moved
    controller_input   ROV: one row per joystick press / release, from the
                       moment it was physically pressed (kernel time stamp of
                       the controller event) to the simulation reading it and
                       the thrusters being set
    summary            count / mean / median / p95 / max of every latency, per mode
  The same columns are used in both modes; controller columns are empty in AUV.
  Live CSV copies of the three data sheets are written as the run goes (they
  survive a crash).

Inputs (all wall-clock / kernel CLOCK_REALTIME stamps on one machine):
  /external_camera/frame_info  real_camera_yolo_detector.py, every frame
  /stonefish/frame_info        stonefish_yolo_detector.py, every frame
      [image_stamp, detector_rx, detection_publish, detected, confidence,
       cx, cy, bbox_w, bbox_h, frame_w, frame_h]
  /usv/control_debug           Task2Manager, every control step
      [42] control start [43] thrusters set [44] Stonefish detection received
      [45] its image stamp [46] /object/position received [47] its image stamp
      [48..53] T200 thrust N [54..56] Fx Fy Fz N [57] yaw torque N m
      [63] mode (0 AUV, 1 ROV) [64..67] pilot input forward, sideways, vertical,
      turn [68] time the step read the pilot input [69] time the real-camera
      detection moved the sim target [70] that detection's image stamp
  /dev/input/eventN            the game controller (raw kernel events)

Image coordinates in the logs: centroid_x -1 (left) .. +1 (right),
centroid_y -1 (bottom) .. +1 (top), (0, 0) = image centre.

usage: system_logger.py [--out DIR] [--controller /dev/input/eventN]
Run it with a Python that has openpyxl (yolo_env); without it only CSVs are written.
"""

import argparse
import csv
import fcntl
import os
import struct
import sys
import threading
import time
from collections import deque
from datetime import datetime

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

STATE_NAMES = [
    'SEARCH', 'YAW_ALIGN', 'YAW_SETTLE', 'VERTICAL_ALIGN', 'VERIFY_CENTER',
    'APPROACH', 'TARGET_LOST_HOLD', 'CALIBRATE', 'STOPPED',
    'TARGET_REACHED', 'BACK_OFF', 'SURFACING', 'SURFACED', 'RETREAT', 'ROV_MANUAL']

THRUSTER_NAMES = ['front_left', 'front_right', 'rear_left', 'rear_right',
                  'middle_left', 'middle_right']
PILOT_INPUTS = ['forward', 'sideways', 'vertical', 'turn']

FRAME_COLUMNS = (
    ['frame_id', 'mode', 'camera_tx_timestamp', 'ros_rx_timestamp', 'transport_latency_ms',
     'detected', 'confidence', 'centroid_x', 'centroid_y', 'bbox_width', 'bbox_height',
     'frame_width', 'frame_height', 'bbox_frame_ratio',
     'detection_publish_timestamp', 'inference_ms', 'sim_rx_timestamp', 'sim_time_s',
     'control_timestamp', 'control_state', 'surge_cmd', 'heave_cmd', 'yaw_cmd',
     'translation_force_N', 'heave_force_N', 'torque_force_Nm']
    + [f'thruster_{n}_N' for n in THRUSTER_NAMES]
    + ['thruster_publish_timestamp', 'ros_rx_to_control_ms', 'control_to_thruster_ms',
       'ros_rx_to_thruster_ms', 'camera_tx_to_thruster_ms']
    + [f'pilot_{n}' for n in PILOT_INPUTS])
REAL_EXTRA = ['sim_target_moved_timestamp', 'sim_rx_to_sim_target_ms', 'camera_tx_to_sim_target_ms']

INPUT_COLUMNS = ['event_id', 'mode', 'input', 'action', 'value', 'pressed_timestamp',
                 'sim_read_timestamp', 'thruster_timestamp', 'pressed_to_sim_ms',
                 'pressed_to_thruster_ms', 'sim_time_s', 'control_state',
                 'surge_cmd', 'heave_cmd', 'yaw_cmd'] + [f'pilot_{n}' for n in PILOT_INPUTS]

LATENCY_COLUMNS = {
    'stonefish_camera': ['transport_latency_ms', 'inference_ms', 'ros_rx_to_control_ms',
                         'control_to_thruster_ms', 'ros_rx_to_thruster_ms', 'camera_tx_to_thruster_ms'],
    'real_camera': ['transport_latency_ms', 'inference_ms', 'ros_rx_to_control_ms',
                    'camera_tx_to_thruster_ms', 'sim_rx_to_sim_target_ms', 'camera_tx_to_sim_target_ms'],
    'controller_input': ['pressed_to_sim_ms', 'pressed_to_thruster_ms'],
}

MATCH_TIMEOUT_SEC = 2.0
CONTROL_HISTORY = 2000          # control steps kept for matching (~20 s at 100 Hz)
STAMP_EPS = 1e-6
STICK_DEADZONE = 0.10           # same as the simulator (ROV_STICK_DEADZONE)


def stamp_text(t):
    """Epoch seconds -> '2026-09-02 23:55:44.517' (local time), '' if unknown."""
    if t is None or t != t or t <= 0.0:
        return ''
    return datetime.fromtimestamp(t).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]


def ms(a, b):
    """(b - a) in milliseconds, '' if either is unknown."""
    if a is None or b is None or a != a or b != b or a <= 0.0 or b <= 0.0:
        return ''
    return round((b - a) * 1000.0, 3)


def num(v, digits):
    return '' if v is None or v != v else round(v, digits)


def mode_name(ctl):
    return 'ROV' if ctl and ctl.get('mode', 0.0) > 0.5 else 'AUV'


class Sheet:
    """Rows kept for the workbook, and mirrored live to a CSV file."""

    def __init__(self, name, columns, csv_path):
        self.name = name
        self.columns = columns
        self.rows = []
        self.csv_path = csv_path
        self.file = open(csv_path, 'w', newline='', buffering=1)
        self.writer = csv.writer(self.file)
        self.writer.writerow(columns)

    def add(self, row):
        self.rows.append(row)
        self.writer.writerow(row)

    def close(self):
        self.file.close()
        if not self.rows:
            os.remove(self.csv_path)


class Loop:
    """One camera -> detector -> controller chain."""

    def __init__(self, sheet, uses_object_topic):
        self.sheet = sheet
        self.uses_object_topic = uses_object_topic   # real camera: /object/position
        self.pending = deque()
        self.frame_id = 0
        self.matched = 0
        self.detected = 0

    def control_uses(self, frame, ctl):
        """Does control step ctl come after this frame reached the simulator?"""
        image_stamp, publish = frame[0], frame[2]
        if self.uses_object_topic:
            if frame[3] > 0.5:   # detections are forwarded on /object/position
                return ctl['object_stamp'] >= image_stamp - STAMP_EPS
            return ctl['control'] >= publish   # nothing sent: next control step
        return ctl['detection_stamp'] >= image_stamp - STAMP_EPS

    def write(self, frame, ctl, mode):
        (image_stamp, rx, publish, detected, conf, cx, cy, bw, bh, fw, fh) = frame[:11]
        self.frame_id += 1
        detected = detected > 0.5
        # -1..+1 over the range the box can move (x right, y up), as both
        # camera views show it. The real detector already reports its point
        # as a pixel of the full frame on that scale.
        if self.uses_object_topic:
            rx_, ry_ = fw / 2.0, fh / 2.0
        else:
            rx_, ry_ = max(fw / 2.0 - bw / 2.0, 1.0), max(fh / 2.0 - bh / 2.0, 1.0)
        cxn = max(-1.0, min(1.0, (cx - fw / 2.0) / rx_)) if detected and fw > 0 else 0.0
        cyn = max(-1.0, min(1.0, -(cy - fh / 2.0) / ry_)) if detected and fh > 0 else 0.0
        row = [self.frame_id, mode, stamp_text(image_stamp), stamp_text(rx), ms(image_stamp, rx),
               int(detected), round(conf, 3), round(cxn, 4) + 0.0, round(cyn, 4) + 0.0,
               round(bw, 1), round(bh, 1), round(fw, 1), round(fh, 1),
               round((bw * bh) / (fw * fh), 8) if fw * fh > 0 else 0.0,
               stamp_text(publish), ms(rx, publish)]
        target_moved = None
        if ctl is None:
            # No control step used it: simulator still calibrating its
            # thrusters (first ~20 sim s), not running, or a newer frame won.
            row += ['', '', '', 'NO_CONTROL']
            row += [''] * (len(FRAME_COLUMNS) - len(row))
        else:
            if self.uses_object_topic:
                same = abs(ctl['object_stamp'] - image_stamp) < STAMP_EPS
                sim_rx = ctl['object_rx'] if same else None
                if detected and abs(ctl['target_moved_stamp'] - image_stamp) < STAMP_EPS:
                    target_moved = ctl['target_moved']
            else:
                same = abs(ctl['detection_stamp'] - image_stamp) < STAMP_EPS
                sim_rx = ctl['detection_rx'] if same else None
            state = int(ctl['state'])
            row += [stamp_text(sim_rx), round(ctl['sim_time'], 3),
                    stamp_text(ctl['control']),
                    STATE_NAMES[state] if 0 <= state < len(STATE_NAMES) else str(state),
                    num(ctl['surge_cmd'], 4), num(ctl['heave_cmd'], 4), num(ctl['yaw_cmd'], 4),
                    num(ctl['fx'], 3), num(ctl['fz'], 3), num(ctl['torque'], 3)]
            row += [num(t, 3) for t in ctl['thrust']]
            row += [stamp_text(ctl['thruster']),
                    ms(rx, ctl['control']), ms(ctl['control'], ctl['thruster']),
                    ms(rx, ctl['thruster']), ms(image_stamp, ctl['thruster'])]
            row += [num(v, 3) for v in ctl['pilot']]
            self.matched += 1
        if self.uses_object_topic:
            # The real camera moves the simulated target (case 6 / ROV mode).
            sim_rx_time = ctl['object_rx'] if ctl and target_moved else None
            row += [stamp_text(target_moved), ms(sim_rx_time, target_moved),
                    ms(image_stamp, target_moved)]
        self.sheet.add(row)
        self.detected += int(detected)


class ControllerReader(threading.Thread):
    """Reads the game controller's raw kernel events (time of the real press)."""

    EV_KEY, EV_ABS = 0x01, 0x03
    ABS_X, ABS_Y, ABS_RX, ABS_RY = 0x00, 0x01, 0x03, 0x04
    BTN_EAST = 0x131   # "B" on an Xbox pad: stop
    EVENT = struct.Struct('llHHi')

    def __init__(self, path, on_event):
        super().__init__(daemon=True)
        self.path = path
        self.on_event = on_event
        self.fd = os.open(path, os.O_RDONLY)
        self.ranges = {}
        for code in (self.ABS_X, self.ABS_Y, self.ABS_RX, self.ABS_RY):
            # EVIOCGABS(code): struct input_absinfo {value, min, max, fuzz, flat, res}
            buf = fcntl.ioctl(self.fd, (2 << 30) | (24 << 16) | (ord('E') << 8) | (0x40 + code),
                              b'\0' * 24)
            _, lo, hi, _, _, _ = struct.unpack('6i', buf)
            self.ranges[code] = (lo, hi)
        self.active = {n: 0 for n in PILOT_INPUTS}   # -1, 0, +1 outside the dead zone

    @staticmethod
    def find():
        """The first joystick in /proc/bus/input/devices (handler jsN)."""
        try:
            blocks = open('/proc/bus/input/devices').read().split('\n\n')
        except OSError:
            return None
        for b in blocks:
            handlers = [l for l in b.splitlines() if l.startswith('H: Handlers=')]
            if handlers and ' js' in handlers[0]:
                for h in handlers[0].split('=')[1].split():
                    if h.startswith('event'):
                        return '/dev/input/' + h
        return None

    def normalise(self, code, value):
        lo, hi = self.ranges[code]
        mid = 0.5 * (lo + hi)
        return max(-1.0, min(1.0, (value - mid) / (0.5 * (hi - lo))))

    def run(self):
        mapping = {self.ABS_Y: ('forward', -1.0), self.ABS_X: ('turn', 1.0),
                   self.ABS_RX: ('sideways', 1.0), self.ABS_RY: ('vertical', 1.0)}
        while True:
            try:
                data = os.read(self.fd, self.EVENT.size * 64)
            except OSError:
                return
            for off in range(0, len(data) - self.EVENT.size + 1, self.EVENT.size):
                sec, usec, etype, code, value = self.EVENT.unpack_from(data, off)
                t = sec + usec * 1e-6
                if etype == self.EV_ABS and code in mapping:
                    name, sign = mapping[code]
                    v = sign * self.normalise(code, value)
                    state = 0 if abs(v) < STICK_DEADZONE else (1 if v > 0 else -1)
                    if state != self.active[name]:
                        action = 'release' if state == 0 else 'press'
                        self.active[name] = state
                        self.on_event(t, name, action, v, state)
                elif etype == self.EV_KEY and code == self.BTN_EAST and value in (0, 1):
                    self.on_event(t, 'stop', 'press' if value else 'release', float(value), 0)


class SystemLogger(Node):

    def __init__(self, out_dir, controller_path):
        super().__init__('system_logger')
        os.makedirs(out_dir, exist_ok=True)
        tag = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.xlsx_path = os.path.join(out_dir, f'system_evaluation_{tag}.xlsx')
        self.sheets = {
            'stonefish_camera': Sheet('stonefish_camera', FRAME_COLUMNS,
                                      os.path.join(out_dir, f'stonefish_evaluation_{tag}.csv')),
            'real_camera': Sheet('real_camera', FRAME_COLUMNS + REAL_EXTRA,
                                 os.path.join(out_dir, f'real_camera_evaluation_{tag}.csv')),
            'controller_input': Sheet('controller_input', INPUT_COLUMNS,
                                      os.path.join(out_dir, f'controller_input_{tag}.csv')),
        }
        self.loops = {
            'real': Loop(self.sheets['real_camera'], True),
            'stonefish': Loop(self.sheets['stonefish_camera'], False),
        }
        self.controls = deque(maxlen=CONTROL_HISTORY)
        self.mode = 'AUV'
        self.lock = threading.Lock()
        self.input_pending = deque()   # controller events waiting for the simulator
        self.input_id = 0

        self.create_subscription(Float64MultiArray, '/external_camera/frame_info',
                                 lambda m: self.on_frame('real', m), 50)
        self.create_subscription(Float64MultiArray, '/stonefish/frame_info',
                                 lambda m: self.on_frame('stonefish', m), 50)
        self.create_subscription(Float64MultiArray, '/usv/control_debug', self.on_control, 200)
        self.create_timer(0.5, self.flush_timeouts)

        path = controller_path or ControllerReader.find()
        self.controller = None
        if path:
            try:
                self.controller = ControllerReader(path, self.on_input)
                self.controller.start()
                self.get_logger().info(f'game controller: {path} (kernel time stamps)')
            except OSError as e:
                self.get_logger().warning(f'cannot read the game controller {path}: {e}')
        else:
            self.get_logger().info('no game controller found: controller_input stays empty')
        self.get_logger().info(f'workbook (on exit): {self.xlsx_path}')

    # ---------------- camera frames ----------------
    def on_frame(self, key, msg):
        if len(msg.data) < 11:
            return
        frame = list(msg.data[:11])
        loop = self.loops[key]
        # The control step may already have arrived (it can beat this message).
        for ctl in self.controls:
            if loop.control_uses(frame, ctl):
                loop.write(frame, ctl, mode_name(ctl))
                return
        loop.pending.append((time.time(), frame))

    # ---------------- controller ----------------
    def on_input(self, t, name, action, value, state):
        with self.lock:
            self.input_id += 1
            self.input_pending.append({'id': self.input_id, 't': t, 'input': name,
                                       'action': action, 'value': value, 'state': state,
                                       'added': time.time()})

    def input_seen(self, ev, ctl):
        """Has control step ctl read this press / release?"""
        if ev['input'] == 'stop' or ctl['mode'] < 0.5 or ctl['pilot_read'] != ctl['pilot_read']:
            return False
        if ctl['pilot_read'] < ev['t'] - 0.002:
            return False
        v = ctl['pilot'][PILOT_INPUTS.index(ev['input'])]
        if v != v:
            return False
        if ev['state'] == 0:
            return abs(v) < STICK_DEADZONE
        return v * ev['state'] >= STICK_DEADZONE * 0.5

    def write_input(self, ev, ctl):
        row = [ev['id'], mode_name(ctl) if ctl else self.mode, ev['input'], ev['action'],
               round(ev['value'], 3), stamp_text(ev['t'])]
        if ctl is None:
            row += [''] * (len(INPUT_COLUMNS) - len(row))
        else:
            state = int(ctl['state'])
            row += [stamp_text(ctl['pilot_read']), stamp_text(ctl['thruster']),
                    ms(ev['t'], ctl['pilot_read']), ms(ev['t'], ctl['thruster']),
                    round(ctl['sim_time'], 3),
                    STATE_NAMES[state] if 0 <= state < len(STATE_NAMES) else str(state),
                    num(ctl['surge_cmd'], 4), num(ctl['heave_cmd'], 4), num(ctl['yaw_cmd'], 4)]
            row += [num(v, 3) for v in ctl['pilot']]
        self.sheets['controller_input'].add(row)

    # ---------------- control steps ----------------
    def on_control(self, msg):
        d = msg.data
        if len(d) < 58:
            return   # simulator without the system-log fields
        nan = float('nan')
        get = lambda i: d[i] if len(d) > i else nan
        ctl = {
            'state': d[0], 'sim_time': d[1],
            'yaw_cmd': d[10], 'heave_cmd': d[11], 'surge_cmd': d[12],
            'control': d[42], 'thruster': d[43],
            'detection_rx': d[44], 'detection_stamp': d[45],
            'object_rx': d[46], 'object_stamp': d[47],
            'thrust': list(d[48:54]), 'fx': d[54], 'fy': d[55], 'fz': d[56], 'torque': d[57],
            'mode': get(63) if get(63) == get(63) else 0.0,
            'pilot': [get(64), get(65), get(66), get(67)],
            'pilot_read': get(68),
            'target_moved': get(69), 'target_moved_stamp': get(70),
        }
        self.mode = mode_name(ctl)
        self.controls.append(ctl)
        for loop in self.loops.values():
            while loop.pending and loop.control_uses(loop.pending[0][1], ctl):
                loop.write(loop.pending.popleft()[1], ctl, mode_name(ctl))
        with self.lock:
            keep = deque()
            while self.input_pending:
                ev = self.input_pending.popleft()
                if self.input_seen(ev, ctl):
                    self.write_input(ev, ctl)
                else:
                    keep.append(ev)
            self.input_pending = keep

    def flush_timeouts(self, everything=False):
        now = time.time()
        for loop in self.loops.values():
            while loop.pending and (everything or now - loop.pending[0][0] > MATCH_TIMEOUT_SEC):
                loop.write(loop.pending.popleft()[1], None, self.mode)
        with self.lock:
            while self.input_pending and (everything or now - self.input_pending[0]['added'] > MATCH_TIMEOUT_SEC):
                self.write_input(self.input_pending.popleft(), None)

    # ---------------- workbook ----------------
    def summary_rows(self):
        rows = []
        for sheet_name, cols in LATENCY_COLUMNS.items():
            sheet = self.sheets[sheet_name]
            mode_idx = sheet.columns.index('mode')
            for mode in ('AUV', 'ROV'):
                for col in cols:
                    i = sheet.columns.index(col)
                    v = sorted(r[i] for r in sheet.rows
                               if r[mode_idx] == mode and isinstance(r[i], (int, float)))
                    if not v:
                        continue
                    rows.append([sheet_name, mode, col, len(v), round(sum(v) / len(v), 3),
                                 round(v[len(v) // 2], 3), round(v[min(len(v) - 1, int(0.95 * len(v)))], 3),
                                 round(v[0], 3), round(v[-1], 3)])
        return rows

    def write_workbook(self):
        try:
            from openpyxl import Workbook
            from openpyxl.styles import Font
        except ImportError:
            print('openpyxl not available (run the logger with yolo_env): CSV files only')
            return False
        wb = Workbook()
        wb.remove(wb.active)
        for name in ('stonefish_camera', 'real_camera', 'controller_input'):
            sheet = self.sheets[name]
            ws = wb.create_sheet(name)
            ws.append(sheet.columns)
            for row in sheet.rows:
                ws.append(row)
            for cell in ws[1]:
                cell.font = Font(bold=True)
            ws.freeze_panes = 'C2'
        ws = wb.create_sheet('summary', 0)
        ws.append(['sheet', 'mode', 'latency', 'count', 'mean_ms', 'median_ms', 'p95_ms', 'min_ms', 'max_ms'])
        for cell in ws[1]:
            cell.font = Font(bold=True)
        for row in self.summary_rows():
            ws.append(row)
        ws.append([])
        ws.append(['Image coordinates: centroid_x -1 (left) .. +1 (right), centroid_y -1 (bottom) .. +1 (top).'])
        ws.append(['Controller columns are empty in AUV mode. pressed_timestamp = kernel time of the controller event.'])
        wb.save(self.xlsx_path)
        return True

    def close(self):
        self.flush_timeouts(everything=True)
        for sheet in self.sheets.values():
            sheet.close()
        for name, sheet in self.sheets.items():
            print(f'{name}: {len(sheet.rows)} rows')
        if any(s.rows for s in self.sheets.values()) and self.write_workbook():
            print(f'workbook: {self.xlsx_path}')
        for row in self.summary_rows():
            print(f'  {row[0]:16s} {row[1]} {row[2]:28s} n={row[3]:5d} median {row[5]:8.1f} ms  p95 {row[6]:8.1f} ms')


def main():
    default_out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs',
                               'system_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    parser = argparse.ArgumentParser(description='System evaluation log (AUV and ROV)')
    parser.add_argument('--out', default=default_out, help='output folder')
    parser.add_argument('--controller', default=None,
                        help='game controller event device (default: the first joystick)')
    args, ros_args = parser.parse_known_args()

    rclpy.init(args=[sys.argv[0]] + ros_args)
    node = SystemLogger(args.out, args.controller)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
