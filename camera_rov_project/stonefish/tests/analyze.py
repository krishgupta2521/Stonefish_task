"""Evaluate one recorded run against simulator ground truth.

Usage: python3 analyze.py <run_dir> <case> [perception_fault]
  perception_fault: the run had NO detector; expect hold + safe surfacing.
Exit code 0 = PASS, 1 = FAIL.
"""
import csv
import math
import sys

import numpy as np

STATE_NAMES = [
    'SEARCH', 'YAW_ALIGN', 'YAW_SETTLE', 'VERTICAL_ALIGN', 'VERIFY_CENTER',
    'APPROACH', 'TARGET_LOST_HOLD', 'CALIBRATE', 'STOPPED',
    'TARGET_REACHED', 'BACK_OFF', 'SURFACING', 'SURFACED', 'RETREAT', 'ROV_MANUAL']
REACHED, SURFACING, SURFACED, SEARCH, APPROACH = 9, 11, 12, 0, 5

# ---- Simulation geometry: KYUBIC V2 (src/kyubic_model.hpp, models/kyubic/) ----
import os
import re as _re

_HERE = os.path.dirname(os.path.abspath(__file__))
_HEADER = open(os.path.join(_HERE, '..', 'src', 'kyubic_model.hpp')).read()


def _vec(name):
    m = _re.search(name + r'\[3\] = \{([^}]*)\}', _HEADER)
    return [float(x) for x in m.group(1).split(',')]


CAMERA_X, _, CAMERA_Z = _vec('CAMERA_POS')                 # ZED Mini, body frame (NED)
FRONT_X = float(_re.search(r'FRONT_X = ([0-9.]+)', _HEADER).group(1))  # recovery hooks
SCALE = float(_re.search(r'SCALE = ([0-9.]+)', _HEADER).group(1))   # robot size x real KYUBIC
# World scale (see Sim:: in Task2Manager.cpp): tuned on KYUBIC x4.275 in a
# 4.275x larger world; lengths scale by L, speeds by sqrt(L) (Froude).
L = SCALE / 4.275
V = math.sqrt(L)
REAL_BUOY = SCALE < 1.5
# The team's buoy (models/yellow_sensor_model.obj), scaled uniformly: real
# bulb 16 cm tall (0.16 / 1.54), or the 1.71 m demo buoy (x0.5). Clearance is
# to a vertical cylinder of the bulb's larger semi-axis over the whole height.
BUOY_SCALE = 0.16 / 1.54 if REAL_BUOY else 0.5
TARGET_RADIUS = 0.47 * BUOY_SCALE
TARGET_TOP, TARGET_BOTTOM = -1.40 * BUOY_SCALE, 2.02 * BUOY_SCALE   # relative to the origin
# The robot centres on the BULB (the YOLO model detects the bulb only): its
# geometric centre is above the target origin.
TARGET_CENTRE_DZ = -0.18 * BUOY_SCALE
TOP_SENSOR_HEIGHT = 0.223 * SCALE         # top of the floaters above the hull axis


def hull_surface_points(spacing=0.02 * SCALE):
    """Points spread over KYUBIC's physics solids (the shapes that collide)."""
    pts = []
    folder = os.path.join(_HERE, '..', 'models', 'kyubic')
    for name in sorted(os.listdir(folder)):
        if not name.endswith('_physics.obj'):
            continue
        v, f = [], []
        for line in open(os.path.join(folder, name)):
            if line.startswith('v '):
                v.append([float(x) for x in line.split()[1:4]])
            elif line.startswith('f '):
                f.append([int(x.split('/')[0]) - 1 for x in line.split()[1:4]])
        v, f = np.array(v), np.array(f)
        for a, b, c in v[f]:
            n = max(1, int(max(np.linalg.norm(b - a), np.linalg.norm(c - a)) / spacing))
            for i in range(n + 1):
                for j in range(n + 1 - i):
                    pts.append(a + (b - a) * i / n + (c - a) * j / n)
    return np.array(pts)


HULL_PTS = hull_surface_points()


def clearance(x, y, depth, yaw, tx, ty, tz):
    """Minimum distance hull -> target (vertical cylinder), metres. <0 = contact."""
    c, s = math.cos(yaw), math.sin(yaw)
    wx = x + c * HULL_PTS[:, 0] - s * HULL_PTS[:, 1]
    wy = y + s * HULL_PTS[:, 0] + c * HULL_PTS[:, 1]
    wz = depth + HULL_PTS[:, 2]
    radial = np.hypot(wx - tx, wy - ty) - TARGET_RADIUS
    z0, z1 = tz + TARGET_TOP, tz + TARGET_BOTTOM
    vertical = np.maximum(np.maximum(z0 - wz, wz - z1), 0.0)
    inside_z = vertical == 0.0
    d = np.where(inside_z, radial, np.hypot(np.maximum(radial, 0.0), vertical))
    return float(d.min())


def report_depth_estimate(d):
    """How far the controller's depth (camera estimate) was from the truth."""
    if 'depth_est' not in d or not np.isfinite(d['depth_est']).any():
        return
    ok = np.isfinite(d['depth_est'])
    camera = 'depth_from_camera' in d and np.nanmax(d['depth_from_camera']) > 0.5
    err = d['depth_est'][ok] - d['depth'][ok]
    print(f"\nDepth used by the controller: {'CAMERA estimate' if camera else 'simulator truth'}"
          f"; error vs truth mean {err.mean():+.3f} m, max |error| {np.abs(err).max():.3f} m, "
          f"final {err[-1]:+.3f} m")


def load(run_dir):
    with open(f'{run_dir}/telemetry.csv') as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def main():
    run_dir, case = sys.argv[1], int(sys.argv[2])
    perception_fault = len(sys.argv) > 3 and sys.argv[3] == 'perception_fault'
    # depth_fault: no camera depth estimate at all -> abort, vertical thrusters
    # off, KYUBIC floats up on its own buoyancy (no "SURFACED" can be confirmed).
    depth_fault = len(sys.argv) > 3 and sys.argv[3] == 'depth_fault'
    d = load(run_dir)
    t = d['t']
    state = d['state'].astype(int)
    checks = []

    def check(name, ok, detail):
        checks.append((name, ok, detail))

    print(f'===== CASE {case}  ({run_dir}) =====')
    print(f'samples={len(t)}  sim time {t[0]:.1f} -> {t[-1]:.1f} s')

    # State timeline
    print('\nState timeline:')
    for i in np.flatnonzero(np.r_[True, state[1:] != state[:-1]]):
        print(f'  t={t[i]:7.2f}  {STATE_NAMES[state[i]]}')

    seen = set(state.tolist())
    final = state[-1]
    aborted = d['aborted'][-1] > 0.5
    print(f'\nFinal state: {STATE_NAMES[final]}   aborted={aborted}')

    has_target = not np.all(np.isnan(d['truth_x']))
    speed = np.abs(d['body_vx'])
    roll = np.degrees(np.abs(d['roll']))
    pitch = np.degrees(np.abs(d['pitch']))
    print(f'Max |surge speed| = {speed.max():.3f} m/s   '
          f'max |roll| = {roll.max():.1f} deg   max |pitch| = {pitch.max():.1f} deg')
    print(f'Max |thrust cmd|: surge {np.abs(d["surge_cmd"]).max():.2f}  '
          f'yaw {np.abs(d["yaw_cmd"]).max():.2f}  heave {np.abs(d["heave_cmd"]).max():.2f}')
    report_depth_estimate(d)

    check('stays upright (|roll|,|pitch| < 10 deg)', roll.max() < 10 and pitch.max() < 10,
          f'{roll.max():.1f} / {pitch.max():.1f} deg')
    check(f'speed never above {0.30 * V:.3f} m/s', speed.max() <= 0.30 * V, f'{speed.max():.3f} m/s')
    if not depth_fault:
        check('ends SURFACED', final == SURFACED, STATE_NAMES[final])
    top = d['depth'][-1] - TOP_SENSOR_HEIGHT
    check('floaters break the surface at the end', top < -0.005 * L, f'{-top:.3f} m above')
    if (state == SURFACED).any():
        sd = d['depth'][state == SURFACED]
        check(f'stays at the surface (depth spread < {0.10 * L:.3f} m)', np.ptp(sd) < 0.10 * L,
              f'{sd.min():.2f}-{sd.max():.2f} m')

    if depth_fault:
        moved = np.hypot(d['x'] - d['x'][0], d['y'] - d['y'][0]).max()
        print(f'\nDepth-estimate fault run: max horizontal drift = {moved:.2f} m')
        ab = d['aborted'] > 0.5
        check('vertical thrust only lifts after the abort (no depth loop)',
              ab.any() and d['heave_cmd'][ab].max() <= 1e-6, '')
        check('did not approach without a depth estimate', APPROACH not in seen, '')
        check('aborted', aborted, f'aborted={aborted}')
    elif perception_fault:
        moved = np.hypot(d['x'] - d['x'][0], d['y'] - d['y'][0]).max()
        print(f'\nPerception fault run: max horizontal drift = {moved:.2f} m')
        check('detector reported missing', (d['detector_alive'] < 0.5).all(), '')
        check('did not search or approach blind', not ({SEARCH, APPROACH} & seen), '')
        check(f'held position (drift < {0.3 * L:.2f} m)', moved < 0.3 * L, f'{moved:.2f} m')
        check('aborted and surfaced', aborted, f'aborted={aborted}')
    elif has_target:
        # Ground-truth geometry, subsampled to ~20 Hz
        idx = np.arange(0, len(t), max(1, len(t) // max(1, int((t[-1] - t[0]) * 20))))
        clr = np.array([clearance(d['x'][i], d['y'][i], d['depth'][i], d['yaw'][i],
                                  d['truth_x'][i], d['truth_y'][i], d['truth_z'][i])
                        for i in idx])
        imin = idx[np.argmin(clr)]
        print(f'\nMin hull-to-target clearance = {clr.min():.3f} m '
              f'(t={t[imin]:.1f}, {STATE_NAMES[state[imin]]})')
        near = clr < 1.0 * L
        near_speed = speed[idx][near].max() if near.any() else 0.0
        print(f'Max speed while within {1.0 * L:.2f} m of the target = {near_speed:.3f} m/s')
        check(f'NO contact with target (clearance > {0.05 * L:.3f} m)', clr.min() > 0.05 * L,
              f'{clr.min():.3f} m')
        check(f'slow near target (< {0.10 * V:.3f} m/s within {1.0 * L:.2f} m)', near_speed < 0.10 * V,
              f'{near_speed:.3f} m/s')

        check('reached the target', REACHED in seen, 'TARGET_REACHED seen' if REACHED in seen else 'never')
        check('not aborted', not aborted, '')
        if REACHED in seen:
            i = np.flatnonzero(state == REACHED)[-1]   # end of the hold = settled pose
            yaw = d['yaw'][i]
            c, s = math.cos(yaw), math.sin(yaw)
            camx = d['x'][i] + c * CAMERA_X
            camy = d['y'][i] + s * CAMERA_X
            camz = d['depth'][i] + CAMERA_Z
            tx, ty = d['truth_x'][i], d['truth_y'][i]
            tz = d['truth_z'][i] + TARGET_CENTRE_DZ
            dx, dy = tx - camx, ty - camy
            along = c * dx + s * dy
            lateral = -s * dx + c * dy
            vertical = tz - camz
            true_gap = along - TARGET_RADIUS - (FRONT_X - CAMERA_X)
            est_gap = d['front_gap'][i]
            clr_reached = clearance(d['x'][i], d['y'][i], d['depth'][i], yaw,
                                    d['truth_x'][i], d['truth_y'][i], d['truth_z'][i])
            print('\nAt TARGET_REACHED (end of hold):')
            print(f'  true front gap      = {true_gap:.3f} m   (estimated {est_gap:.3f} m)')
            print(f'  hull clearance      = {clr_reached:.3f} m')
            print(f'  lateral offset      = {lateral:+.3f} m   (camera axis vs target centre)')
            print(f'  vertical offset     = {vertical:+.3f} m   (+ = target centre deeper)')
            print(f'  image error         = e_x {d["e_x"][i]:+.0f} px, e_y {d["e_y"][i]:+.0f} px')
            print(f'  speed               = {d["body_vx"][i]:+.3f} m/s')
            if case == 6:
                # Moving target: stops where the camera still sees it (1.0 m).
                check(f'stops at moving-target standoff (true gap {0.6 * L:.2f}-{1.4 * L:.2f} m)',
                      0.6 * L <= true_gap <= 1.4 * L, f'{true_gap:.3f} m')
            else:
                check(f'stops close to target (true gap {0.05 * L:.3f}-{0.40 * L:.3f} m)',
                      0.05 * L <= true_gap <= 0.40 * L, f'{true_gap:.3f} m')
            check(f'centred horizontally (|lateral| < {0.10 * L:.3f} m)', abs(lateral) < 0.10 * L,
                  f'{lateral:+.3f} m')
            # Vertical offset is measured to the bulb's centre (TARGET_CENTRE_DZ).
            if case == 6:
                # Moving target (+-0.28 m in depth): once it fills the view
                # there is no vertical information, so require the camera
                # axis to be on the target body.
                half_height = 0.5 * (TARGET_BOTTOM - TARGET_TOP)
                check(f'camera axis on the moving target body (|vertical| < {half_height:.2f} m)',
                      abs(vertical) < half_height, f'{vertical:+.3f} m')
            else:
                check(f'centred vertically (|vertical| < {0.20 * L:.3f} m)', abs(vertical) < 0.20 * L,
                      f'{vertical:+.3f} m')

        # Range-estimate quality during the approach
        ap = state == APPROACH
        if ap.any():
            errs = []
            for i in np.flatnonzero(ap)[::10]:
                yaw = d['yaw'][i]
                c, s = math.cos(yaw), math.sin(yaw)
                along = (c * (d['truth_x'][i] - d['x'][i]) + s * (d['truth_y'][i] - d['y'][i])
                         - FRONT_X - TARGET_RADIUS)
                if d['have_range'][i] > 0.5:
                    errs.append(d['front_gap'][i] - along)
            if errs:
                errs = np.array(errs)
                print(f'\nRange estimate error during approach: mean {errs.mean():+.3f} m, '
                      f'max |err| {np.abs(errs).max():.3f} m')
    else:
        yaw_search = np.unwrap(d['yaw'][state == SEARCH])
        turned = np.abs(np.diff(yaw_search)).sum() if len(yaw_search) > 1 else 0.0
        print(f'\nSearch turned {math.degrees(turned):.0f} deg')
        check('searched a full turn', math.degrees(turned) >= 350, f'{math.degrees(turned):.0f} deg')
        check('aborted safely (no false target)', aborted and REACHED not in seen and APPROACH not in seen,
              f'aborted={aborted}')

    print('\nChecks:')
    ok_all = True
    for name, ok, detail in checks:
        ok_all &= ok
        print(f'  [{"PASS" if ok else "FAIL"}] {name}  {detail}')
    print(f'\nRESULT CASE {case}: {"PASS" if ok_all else "FAIL"}')
    sys.exit(0 if ok_all else 1)


if __name__ == '__main__':
    main()
