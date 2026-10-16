#!/usr/bin/env python3
"""Build the real-size target buoy for Stonefish.

Real buoy (measured by the team): 40 cm long in total including the pole,
bulb 10 cm wide and 16 cm tall, pole 8 cm wide.

The bulb is the yellow body of yellow_sensor_model.obj (the shape the
Stonefish YOLO model was trained on), rescaled to 10 x 10 x 16 cm. The pole is
a new 8 cm cylinder below it. Output is in metres, long axis Z with the bulb
at +Z and the origin at the middle of the 40 cm (the bounding-box centre), so
the scenario places it with scale 1 and the same pi pitch as before.

usage: make_buoy.py models/yellow_sensor_model.obj models/buoy_real.obj
"""
import sys

import numpy as np

TOTAL_LENGTH = 0.40
BULB_WIDTH = 0.10
BULB_HEIGHT = 0.16
POLE_DIAMETER = 0.08
POLE_SEGMENTS = 32


def read_group(path, group):
    verts, faces, current = [], [], None
    for line in open(path):
        p = line.split()
        if not p:
            continue
        if p[0] == 'v':
            verts.append([float(x) for x in p[1:4]])
        elif p[0] in ('g', 'usemtl'):
            current = p[1]
        elif p[0] == 'f' and current == group:
            idx = [int(x.split('/')[0]) - 1 for x in p[1:]]
            for k in range(1, len(idx) - 1):   # fan-triangulate
                faces.append([idx[0], idx[k], idx[k + 1]])
    verts = np.array(verts)
    faces = np.array(faces)
    used = np.unique(faces)
    remap = -np.ones(len(verts), dtype=int)
    remap[used] = np.arange(len(used))
    return verts[used], remap[faces]


def main(src, dst):
    v, f = read_group(src, 'yellow')
    f = f[:, ::-1]   # the source winds the bulb clockwise; make its normals face out
    lo, hi = v.min(0), v.max(0)
    centre = 0.5 * (lo + hi)
    # Rescale each axis so the bulb is round, BULB_WIDTH wide and BULB_HEIGHT tall,
    # with its top at +TOTAL_LENGTH/2.
    v = (v - centre) * np.array([BULB_WIDTH, BULB_WIDTH, BULB_HEIGHT]) / (hi - lo)
    v[:, 2] += 0.5 * TOTAL_LENGTH - 0.5 * BULB_HEIGHT

    # Pole: from the bottom of the 40 cm up into the bulb, to where the bulb is
    # as wide as the pole, so there is no visible gap.
    r_pole = 0.5 * POLE_DIAMETER
    radial = np.hypot(v[:, 0], v[:, 1])
    z_top = v[radial >= r_pole, 2].min() + 0.005
    z_bot = -0.5 * TOTAL_LENGTH
    a = np.linspace(0, 2 * np.pi, POLE_SEGMENTS, endpoint=False)
    ring = np.c_[r_pole * np.cos(a), r_pole * np.sin(a)]
    n = len(v)
    pv = np.vstack([np.c_[ring, np.full(POLE_SEGMENTS, z_bot)],
                    np.c_[ring, np.full(POLE_SEGMENTS, z_top)],
                    [[0, 0, z_bot]]])
    pf = []
    for i in range(POLE_SEGMENTS):
        j = (i + 1) % POLE_SEGMENTS
        b0, b1, t0, t1 = n + i, n + j, n + POLE_SEGMENTS + i, n + POLE_SEGMENTS + j
        pf += [[b0, b1, t1], [b0, t1, t0], [n + 2 * POLE_SEGMENTS, b1, b0]]
    v = np.vstack([v, pv])
    f = np.vstack([f, np.array(pf)])

    # Per-face normals: the Stonefish OBJ loader renders faces without them black.
    tri = v[f]
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
    with open(dst, 'w') as out:
        out.write('# Real-size target buoy, metres, long axis Z, bulb at +Z, '
                  'origin at the bounding-box centre (tools/make_buoy.py)\n')
        for x, y, z in v:
            out.write(f'v {x:.5f} {y:.5f} {z:.5f}\n')
        for x, y, z in nrm:
            out.write(f'vn {x:.4f} {y:.4f} {z:.4f}\n')
        for k, (i, j, l) in enumerate(f, start=1):
            out.write(f'f {i + 1}//{k} {j + 1}//{k} {l + 1}//{k}\n')
    lo, hi = v.min(0), v.max(0)
    print(f'{dst}: {len(v)} vertices, {len(f)} faces, '
          f'size {hi[0] - lo[0]:.3f} x {hi[1] - lo[1]:.3f} x {hi[2] - lo[2]:.3f} m, '
          f'pole z {z_bot:.3f}..{z_top:.3f}')


if __name__ == '__main__':
    main(*sys.argv[1:3])
