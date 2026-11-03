#!/usr/bin/env python3
"""Split the buoy model into one OBJ per material, so Stonefish (one look per
object) can draw it in its real colours: yellow bulb, dark grey pole, white
mounts, metal bracket. Without this the whole buoy rendered yellow.

Every piece's faces are turned to face outwards: the source model's bulb
(and the small metal pieces) were inside out, so the renderer culled their
front and showed the pole through the bulb.

The parts keep the source model's units and origin; the scenario scales them
uniformly (x0.5 = the 1.71 m demo buoy; 0.16 / 1.54 = the real buoy, whose
bulb is 16 cm tall and 9.8 cm wide).

usage: split_buoy.py models/yellow_sensor_model.obj models/buoy/
"""
import os
import sys

import numpy as np


def orient_outwards(verts, faces):
    """Reverse the faces whose winding points into their piece (pieces =
    faces connected by shared vertex positions; all are convex here)."""
    pos = np.array([[float(x) for x in v.split()[:3]] for v in verts])
    key = [tuple(np.round(p, 5)) for p in pos]
    parent = list(range(len(faces)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    owner = {}
    for fi, face in enumerate(faces):
        for c in face:
            k = key[c[0] - 1]
            if k in owner:
                parent[find(fi)] = find(owner[k])
            else:
                owner[k] = fi
    pieces = {}
    for fi in range(len(faces)):
        pieces.setdefault(find(fi), []).append(fi)
    flipped = set()
    for members in pieces.values():
        centre = np.mean([pos[c[0] - 1] for fi in members for c in faces[fi]], axis=0)
        for fi in members:
            p = [pos[c[0] - 1] for c in faces[fi]]
            normal = np.cross(p[1] - p[0], p[2] - p[0])
            if np.dot(normal, np.mean(p, axis=0) - centre) < 0.0:
                flipped.add(fi)
    return flipped


def main(src, out_dir):
    verts, normals, groups, current = [], [], {}, None
    for line in open(src):
        p = line.split()
        if not p:
            continue
        if p[0] == 'v':
            verts.append(line.split(None, 1)[1].strip())
        elif p[0] == 'vn':
            normals.append(line.split(None, 1)[1].strip())
        elif p[0] == 'g':
            current = p[1]
            groups.setdefault(current, [])
        elif p[0] == 'f':
            groups[current].append([tuple(int(x) if x else 0 for x in c.split('/')) + (0, 0)
                                    for c in p[1:]])
    os.makedirs(out_dir, exist_ok=True)
    for name, faces in groups.items():
        flipped = orient_outwards(verts, faces)
        vmap, nmap, vout, nout = {}, {}, [], []
        lines = []
        for fi, face in enumerate(faces):
            flip = fi in flipped
            idx = []
            for c in (reversed(face) if flip else face):
                v, n = c[0], c[2]
                if v not in vmap:
                    vmap[v] = len(vout) + 1
                    vout.append(verts[v - 1])
                if n and (n, flip) not in nmap:
                    nmap[(n, flip)] = len(nout) + 1
                    nx, ny, nz = (float(x) for x in normals[n - 1].split()[:3])
                    nout.append(f'{-nx:.6f} {-ny:.6f} {-nz:.6f}' if flip else normals[n - 1])
                idx.append(f'{vmap[v]}//{nmap[(n, flip)]}' if n else f'{vmap[v]}')
            lines.append('f ' + ' '.join(idx))
        path = os.path.join(out_dir, name + '.obj')
        with open(path, 'w') as f:
            f.write(f'# {name} part of {os.path.basename(src)} (tools/split_buoy.py)\n')
            f.writelines(f'v {v}\n' for v in vout)
            f.writelines(f'vn {n}\n' for n in nout)
            f.writelines(l + '\n' for l in lines)
        print(f'{path}: {len(vout)} vertices, {len(faces)} faces ({len(flipped)} turned outwards)')


if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
