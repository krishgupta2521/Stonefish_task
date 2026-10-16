"""Convert the KYUBIC V2 STEP assembly into Stonefish models.

Writes to stonefish/models/kyubic/:
  <group>_visual.obj   CAD tessellation of the parts you see (graphics)
  <group>_physics.obj  closed boxes / cylinders sized from the CAD (buoyancy, drag, mass)
and prints the thruster, camera and sensor poses in the Stonefish body frame.

Body frame (Stonefish, NED): x forward, y right, z down, metres.
CAD frame (this STEP): millimetres, +Z forward, +Y up, +X left.
Origin: the control-hull axis at CAD (X 0, Y 163.2, Z 0).

Needs OpenCascade bindings:  python -m venv cadenv && cadenv/bin/pip install cadquery-ocp numpy
Usage: cadenv/bin/python tools/kyubic_step_export.py "<KYUBIC .step>" models/kyubic [scale]

scale (default 1) enlarges the whole robot geometrically: every length is
multiplied by it, volumes and masses by scale^3 (same densities, so the same
trim). The header records it as kyubic::SCALE; the simulator scales the
thrusters to match.
"""
import math
import os
import re
import sys
import time

import numpy as np
from OCP.BRep import BRep_Tool
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.Bnd import Bnd_Box
from OCP.OCP.collections import Sequence_TDF_Label
from OCP.STEPCAFControl import STEPCAFControl_Reader
from OCP.TCollection import TCollection_ExtendedString
from OCP.TDataStd import TDataStd_Name
from OCP.TDF import TDF_Label
from OCP.TDocStd import TDocStd_Document
from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
from OCP.TopExp import TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS
from OCP.XCAFApp import XCAFApp_Application
from OCP.XCAFDoc import XCAFDoc_DocumentTool, XCAFDoc_ShapeTool

SCALE = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
M_PER_MM = SCALE / 1000.0                        # CAD millimetres -> simulator metres
ORIGIN_MM = np.array([0.0, 163.2, 0.0])          # control-hull axis
LINEAR_DEFLECTION_MM = 2.0                       # tessellation accuracy (visual only)
ANGULAR_DEFLECTION = 0.8
MIN_VISIBLE_PART_MM = 12.0                       # skip screws, SMD parts, o-rings
CLUSTER_GRID_M = 0.003 * SCALE                   # visual simplification: merge vertices on a 3 mm (CAD) grid


def cad_to_body(p_mm):
    """CAD (mm, X left, Y up, Z forward) -> body (m, x fwd, y right, z down)."""
    d = (np.asarray(p_mm, dtype=float) - ORIGIN_MM) * M_PER_MM
    return np.stack([d[..., 2], -d[..., 0], -d[..., 1]], axis=-1)


def decode(name):
    return re.sub(r'\\X2\\([0-9A-F]+)\\X0\\',
                  lambda m: ''.join(chr(int(m.group(1)[i:i + 4], 16))
                                    for i in range(0, len(m.group(1)), 4)), name)


# ---------------------------------------------------------------- reading
def read_leaves(path):
    app = XCAFApp_Application.GetApplication_s()
    doc = TDocStd_Document(TCollection_ExtendedString("MDTV-XCAF"))
    app.InitDocument(doc)
    reader = STEPCAFControl_Reader()
    reader.SetNameMode(True)
    reader.ReadFile(path)
    reader.Transfer(doc)
    tool = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())

    def name(label):
        attr = TDataStd_Name()
        return decode(attr.Get().ToExtString()) if label.FindAttribute(TDataStd_Name.GetID_s(), attr) else ""

    leaves = []

    def walk(label, loc, names):
        if XCAFDoc_ShapeTool.IsAssembly_s(label):
            comps = Sequence_TDF_Label()
            XCAFDoc_ShapeTool.GetComponents_s(label, comps, False)
            for i in range(1, comps.Length() + 1):
                comp = comps.Value(i)
                ref = TDF_Label()
                XCAFDoc_ShapeTool.GetReferredShape_s(comp, ref)
                walk(ref, loc.Multiplied(XCAFDoc_ShapeTool.GetLocation_s(comp)),
                     names + [name(comp) or name(ref)])
        else:
            shape = XCAFDoc_ShapeTool.GetShape_s(label)
            if not shape.IsNull():
                leaves.append((names, name(label), shape.Moved(loc)))

    roots = Sequence_TDF_Label()
    tool.GetFreeShapes(roots)
    for i in range(1, roots.Length() + 1):
        walk(roots.Value(i), TopLoc_Location(), [name(roots.Value(i))])
    return leaves


def bbox(shape):
    box = Bnd_Box()
    BRepBndLib.Add_s(shape, box)
    if box.IsVoid():
        return None
    a, b = box.CornerMin(), box.CornerMax()
    return np.array([a.X(), a.Y(), a.Z(), b.X(), b.Y(), b.Z()])


# ---------------------------------------------------------------- grouping
def inside_cylinder(bb, axis_x, axis_y, radius, z0, z1):
    dx = max(abs(bb[0] - axis_x), abs(bb[3] - axis_x))
    dy = max(abs(bb[1] - axis_y), abs(bb[4] - axis_y))
    return math.hypot(dx, dy) < radius and bb[2] > z0 and bb[5] < z1


def group_of(names, product, bb):
    p = ' / '.join(names)
    if 'T200' in p:
        return 'thrusters'
    if 'Floater' in p:
        return 'floaters'
    if 'Pathfinder' in p:
        return 'dvl'
    if 'Claw' in p:
        return 'claw'
    if 'Battery hull' in p:
        for ax in (-175.5, 175.4):
            if inside_cylinder(bb, ax, 43.6, 42.0, -112.0, 245.0):
                return None                      # battery and parts inside the tube
        return 'battery_hulls'
    if 'Control hull' in p:
        if 'Injection dome' in product or 'Dome top' in product:
            return None                          # clear acrylic: the camera looks through it
        if inside_cylinder(bb, 0.0, 163.2, 108.0, -150.0, 280.0):
            return None                          # electronics inside the hull
        return 'control_hull'
    return 'frame'


# ---------------------------------------------------------------- meshing
def triangulate(shape):
    BRepMesh_IncrementalMesh(shape, LINEAR_DEFLECTION_MM, False, ANGULAR_DEFLECTION, True)
    verts, tris = [], []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = TopoDS.Face(exp.Current())
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(face, loc)
        if tri is not None:
            trsf = loc.Transformation()
            base = len(verts)
            for i in range(1, tri.NbNodes() + 1):
                verts.append(tri.Node(i).Transformed(trsf).Coord())
            flip = face.Orientation() == TopAbs_REVERSED
            for i in range(1, tri.NbTriangles() + 1):
                a, b, c = tri.Triangle(i).Get()
                tris.append((base + a - 1, base + (c if flip else b) - 1, base + (b if flip else c) - 1))
        exp.Next()
    return np.array(verts, dtype=float), np.array(tris, dtype=np.int64)


def simplify(v, t, grid):
    """Vertex clustering: snap vertices to a grid, drop collapsed triangles."""
    keys = np.floor(v / grid).astype(np.int64)
    _, rep_idx, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    inverse = inverse.reshape(-1)
    nv = v[rep_idx]
    nt = inverse[t]
    keep = (nt[:, 0] != nt[:, 1]) & (nt[:, 1] != nt[:, 2]) & (nt[:, 0] != nt[:, 2])
    nt = nt[keep]
    nt = np.unique(np.sort(nt, axis=1), axis=0, return_index=True)[1]
    return nv, inverse[t][keep][np.sort(nt)]


def write_obj(path, meshes):
    """meshes: list of (vertices_body_m, triangles).
    Writes one normal per face: Stonefish lights a mesh only through the OBJ's
    normals - without them every vertex normal is zero and the part renders black."""
    with open(path, 'w') as f:
        f.write('# KYUBIC V2, generated by tools/kyubic_step_export.py\n')
        v_off, n_off = 1, 1
        for v, t in meshes:
            for x, y, z in v:
                f.write(f'v {x:.5f} {y:.5f} {z:.5f}\n')
            tri = v[t]
            n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
            for nx, ny, nz in n:
                f.write(f'vn {nx:.4f} {ny:.4f} {nz:.4f}\n')
            for i, (a, b, c) in enumerate(t):
                k = n_off + i
                f.write(f'f {a + v_off}//{k} {b + v_off}//{k} {c + v_off}//{k}\n')
            v_off += len(v)
            n_off += len(t)


# The CAD-to-body map (X,Y,Z) -> (Z,-X,-Y) is a proper rotation (det +1), so
# triangle winding (outward normals) is preserved.

# ---------------------------------------------------------------- physics solids
def box_mesh(center_mm, size_mm):
    c, s = np.asarray(center_mm, float), np.asarray(size_mm, float) / 2
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * s + c
    # vertex index = 4*ix + 2*iy + iz
    faces = [(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1),
             (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3)]
    return cad_to_body(corners), np.array(faces)


def cylinder_mesh(center_mm, radius_mm, length_mm, axis, n=24):
    """Closed cylinder along CAD axis 'x', 'y' or 'z'."""
    c = np.asarray(center_mm, float)
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.stack([np.cos(ang), np.sin(ang)], 1) * radius_mm
    h = length_mm / 2
    pts = []
    for zz in (-h, h):
        for u, w in ring:
            local = {'x': (zz, u, w), 'y': (w, zz, u), 'z': (u, w, zz)}[axis]
            pts.append(c + np.array(local))
    pts.append(c + np.array({'x': (-h, 0, 0), 'y': (0, -h, 0), 'z': (0, 0, -h)}[axis]))
    pts.append(c + np.array({'x': (h, 0, 0), 'y': (0, h, 0), 'z': (0, 0, h)}[axis]))
    bot, top = 2 * n, 2 * n + 1
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces += [(i, j, n + j), (i, n + j, n + i), (bot, j, i), (top, n + i, n + j)]
    faces = np.array(faces)
    v = cad_to_body(np.array(pts))
    # make every triangle face outward (winding depends on the axis permutation)
    centre = cad_to_body(c)
    tri = v[faces]
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    outward = np.einsum('ij,ij->i', normal, tri.mean(1) - centre) > 0
    faces[~outward] = faces[~outward][:, ::-1]
    return v, faces


def oriented(v, faces, centre_body):
    tri = v[faces]
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    outward = np.einsum('ij,ij->i', normal, tri.mean(1) - centre_body) > 0
    faces = faces.copy()
    faces[~outward] = faces[~outward][:, ::-1]
    return v, faces


def sphere_mesh(center_mm, radius_mm, n=12):
    c = np.asarray(center_mm, float)
    pts = [c + (0, 0, -radius_mm), c + (0, 0, radius_mm)]
    for i in range(1, n):
        th = np.pi * i / n
        for j in range(2 * n):
            ph = np.pi * j / n
            pts.append(c + radius_mm * np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)]))
    faces = []
    ring = lambda i, j: 2 + (i - 1) * 2 * n + (j % (2 * n))
    for j in range(2 * n):
        faces.append((0, ring(1, j + 1), ring(1, j)))
        faces.append((1, ring(n - 1, j), ring(n - 1, j + 1)))
        for i in range(1, n - 1):
            faces += [(ring(i, j), ring(i, j + 1), ring(i + 1, j + 1)), (ring(i, j), ring(i + 1, j + 1), ring(i + 1, j))]
    return oriented(cad_to_body(np.array(pts)), np.array(faces), cad_to_body(c))


# Physics primitives in CAD millimetres. Each becomes its own Stonefish solid,
# so each gets its own added mass and drag (one shape fitted around a whole
# group gave the open frame ~900 kg of added mass in heave).
def B(center, size):
    return ('box', center, size)


def C(center, radius, length, axis):
    return ('cyl', center, radius, length, axis)


def S(center, radius):
    return ('sphere', center, radius)


def primitive_mesh(p):
    if p[0] == 'box':
        v, f = box_mesh(p[1], p[2])
        return oriented(v, f, cad_to_body(p[1]))
    if p[0] == 'cyl':
        return cylinder_mesh(p[1], p[2], p[3], p[4])
    return sphere_mesh(p[1], p[2])


EXT = 13.5        # 20x20 extrusion: ~1.8 cm^2 of metal (CAD volume / length) = a 13.5 mm square


def foam_box(center, footprint, volume_l):
    """Box over the foam block's footprint, height chosen to give the CAD volume."""
    h = volume_l * 1e6 / (footprint[0] * footprint[1])
    return B(center, (footprint[0], h, footprint[1]))


# Every solid's volume is the CAD's: solid parts use their real volume (plates
# as an equivalent thickness), sealed hulls their outer envelope.
T200_SOLID_RADIUS = (3 * 0.195e6 / (4 * np.pi)) ** (1 / 3)    # 0.195 L sphere per T200
PHYSICS = {
    # 8 in Blue Robotics tube (OD 215.9 mm) from end cap to dome ring, then the
    # acrylic dome as a cylinder of the same volume (spherical cap: 2.42 L)
    'control_hull': [C((0, 163.2, 14.5), 108.0, 323.0, 'z'), C((0, 163.2, 209.0), 108.0, 66.0, 'z')],
    # 3 in WTE3 tubes (OD 88.9 mm) with domes
    'battery_hulls': [C((-175.5, 43.6, 59.0), 44.45, 335.0, 'z'), C((175.4, 43.6, 59.0), 44.45, 335.0, 'z')],
    # per corner: foam block + printed case; upper 1.88 L, lower 1.20 L (CAD solids)
    'floaters': [foam_box((s * 145, 347.7, zc), (138, 140), 1.88) for s in (1, -1) for zc in (162.9, -168.6)]
              + [foam_box((s * 165, 270.2, zc), (138, 140), 1.20) for s in (1, -1) for zc in (177.9, -188.6)],
    # plates at the thickness that gives their CAD volume (they have cut-outs)
    'frame': [B((270, 144.2, 0.9), (5.83, 390, 570)), B((-270, 144.2, 0.9), (5.83, 390, 570)),
              B((0, -5.8, 3.9), (550, 6.77, 340)),
              B((0, 304.2, 145.9), (550, 7.90, 140)), B((0, 304.2, -151.6), (550, 9.36, 155)),
              B((255, -20.8, 0.9), (EXT, EXT, 570)), B((-255, -20.8, 0.9), (EXT, EXT, 570)),
              B((255, 289.2, 0.9), (EXT, EXT, 570)), B((-255, 289.2, 0.9), (EXT, EXT, 570)),
              B((255, 134.2, -191.6), (EXT, 290, EXT)), B((-255, 134.2, -191.6), (EXT, 290, EXT)),
              B((255, 134.2, 193.4), (EXT, 290, EXT)), B((-255, 134.2, 193.4), (EXT, 290, EXT)),
              B((0, -20.8, 193.4), (490, EXT, EXT)), B((0, -20.8, -191.6), (490, EXT, EXT)),
              B((0, 289.2, 90.9), (490, EXT, EXT)), B((0, 289.2, -94.1), (490, EXT, EXT)),
              B((0, -40.8, 193.4), (510, EXT, EXT)), B((0, -40.8, -191.6), (510, EXT, EXT)),
              B((255, -40.8, 0), (EXT, EXT, 315)), B((-255, -40.8, 0), (EXT, EXT, 315))],
    'claw': [B((0, 269.2, 275.9), (530, EXT, EXT)), B((95, 166.7, 275.9), (EXT, 185, EXT)),
             B((-95, 166.7, 275.9), (EXT, 185, EXT)), B((0, 64.2, 275.9), (490, EXT, EXT)),
             B((122.6, 64.2, 325.9), (245.1, 2.3, 80)), B((-122.6, 64.2, 325.9), (245.1, 2.3, 80))],
    'dvl': [C((-2.0, -13.3, 62.2), 50.0, 172.7, 'z')],
    # T200: 0.195 L solid each (the rest of the thruster floods)
    'thrusters': [S(c, T200_SOLID_RADIUS) for c in [(197.8, 218.4, 0.9), (-197.8, 218.4, 0.9),
                                                    (194.6, 155.2, 126.8), (-194.6, 155.2, 126.8),
                                                    (-194.6, 155.2, -125.0), (194.6, 155.2, -125.0)]],
}


# Effective densities (kg/m^3) = estimated mass in air / displaced volume.
# ASSUMPTIONS - replace with weighed values when available:
#   frame, claw   aluminium 2700
#   floaters      foam 200 kg/m^3 (8.1 L) + printed cases 3.8 L at 50 % infill PLA
#   control hull  ~6.5 kg: tube, end caps, NUC, Jetson, boards (14.3 L sealed)
#   battery hull  ~1.8 kg each: WTE3 tube, 6S 6 Ah LiPo, end caps
#   DVL           ~2.0 kg Teledyne Pathfinder
#   T200          0.344 kg each (Blue Robotics datasheet)
DENSITY = {'frame': 2700.0, 'claw': 2700.0, 'floaters': 330.0, 'control_hull': 455.0,
           'battery_hulls': 865.0, 'dvl': 1475.0, 'thrusters': 1755.0}
TARGET_NET_BUOYANCY_KG = 0.5 * SCALE ** 3   # slightly positive, as a real AUV is trimmed
WATER_DENSITY = 1000.0

# Poses taken from the CAD (mm) for the generated header
# Thruster axes are fitted from each T200's geometry (fit_thruster_axis):
# the CAD bounding box is misleading because the T200 model includes its mount.
# Detection camera: ZED Mini LEFT lens (front face; 63 mm baseline, CAD +X = left).
# The OAK-1 (0, 89.9, 231.2) is not used: the claw hides ~40 % of its view.
CAMERA_CAD = (39.2, 163.4, 241.85)
FRONT_CAD_Z = 365.9                    # recovery hooks: front-most point
DVL_CAD = (-2.0, -13.3, 62.2)


def fit_thruster_axis(v, t):
    """Axis of a surface of revolution: every surface-normal line meets the axis.
    Smallest eigenvector of the 6x6 line-complex fit (iteratively reweighted so
    the flat mounting faces count as outliers). Returns direction a, point c (mm)."""
    tri = v[t]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = np.linalg.norm(n, axis=1)
    ok = area > 1e-9
    n, p, area = n[ok] / area[ok, None], tri.mean(1)[ok], area[ok]
    lines = np.hstack([np.cross(p, n), n])
    w = area.copy()
    for _ in range(8):
        m = (lines * w[:, None]).T @ lines
        a_, b_, c_ = m[:3, :3], m[:3, 3:], m[3:, 3:]
        s_ = a_ - b_ @ np.linalg.solve(c_, b_.T)
        ev, evec = np.linalg.eigh(s_)
        a = evec[:, 0]
        b = -np.linalg.solve(c_, b_.T @ a)
        res = np.abs(lines @ np.concatenate([a, b]))
        w = area / (1.0 + (res / (np.median(res) + 1e-9)) ** 2)
    return a, np.cross(a, b)


def thruster_mounts(t200_shapes):
    """Body-frame mount (position on the axis, unit push direction) per T200.
    Corner thrusters push forward (+x), middle ones push down (+z)."""
    mounts = []
    for shape in t200_shapes:
        v, t = triangulate(shape)
        a, c = fit_thruster_axis(v, t)
        bb = bbox(shape)
        centre = (bb[:3] + bb[3:]) / 2
        on_axis = c + np.dot(centre - c, a) * a
        pos = cad_to_body(on_axis)
        d = np.array([a[2], -a[0], -a[1]])
        vertical = abs(d[2]) > abs(d[0])
        if (d[2] if vertical else d[0]) < 0:
            d = -d
        side = 'Left' if pos[1] < 0 else 'Right'
        where = 'Middle' if vertical else ('Front' if pos[0] > 0 else 'Rear')
        mounts.append((f'Thruster{where}{side}', pos, d, vertical))
    order = ['FrontLeft', 'FrontRight', 'RearLeft', 'RearRight', 'MiddleLeft', 'MiddleRight']
    return sorted(mounts, key=lambda m: order.index(m[0][len('Thruster'):]))


def mesh_volume_and_centroid(meshes):
    vol, mom = 0.0, np.zeros(3)
    for v, f in meshes:
        a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
        dv = np.einsum('ij,ij->i', a, np.cross(b, c)) / 6.0
        vol += dv.sum()
        mom += ((a + b + c) / 4.0 * dv[:, None]).sum(0)
    return vol, mom / vol


def main():
    step, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    leaves = read_leaves(step)
    print(f'{len(leaves)} parts read in {time.time() - t0:.0f} s')

    mounts = thruster_mounts([shape for names, _, shape in leaves if 'T200' in ' / '.join(names)])
    for name, pos, d, vertical in mounts:
        print(f'{name:22s} pos {np.round(pos, 4)}  push {np.round(d, 3)}')

    groups = {}
    skipped = 0
    for names, product, shape in leaves:
        bb = bbox(shape)
        if bb is None:
            continue
        g = group_of(names, product, bb)
        if g is None or np.linalg.norm(bb[3:] - bb[:3]) < MIN_VISIBLE_PART_MM:
            skipped += 1
            continue
        v, t = triangulate(shape)
        if len(t):
            groups.setdefault(g, []).append((cad_to_body(v), t))
    print(f'hidden inside hulls or transparent: {skipped} parts')

    for g, meshes in sorted(groups.items()):
        v = np.concatenate([m[0] for m in meshes])
        offs = np.cumsum([0] + [len(m[0]) for m in meshes[:-1]])
        t = np.concatenate([m[1] + o for m, o in zip(meshes, offs)])
        before = len(t)
        v, t = simplify(v, t, CLUSTER_GRID_M)
        write_obj(os.path.join(out, f'{g}_visual.obj'), [(v, t)])
        print(f'{g:14s} visual: {before:7d} -> {len(t):7d} triangles')
    total_v, total_m, mom_b, mom_g = 0.0, 0.0, np.zeros(3), np.zeros(3)
    write_obj(os.path.join(out, 'tiny_physics.obj'), [box_mesh(ORIGIN_MM, (10, 10, 10))])
    for g, prims in sorted(PHYSICS.items()):
        meshes = [primitive_mesh(p) for p in prims]
        write_obj(os.path.join(out, f'{g}_physics.obj'), meshes)
        vol, cen = mesh_volume_and_centroid(meshes)
        m = vol * DENSITY[g]
        total_v += vol
        total_m += m
        mom_b += cen * vol
        mom_g += cen * m
        print(f'{g:14s} physics {vol * 1000:6.2f} L  {m:6.2f} kg  centroid (body m) {np.round(cen, 3)}')
    cb = mom_b / total_v
    displaced = total_v * WATER_DENSITY
    ballast = displaced - TARGET_NET_BUOYANCY_KG - total_m
    # put the ballast under the centre of buoyancy (level trim), on the bottom plate
    cg_no_ballast = mom_g / total_m
    bottom_z = cad_to_body((0.0, -2.0, 0.0))[2]
    target_xy = (cb * (total_m + ballast) - cg_no_ballast * total_m)[:2] / ballast
    ballast_pos = np.array([target_xy[0], target_xy[1], bottom_z])
    cg = (mom_g + ballast_pos * ballast) / (total_m + ballast)
    print(f'displaced {displaced:.2f} kg, structure {total_m:.2f} kg, trim ballast {ballast:.2f} kg '
          f'at {np.round(ballast_pos, 3)}')
    print(f'centre of buoyancy {np.round(cb, 3)}, centre of gravity {np.round(cg, 3)}, '
          f'righting lever {cg[2] - cb[2]:.3f} m')
    write_header(os.path.join(os.path.dirname(os.path.abspath(out)), '..', 'src', 'kyubic_model.hpp'),
                 ballast, ballast_pos, displaced, total_m + ballast, cb, cg, mounts)
    print(f'done in {time.time() - t0:.0f} s')


def write_header(path, ballast, ballast_pos, displaced, mass, cb, cg, mounts):
    def vec(p):
        return f'{p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}'
    lines = [
        '#pragma once',
        '// KYUBIC V2 model data - GENERATED by tools/kyubic_step_export.py from',
        '// "KYUBIC V2 2025 2026Iwakuni.step". Do not edit by hand; re-run the tool.',
        '// Body frame: x forward, y right, z down [m]; origin = control-hull axis.',
        '',
        'namespace kyubic {',
        '',
        f'// Geometric scale of the whole robot (1 = the real KYUBIC). Lengths x SCALE,',
        f'// volumes and masses x SCALE^3.',
        f'constexpr double SCALE = {SCALE:.4f};',
        '',
        'struct Group { const char* name; double density; };',
        '// Effective densities [kg/m^3]: estimated mass in air / displaced volume.',
        'constexpr Group GROUPS[] = {',
    ]
    for g in ['control_hull', 'frame', 'floaters', 'battery_hulls', 'dvl', 'thrusters', 'claw']:
        lines.append(f'    {{"{g}", {DENSITY[g]:.1f}}},')
    lines += ['};', '',
              f'// Trim ballast so the robot floats level with {TARGET_NET_BUOYANCY_KG:.2f} kg net buoyancy.',
              f'constexpr double BALLAST_KG = {ballast:.3f};',
              f'constexpr double BALLAST_POS[3] = {{{vec(ballast_pos)}}};',
              f'constexpr double DISPLACED_KG = {displaced:.3f};   // fresh water',
              f'constexpr double MASS_KG = {mass:.3f};',
              f'constexpr double CENTRE_OF_BUOYANCY[3] = {{{vec(cb)}}};',
              f'constexpr double CENTRE_OF_GRAVITY[3] = {{{vec(cg)}}};',
              '',
              'struct ThrusterMount { const char* name; double pos[3]; double dir[3]; bool vertical; };',
              '// pos = on the thruster axis; dir = unit push direction for a positive command.',
              '// Four corner thrusters vectored 30 deg in the horizontal plane (surge, sway, yaw);',
              '// two middle thrusters tilted 30 deg inward from vertical (heave, roll).',
              'constexpr ThrusterMount THRUSTERS[] = {']
    for name, pos, d, vertical in mounts:
        lines.append(f'    {{"{name}", {{{vec(pos)}}}, {{{vec(d)}}}, {"true" if vertical else "false"}}},')
    lines += ['};',
              'constexpr unsigned THRUSTER_COUNT = sizeof(THRUSTERS) / sizeof(THRUSTERS[0]);', '',
              '// Physics solids (buoyancy, drag, added mass, collisions), one Stonefish solid each.',
              '// type: b = box (size = x, y, z), c = cylinder (size = radius, length; axis),',
              '// s = sphere (size = radius). Each part uses its group density.',
              'struct Primitive { const char* group; char type; double center[3]; double size[3]; char axis; };',
              'constexpr Primitive PRIMITIVES[] = {']
    body_axis = {'z': 'x', 'x': 'y', 'y': 'z'}
    for g, prims in PHYSICS.items():
        for p in prims:
            c = cad_to_body(p[1])
            if p[0] == 'box':
                sx, sy, sz = p[2]
                size, typ, ax = (sz * M_PER_MM, sx * M_PER_MM, sy * M_PER_MM), 'b', 'x'
            elif p[0] == 'cyl':
                size, typ, ax = (p[2] * M_PER_MM, p[3] * M_PER_MM, 0.0), 'c', body_axis[p[4]]
            else:
                size, typ, ax = (p[2] * M_PER_MM, 0.0, 0.0), 's', 'x'
            lines.append(f'    {{"{g}", \'{typ}\', {{{vec(c)}}}, {{{size[0]:.5f}, {size[1]:.5f}, {size[2]:.5f}}}, \'{ax}\'}},')
    lines += ['};',
              'constexpr unsigned PRIMITIVE_COUNT = sizeof(PRIMITIVES) / sizeof(PRIMITIVES[0]);', '',
              f'constexpr double CAMERA_POS[3] = {{{vec(cad_to_body(CAMERA_CAD))}}};   // ZED Mini left lens, looking forward',
              f'constexpr double FRONT_X = {(FRONT_CAD_Z - ORIGIN_MM[2]) * M_PER_MM:.4f};   // front-most point (recovery hooks)',
              f'constexpr double DVL_POS[3] = {{{vec(cad_to_body(DVL_CAD))}}};',
              '',
              '}  // namespace kyubic', '']
    with open(os.path.normpath(path), 'w') as f:
        f.write('\n'.join(lines))
    print('wrote', os.path.normpath(path))


if __name__ == '__main__':
    main()
