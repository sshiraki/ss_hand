#!/usr/bin/env python3
"""Carpus / wrist base v10.1.

Changes vs v4.1:
- Finger roots at pitch 11.5mm with a human-like fan (each finger yawed
  about its root: index +4.5, middle +1.5, ring -1.5, pinky -4.5 deg),
  so the MCP joints sit ~14.6mm apart and clear each other, while the
  roots stay compact. Fork slots and pin holes are yawed to match.
- A palm plate under the carpal cluster ties everything together.
- The thumb saddle cup is replaced by a CMC PEDESTAL: a fork at point C
  whose pin (dia 2.2) runs along CMC axis 1, holding the saddle block
  of the 2-axis thumb CMC (see cmc_params.py). The pedestal arm reaches
  back into the cluster along -u0.
Kept from v2/v4.1: cluster ellipsoids, fork slot depth (y 4..20), wrist
centre hole dia 6, M3 holes, flat wrist facet.
"""
import os, math
import numpy as np
import trimesh
import cmc_params as P

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "stl")
os.makedirs(OUT, exist_ok=True)
PIN_R = 1.10
CLEAR = 0.35

def ell(cx, cy, cz, rx, ry, rz):
    m = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    m.apply_scale([rx, ry, rz]); m.apply_translation([cx, cy, cz]); return m

def ell_b(center, radii, basis):
    m = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    m.apply_scale(list(radii))
    T = np.eye(4); T[:3, :3] = basis
    m.apply_transform(T); m.apply_translation(list(center)); return m

def sph(cx, cy, cz, r):
    m = trimesh.creation.icosphere(subdivisions=3, radius=r)
    m.apply_translation([cx, cy, cz]); return m

def box(cx, cy, cz, sx, sy, sz):
    m = trimesh.creation.box(extents=[sx, sy, sz]); m.apply_translation([cx, cy, cz]); return m

def cyl_x(cx, cy, cz, r, length, sections=48):
    m = trimesh.creation.cylinder(radius=r, height=length, sections=sections)
    m.apply_transform(trimesh.transformations.rotation_matrix(math.radians(90), [0, 1, 0]))
    m.apply_translation([cx, cy, cz]); return m

def cyl_y(cx, cy, cz, r, length, sections=32):
    m = trimesh.creation.cylinder(radius=r, height=length, sections=sections)
    m.apply_transform(trimesh.transformations.rotation_matrix(math.radians(90), [1, 0, 0]))
    m.apply_translation([cx, cy, cz]); return m

def cyl_along(center, direction, r, length, sections=32):
    d = np.asarray(direction, float); d = d / np.linalg.norm(d)
    m = trimesh.creation.cylinder(radius=r, height=length, sections=sections)
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(z, d)
    if np.linalg.norm(axis) > 1e-9:
        ang = np.arccos(np.clip(np.dot(z, d), -1, 1))
        m.apply_transform(trimesh.transformations.rotation_matrix(ang, axis / np.linalg.norm(axis)))
    elif np.dot(z, d) < 0:
        m.apply_transform(trimesh.transformations.rotation_matrix(math.pi, [1, 0, 0]))
    m.apply_translation(list(center)); return m

def union(meshes): return trimesh.boolean.union(meshes, engine="manifold")
def diff(mesh, cutters): return trimesh.boolean.difference([mesh] + list(cutters), engine="manifold")
def inter(meshes): return trimesh.boolean.intersection(meshes, engine="manifold")

def rounded_plate(cx=1.5, cy=5.0, sx=49.0, sy=28.0, sz=9.0):
    """Palm plate with every corner/edge rounded (v10.1, user request:
    the old plain box stuck out past the carpal cluster on the pinky
    side with sharp corners). Built as the intersection of three
    rounded-rectangle extrusions: plan corners R=9, thickness edges
    R=4 in both directions. Same outer size as the old box."""
    R, RB = 9.0, 4.0
    # plan extrusion: boxes + 4 vertical (Z) corner cylinders
    def cyl_z(x, y, r, h):
        m = trimesh.creation.cylinder(radius=r, height=h, sections=40)
        m.apply_translation([x, y, 0]); return m
    def cyl_y2(x, z, r, h):
        m = trimesh.creation.cylinder(radius=r, height=h, sections=32)
        m.apply_transform(trimesh.transformations.rotation_matrix(math.radians(90), [1, 0, 0]))
        m.apply_translation([x, cy, z]); return m
    def cyl_x2(y, z, r, h):
        m = trimesh.creation.cylinder(radius=r, height=h, sections=32)
        m.apply_transform(trimesh.transformations.rotation_matrix(math.radians(90), [0, 1, 0]))
        m.apply_translation([cx, y, z]); return m
    A = union([box(cx, cy, 0, sx - 2 * R, sy, 14), box(cx, cy, 0, sx, sy - 2 * R, 14)] +
              [cyl_z(cx + a * (sx / 2 - R), cy + b * (sy / 2 - R), R, 14)
               for a in (-1, 1) for b in (-1, 1)])
    B = union([box(cx, cy, 0, sx - 2 * RB, 36, sz), box(cx, cy, 0, sx, 36, sz - 2 * RB)] +
              [cyl_y2(cx + a * (sx / 2 - RB), b * (sz / 2 - RB), RB, 36)
               for a in (-1, 1) for b in (-1, 1)])
    C = union([box(cx, cy, 0, 56, sy - 2 * RB, sz), box(cx, cy, 0, 56, sy, sz - 2 * RB)] +
              [cyl_x2(cy + a * (sy / 2 - RB), b * (sz / 2 - RB), RB, 56)
               for a in (-1, 1) for b in (-1, 1)])
    return inter([inter([A, B]), C])

def yaw(mesh, deg, point):
    mesh.apply_transform(trimesh.transformations.rotation_matrix(
        math.radians(deg), [0, 0, 1], point=list(point)))
    return mesh

META = {"index": (62, 11.0, 9.5), "middle": (64, 11.5, 10.0), "ring": (60, 11.0, 9.5),
        "pinky": (54, 10.0, 8.8), "thumb": (42, 13.0, 11.0)}
XPOS = {"index": -17.25, "middle": -5.75, "ring": 5.75, "pinky": 17.25}
FAN = {"index": 4.5, "middle": 1.5, "ring": -1.5, "pinky": -4.5}

parts = [
    ell(-17, -3.0, 0,    10.0, 7.5, 7.0),   # scaphoid
    ell(-6,  -4.5, 0.5,   8.5, 7.0, 6.5),   # lunate
    ell(6,   -4.5, 0,     8.0, 6.5, 6.5),   # triquetrum
    ell(13,  -3.0, -4.5,  4.5, 4.5, 4.5),   # pisiform (palmar)
    ell(-19,  4.0, -0.5,  7.0, 6.0, 6.0),   # trapezium
    ell(-11,  5.5, 0,     6.0, 5.5, 5.5),   # trapezoid
    ell(0,    6.0, 0,     8.5, 7.0, 7.0),   # capitate
    ell(12,   5.5, -0.5,  9.0, 6.5, 6.5),   # hamate
    ell(16,   4.0, -6.0,  3.5, 3.0, 3.0),   # hamulus (palmar hook)
    rounded_plate(),                        # palm plate v10.1: fully rounded (was a sharp-cornered box)
]
cutters = []
for f, x in XPOS.items():
    w, t = META[f][1], META[f][2]
    slot_w = (w * 0.28) * 2 + CLEAR * 2 + 1.2
    cheek_cx = slot_w / 2 + 1.1
    for s in (-1, 1):
        parts.append(yaw(ell(x + s * cheek_cx, 10.0, 0, 1.6, 5.8, t * 0.52), FAN[f], (x, 10, 0)))
    cutters.append(yaw(box(x, 12.0, 0, slot_w, 16.0, 30.0), FAN[f], (x, 10, 0)))
    cutters.append(yaw(cyl_x(x, 10.0, 0, PIN_R, w * 2.4), FAN[f], (x, 10, 0)))

# ---- thumb CMC pedestal ----
R0, A1, A2, u0 = P.frames(P.RHO, P.KAPPA)
e3 = np.cross(A1, A2)
basis_ped = np.column_stack([A1, A2, e3])
# ---- v11 ball-joint socket (user direction, 2026-11-08) ----
# The 2-pin universal joint (fork cheeks + loose cross block) is
# replaced by a ball joint at the same centre C: the metacarpal ends
# in a dia-13 ball, the base carries a snap-fit socket.
# - cup: sphere r=8.75 at C (merges with the carpal cluster / palm
#   plate edge by ~330mm^3 -- measured, no struts needed, and it
#   clears the index/middle metacarpals entirely)
# - cavity r=6.75 (ball r=6.5 + 0.25 clearance), mouth cone half
#   angle 70deg from the +u0 axis: rim circle dia = 2*6.75*sin70 =
#   12.68 < ball dia 13 -> the ball snaps in and is retained;
#   6 axial slits (azimuth offset 30deg so none sits on the palmar
#   tendon line) let the rim flex for the snap.
# Socket cuts are applied after the union, below.
ped_parts = [sph(P.C[0], P.C[1], P.C[2], 8.75)]

cutters.append(cyl_y(0, -10, 0, 3.0, 10.0))                       # wrist centre hole dia 6
for sx in (-14.0, 14.0):                                          # M3 holes
    mc = trimesh.creation.cylinder(radius=1.6, height=34.0, sections=20)
    mc.apply_translation([sx, 0, 0])
    cutters.append(mc)
cutters.append(box(0, -16.5, 0, 90.0, 9.0, 44.0))                 # flat wrist facet

mesh = union([diff(union(parts), cutters)] + ped_parts)

# ---- flexor tendon tunnels (v10.1, user request) ----
# Each finger's tendon runs on the line z = -r_guide along its (fanned)
# bone axis -- the same line it takes along the metacarpal to the
# mid-shaft guide hole. Proximal of the fork slot that line passes
# through the carpal cluster, so a dia-2.0 tunnel is cut along it
# (guide holes are dia 1.8; the tunnel gets +0.2 for threading through
# up to ~14mm of material). Index/pinky axes are shifted 0.3mm outward
# so the tunnel stays clear of the M3 mounting holes at x=+/-14
# (fan-aligned axes would kiss them by 0.07mm^3).
TENDON_TUNNEL_R = 1.0
R_GUIDE = {"index": 5.75, "middle": 6.0, "ring": 5.75, "pinky": 5.3}
X_OFF = {"index": -0.3, "pinky": 0.3}
tendon_cutters = []
for f, x in XPOS.items():
    d = trimesh.transformations.rotation_matrix(math.radians(FAN[f]), [0, 0, 1])[:3, :3] @ np.array([0.0, 1.0, 0.0])
    root = np.array([x + X_OFF.get(f, 0.0), 10.0, -R_GUIDE[f]])
    tendon_cutters.append(cyl_along(root - 9.0 * d, d, TENDON_TUNNEL_R, 30.0))
# Thumb: entry tunnel along the metacarpal axis at rest (u0), on the
# frame-M line z=-8.3 (v11: pushed out from -6.5 so the tunnel
# clears the CMC ball, r=6.5 at C, by 0.8mm; it grooves the socket
# cup's outer wall and joins the metacarpal guide line in open air).
# Fixed in the base
# like the carpal tunnel in a real hand; when the CMC rotates the
# tendon slides and its angle changes, as in the real hand.
tendon_cutters.append(cyl_along(P.C + R0 @ np.array([0.0, -6.0, -8.3]), u0,
                                TENDON_TUNNEL_R, 28.0))

# ---- v11 socket cuts: cavity + mouth cone + flex slits ----
mesh = diff(mesh, [sph(P.C[0], P.C[1], P.C[2], 6.75)])          # cavity
_cone = trimesh.creation.cone(radius=11.0 * math.tan(math.radians(70.0)),
                              height=11.0, sections=48)
# trimesh cone: base (full radius) at z=0, apex at z=+h. Map local
# +Z -> -u0 and put the cone origin 11mm along +u0 from C, so the
# apex lands on C and the cone opens along +u0 (mouth, half-angle 70).
ZM = np.cross(A2, u0)  # frame-M +Z in base frame (unit, perp to u0 and A2)
_basis_cone = np.column_stack([A2, ZM, -u0])
_T = np.eye(4); _T[:3, :3] = _basis_cone
_cone.apply_transform(_T)
_cone.apply_translation(list(P.C + 11.0 * u0))
mesh = diff(mesh, [_cone])                                       # mouth
for k in range(6):  # flex slits, azimuth 30+60k deg around u0
    phi = math.radians(30.0 + 60.0 * k)
    d_r = math.cos(phi) * A2 + math.sin(phi) * ZM                # radial dir
    d_t = -math.sin(phi) * A2 + math.cos(phi) * ZM               # tangential dir
    b = trimesh.creation.box(extents=[1.5, 4.2, 5.2])
    b.apply_translation([0.0, 7.3, -0.2])
    Tb = np.eye(4); Tb[:3, :3] = np.column_stack([d_t, d_r, u0])
    b.apply_transform(Tb); b.apply_translation(list(P.C))
    mesh = diff(mesh, [b])                                       # sequential
mesh = diff(mesh, tendon_cutters).process()
try:
    trimesh.repair.fill_holes(mesh)
    mesh = mesh.process()
except Exception:
    pass
if not mesh.is_watertight:
    # STL round-trip + merge/fill (the recipe that closed v4.x cracks)
    try:
        import io
        buf = io.BytesIO(); mesh.export(buf, file_type="stl"); buf.seek(0)
        m2 = trimesh.load(buf, file_type="stl")
        m2.merge_vertices()
        trimesh.repair.fill_holes(m2)
        m2 = m2.process()
        if m2.is_watertight:
            mesh = m2
    except Exception:
        pass
mesh.export(os.path.join(OUT, "carpus_wrist_base.stl"))
comps = [c for c in mesh.split(only_watertight=False) if abs(float(c.volume)) > 1.0]
print("carpus_wrist_base.stl watertight:", bool(mesh.is_watertight),
      "| components:", len(comps),
      "| size:", [round(float(s), 1) for s in (mesh.bounds[1] - mesh.bounds[0])],
      "| vol cm3:", round(float(mesh.volume) / 1000.0, 2))
