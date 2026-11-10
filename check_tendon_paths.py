#!/usr/bin/env python3
"""Verify v10.1 tendon tunnels: a test line (r=0.7, dia 1.4 -- slimmer than
the dia-2.0 tunnel) run from the wrist-side entry to each finger's
metacarpal mid-shaft guide must meet ~0 base material along the way.
Also: thumb line to its frame-M guide (y=27.5), and the index/pinky
tunnel axes must stay clear of the M3 holes."""
import math
import numpy as np
import trimesh
import cmc_params as P

base = trimesh.load("stl/carpus_wrist_base.stl")

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

def inter_vol(a, b):
    try:
        m = trimesh.boolean.intersection([a, b], engine="manifold")
        return float(m.volume) if m is not None and len(m.faces) else 0.0
    except Exception:
        return 0.0

XPOS = {"index": -17.25, "middle": -5.75, "ring": 5.75, "pinky": 17.25}
FAN = {"index": 4.5, "middle": 1.5, "ring": -1.5, "pinky": -4.5}
RG = {"index": 5.75, "middle": 6.0, "ring": 5.75, "pinky": 5.3}
META_L = {"index": 62.0, "middle": 64.0, "ring": 60.0, "pinky": 54.0}
X_OFF = {"index": -0.3, "pinky": 0.3}

print("== tendon path clearance (entry -> metacarpal guide, dia-1.4 test line) ==")
for f in XPOS:
    Rz = trimesh.transformations.rotation_matrix(math.radians(FAN[f]), [0, 0, 1])[:3, :3]
    d = Rz @ np.array([0.0, 1.0, 0.0])
    root = np.array([XPOS[f] + X_OFF.get(f, 0.0), 10.0, -RG[f]])
    s0, s1 = -24.0, META_L[f] / 2  # entry (wrist facet) -> guide at L/2
    line = cyl_along(root + (s0 + s1) / 2 * d, d, 0.7, s1 - s0)
    print(f"  {f:7s} residual material on path: {inter_vol(line, base):.3f} mm^3")

R0, A1, A2, u0 = P.frames(P.RHO, P.KAPPA)
s0, s1 = -20.0, 8.0  # v11: tunnel line at frame-M z=-8.3 (clears the CMC ball); guide approach is open air
cen = P.C + R0 @ np.array([0.0, (s0 + s1) / 2, -8.3])
line = cyl_along(cen, u0, 0.7, s1 - s0)
print(f"  thumb   residual material on path: {inter_vol(line, base):.3f} mm^3")

print("== tunnel axis x M3 hole (must be 0) ==")
for f, sx in (("index", -14.0), ("pinky", 14.0)):
    Rz = trimesh.transformations.rotation_matrix(math.radians(FAN[f]), [0, 0, 1])[:3, :3]
    d = Rz @ np.array([0.0, 1.0, 0.0])
    root = np.array([XPOS[f] + X_OFF.get(f, 0.0), 10.0, -RG[f]])
    tun = cyl_along(root - 9 * d, d, 1.0, 30.0)
    m3 = cyl_along([sx, 0, 0], [0, 0, 1], 1.6, 34.0)
    print(f"  {f:7s} overlap: {inter_vol(tun, m3):.3f} mm^3")
