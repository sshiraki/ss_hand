#!/usr/bin/env python3
"""Ball-joint CMC figure for v11:
1. Section through C along the (u0, A2) plane: ball in the socket,
   rim narrower than the ball (snap retention), flex slits.
2. Rest pose, whole CMC area.
3. Tilted 40deg toward the pinch direction (bone axis tilt), showing
   the range the socket allows in every direction (~45deg).
Writes ball_joint_v11.png."""
import os, math, re
import numpy as np
import trimesh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import cmc_params as P

plt.rcParams["font.family"] = ["Noto Sans CJK JP", "Noto Serif CJK JP", "DejaVu Sans"]
HERE = os.path.dirname(__file__)
STL = os.path.join(HERE, "stl")
src = open(os.path.join(HERE, "verify_joints_v9.py")).read()
PIVOT = eval(re.search(r"PIVOT = (\{[^}]+\})", src).group(1))
PIVOT["thumb_metacarpal"] = P.META_PIVOT
R0, A1, A2, u0 = P.frames(P.RHO, P.KAPPA)
ZM = np.cross(A2, u0)
BASE = (0.78, 0.74, 0.68); BONE = (0.92, 0.85, 0.74); BALLC = (0.95, 0.45, 0.25)

def load(n): return trimesh.load(os.path.join(STL, n + ".stl"))

def T_of(R, t):
    M = np.eye(4); M[:3, :3] = R; M[:3, 3] = t
    return M

def draw(ax, mesh, color):
    tris = mesh.vertices[mesh.faces]
    shade = 0.55 + 0.45 * np.abs(mesh.face_normals[:, 2])
    cols = np.stack([shade * color[0], shade * color[1], shade * color[2]], axis=1)
    ax.add_collection3d(Poly3DCollection(tris, facecolors=cols, edgecolors="none"))

def build_chain(angles):
    names = ["thumb_metacarpal", "thumb_proximal", "thumb_distal"]
    meshes, O = [], np.zeros(3)
    Rcum = np.eye(4)
    for i, n in enumerate(names):
        m = load(n)
        m.apply_transform(Rcum); m.apply_translation(O)
        meshes.append(m)
        if i < len(angles):
            O = O + Rcum[:3, :3] @ np.array([0.0, PIVOT[n], 0.0])
            Rcum = Rcum @ trimesh.transformations.rotation_matrix(math.radians(-angles[i]), [1, 0, 0])
    return meshes

def panel(ax, center, r, elev, azim, title):
    c = np.array(center, dtype=float)
    ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r); ax.set_zlim(c[2] - r, c[2] + r)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off(); ax.set_title(title, fontsize=10)

def cut_half(mesh, normal, point):
    """Remove the half of mesh on the +normal side of the plane."""
    box = trimesh.creation.box(extents=[90, 90, 46])
    n = np.asarray(normal, float); n = n / np.linalg.norm(n)
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(z, n)
    if np.linalg.norm(axis) > 1e-9:
        ang = np.arccos(np.clip(np.dot(z, n), -1, 1))
        box.apply_transform(trimesh.transformations.rotation_matrix(ang, axis / np.linalg.norm(axis)))
    elif np.dot(z, n) < 0:
        box.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))
    box.apply_translation(list(np.asarray(point, float) + 23 * n))
    return trimesh.boolean.difference([mesh, box], engine="manifold")

fig = plt.figure(figsize=(15.5, 5.4))

# 1. section: cut away the +ZM half; look from +ZM
ax = fig.add_subplot(1, 3, 1, projection="3d")
draw(ax, cut_half(load("carpus_wrist_base"), ZM, P.C), BASE)
meta = load("thumb_metacarpal"); meta.apply_transform(T_of(R0, P.C))
draw(ax, cut_half(meta, ZM, P.C), BALLC)
panel(ax, (-29.5, 9.0, -4.5), 17, 37, 24,
      "断面: ボール Ø13 / 受け口 Ø12.7\n押し込むとスリットが開いてはまる")

# 2. rest
ax = fig.add_subplot(1, 3, 2, projection="3d")
draw(ax, load("carpus_wrist_base"), BASE)
for m in build_chain([0, 0]):
    m.apply_transform(T_of(R0, P.C)); draw(ax, m, BONE)
panel(ax, (-27.0, 14.0, -4.0), 27, 55, -100, "静止姿勢: ピンもブロックも無し")

# 3. tilt 40deg about the axis that takes u0 toward the index finger
ax = fig.add_subplot(1, 3, 3, projection="3d")
draw(ax, load("carpus_wrist_base"), BASE)
axis = np.cross(u0, np.array([-17.25, 10.0, 0.0]) - P.C)
axis = axis / np.linalg.norm(axis)
Rt = trimesh.transformations.rotation_matrix(math.radians(40), axis, point=P.C)[:3, :3]
chain = build_chain([0, 0])
for m in chain:
    m.apply_transform(T_of(Rt @ R0, P.C))
    draw(ax, m, BONE)
panel(ax, (-26.0, 16.0, -4.0), 30, 55, -100, "40°傾けた姿勢: 全方向に約45°まで可動")

fig.tight_layout()
fig.savefig(os.path.join(HERE, "ball_joint_v11.png"), dpi=150, bbox_inches="tight")
print("saved ball_joint_v11.png")
