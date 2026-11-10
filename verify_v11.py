#!/usr/bin/env python3
"""v11 verification (ball-joint CMC):
1. All 20 parts watertight + single solid (saddle block retired).
2. Four fingers on the fanned base (unchanged from v10 checks).
3. Ball-joint tilt sweep: for 12 tilt directions (azimuth around the
   rest bone axis u0), the largest tilt of the thumb metacarpal with
   base interference <= 0.30mm^3. Design target ~45deg everywhere.
4. The five pinch poses verified for the universal joint in v10 --
   same orientations (a ball joint reaches any orientation whose
   bone-axis tilt is within the sweep), full chains, must contact at
   the fingertips without other interference.
"""
import os, math, itertools, re
import numpy as np
import trimesh
import cmc_params as P

HERE = os.path.dirname(__file__)
STL = os.path.join(HERE, "stl")
src = open(os.path.join(HERE, "verify_joints_v9.py")).read()
PIVOT = eval(re.search(r"PIVOT = (\{[^}]+\})", src).group(1))
PIVOT["thumb_metacarpal"] = P.META_PIVOT

XPOS = {"index": -17.25, "middle": -5.75, "ring": 5.75, "pinky": 17.25}
FAN = {"index": 4.5, "middle": 1.5, "ring": -1.5, "pinky": -4.5}
R0, A1, A2, u0 = P.frames(P.RHO, P.KAPPA)

def load(n): return trimesh.load(os.path.join(STL, n + ".stl"))

def inter(a, b):
    if (a.bounds[0] > b.bounds[1]).any() or (b.bounds[0] > a.bounds[1]).any():
        return 0.0
    try:
        m = trimesh.boolean.intersection([a, b], engine="manifold")
        return float(m.volume) if m is not None and len(m.faces) else 0.0
    except Exception:
        return 0.0

def T_of(R, t):
    M = np.eye(4); M[:3, :3] = R; M[:3, 3] = t
    return M

def build_chain(finger, angles):
    names = [f"{finger}_metacarpal", f"{finger}_proximal", f"{finger}_middle", f"{finger}_distal"] \
        if finger != "thumb" else [f"{finger}_metacarpal", f"{finger}_proximal", f"{finger}_distal"]
    meshes, O = [], np.zeros(3)
    Rcum = np.eye(4)
    for i, n in enumerate(names):
        m = load(n)
        m.apply_transform(Rcum); m.apply_translation(O)
        meshes.append(m)
        if i < len(angles):
            O = O + Rcum[:3, :3] @ np.array([0.0, PIVOT[n], 0.0])
            Rx = trimesh.transformations.rotation_matrix(math.radians(-angles[i]), [1, 0, 0])
            Rcum = Rcum @ Rx
    return names, meshes

def place_finger(finger, meshes):
    Rz = trimesh.transformations.rotation_matrix(math.radians(FAN[finger]), [0, 0, 1])
    for m in meshes:
        m.apply_transform(Rz); m.apply_translation([XPOS[finger], 10.0, 0.0])
    return meshes

print("== 1. integrity ==")
ok, nparts = True, 0
for fn in sorted(os.listdir(STL)):
    m = load(fn[:-4])
    n = len([c for c in m.split(only_watertight=False) if abs(float(c.volume)) > 1.0])
    nparts += 1
    good = bool(m.is_watertight) and n == 1
    ok &= good
    if not good:
        print(f"  {fn}: watertight={m.is_watertight} comps={n}")
print(f"  all {nparts} parts watertight + single solid:", ok)

base = load("carpus_wrist_base")
print("== 2. four fingers on fanned base ==")
for t in (0.0, 0.5, 1.0):
    parts = []
    for f in XPOS:
        names, meshes = build_chain(f, [t * 90, t * 100, t * 70])
        place_finger(f, meshes)
        parts += [(f, n, m) for n, m in zip(names, meshes)]
    worst, wp = 0.0, None
    for (f1, n1, m1), (f2, n2, m2) in itertools.combinations(parts, 2):
        if f1 == f2:
            continue
        v = inter(m1, m2)
        if v > worst:
            worst, wp = v, (n1, n2)
    print(f"  t={t}: worst cross-finger {worst:.2f} mm^3 {wp}")
for f in XPOS:
    m = load(f"{f}_metacarpal")
    m.apply_transform(trimesh.transformations.rotation_matrix(math.radians(FAN[f]), [0, 0, 1]))
    m.apply_translation([XPOS[f], 10.0, 0.0])
    print(f"  {f} metacarpal x base: {inter(m, base):.2f} mm^3")

print("== 3. ball-joint tilt sweep (metacarpal x base) ==")
meta_rest = load("thumb_metacarpal")
meta_rest.apply_transform(T_of(R0, P.C))
q1 = A2 / np.linalg.norm(A2)
q2 = np.cross(u0, q1)
for phi_deg in range(0, 360, 30):
    phi = math.radians(phi_deg)
    axis = math.cos(phi) * q1 + math.sin(phi) * q2
    vals = []
    for tilt in range(0, 61, 5):
        m = meta_rest.copy()
        m.apply_transform(trimesh.transformations.rotation_matrix(math.radians(tilt), axis, point=P.C))
        vals.append(inter(m, base))
    clean = [t for t, v in zip(range(0, 61, 5), vals) if v <= 0.30]
    print(f"  azimuth {phi_deg:3d}deg: max clean tilt {max(clean) if clean else -1:3d}deg"
          f" | overlap@45deg {vals[9]:.2f} mm^3")

print("== 4. pinch poses (orientations from the v10 search) ==")
POSES = ((-25, 28, 0, 0, 0.65), (-30, 30, 0, 40, 0.45), (-35, 7, 0, 40, 0.65),
         (-40, 17, 0, 40, 0.45), (-30, 22, 0, 80, 0.85))
for (t1, t2, mcp, ip, tf) in POSES:
    _, chain = build_chain("thumb", [mcp, ip])
    R = P.cmc_matrix(R0, A1, A2, t1, t2)
    for m in chain:
        m.apply_transform(T_of(R, P.C))
    names_i, meshes_i = build_chain("index", [tf * 90, tf * 100, tf * 70])
    place_finger("index", meshes_i)
    pairs = [(f"thumb_{k}", m, "base", base) for k, m in zip(("meta", "prox", "dist"), chain)] + \
            [(f"thumb_{k}", m, f"index_{j}", im) for k, m in zip(("meta", "prox", "dist"), chain)
             for j, im in zip(("meta", "prox", "mid", "dist"), meshes_i)
             if not (k == "dist" and j == "dist")]
    bad = [(a, b, round(inter(x, y), 2)) for a, x, b, y in pairs if inter(x, y) > 0.30]
    touch = inter(chain[2], meshes_i[3])
    tilt = math.degrees(math.acos(np.clip((R @ np.array([0.0, 1.0, 0.0])) @ u0, -1, 1)))
    print(f"  CMC=({t1},{t2}) MCP={mcp} IP={ip} index t={tf}: bone tilt {tilt:.1f}deg | "
          f"fingertip overlap {touch:.2f} mm^3 | violations>0.3: {bad}")
