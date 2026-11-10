#!/usr/bin/env python3
"""v11 thumb metacarpal: ball-joint end (user direction, 2026-11-08).

Frame M as before (origin at CMC centre C, bone along +Y). The v9
metacarpal is reused shifted +6.5mm (MCP pivot stays at y=48.5, so
the v9 proximal/distal fit unchanged), but its wide basal knob is
cut away at y=12 and replaced by:
- ball: sphere r=6.5 (dia 13) at the origin -- rotates in the base
  socket (cavity r=6.75, rim dia 12.68, snap fit)
- neck waist: cylinder r=2.8, y 2..8.6 -- this radius sets the tilt
  range: theta_max ~= 70deg - asin(2.8/6.75) ~= 45deg in every
  direction, which covers all five verified pinch poses (max
  resultant tilt ~43.5deg)
- flare: cone frustum r2.8 @ y=8.6 -> r5.4 @ y=13.5, plus a cap
  cylinder, blending the neck into the shaft. All flare points sit
  >= 9.0mm from C, outside the socket cup (r=8.75).
No pins, no fork, no relief carving: a sphere centred at C clears
itself in every direction.
"""
import os
import numpy as np
import trimesh

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "stl")
os.makedirs(OUT, exist_ok=True)
V9 = os.path.join(HERE, "..", "hand3d_v9", "stl")
import cmc_params as P

def union(meshes): return trimesh.boolean.union(meshes, engine="manifold")
def diff(mesh, cutters): return trimesh.boolean.difference([mesh] + list(cutters), engine="manifold")

def finish(mesh, name):
    mesh = mesh.process()
    try:
        trimesh.repair.fill_holes(mesh)
        mesh = mesh.process()
    except Exception:
        pass
    if not mesh.is_watertight:
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
    comps = [c for c in mesh.split(only_watertight=False) if abs(float(c.volume)) > 1.0]
    mesh.export(os.path.join(OUT, name + ".stl"))
    print(f"{name}: watertight={bool(mesh.is_watertight)} components={len(comps)} "
          f"vol={float(mesh.volume)/1000:.2f}cm3 size={[round(float(s),1) for s in (mesh.bounds[1]-mesh.bounds[0])]}")

def cyl_y(r, y0, y1, sections=40):
    m = trimesh.creation.cylinder(radius=r, height=y1 - y0, sections=sections)
    m.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    m.apply_translation([0, (y0 + y1) / 2, 0])
    return m

# shaft: v9 bone, same placement as v10 (MCP pivot at y=48.5)
bone = trimesh.load(os.path.join(V9, "thumb_metacarpal.stl"))
bone.apply_translation([0, P.META_SHIFT, 0])
cut = trimesh.creation.box(extents=[40, 30, 40]); cut.apply_translation([0, 12.0 - 15.0, 0])
bone = diff(bone, [cut])                       # remove basal knob (y < 12)

ball = trimesh.creation.icosphere(subdivisions=4, radius=6.5)
waist = cyl_y(2.8, 2.0, 8.6)
# flare cone: apex at y=3.32, base r=5.4 at y=13.5 (trimesh cone:
# base at local z=0, apex at z=+h; rotX(+90deg) maps +Z -> -Y, so
# with the base placed at y=13.5 the apex lands at y=13.5-10.18=3.32
# and the radius at height y is 5.4*(y-3.32)/10.18: 2.8 at y=8.6,
# matching the waist.)
flare = trimesh.creation.cone(radius=5.4, height=13.5 - 3.32, sections=48)
flare.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
flare.apply_translation([0, 13.5, 0])
cap = cyl_y(5.4, 12.0, 14.5)

meta = union([bone, ball, waist, flare, cap])
finish(meta, "thumb_metacarpal")
