#!/usr/bin/env python3
"""Shared CMC (thumb trapeziometacarpal) design parameters for v10.

The thumb CMC is a 2-axis universal joint at point C (saddle-joint
equivalent, user chose option C on 2026-11-08, variable so the working
angles can be studied before deciding any fixed angle).

Frames:
- Carpus/base frame: X = finger spread (index side -X), Y = proximal->
  distal, Z+ = dorsal.
- Frame M: the thumb metacarpal's rest frame. R0 maps M -> base frame.
  In M, the metacarpal points +Y, its MCP/IP pin axis is +X, dorsal +Z
  (same convention as every other bone).
- CMC axes: axis 2 (metacarpal side) = M's +X. Axis 1 (base side) =
  R0 @ (0, cos kappa, sin kappa), perpendicular to axis 2 at rest.
- World transform of a frame-M point p at CMC angles (t1, t2):
    C + Rot(A1, t1) @ Rot(A2, t2) @ R0 @ p
  (Rot(A2) uses A2 at rest = R0 @ +X; the composition gives the correct
  universal-joint kinematics.)
- The v9 thumb metacarpal mesh is reused shifted +META_SHIFT along Y in
  frame M (base ball cut off, fork added), so its MCP pivot sits at
  META_PIVOT from C.
"""
import math
import numpy as np
import trimesh

C = np.array([-30.5, 7.5, -4.6])          # CMC centre, base frame
ALPHA = 38.0                              # metacarpal azimuth toward -X (deg)
BETA = -32.0                              # palmar tilt (deg)
RHO = 45.0                                # pronation (deg) — chosen by the
KAPPA = 65.0                              # pinch search, 2026-11-08: with these,
                                          # thumb-tip contact with index/middle
                                          # sits at CMC ~ (-35, +35)
META_SHIFT = 6.5                          # v9 metacarpal shift in frame M
META_PIVOT = 42.0 + META_SHIFT            # C -> MCP pivot distance

def rot(axis, deg):
    return trimesh.transformations.rotation_matrix(math.radians(deg), axis)[:3, :3]

def frames(rho_deg, kappa_deg):
    """Return (R0, A1, A2_rest, u0) for pronation rho and axis-1 tilt kappa."""
    R_pre = rot([0, 0, 1], ALPHA) @ rot([1, 0, 0], BETA)
    u0 = R_pre @ np.array([0.0, 1.0, 0.0])
    R0 = rot(u0, rho_deg) @ R_pre
    A2 = R0 @ np.array([1.0, 0.0, 0.0])
    A1 = R0 @ np.array([0.0, math.cos(math.radians(kappa_deg)),
                        math.sin(math.radians(kappa_deg))])
    return R0, A1, A2, u0

def cmc_matrix(R0, A1, A2, t1, t2):
    """3x3 rotation mapping frame-M directions to base frame at (t1, t2)."""
    return rot(A1, t1) @ rot(A2, t2) @ R0
