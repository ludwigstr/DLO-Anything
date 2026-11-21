"""U scenario: cable 2 is fixed at both ends and hangs as a U; cable 1 is pulled through the U.

Cable 1 keeps the usual hand and connect constraint. Cable 2 starts as a catenary between its two
fixed points, so it has no drop phase. Both cables are settled with extra velocity damping before
recording starts. Cable 2 always has the hand pose of its first fixed point in the data arrays.
"""
from __future__ import annotations
import numpy as np
import mujoco

from .config import SimConfig, sample_material, material_to_vec
from .two_cable_env import TwoCableSim, build_xml2

U_MODES = ["straight", "angled", "arc", "s_curve", "there_back"]
SETTLE_STEPS = 200
SETTLE_VEL_SCALE = 0.9
# Stiff cable contact: with the MuJoCo defaults the cables pass through each other under load
# (centre lines closer than 1 mm). With these values they stay about one diameter apart.
GEOM_ATTRS = 'solref="0.004 1" solimp="0.98 0.995 0.0005 0.5 2" margin="0.001"'


def catenary(span: float, length: float, n: int):
    """Equal-arc-length points (x along the span from 0 to span, z <= 0) of a catenary, and its sag."""
    lo, hi = 1e-3, 1e3
    for _ in range(200):
        a = np.sqrt(lo * hi)
        if 2 * a * np.sinh(span / (2 * a)) > length:
            lo = a
        else:
            hi = a
    arc = np.linspace(-length / 2, length / 2, n)
    x = a * np.arcsinh(arc / a)
    sag = a * (np.cosh(span / (2 * a)) - 1)
    z = a * (np.cosh(x / a) - 1) - sag
    return x + span / 2, z, sag


def u_geometry(cfg: SimConfig, mat2: dict, rng):
    """Draws the U: its length, span, height and position. Sets length and height in mat2."""
    mat2["length"] = float(rng.uniform(0.8, 1.6))
    span = float(rng.uniform(0.45, 0.8)) * mat2["length"]
    x, z, sag = catenary(span, mat2["length"], cfg.n_nodes)
    z_bottom = float(rng.uniform(0.10, 0.35))
    mat2["height"] = z_bottom + sag
    phi = float(rng.uniform(-np.pi, np.pi))
    centre = rng.uniform(-0.3, 0.3, 2)
    e = np.array([np.cos(phi), np.sin(phi)])
    return dict(span=span, phi=phi, centre=centre, e=e, n=np.array([-e[1], e[0]]), z_bottom=z_bottom, sag=sag,
                a=centre - 0.5 * span * e, b=centre + 0.5 * span * e, x=x, z=z)


def u_xml(cfg: SimConfig, mats, g, start1, away):
    """Cable 1 starts straight and horizontal, pointing away from the U. Cable 2 starts as catenary."""
    L1 = mats[0]["length"]
    d = np.linspace(0, L1, cfg.n_nodes)
    az = float(away[2]) if len(away) > 2 else 0.0
    v1 = [(float(away[0] * s), float(away[1] * s), float(az * s)) for s in d]
    v2 = [(float(xi * g["e"][0]), float(xi * g["e"][1]), float(zi)) for xi, zi in zip(g["x"], g["z"])]
    extra_body = (f'<body name="anchor_b" mocap="true" pos="{g["b"][0]:.6f} {g["b"][1]:.6f} {mats[1]["height"]:.6f}">'
                  '<geom type="sphere" size="0.008" rgba="0.9 0.6 0.4 0.5" contype="0" conaffinity="0"/></body>')
    extra_eq = '<connect name="right_boundary_Wire2" body1="anchor_b" body2="Wire2B_last" anchor="0 0 0" solref="0.02 1"/>'
    xml = build_xml2(cfg, mats, (start1, g["a"]), vertices=[v1, v2], flat=(False, True), extra_bodies=extra_body, extra_eqs=extra_eq, geom_attrs=GEOM_ATTRS)
    left = 'body1="hand2" body2="Wire2B_first" anchor="0 0 0" solref="0.05 1"'
    assert left in xml
    return xml.replace(left, 'body1="hand2" body2="Wire2B_first" anchor="0 0 0" solref="0.02 1"')  # both ends of the U equally stiff


def _curve(mode, g, rng, side):
    """Hand path as (x along the span, y normal to the U) over u in [0,1] and an optional return."""
    w = g["span"]
    xc = rng.uniform(-0.35, 0.35) * w
    d0, d1 = rng.uniform(0.3, 0.7), rng.uniform(0.3, 0.9)
    u = np.linspace(0, 1, 600)
    y = -d0 + u * (d0 + d1)
    if mode == "straight":
        x = xc * np.ones_like(u)
    elif mode == "angled":
        x = xc + np.tan(rng.uniform(-0.7, 0.7)) * y
    elif mode == "arc":
        x = xc + rng.uniform(-0.25, 0.25) * (1 - (2 * u - 1) ** 2)
    elif mode == "s_curve":
        x = xc + rng.uniform(0.05, 0.2) * rng.choice([-1, 1]) * np.sin(2 * np.pi * rng.uniform(1.0, 2.0) * u)
    else:
        xc2 = rng.uniform(-0.35, 0.35) * w
        x0 = xc * np.ones_like(u); y2 = y[::-1]
        x2 = np.linspace(xc, xc2, len(u))
        return np.column_stack([x0, side * y]), np.column_stack([x2, side * y2])
    return np.column_stack([x, side * y]), None


def _follow(path, speed, ramp, dt):
    """Positions along a polyline with cosine speed ramps."""
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    arc = np.r_[0.0, np.cumsum(seg)]
    total = arc[-1]
    ramp_eff = min(ramp, total / speed)
    n_ramp = max(1, int(round(ramp_eff / dt)))
    n_cruise = max(0, int(round((total - speed * ramp_eff) / speed / dt)))
    prof = np.concatenate([0.5 * (1 - np.cos(np.pi * np.arange(n_ramp) / n_ramp)), np.ones(n_cruise),
                           0.5 * (1 + np.cos(np.pi * np.arange(n_ramp) / n_ramp))]) * speed
    s = np.clip(np.cumsum(prof * dt), 0, total)
    return np.column_stack([np.interp(s, arc, path[:, 0]), np.interp(s, arc, path[:, 1])])


def u_trajectory(cfg: SimConfig, g, rng, n_steps, mode=None):
    """World hand 1 path (T,3) that starts on one side of the U and crosses it, and the mode."""
    mode = mode or U_MODES[rng.integers(len(U_MODES))]
    side = float(rng.choice([-1.0, 1.0]))
    p1, p2 = _curve(mode, g, rng, side)
    speed, ramp = rng.uniform(0.15, 0.8), rng.uniform(0.2, 0.8)
    dt = cfg.control_dt
    seq = [np.repeat(p1[:1], int(round(rng.uniform(0.2, 1.0) / dt)), axis=0), _follow(p1, speed, ramp, dt)]
    if p2 is not None:
        seq.append(np.repeat(seq[-1][-1:], int(round(rng.uniform(0.0, 0.8) / dt)), axis=0))
        seq.append(_follow(p2, speed, ramp, dt))
    local = np.concatenate(seq)
    if len(local) < n_steps:
        local = np.concatenate([local, np.repeat(local[-1:], n_steps - len(local), axis=0)])
    local = local[:n_steps]
    xy = g["centre"] + local[:, 0, None] * g["e"] + local[:, 1, None] * g["n"]
    return np.column_stack([xy, np.zeros(n_steps)]), mode, side


def run_episode_u(cfg: SimConfig, mats, g, traj1, away, record_frames=None):
    T = traj1.shape[0]
    start1 = traj1[0, :2]
    sim = TwoCableSim(cfg, mats, xml=u_xml(cfg, mats, g, start1, away))
    N = cfg.n_nodes
    traj2 = np.zeros((T, 3)); traj2[:, :2] = g["a"]
    traj = np.stack([traj1, traj2])
    pos = np.zeros((2, T, N, 3), np.float32); force = np.zeros((2, T, N, 3), np.float32); n_contact = np.zeros(T, np.int16)
    floor = np.zeros((2, T, N, 3), np.float32)
    for c in (0, 1):
        sim.set_gripper(c, *traj[c, 0])
    mujoco.mj_forward(sim.model, sim.data)
    for _ in range(SETTLE_STEPS):
        sim.step_control()
        sim.data.qvel[:] *= SETTLE_VEL_SCALE
    ok = sim.is_stable()
    nodes0 = sim.all_node_positions()
    gap0 = float(np.linalg.norm(nodes0[0, :, None] - nodes0[1, None], axis=-1).min())
    if gap0 < 0.02:
        ok = False
    frames = []
    for t in range(T):
        for c in (0, 1):
            sim.set_gripper(c, *traj[c, t])
        if t > 0:
            sim.step_control()
        else:
            mujoco.mj_forward(sim.model, sim.data)
        nodes = sim.all_node_positions()
        pos[:, t] = nodes
        f, cons, ff = sim.contact_forces(nodes, floor=True)
        force[:, t] = f; floor[:, t] = ff
        n_contact[t] = len(cons)
        if record_frames is not None:
            frames.append(record_frames(sim, t))
        if not sim.is_stable():
            ok = False
            break
    if ok and np.abs(np.diff(pos, axis=1)).max() > 0.1:
        ok = False
    action = np.zeros((2, T, 3), np.float32)
    action[:, :-1] = traj[:, 1:] - traj[:, :-1]
    out = dict(pose=traj.astype(np.float32), pos=pos, action=action, force=force, force_floor=floor, n_contact=n_contact,
               material=np.stack([material_to_vec(m) for m in mats]), ok=ok, gap0=gap0,
               anchor_b=np.r_[g["b"], mats[1]["height"]].astype(np.float32))
    if record_frames is not None:
        out["frames"] = frames
    return out, sim


FREE_SHARE = 0.2      # share of episodes in which cable 1 swings freely in the air, away from the U


def free_trajectory(cfg: SimConfig, g, L1, rng, n_steps):
    """Hand path far from the U with sharp starts, stops and turns: only the cable's inertia shapes it."""
    dt = cfg.control_dt
    seg_u = (g["a"], g["b"])

    def dist_u(p):
        a, b = seg_u; ab = b - a
        w = np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0, 1)
        return np.linalg.norm(p - (a + w * ab))

    clear = L1 + 0.35
    for _ in range(200):
        ang = rng.uniform(-np.pi, np.pi)
        start = g["centre"] + rng.uniform(clear + 0.3, clear + 0.9) * np.array([np.cos(ang), np.sin(ang)])
        pts = [start]
        for _ in range(int(rng.integers(3, 7))):
            pts.append(pts[-1] + rng.uniform(0.15, 0.6) * np.array([np.cos(a := rng.uniform(-np.pi, np.pi)), np.sin(a)]))
        pts = np.array(pts)
        dense = np.concatenate([np.linspace(pts[i], pts[i + 1], 50) for i in range(len(pts) - 1)])
        if min(dist_u(p) for p in dense) > clear:
            break
    else:
        return None
    speed, ramp = rng.uniform(0.3, 1.2), rng.uniform(0.08, 0.4)
    seq = [np.repeat(pts[:1], int(round(rng.uniform(0.2, 0.6) / dt)), axis=0)]
    for i in range(len(pts) - 1):                      # stop at every waypoint: the cable swings on
        seq.append(_follow(pts[i:i + 2], speed, ramp, dt))
        seq.append(np.repeat(seq[-1][-1:], int(round(rng.uniform(0.0, 0.5) / dt)), axis=0))
    xy = np.concatenate(seq)
    if len(xy) < n_steps:
        xy = np.concatenate([xy, np.repeat(xy[-1:], n_steps - len(xy), axis=0)])
    return np.column_stack([xy[:n_steps], np.zeros(n_steps)])


def sample_episode_u(cfg: SimConfig, rng, free=None):
    mats = [sample_material(rng), sample_material(rng)]
    g = u_geometry(cfg, mats[1], rng)
    free = rng.uniform() < FREE_SHARE if free is None else free
    if free:
        mats[0]["length"] = float(rng.uniform(0.6, 1.3))
        mats[0]["height"] = mats[0]["length"] + float(rng.uniform(0.1, 0.3))   # hangs clear of the floor
        traj = free_trajectory(cfg, g, mats[0]["length"], rng, cfg.episode_steps)
        if traj is not None:
            away = np.array([0.0, 0.0, -1.0])                                  # starts hanging straight down
            return mats, g, traj, "free", away
        mats = [sample_material(rng), mats[1]]
    H1 = float(rng.uniform(g["z_bottom"] + 0.05, max(mats[1]["height"] - 0.02, g["z_bottom"] + 0.06)))
    mats[0]["height"] = H1
    mats[0]["length"] = float(rng.uniform(max(0.6, H1 - g["z_bottom"] + 0.05), 2.0))
    traj, mode, side = u_trajectory(cfg, g, rng, cfg.episode_steps)
    away = -side * g["n"]                      # cable 1 starts pointing away from the U
    return mats, g, traj, mode, away


def generate_episode_u(cfg: SimConfig, seed: int):
    rng = np.random.default_rng(seed)
    mats, g, traj, mode, away = sample_episode_u(cfg, rng)
    out, _ = run_episode_u(cfg, mats, g, traj, away)
    out["seed"] = seed
    out["mode"] = mode
    return out
