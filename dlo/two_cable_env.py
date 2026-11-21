"""Two cables, two hands, one plane: MuJoCo simulation with cable-cable contact forces.

Extends `cable_env` (one cable on a point-constrained mocap hand) by a second, independent
cable on a second hand. Everything of the first cable (geometry, integrator, air drag, connect
constraint, condim=1) is unchanged, so when the cables never touch each cable moves exactly as
in the single-cable simulation. New per step:
  * node positions of both cables
  * the contact force that the OTHER cable exerts on every node (world frame, N)
The per-node force is the sum of all cable-cable contact forces acting on the segment bodies,
split between the two end nodes of a segment by the contact position along the segment.
Floor contact and the hand constraint are not part of this force (the model already learns
them from the single-cable data); it is the "external force from another object" channel.
"""
from __future__ import annotations
import os
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import mujoco

from .config import SimConfig, sample_material, material_to_vec
from .cable_env import cable_vertices, yaw_to_quat, random_trajectory, sample_traj_params, CableSim

RGBA = ["0 0 1 1", "0 0.65 0.1 1"]


def build_xml2(cfg: SimConfig, mats: list[dict], start_xy=((0.0, 0.0), (0.0, 0.0)), vertices=None, flat=(False, False),
               extra_bodies: str = "", extra_eqs: str = "", geom_attrs: str = "") -> str:
    """MJCF with two cables (Wire1, Wire2), each on its own mocap hand (hand1, hand2).
    vertices: optional initial shape per cable relative to the hand; flat: stress-free shape is
    straight even when the initial shape is curved; extra_*: additional bodies and constraints."""
    cables, hands, eqs = [], [], []
    for c, (mat, xy) in enumerate(zip(mats, start_xy), start=1):
        vtx = "\n              ".join(f"{x:.6f} {y:.6f} {z:.6f}"
                                      for x, y, z in (vertices[c - 1] if vertices is not None and vertices[c - 1] is not None
                                                      else cable_vertices(cfg, mat.get("length", cfg.cable_length))))
        H = mat.get("height", cfg.gripper_height)
        flat_cfg = '<config key="flat" value="true"/>' if flat[c - 1] else ""
        cables.append(f"""
    <composite prefix="Wire{c}" type="cable"
      vertex="{vtx}"
      offset="{xy[0]:.6f} {xy[1]:.6f} {H}" initial="free">
      <plugin plugin="mujoco.elasticity.cable">
        <config key="twist" value="{mat['twist']:.6g}"/>
        <config key="bend" value="{mat['bend']:.6g}"/>
        <config key="vmax" value="0.05"/>
        {flat_cfg}
      </plugin>
      <joint kind="main" damping="{mat['damping']:.6g}"/>
      <geom type="capsule" size="{cfg.cable_radius}" density="{mat['density']:.6g}" rgba="{RGBA[c - 1]}" condim="1" {geom_attrs}/>
    </composite>""")
        hands.append(f"""
    <body name="hand{c}" mocap="true" pos="{xy[0]:.6f} {xy[1]:.6f} {H}">
      <geom type="sphere" size="0.008" rgba="0.9 0.6 0.4 0.5" contype="0" conaffinity="0"/>
    </body>""")
        eqs.append(f'    <connect name="left_boundary_Wire{c}" body1="hand{c}" body2="Wire{c}B_first" anchor="0 0 0" solref="0.05 1"/>')
    return f"""
<mujoco model="cable_hang2">
  <extension>
    <plugin plugin="mujoco.elasticity.cable"/>
  </extension>

  <compiler angle="radian" autolimits="true"/>
  <option gravity="0.000 0.000 -9.810"
          timestep="{cfg.timestep}" integrator="{cfg.integrator}" density="1.2" viscosity="1.8e-5">
    <flag autoreset="disable"/>
  </option>
  <visual>
    <global offwidth="1280" offheight="720"/>
    <quality shadowsize="2048"/>
    <headlight ambient="0.4 0.4 0.4" diffuse="0.28 0.28 0.28" specular="0 0 0"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.55 0.70 0.90" rgb2="0.92 0.95 1.0" width="256" height="256"/>
    <material name="floor" rgba="1 1 1 1" specular="0" shininess="0" reflectance="0"/>
  </asset>
  <worldbody>

    <light pos="0.5 1.0 0.5" dir="-0.4 -0.8 -0.45" directional="true"/>
{''.join(cables)}

    <camera name="cam" pos="0 0 0" fovy="45.00"/>
    <geom name="wall" type="plane" pos="0 0 0"
          size="6 6 0.1" material="floor"/>
{''.join(hands)}
{extra_bodies}
  </worldbody>

  <equality>
{chr(10).join(eqs)}
{extra_eqs}
  </equality>
</mujoco>
"""


class TwoCableSim:
    """MjModel/MjData wrapper for two cables with per-node cable-cable contact forces."""

    def __init__(self, cfg: SimConfig, mats: list[dict], start_xy=((0.0, 0.0), (0.0, 0.0)), xml: str | None = None):
        self.cfg = cfg
        self.mats = mats
        self.heights = [float(m.get("height", cfg.gripper_height)) for m in mats]
        self.model = mujoco.MjModel.from_xml_string(xml or build_xml2(cfg, mats, start_xy))
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.hand_bodies = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"hand{c}") for c in (1, 2)]
        self.mocap_ids = [m.body_mocapid[b] for b in self.hand_bodies]
        self.cable_bodies, self.last_geom, self.cable_geoms = [], [], []
        for c in (1, 2):
            bodies = [i for i in range(m.nbody) if m.body(i).name.startswith(f"Wire{c}B_")]
            assert len(bodies) == cfg.n_segments, (len(bodies), cfg.n_segments)
            geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] in set(bodies)]
            self.cable_bodies.append(bodies)
            self.cable_geoms.append(geoms)
            self.last_geom.append([g for g in geoms if m.geom_bodyid[g] == bodies[-1]][0])
        # geom id -> (cable index, segment index)
        self.geom_seg = {}
        for c in (0, 1):
            for s, b in enumerate(self.cable_bodies[c]):
                for g in self.cable_geoms[c]:
                    if m.geom_bodyid[g] == b:
                        self.geom_seg[g] = (c, s)
        self._f6 = np.zeros(6)

    # ---- state access -------------------------------------------------------------------
    def node_positions(self, c: int) -> np.ndarray:
        d = self.data
        bodies = self.cable_bodies[c]
        pos = d.xpos[bodies]
        end = 2.0 * d.geom_xpos[self.last_geom[c]] - d.xpos[bodies[-1]]
        return np.concatenate([pos, end[None]], axis=0)

    def all_node_positions(self) -> np.ndarray:
        return np.stack([self.node_positions(0), self.node_positions(1)])

    def set_gripper(self, c: int, x: float, y: float, yaw: float):
        mid = self.mocap_ids[c]
        self.data.mocap_pos[mid] = [x, y, self.heights[c]]
        self.data.mocap_quat[mid] = yaw_to_quat(yaw if self.cfg.use_yaw else 0.0)

    def step_control(self):
        for _ in range(self.cfg.substeps):
            mujoco.mj_step(self.model, self.data)

    def is_stable(self) -> bool:
        d = self.data
        if sum(w.number for w in d.warning) > 0:
            return False
        if not np.all(np.isfinite(d.qpos)) or not np.all(np.isfinite(d.qvel)):
            return False
        return bool(np.abs(d.xpos).max() < 5.0 and np.abs(d.qvel).max() < 1e4)

    # ---- contact forces ------------------------------------------------------------------
    def contact_forces(self, nodes: np.ndarray | None = None, floor: bool = False):
        """Per-node force from cable-cable contacts at the current physics state.

        Returns force (2, N, 3) [N, world frame] and the list of contacts
        [(cable_a, seg_a, cable_b, seg_b, pos(3), force_on_a(3))].
        MuJoCo convention: the contact frame's first axis is the normal pointing from geom1 to
        geom2 and mj_contactForce returns the force acting on geom2 (geom1 gets the opposite).
        """
        m, d = self.model, self.data
        N = self.cfg.n_nodes
        force = np.zeros((2, N, 3))
        floor_force = np.zeros((2, N, 3))
        contacts = []
        if nodes is None:
            nodes = self.all_node_positions()
        for i in range(d.ncon):
            con = d.contact[i]
            a, b = self.geom_seg.get(con.geom1), self.geom_seg.get(con.geom2)
            if floor and (a is None) != (b is None):        # cable - floor: force on the cable segment
                mujoco.mj_contactForce(m, d, i, self._f6)
                f2 = np.asarray(con.frame).reshape(3, 3).T @ self._f6[:3]
                (c, s), f = (b, f2) if a is None else (a, -f2)
                p0, p1 = nodes[c, s], nodes[c, s + 1]
                seg = p1 - p0
                w = float(np.clip(np.dot(np.asarray(con.pos) - p0, seg) / max(np.dot(seg, seg), 1e-12), 0.0, 1.0))
                floor_force[c, s] += (1 - w) * f
                floor_force[c, s + 1] += w * f
                continue
            if a is None or b is None or a[0] == b[0]:
                continue                                   # floor contact or self contact
            mujoco.mj_contactForce(m, d, i, self._f6)
            frame = np.asarray(con.frame).reshape(3, 3)    # rows = normal, tangent1, tangent2
            f_on_2 = frame.T @ self._f6[:3]               # world frame, force on geom2
            p = np.asarray(con.pos)
            for (c, s), f in ((b, f_on_2), (a, -f_on_2)):
                p0, p1 = nodes[c, s], nodes[c, s + 1]
                seg = p1 - p0
                w = float(np.clip(np.dot(p - p0, seg) / max(np.dot(seg, seg), 1e-12), 0.0, 1.0))
                force[c, s] += (1 - w) * f
                force[c, s + 1] += w * f
            contacts.append((a[0], a[1], b[0], b[1], p.copy(), -f_on_2))
        if floor:
            return force, contacts, floor_force
        return force, contacts


# ---- hand trajectories -------------------------------------------------------------------
TRAJ_MODES = ["cross", "near", "follow", "park"]


def _shifted_cfg(cfg: SimConfig, half: float) -> SimConfig:
    """Copy of cfg with a smaller workspace (targets are shifted to the shared centre afterwards)."""
    c = SimConfig(**{k: getattr(cfg, k) for k in cfg.__dataclass_fields__})
    c.workspace = half
    return c


def two_hand_trajectories(cfg: SimConfig, rng: np.random.Generator, n_steps: int, mats: list[dict],
                          mode: str | None = None):
    """Two random hand trajectories that make the cables meet. Modes (drawn per episode):
      cross  (35 %): hand 2 is parked (static) at a point hand 1 passes later (random time in
                     the episode, offset <= 0.15 m): cable 1 is dragged across cable 2
      near   (30 %): both hands move inside a small shared workspace (half-size 0.25..0.6 m)
                     centred at a random point: frequent crossings, cables lie on each other
      follow (15 %): hand 2 copies hand 1's trajectory shifted by 0.1..0.6 m: the cables are
                     dragged side by side
      park   (20 %): hand 2 is static 0.1..0.4 m from hand 1's start, hand 1 moves in a small
                     workspace around it
    Returns traj (2, n_steps, 3) [x, y, yaw] and the mode string.
    """
    ws = cfg.workspace
    if mode is None:
        mode = rng.choice(TRAJ_MODES, p=[0.35, 0.30, 0.15, 0.20])
    L = max(m.get("length", cfg.cable_length) for m in mats)

    def offset(lo, hi):
        r, ang = rng.uniform(lo, hi), rng.uniform(-np.pi, np.pi)
        return np.array([r * np.cos(ang), r * np.sin(ang)])

    if mode == "cross":
        t1 = random_trajectory(cfg, rng, n_steps)
        t_star = int(rng.integers(int(0.15 * n_steps), int(0.85 * n_steps)))
        start2 = t1[t_star, :2] + offset(0.0, 0.15)
        t2 = np.zeros_like(t1); t2[:, :2] = start2
    elif mode == "near":
        half = rng.uniform(0.25, 0.6)
        centre = rng.uniform(-(ws - half), ws - half, 2)
        c2 = _shifted_cfg(cfg, half)
        t1 = random_trajectory(c2, rng, n_steps, start_xy=rng.uniform(-half, half, 2))
        t2 = random_trajectory(c2, rng, n_steps, start_xy=rng.uniform(-half, half, 2))
        t1[:, :2] += centre; t2[:, :2] += centre
    elif mode == "follow":
        t1 = random_trajectory(cfg, rng, n_steps)
        off = offset(0.1, 0.6)
        t2 = t1.copy(); t2[:, :2] = t1[:, :2] + off
    elif mode == "park":
        half = rng.uniform(0.4, 0.8)
        centre = rng.uniform(-(ws - half), ws - half, 2)
        c2 = _shifted_cfg(cfg, half)
        t1 = random_trajectory(c2, rng, n_steps, start_xy=rng.uniform(-half, half, 2))
        t1[:, :2] += centre
        t2 = np.zeros_like(t1); t2[:, :2] = t1[0, :2] + offset(0.1, min(0.4, 0.9 * L))
    else:
        raise ValueError(mode)
    traj = np.stack([t1, t2])
    traj[:, :, :2] = np.clip(traj[:, :, :2], -ws - 0.5, ws + 0.5)
    return traj, mode


# ---- "drive" scenario: one cable moves and is driven against the other, resting one -----------
def rest_nodes(cfg: SimConfig, mat: dict, xy) -> np.ndarray:
    """Settled shape (N,3) of a single cable whose hand rests at xy (settled at the origin and
    shifted: the plane is translation invariant)."""
    sim = CableSim(cfg, mat)
    sim.set_gripper(0.0, 0.0, 0.0)
    mujoco.mj_forward(sim.model, sim.data)
    for _ in range(cfg.settle_steps):
        sim.step_control()
    nodes = sim.node_positions().copy()
    nodes[:, :2] += np.asarray(xy)
    return nodes


def _straight(xy, tgt, speed, ramp, dt):
    """Positions of a straight move from xy to tgt with cosine ramps, as in random_trajectory."""
    dist = float(np.linalg.norm(tgt - xy))
    if dist < 1e-6:
        return np.zeros((0, 2))
    direction = (tgt - xy) / dist
    ramp_eff = min(ramp, dist / speed)
    n_ramp = max(1, int(round(ramp_eff / dt)))
    n_cruise = max(0, int(round((dist - speed * ramp_eff) / speed / dt)))
    prof = np.concatenate([0.5 * (1 - np.cos(np.pi * np.arange(n_ramp) / n_ramp)), np.ones(n_cruise),
                           0.5 * (1 + np.cos(np.pi * np.arange(n_ramp) / n_ramp))]) * speed
    return xy + np.cumsum(prof * dt)[:, None] * direction


def drive_trajectories(cfg: SimConfig, rng: np.random.Generator, n_steps: int, mats: list[dict]):
    """Hand 2 stays still; hand 1 starts clear of cable 2 and repeatedly drives through a point of
    cable 2's settled shape, overshooting it, with a pause after every pass. Target points are
    limited to the heights cable 1 can reach. Returns traj (2,T,3), or None if the materials
    allow no contact (the caller redraws them)."""
    H1, L1 = mats[0]["height"], mats[0]["length"]
    start2 = rng.uniform(-0.5 * cfg.workspace, 0.5 * cfg.workspace, 2)
    R2 = rest_nodes(cfg, mats[1], start2)
    z_min = max(0.0, H1 - 0.95 * L1)
    cand = np.where((R2[:, 2] >= z_min) & (R2[:, 2] <= H1))[0]
    if len(cand) == 0:
        return None
    R1_rel = rest_nodes(cfg, mats[0], (0.0, 0.0))
    P = sample_traj_params(cfg, rng, "long")
    for _ in range(30):
        target = R2[cand[rng.integers(len(cand))], :2]
        ang = rng.uniform(-np.pi, np.pi)
        start1 = target + rng.uniform(0.15, 0.9) * np.array([np.cos(ang), np.sin(ang)])
        gap = np.linalg.norm((R1_rel + np.r_[start1, 0.0])[:, None] - R2[None], axis=-1).min()
        if gap > 0.05:
            break
    else:
        return None
    t1 = np.zeros((n_steps, 3))
    xy, t = start1.copy(), 0
    ws = cfg.workspace + 0.5
    while t < n_steps:
        direction = (target - xy) / max(np.linalg.norm(target - xy), 1e-6)
        end = np.clip(target + rng.uniform(0.1, 0.8) * direction, -ws, ws)
        path = _straight(xy, end, rng.uniform(*P["speed"]), P["ramp"], cfg.control_dt)
        pause = int(round(rng.uniform(0.0, P["max_pause"]) / cfg.control_dt))
        if len(path):
            xy = path[-1]
        seq = np.concatenate([path, np.repeat(xy[None], pause, axis=0)])
        n = min(len(seq), n_steps - t)
        t1[t:t + n, :2] = seq[:n]
        t += n
        if n == 0:
            t1[t:] = t1[max(t - 1, 0)]
            t1[t:, :2] = xy
            t += 1
        target = R2[cand[rng.integers(len(cand))], :2]
    t2 = np.zeros_like(t1); t2[:, :2] = start2
    # the settle phase holds the hand at traj[0]: start the record at the start position
    t1 = np.concatenate([np.r_[start1, 0.0][None], t1[:-1]])
    return np.stack([t1, t2])


# ---- episode rollout ----------------------------------------------------------------------
def run_episode2(cfg: SimConfig, mats: list[dict], traj: np.ndarray, record_frames=None):
    """Simulate both cables along traj (2, T, 3). Returns a dict of arrays:
    pose (2,T,3), pos (2,T,N,3), action (2,T,3), force (2,T,N,3) [N], n_contact (T,),
    material (2,6), ok.
    """
    T = traj.shape[1]
    sim = TwoCableSim(cfg, mats, start_xy=(traj[0, 0, :2], traj[1, 0, :2]))
    N = cfg.n_nodes
    pos = np.zeros((2, T, N, 3), np.float32)
    force = np.zeros((2, T, N, 3), np.float32)
    n_contact = np.zeros(T, np.int16)
    for c in (0, 1):
        sim.set_gripper(c, *traj[c, 0])
    mujoco.mj_forward(sim.model, sim.data)
    for _ in range(cfg.settle_steps):
        sim.step_control()
    ok = sim.is_stable()
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
        f, cons = sim.contact_forces(nodes)
        force[:, t] = f
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
    out = dict(pose=traj.astype(np.float32), pos=pos, action=action, force=force, n_contact=n_contact,
               material=np.stack([material_to_vec(m) for m in mats]), ok=ok)
    if record_frames is not None:
        out["frames"] = frames
    return out, sim


def generate_episode2(cfg: SimConfig, seed: int, scenario: str = "mixed"):
    rng = np.random.default_rng(seed)
    mats = [sample_material(rng), sample_material(rng)]
    if scenario == "drive":
        traj = None
        while traj is None:
            traj = drive_trajectories(cfg, rng, cfg.episode_steps, mats)
            if traj is None:
                mats = [sample_material(rng), sample_material(rng)]
        mode = "drive"
    else:
        traj, mode = two_hand_trajectories(cfg, rng, cfg.episode_steps, mats)
    out, _ = run_episode2(cfg, mats, traj)
    out["seed"] = seed
    out["mode"] = mode
    return out
