"""MuJoCo cable simulation.

A cable (MuJoCo `composite type="cable"` with the elasticity plugin) is rigidly attached to a
mocap "gripper" body held at a fixed height above a plane. The cable is longer than the gripper
height, so the lower part rests on / drags across the plane. The gripper follows a random smooth
trajectory in (x, y, yaw). Material parameters (bending/torsional stiffness, damping, density,
friction) are sampled per episode and baked into the MJCF model.
"""
from __future__ import annotations

import os
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import mujoco

from .config import SimConfig, MATERIAL_NAMES, sample_material, material_to_vec


def yaw_to_quat(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def cable_vertices(cfg: SimConfig, length: float | None = None) -> list[tuple[float, float, float]]:
    """Initial cable shape (user model): straight horizontal line along the xy diagonal at the
    hand height; the cable falls onto the plane during the settle phase."""
    L, N = (length or cfg.cable_length), cfg.n_segments
    a = L / N / np.sqrt(2.0)
    return [(a * i, -a * i, 0.0) for i in range(N + 1)]


def build_xml(cfg: SimConfig, mat: dict) -> str:
    """MJCF of the user's `cable_hang` model with the material parameters substituted.
    Physics is unchanged from the user's file (Euler, 2 ms, air drag, condim=1, connect
    constraint); added: autoreset disabled, solref 0.05 on the connect constraint (40 segments: the
    default 0.02 blows up during the initial drop of heavy, very soft cables), a white floor,
    sky gradient, brighter headlight and a larger offscreen buffer."""
    vtx = "\n              ".join(f"{x:.6f} {y:.6f} {z:.6f}"
                                  for x, y, z in cable_vertices(cfg, mat.get("length", cfg.cable_length)))
    H = mat.get("height", cfg.gripper_height)
    return f"""
<mujoco model="cable_hang">
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

    <!-- 001 wire -->
    <composite prefix="Wire1" type="cable"
      vertex="{vtx}"
      offset="0 0 {H}" initial="free">
      <plugin plugin="mujoco.elasticity.cable">
        <config key="twist" value="{mat['twist']:.6g}"/>
        <config key="bend" value="{mat['bend']:.6g}"/>
        <config key="vmax" value="0.05"/>
      </plugin>
      <joint kind="main" damping="{mat['damping']:.6g}"/>
      <geom type="capsule" size="{cfg.cable_radius}" density="{mat['density']:.6g}" rgba="0 0 1 1" condim="1"/>
    </composite>

    <camera name="cam" pos="0 0 0" fovy="45.00"/>
    <geom name="wall" type="plane" pos="0 0 0"
          size="6 6 0.1" material="floor"/>
    <body name="hand" mocap="true" pos="0 0 {H}">
      <geom type="sphere" size="0.008" rgba="0.9 0.6 0.4 0.5" contype="0" conaffinity="0"/>
    </body>
  </worldbody>

  <equality>
    <!-- Connections for the endpoints -->
    <connect name="left_boundary_Wire1" body1="hand" body2="Wire1B_first" anchor="0 0 0" solref="0.05 1"/>
  </equality>
</mujoco>
"""


class CableSim:
    """Thin wrapper around MjModel/MjData for one material configuration."""

    def __init__(self, cfg: SimConfig, mat: dict):
        self.cfg = cfg
        self.mat = mat
        self.height = float(mat.get("height", cfg.gripper_height))
        self.length = float(mat.get("length", cfg.cable_length))
        self.model = mujoco.MjModel.from_xml_string(build_xml(cfg, mat))
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.gripper_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "hand")
        # cable bodies in order: Wire1B_first, Wire1B_1, ..., Wire1B_last  (= n_segments bodies)
        self.cable_bodies = [i for i in range(m.nbody) if m.body(i).name.startswith("Wire1B_")]
        assert len(self.cable_bodies) == cfg.n_segments, (len(self.cable_bodies), cfg.n_segments)
        self.last_geom = [g for g in range(m.ngeom) if m.geom_bodyid[g] == self.cable_bodies[-1]][0]
        self.cable_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] in set(self.cable_bodies)]

    # ---- state access -------------------------------------------------------------------
    def node_positions(self) -> np.ndarray:
        """(n_nodes, 3): start point of each segment plus the end point of the last segment."""
        d, m = self.data, self.model
        pos = d.xpos[self.cable_bodies]
        g = self.last_geom
        # the capsule centre is the midpoint between the body origin and the segment end point
        end = 2.0 * d.geom_xpos[g] - d.xpos[self.cable_bodies[-1]]
        return np.concatenate([pos, end[None]], axis=0)

    def set_gripper(self, x: float, y: float, yaw: float):
        self.data.mocap_pos[0] = [x, y, self.height]
        self.data.mocap_quat[0] = yaw_to_quat(yaw if self.cfg.use_yaw else 0.0)

    def step_control(self):
        for _ in range(self.cfg.substeps):
            mujoco.mj_step(self.model, self.data)

    def is_stable(self) -> bool:
        """True while no MuJoCo warning fired and the state is finite. Autoreset is disabled in the
        model so an instability cannot silently reset the state (which would clear the warnings)."""
        d = self.data
        if sum(w.number for w in d.warning) > 0:
            return False
        if not np.all(np.isfinite(d.qpos)) or not np.all(np.isfinite(d.qvel)):
            return False
        return bool(np.abs(d.xpos).max() < 5.0 and np.abs(d.qvel).max() < 1e4)


# ---- random trajectories ------------------------------------------------------------------
def sample_traj_params(cfg: SimConfig, rng: np.random.Generator, style: str) -> dict:
    """Per-episode trajectory parameters drawn from the ranges in cfg."""
    u = rng.uniform
    lo = u(*cfg.speed_lo_range); hi = max(lo + 0.1, u(*cfg.speed_hi_range))
    P = dict(style=style, speed=(lo, hi))
    if style == "long":
        P.update(min_move=u(*cfg.long_min_move_range), ramp=u(*cfg.long_ramp_range), max_pause=u(*cfg.long_pause_range))
    elif style == "diverse":
        a, b = sorted(u(*cfg.div_move_range, size=2)); b = max(b, a + 0.05)
        P.update(move=(a, b), ramp=u(*cfg.div_ramp_range), max_pause=u(*cfg.div_pause_range), max_turn=u(*cfg.div_turn_range))
    elif style == "varied":
        P.update(move=cfg.var_move_range, ramp=u(*cfg.div_ramp_range), max_pause=u(*cfg.long_pause_range), max_turn=180.0)
    else:
        raise ValueError(style)
    return P


def random_trajectory(cfg: SimConfig, rng: np.random.Generator, n_steps: int,
                      start_xy=None, start_yaw=None, style: str | None = None, params: dict | None = None) -> np.ndarray:
    """Random hand trajectory. Returns (n_steps, 3) array [x, y, yaw]. Yaw stays 0 unless cfg.use_yaw.

    style "long":    straight moves (>= min_move) to random waypoints in the workspace.
    style "diverse": short moves whose direction changes by a random angle (+-max_turn) after
                     every move -> many direction changes per second.
    style "varied":  like "diverse" but every move has a random length between 3 cm and 2 m.
    style "mix":     one of the three, chosen per episode.
    Every move has a random cruise speed inside the episode's speed range, cosine ramps and an
    optional pause. The episode parameters come from `params` or `sample_traj_params`.
    """
    style = style or cfg.style
    if style == "mix":
        style = ["long", "diverse", "varied"][int(rng.integers(3))]
    P = params or sample_traj_params(cfg, rng, style)
    long = style == "long"
    dt = cfg.control_dt
    ws = cfg.workspace
    xy = np.array(start_xy if start_xy is not None else rng.uniform(-0.5 * ws, 0.5 * ws, 2), np.float64)
    out = np.zeros((n_steps, 3))
    t = 0
    yaw = float(start_yaw if start_yaw is not None else 0.0)
    heading = rng.uniform(-np.pi, np.pi)
    ramp, max_pause = P["ramp"], P["max_pause"]
    while t < n_steps:
        if long:
            for _ in range(100):
                tgt = rng.uniform(-ws, ws, 2)
                if np.linalg.norm(tgt - xy) >= P["min_move"]:
                    break
            dist = np.linalg.norm(tgt - xy)
            direction = (tgt - xy) / dist
        else:
            # new heading = old heading + random turn; steer back when the move would leave the workspace
            for _ in range(100):
                heading = heading + np.deg2rad(rng.uniform(-P["max_turn"], P["max_turn"]))
                dist = rng.uniform(*P["move"])
                direction = np.array([np.cos(heading), np.sin(heading)])
                if np.abs(xy + direction * dist).max() <= ws:
                    break
                heading = np.arctan2(-xy[1], -xy[0])   # point towards the centre and retry
        speed = rng.uniform(*P["speed"])
        ramp_eff = min(ramp, dist / speed)        # short moves: shorter ramps so the move does not overshoot
        n_ramp = max(1, int(round(ramp_eff / dt)))
        n_cruise = max(0, int(round((dist - speed * ramp_eff) / speed / dt)))   # ramp up+down cover speed*ramp
        prof = np.concatenate([0.5 * (1 - np.cos(np.pi * np.arange(n_ramp) / n_ramp)),
                               np.ones(n_cruise),
                               0.5 * (1 + np.cos(np.pi * np.arange(n_ramp) / n_ramp))]) * speed
        for v in prof:
            if t >= n_steps:
                break
            xy = xy + direction * v * dt
            out[t] = (xy[0], xy[1], yaw)
            t += 1
        for _ in range(int(round(rng.uniform(0.0, max_pause) / dt))):
            if t >= n_steps:
                break
            out[t] = (xy[0], xy[1], yaw)
            t += 1
    return out


# ---- episode rollout ----------------------------------------------------------------------
def run_episode(cfg: SimConfig, mat: dict, traj: np.ndarray, record_frames=None):
    """Simulate one episode following `traj` (n_steps, 3). Returns dict of arrays.

    pose[t]  : gripper (x, y, yaw) at control step t
    pos[t]   : node positions (n_nodes, 3) at control step t
    action[t]: pose[t+1] - pose[t]  (last action is zero)
    """
    sim = CableSim(cfg, mat)
    n = len(traj)
    pos = np.zeros((n, cfg.n_nodes, 3), np.float32)
    # settle with gripper at the start pose
    sim.set_gripper(*traj[0])
    mujoco.mj_forward(sim.model, sim.data)
    for _ in range(cfg.settle_steps):
        sim.step_control()
    ok = sim.is_stable()
    frames = []
    for t in range(n):
        sim.set_gripper(*traj[t])
        if t > 0:
            sim.step_control()
        else:
            mujoco.mj_forward(sim.model, sim.data)
        pos[t] = sim.node_positions()
        if record_frames is not None:
            frames.append(record_frames(sim, t))
        if not sim.is_stable():
            ok = False
            break
    # second guard: no node may jump more than 20 cm within one control step (silent resets/NaNs)
    if ok and np.abs(np.diff(pos, axis=0)).max() > 0.1:
        ok = False
    action = np.zeros((n, 3), np.float32)
    action[:-1] = traj[1:] - traj[:-1]
    out = dict(pose=traj.astype(np.float32), pos=pos, action=action,
               material=material_to_vec(mat), ok=ok)
    if record_frames is not None:
        out["frames"] = frames
    return out, sim


def generate_episode(cfg: SimConfig, seed: int, mat: dict | None = None):
    rng = np.random.default_rng(seed)
    if mat is None:
        mat = sample_material(rng)
    traj = random_trajectory(cfg, rng, cfg.episode_steps)
    out, _ = run_episode(cfg, mat, traj)
    out["seed"] = seed
    return out
