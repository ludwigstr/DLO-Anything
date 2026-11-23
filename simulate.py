"""Simulation data for DLO-Anything (MuJoCo): a robot pulls one cable across a second cable that hangs as a U.

    python simulate.py --set F --scenes 12000 --out data/F_12000.npz --seed 1000000      # training set, eight motion kinds
    python simulate.py --set A --scenes 300 --out data/val_designed.npz --seed 2000000   # checkpoint selection
    python simulate.py --held-out --scenes 300 --out data/val.npz --seed 3000000         # validation: circle, sine, jerk
    python simulate.py --held-out --scenes 340 --out data/test.npz --seed 4000000        # test: circle, sine, jerk

Training sets (every kind with the same share):
    A  designed patterns (straight, angled, arc, S-curve, there and back, free swing)
    B  A + random waypoints          C  B + spline curves          D  C + varying speed
    E  D + retraced paths            F  E + loops (curl), weaving, bursts
Circle, sine and jerk motions are held out: `--set` never produces them, `--held-out` produces only them.
Unstable scenes and scenes in which the pulled cable winds around the U by more than one turn are dropped.

Arrays (E scenes, cable 0 = pulled / 1 = fixed U, T = 400 steps at 50 Hz, N = 41 nodes):
    pos (E,2,T,N,3)  pose (E,2,T,3) gripper x, y, yaw  action (E,2,T,3)  material (E,2,6)
    force_cable, force_floor (E,2,T,N,3) contact forces [N] (evaluation and videos only, never a model input)
    mode (E) motion pattern   kind (E) motion kind   seed (E)   winding (E) degrees"""
from __future__ import annotations
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import argparse
import mujoco
import numpy as np
import time
import torch
from dataclasses import dataclass
from multiprocessing import Pool
from model import MATERIAL_NAMES, MATERIAL_RANGES, closest_on_cable


@dataclass
class SimConfig:
    # Cable geometry
    n_segments: int = 40          # cable segments
    cable_length: float = 1.3153  # [m]
    cable_radius: float = 0.003   # [m]
    gripper_height: float = 0.5   # [m] fixed height of the hand above the plane
    # Integration
    timestep: float = 0.002       # physics step [s]
    integrator: str = "Euler"     # MuJoCo integrator
    control_dt: float = 0.02      # control / data step [s]  (10 physics steps)
    # Workspace (gripper xy is kept inside a square of this half-size)
    workspace: float = 1.2        # [m]
    # Random trajectory: long straight moves between random waypoints
    max_speed: float = 1.0        # [m/s] cruise speed is uniform in [min_speed, max_speed]
    min_speed: float = 0.4        # [m/s]
    min_move: float = 1.2         # [m] minimum length of one straight move
    ramp_time: float = 0.6        # [s] cosine acceleration / deceleration ramp
    max_pause: float = 0.6        # [s] random pause between moves (uniform in [0, max_pause])
    # Trajectory style: "long" = long straight moves, "diverse" = short moves with frequent
    # direction changes (test videos), "varied" = every move has a random length between short and
    # long, "mix" = one of the three per episode. The per-episode trajectory parameters (speed
    # range, move lengths, ramps, pauses, turn angle) are themselves sampled from the ranges below.
    style: str = "long"
    speed_lo_range: tuple = (0.1, 0.5)      # [m/s] lower end of the per-episode cruise-speed range
    speed_hi_range: tuple = (0.6, 1.6)      # [m/s] upper end
    long_min_move_range: tuple = (0.5, 1.6) # [m] minimum move length of the long style
    long_ramp_range: tuple = (0.2, 1.0)     # [s]
    long_pause_range: tuple = (0.0, 1.2)    # [s] maximum pause
    div_move_range: tuple = (0.03, 0.8)     # [m] per-episode move-length range is drawn inside this
    div_ramp_range: tuple = (0.06, 0.4)     # [s]
    div_pause_range: tuple = (0.0, 0.5)     # [s] maximum pause
    div_turn_range: tuple = (40.0, 180.0)   # [deg] maximum direction change per move
    var_move_range: tuple = (0.03, 2.0)     # [m] per-move length of the varied style
    use_yaw: bool = False         # the hand is a point constraint (connect), so yaw has no effect
    max_yaw_rate: float = 2.0     # [rad/s] (only used for action scaling)
    # Episode
    episode_steps: int = 400      # control steps (8 s at 50 Hz)
    settle_steps: int = 150       # control steps with the hand at rest before recording starts (cable falls onto the plane)

    @property
    def n_nodes(self) -> int:
        return self.n_segments + 1

    @property
    def substeps(self) -> int:
        return int(round(self.control_dt / self.timestep))


def sample_material(rng: np.random.Generator) -> dict:
    out = {}
    for k in MATERIAL_NAMES:
        lo, hi, kind = MATERIAL_RANGES[k]
        if kind == "log":
            out[k] = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        else:
            out[k] = float(rng.uniform(lo, hi))
    return out


def material_to_vec(mat: dict) -> np.ndarray:
    return np.array([mat[k] for k in MATERIAL_NAMES], dtype=np.float32)


def yaw_to_quat(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def cable_vertices(cfg: SimConfig, length: float | None = None) -> list[tuple[float, float, float]]:
    """Initial cable shape: straight horizontal line along the xy diagonal at the
    hand height; the cable falls onto the plane during the settle phase."""
    L, N = (length or cfg.cable_length), cfg.n_segments
    a = L / N / np.sqrt(2.0)
    return [(a * i, -a * i, 0.0) for i in range(N + 1)]


def build_xml(cfg: SimConfig, mat: dict) -> str:
    """MJCF of the hanging-cable model with the material parameters substituted.
    Physics of the base model (Euler, 2 ms, air drag, condim=1, connect
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


def _path(fam, g, rng, side):
    """Gripper path pieces in U coordinates (x along the span, y along the normal), speed range, pause range."""
    w = g["span"]
    xc = rng.uniform(-0.35, 0.35) * w
    line = lambda p, q: np.linspace(p, q, 600)
    if fam == "far":
        d0, d1 = rng.uniform(1.0, 1.5), rng.uniform(0.3, 0.6)
        t = np.tan(rng.uniform(-0.3, 0.3))
        return [line([xc - t * d0, -side * d0], [xc + t * d1, side * d1])], (0.4, 0.8), (0, 0)
    if fam == "near":
        d0, d1 = rng.uniform(0.08, 0.2), rng.uniform(0.3, 0.9)
        t = np.tan(rng.uniform(-0.5, 0.5))
        return [line([xc - t * d0, -side * d0], [xc + t * d1, side * d1])], (0.15, 0.8), (0, 0)
    if fam == "miss":
        xs = rng.choice([-1, 1]) * (0.5 * w + rng.uniform(0.25, 0.6))
        d0, d1 = rng.uniform(0.3, 0.7), rng.uniform(0.3, 0.9)
        t = np.tan(rng.uniform(-0.3, 0.3)) * (-np.sign(xs))          # never steers towards the U
        return [line([xs + t * d0, -side * d0], [xs - t * d1, side * d1])], (0.15, 0.8), (0, 0)
    if fam == "oblique":
        d0, d1 = rng.uniform(0.3, 0.6), rng.uniform(0.3, 0.6)
        t = np.tan(np.radians(rng.uniform(55, 72))) * rng.choice([-1, 1])
        return [line([xc - t * d0, -side * d0], [xc + t * d1, side * d1])], (0.3, 0.8), (0, 0)
    if fam == "multi":
        n = int(rng.integers(3, 5)); pts = [[xc, -side * rng.uniform(0.3, 0.6)]]
        for i in range(n):
            s = side if i % 2 == 0 else -side
            pts.append([rng.uniform(-0.35, 0.35) * w, s * rng.uniform(0.25, 0.6)])
        return [line(pts[i], pts[i + 1]) for i in range(n)], (0.5, 0.9), (0.2, 1.5)
    if fam == "random":
        # broad training scenes: random waypoints anywhere around the U, no designed crossing
        r = 0.5 * w + 0.3
        ang = rng.uniform(-np.pi, np.pi); d = rng.uniform(0.2, 1.6)
        pts = [[d * np.cos(ang), d * np.sin(ang)]]
        for _ in range(int(rng.integers(3, 7))):
            a2, d2 = rng.uniform(-np.pi, np.pi), r * np.sqrt(rng.uniform())
            pts.append([d2 * np.cos(a2), d2 * np.sin(a2)])
        return [line(pts[i], pts[i + 1]) for i in range(len(pts) - 1)], (0.15, 1.0), (0.0, 1.5)
    raise ValueError(fam)

SMOOTH = ["spline", "varspeed", "retrace", "curl", "weave"]     # further training motions (never circle, sine or jerk: those stay validation / test)
A_TAN, A_LAT = 4.0, 6.0        # [m/s^2] limits of the smooth families along and across the curve (the jerk motions reach 60 m/s^2)


def _smooth(fam, g, rng, cfg):
    """Gripper positions (T, 2) in U coordinates on a smooth random curve through 4-8 waypoints around the U.
    spline: one move at constant cruise speed.  varspeed: the speed changes smoothly along the curve, including stops.
    retrace: the gripper moves forward, turns back along the same curve for a part of the way and moves on.
    The speed is limited in tight bends and its change per step is limited, so that no abrupt motion occurs."""
    from scipy.interpolate import CubicSpline
    dt, T, r = cfg.control_dt, cfg.episode_steps, 0.5 * g["span"] + 0.3
    if fam == "curl":                    # loops around a point near the U with a radius that changes from waypoint to waypoint
        c, th, sg = np.array([rng.uniform(-0.3, 0.3) * g["span"], rng.uniform(-0.25, 0.25)]), rng.uniform(-np.pi, np.pi), rng.choice([-1, 1])
        pts = []
        for _ in range(int(rng.integers(6, 15))):
            rad = rng.uniform(0.1, 0.5); pts.append(c + rad * np.array([np.cos(th), np.sin(th)])); th += sg * np.radians(rng.uniform(40, 110))
    elif fam == "weave":                 # travel along a random direction with irregular swings to both sides
        c, th = np.array([rng.uniform(-0.3, 0.3) * g["span"], rng.uniform(-0.2, 0.2)]), rng.uniform(-np.pi, np.pi)
        e, nn = np.array([np.cos(th), np.sin(th)]), np.array([-np.sin(th), np.cos(th)])
        total = rng.uniform(0.8, 2.0); a, sg, pts = -rng.uniform(0.3, 0.7) * total, rng.choice([-1, 1]), []
        a0 = a
        while a < a0 + total:
            pts.append(c + a * e + sg * rng.uniform(0.03, 0.3) * nn); a += rng.uniform(0.1, 0.35); sg = -sg
    else:
        ang, d = rng.uniform(-np.pi, np.pi), rng.uniform(0.2, 1.6)
        pts = [[d * np.cos(ang), d * np.sin(ang)]]
        for _ in range(int(rng.integers(3, 8))):
            a2, d2 = rng.uniform(-np.pi, np.pi), r * np.sqrt(rng.uniform())
            pts.append([d2 * np.cos(a2), d2 * np.sin(a2)])
    pts = np.array(pts); chord = np.r_[0, np.cumsum(np.maximum(np.linalg.norm(np.diff(pts, axis=0), axis=1), 1e-3))]
    cs = CubicSpline(chord, pts, bc_type="clamped"); u = np.linspace(0, chord[-1], 6000)
    dense, d1, d2 = cs(u), cs(u, 1), cs(u, 2)
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))]; L = arc[-1]; ds = np.maximum(np.diff(arc), 1e-9)
    kappa = np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) / np.maximum(np.linalg.norm(d1, axis=1), 1e-6) ** 3
    vlat = np.clip(np.sqrt(A_LAT / np.maximum(kappa, 1e-6)), 0.03, 2.0)
    fwd, bwd = vlat.copy(), vlat.copy(); fwd[-1] = 0.0; bwd[0] = 0.0          # speed limits: brake before bends and before the ends
    for i in range(len(u) - 2, -1, -1): fwd[i] = min(fwd[i], np.sqrt(fwd[i + 1] ** 2 + 2 * A_TAN * ds[i]))
    for i in range(1, len(u)): bwd[i] = min(bwd[i], np.sqrt(bwd[i - 1] ** 2 + 2 * A_TAN * ds[i - 1]))
    t = np.arange(T) * dt; hold = rng.uniform(0.2, 1.0)
    if fam in ("spline", "curl", "weave"):
        want = np.full(T, rng.uniform(0.15, 1.0))
    else:                                                                    # smooth random profile between knots
        tk, vk, sign = [hold], [0.0], 1.0
        while tk[-1] < T * dt:
            if fam == "varspeed":
                tk.append(tk[-1] + rng.uniform(0.4, 1.2)); vk.append(0.0 if rng.uniform() < 0.2 else rng.uniform(0.1, 1.2))
            else:                                                            # forward 1-3 s, back 0.5-1.5 s, ...
                dur, v = rng.uniform(1.0, 3.0) if sign > 0 else rng.uniform(0.5, 1.5), sign * rng.uniform(0.2, 1.0)
                tk += [tk[-1] + 0.4, tk[-1] + 0.4 + dur]; vk += [v, v]; sign = -sign
        tk, vk = np.array(tk), np.array(vk); i = np.clip(np.searchsorted(tk, t, side="right") - 1, 0, len(tk) - 2)
        w = np.clip((t - tk[i]) / (tk[i + 1] - tk[i]), 0, 1); want = vk[i] + (vk[i + 1] - vk[i]) * 0.5 * (1 - np.cos(np.pi * w))
    want[t < hold] = 0.0
    s, v, out = 0.0, 0.0, np.zeros(T)
    for k in range(T):
        out[k] = s
        v += np.clip(want[k] - v, -A_TAN * dt, A_TAN * dt)
        v = np.clip(v, -np.interp(s, arc, bwd), np.interp(s, arc, fwd))
        s = float(np.clip(s + v * dt, 0.0, L))
    return np.column_stack([np.interp(out, arc, dense[:, 0]), np.interp(out, arc, dense[:, 1])])


def _burst(g, rng, cfg):
    """Short straight moves of 5-30 cm with pauses. Every move ramps up over 3-7 control steps (0.06-0.14 s);
    the held-out jerk motions reach their speed within a single step."""
    dt, T, r = cfg.control_dt, cfg.episode_steps, 0.5 * g["span"] + 0.3
    ang, d = rng.uniform(-np.pi, np.pi), rng.uniform(0.15, 0.7)
    cur = np.array([d * np.cos(ang), d * np.sin(ang)]); seq = [np.repeat(cur[None], int(round(rng.uniform(0.2, 1.0) / dt)), axis=0)]
    while sum(len(x) for x in seq) < T:
        for _ in range(20):
            a2 = rng.uniform(-np.pi, np.pi); nxt = cur + rng.uniform(0.05, 0.3) * np.array([np.cos(a2), np.sin(a2)])
            if np.linalg.norm(nxt) < r + 0.2: break
        seq.append(_follow(np.linspace(cur, nxt, 50), rng.uniform(0.4, 1.2), rng.uniform(0.06, 0.14), dt))
        seq.append(np.repeat(seq[-1][-1:], int(round(rng.uniform(0.04, 0.4) / dt)), axis=0)); cur = seq[-1][-1] if len(seq[-1]) else seq[-2][-1]
    return np.concatenate(seq)[:T]

MOTION = ["circle", "sine", "jerk"]


def _motion(fam, g, rng, side, cfg):
    """Gripper positions (T,2) in U coordinates for the motion families, sampled directly in time."""
    w, dt, n = g["span"], cfg.control_dt, cfg.episode_steps
    xc = rng.uniform(-0.3, 0.3) * w
    wait = int(round(rng.uniform(0.2, 0.8) / dt))
    t = np.clip(np.arange(n) - wait, 0, None) * dt
    ramp = rng.uniform(0.3, 0.6)
    tau = np.where(t < ramp, ramp * (t / ramp) ** 2 / 2 * 2 * 0.5, t - ramp / 2)      # time with a smooth start (speed ramps linearly)
    if fam == "circle":
        r, v = rng.uniform(0.15, 0.45), rng.uniform(0.3, 0.9)
        yc = -side * rng.uniform(0.0, 0.7) * r                                         # the circle cuts the plane of the U
        phi = rng.choice([-1, 1]) * v / r * tau
        return np.column_stack([xc + r * np.sin(phi), yc - side * r * np.cos(phi)])     # starts at the point farthest in front of the U
    if fam == "sine":
        f = rng.uniform(0.4, 1.2); a = min(rng.uniform(0.1, 0.3), 0.16 / f)            # peak sideways speed <= 1 m/s
        d0, v = rng.uniform(0.25, 0.5), rng.uniform(0.1, 0.3)
        grow = np.clip(t / (2 * ramp), 0, 1)
        return np.column_stack([xc + a * grow * np.sin(2 * np.pi * f * t), side * (-d0 + v * tau)])
    if fam == "jerk":
        p = np.array([xc, -side * rng.uniform(0.3, 0.6)]); seq = [np.repeat(p[None], wait, axis=0)]
        while sum(len(q) for q in seq) < n:
            ang = rng.uniform(-1.2, 1.2) + (np.pi if rng.uniform() < 0.25 else 0.0)    # mostly towards the U, sometimes back
            q = p + rng.uniform(0.05, 0.3) * np.array([np.sin(ang), side * np.cos(ang)])
            q[0] = np.clip(q[0], -0.5 * w - 0.2, 0.5 * w + 0.2); q[1] = np.clip(q[1], -0.8, 0.8)
            seq.append(_follow(np.linspace(p, q, 50), rng.uniform(0.5, 1.2), dt, dt))   # full speed within one control step
            seq.append(np.repeat(seq[-1][-1:], int(round(rng.uniform(0.04, 0.4) / dt)), axis=0)); p = seq[-1][-1]
        return np.concatenate(seq)[:n]
    raise ValueError(fam)


def sample(cfg, rng, fam):
    mats = [sample_material(rng), sample_material(rng)]
    g = u_geometry(cfg, mats[1], rng)
    H1 = float(rng.uniform(g["z_bottom"] + 0.05, max(mats[1]["height"] - 0.02, g["z_bottom"] + 0.06)))
    mats[0]["height"] = H1
    mats[0]["length"] = float(rng.uniform(max(0.6, H1 - g["z_bottom"] + 0.05), 2.0))
    side = float(rng.choice([-1.0, 1.0]))
    if fam in MOTION:
        local = _motion(fam, g, rng, side, cfg)
        xy = g["centre"] + local[:, 0, None] * g["e"] + local[:, 1, None] * g["n"]
        return mats, g, np.column_stack([xy, np.zeros(len(xy))]), -side * g["n"]
    if fam in SMOOTH or fam == "burst":
        local = _burst(g, rng, cfg) if fam == "burst" else _smooth(fam, g, rng, cfg)
        xy = g["centre"] + local[:, 0, None] * g["e"] + local[:, 1, None] * g["n"]
        v = xy[0] - g["centre"]                              # cable starts pointing away from the centre of the U
        return mats, g, np.column_stack([xy, np.zeros(len(xy))]), v / (np.linalg.norm(v) + 1e-9)
    pieces, sp, pause = _path(fam, g, rng, side)
    speed, ramp, dt = rng.uniform(*sp), rng.uniform(0.2, 0.8), cfg.control_dt
    seq = [np.repeat(pieces[0][:1], int(round(rng.uniform(0.2, 1.0) / dt)), axis=0)]
    for p in pieces:
        seq.append(_follow(p, speed, ramp, dt))
        seq.append(np.repeat(seq[-1][-1:], int(round(rng.uniform(*pause) / dt)), axis=0))
    local = np.concatenate(seq); n = cfg.episode_steps
    if len(local) < n:
        local = np.concatenate([local, np.repeat(local[-1:], n - len(local), axis=0)])
    local = local[:n]
    xy = g["centre"] + local[:, 0, None] * g["e"] + local[:, 1, None] * g["n"]
    away = -side * g["n"]
    if fam == "random":                                   # cable starts pointing away from the centre of the U
        v = xy[0] - g["centre"]; away = v / (np.linalg.norm(v) + 1e-9)
    return mats, g, np.column_stack([xy, np.zeros(n)]), away


KINDS = ["designed", "random", "spline", "varspeed", "retrace", "curl", "weave", "burst"]     # training kinds in the order of the sets A to F
HELD_OUT = ["circle", "sine", "jerk"]                                                          # validation and test only
SETS = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 8}                                        # set -> number of kinds
MAX_WINDING = 360.0                                                                            # [deg] more is a wrap around the U
CFG = SimConfig()
CFG.style = "mix"


def scene(job):
    seed, kind = job
    try:
        if kind == "designed":
            out = generate_episode_u(CFG, seed)
        else:
            mats, g, traj, away = sample(CFG, np.random.default_rng(seed), kind)
            out, _ = run_episode_u(CFG, mats, g, traj, away)
            out["mode"] = kind
        out["seed"] = seed; out["kind"] = kind
        return out
    except Exception as e:
        return dict(ok=False, seed=seed, kind=kind, error=str(e))


@torch.no_grad()
def winding(pos, near=0.03):
    """How far the pulled cable winds around the U [deg]. pos (2,T,N,3). Nodes of the pulled cable within `near` of the U:
    their position relative to the closest point on the U, projected onto the plane normal to the U, sweeps an angle along
    the run of consecutive close nodes. Lying over the U gives up to about 180 degrees, a wrap clearly more."""
    p = torch.as_tensor(np.asarray(pos), dtype=torch.float32); p0, p1 = p[0], p[1]
    q, arc, gap, i, ti = closest_on_cable(p0, p1)
    seg = p1[:, 1:] - p1[:, :-1]
    tan = torch.gather(seg, 1, i[..., None].expand(-1, -1, 3)); tan = tan / tan.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    close = gap < near
    r = p0 - q; r = r - (r * tan).sum(-1, keepdim=True) * tan
    a, b = r[:, :-1], r[:, 1:]
    d = torch.atan2((torch.cross(a, b, dim=-1) * tan[:, :-1]).sum(-1), (a * b).sum(-1)) * (close[:, :-1] & close[:, 1:])
    best = torch.zeros(d.shape[0]); run = torch.zeros_like(best)
    for k in range(d.shape[-1]):
        run = torch.where(d[:, k] != 0, run + d[:, k], torch.zeros_like(run)); best = torch.maximum(best, run.abs())
    return float(torch.rad2deg(best.max()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", choices=list(SETS), help="training set A to F")
    ap.add_argument("--kinds", help="comma-separated motion kinds instead of --set: " + ", ".join(KINDS))
    ap.add_argument("--held-out", action="store_true", help="validation / test scenes: circle, sine and jerk motions")
    ap.add_argument("--scenes", type=int, default=1000); ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1000000, help="use a different seed range for every file")
    ap.add_argument("--workers", type=int, default=os.cpu_count() // 2)
    ap.add_argument("--slim", action="store_true", help="do not store the floor forces (smaller training files)")
    args = ap.parse_args()
    kinds = HELD_OUT if args.held_out else args.kinds.split(",") if args.kinds else KINDS[:SETS[args.set or "A"]]
    assert all(k in KINDS + HELD_OUT for k in kinds), kinds
    assert args.held_out or not set(kinds) & set(HELD_OUT), "circle, sine and jerk motions are validation and test data only"
    per = args.scenes // len(kinds); got = {k: [] for k in kinds}; nxt = {k: args.seed + 1000000 * i for i, k in enumerate(kinds)}
    dropped = dict(unstable=0, wrap=0); t0 = time.time(); torch.set_num_threads(4)
    with Pool(args.workers) as pool:
        while any(len(v) < per for v in got.values()):
            jobs = []
            for k in kinds:
                n = int((per - len(got[k])) * 1.25) + 4 if len(got[k]) < per else 0
                jobs += [(nxt[k] + j, k) for j in range(n)]; nxt[k] += n
            for ep in pool.imap_unordered(scene, jobs, chunksize=1):
                if not ep["ok"]:
                    dropped["unstable"] += 1; continue
                ep["winding"] = winding(ep["pos"])
                if ep["winding"] > MAX_WINDING:
                    dropped["wrap"] += 1; continue
                got[ep["kind"]].append(ep)
            print({k: min(len(v), per) for k, v in got.items()}, f"{time.time() - t0:.0f}s", flush=True)
    eps = []
    for j in range(per):                                    # interleaved: every prefix of the file has the same shares
        eps += [sorted(got[k], key=lambda e: e["seed"])[j] for k in kinds]
    st = lambda key, dtype=None: np.stack([e[key] for e in eps]).astype(dtype) if dtype else np.stack([e[key] for e in eps])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out, pos=st("pos"), pose=st("pose"), action=st("action"), material=st("material"), force_cable=st("force", np.float16),
             **({} if args.slim else dict(force_floor=st("force_floor", np.float16))), n_contact=st("n_contact"), anchor_b=st("anchor_b"),
             mode=np.array([str(e["mode"]) for e in eps]), kind=np.array([e["kind"] for e in eps]), seed=np.array([e["seed"] for e in eps]),
             winding=np.array([e["winding"] for e in eps], np.float32), control_dt=CFG.control_dt)
    touch = np.mean([np.abs(e["force"][1]).sum() > 0 for e in eps])
    print(f"saved {args.out}: {len(eps)} scenes, {per} per kind {kinds} | dropped {dropped} | scenes with cable contact {100 * touch:.0f} % | {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
