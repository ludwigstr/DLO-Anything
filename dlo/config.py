"""Central configuration shared by simulation, data generation, training and video rendering."""
from dataclasses import dataclass, field, asdict
import numpy as np


@dataclass
class SimConfig:
    # Cable geometry
    n_segments: int = 40          # cable segments (user model had 10; doubled twice on request)
    cable_length: float = 1.3153  # [m] (user model: 10 segments of 0.093*sqrt(2) m)
    cable_radius: float = 0.003   # [m]
    gripper_height: float = 0.5   # [m] fixed height of the hand above the plane
    # Integration
    timestep: float = 0.002       # physics step [s] (user model)
    integrator: str = "Euler"     # MuJoCo integrator (user model: default Euler)
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
    # range, move lengths, ramps, pauses, turn angle) are themselves sampled from the ranges below
    # (v4, "mehr Diversität"); the fixed v3 values were speed 0.4..1.0, long moves >= 1.2 m,
    # ramp 0.6 s, pause <= 0.6 s / diverse moves 0.1..0.4 m, ramp 0.15 s, pause <= 0.15 s, turn 150°.
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


# Material parameter ranges (sampled log-uniformly for stiffness/density, uniformly otherwise).
# Names are the order used in the data arrays and in the model's material tokens.
MATERIAL_NAMES = ["bend", "twist", "damping", "density", "length", "height"]
MATERIAL_DEFAULTS = {"length": 1.3153, "height": 0.5}   # values of the v3 datasets (4 material columns)
MATERIAL_RANGES = {
    # v4 (2026-10-28): ranges widened by ~1 decade in each direction ("mehr Diversität"); the v3
    # ranges were bend 1.5e4..5e5, twist 2.5e3..1e5, damping 0.0025..0.025, density 500..2000
    "bend":     (5e3, 5e6, "log"),    # cable plugin `bend` [Pa]; floppy rope .. visibly stiff. User model had 6e7
    "twist":    (1e3, 1e6, "log"),    # cable plugin `twist` [Pa] (user model: bend/6)
    "damping":  (0.001, 0.08, "log"), # joint damping [N m s/rad]; main lever for the "stiff / viscous" look (user model 0.15)
    "density":  (300.0, 4000.0, "log"),  # capsule density [kg/m^3]; user model: default 1000
    "length":   (0.6, 2.0, "lin"),    # cable length [m] (user model 1.3153); 40 segments always
    "height":   (0.2, 0.9, "lin"),    # hand height above the plane [m] (user model 0.5)
}


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


def pad_material(vec: np.ndarray) -> np.ndarray:
    """Datasets written before length/height were parameters have 4 material columns; append
    the fixed values they were created with."""
    vec = np.asarray(vec, dtype=np.float32)
    if vec.shape[-1] == len(MATERIAL_NAMES):
        return vec
    extra = np.array([MATERIAL_DEFAULTS[k] for k in MATERIAL_NAMES[vec.shape[-1]:]], np.float32)
    return np.concatenate([vec, np.broadcast_to(extra, vec.shape[:-1] + extra.shape)], axis=-1)


def material_dict(vec) -> dict:
    return dict(zip(MATERIAL_NAMES, map(float, pad_material(np.asarray(vec)))))


def normalize_material(vec: np.ndarray) -> np.ndarray:
    """Map raw material values to roughly [-1, 1] (log scale for log-sampled params)."""
    vec = np.asarray(vec, dtype=np.float64)
    out = np.empty_like(vec)
    for i, k in enumerate(MATERIAL_NAMES):
        lo, hi, kind = MATERIAL_RANGES[k]
        if kind == "log":
            out[..., i] = 2 * (np.log(vec[..., i]) - np.log(lo)) / (np.log(hi) - np.log(lo)) - 1
        else:
            out[..., i] = 2 * (vec[..., i] - lo) / (hi - lo) - 1
    return out.astype(np.float32)


@dataclass
class ModelConfig:
    context_len: int = 10      # past control steps given to the model
    horizon: int = 100         # future control steps to predict
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    dropout: float = 0.0
    n_nodes: int = 41             # cable nodes (SimConfig.n_segments + 1)
    n_material: int = 6
    force: bool = False           # external per-node contact-force channel (context + future forces)
    contact_in: int = 0           # feature size of the soft-contact tokens (0 = no contact tokens), see soft_contact.py
    contact_fut_in: int = 0       # per-step contact features over the forecast horizon (0 = none), see soft_contact_iter.py
    rigid_in: int = 0             # feature size of the rigid-contact tokens (0 = none): one floor patch below every node
    residual_vel: bool = False    # one-step model: the network output is added to the constant-velocity extrapolation


def asdict_safe(obj):
    return asdict(obj)
