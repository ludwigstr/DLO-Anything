"""Windowed dataset + gripper-frame normalisation shared by training and inference."""
from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset

from .config import SimConfig, ModelConfig, normalize_material, pad_material

POS_SCALE = 1.0 / 0.5            # positions in hand frame are O(0.5 m) (1.3 m cable)
_cfg = SimConfig()
ACT_SCALE = np.array([1 / (_cfg.max_speed * _cfg.control_dt),
                      1 / (_cfg.max_speed * _cfg.control_dt),
                      1 / (_cfg.max_yaw_rate * _cfg.control_dt)], np.float32)
VEL_SCALE = 1 / (_cfg.max_speed * _cfg.control_dt)
FORCE_SCALE = 1.0 / 0.05          # contact forces: median ~30 mN, p95 ~150 mN -> O(1); clipped at +-20
FORCE_CLIP = 20.0


def rot2(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]])


def to_gripper_frame(pos, pose0):
    """pos (..., 3) world -> frame of gripper pose0 = (x, y, yaw). z unchanged."""
    out = np.array(pos, dtype=np.float32, copy=True)
    out[..., :2] = (out[..., :2] - pose0[:2]) @ rot2(pose0[2])   # == R^T @ (p - t)
    return out


def from_gripper_frame(pos_f, pose0):
    out = np.array(pos_f, dtype=np.float32, copy=True)
    out[..., :2] = out[..., :2] @ rot2(pose0[2]).T + pose0[:2]
    return out


def force_to_frame(f, pose0):
    """External forces (..., 3) world -> gripper frame (rotation only), scaled and clipped."""
    out = np.array(f, dtype=np.float32, copy=True)
    out[..., :2] = out[..., :2] @ rot2(pose0[2])
    return np.clip(out * FORCE_SCALE, -FORCE_CLIP, FORCE_CLIP)


def make_sample(pos, pose, action, material, t0, K, H, force=None):
    """Build one model input/target from an episode.

    pos (T, N, 3), pose (T, 3), action (T, 3), material (6,), force (T, N, 3) [N] or None
    context = steps t0-K+1 .. t0, target = steps t0+1 .. t0+H, actions = action[t0 .. t0+H-1]
    Everything is expressed in the gripper frame at t0. With `force`, the sample also carries
    force_ctx (N, K*3): forces on each node over the context steps, and force_fut (H, N, 3):
    forces over the target steps (teacher forcing of the contact channel); without it both are 0.
    """
    pose0 = pose[t0]
    ctx = to_gripper_frame(pos[t0 - K + 1:t0 + 1], pose0)            # (K, N, 3)
    vel = np.diff(ctx, axis=0, prepend=ctx[:1])                        # (K, N, 3)
    node_feat = np.concatenate([ctx * POS_SCALE, vel * VEL_SCALE], -1)  # (K, N, 6)
    node_feat = node_feat.transpose(1, 0, 2).reshape(pos.shape[1], K * 6)

    R = rot2(pose0[2])
    act = np.array(action[t0:t0 + H], dtype=np.float32, copy=True)     # (H, 3)
    act[:, :2] = act[:, :2] @ R                                         # rotate dx,dy into frame
    fut_pose = pose[t0 + 1:t0 + H + 1]                                  # (H, 3)
    rel_xy = (fut_pose[:, :2] - pose0[:2]) @ R
    rel_yaw = fut_pose[:, 2] - pose0[2]
    act_feat = np.concatenate([act * ACT_SCALE, rel_xy * POS_SCALE,
                               np.sin(rel_yaw)[:, None], np.cos(rel_yaw)[:, None]], -1)  # (H, 7)

    target = to_gripper_frame(pos[t0 + 1:t0 + H + 1], pose0) * POS_SCALE   # (H, N, 3)
    cur = ctx[-1] * POS_SCALE                                              # (N, 3)
    N = pos.shape[1]
    if force is None:
        f_ctx = np.zeros((N, K * 3), np.float32)
        f_fut = np.zeros((H, N, 3), np.float32)
    else:
        f_ctx = force_to_frame(force[t0 - K + 1:t0 + 1], pose0).transpose(1, 0, 2).reshape(N, K * 3)
        f_fut = force_to_frame(force[t0 + 1:t0 + H + 1], pose0)
    return dict(
        node=torch.from_numpy(node_feat.astype(np.float32)),
        act=torch.from_numpy(act_feat.astype(np.float32)),
        mat=torch.from_numpy(normalize_material(material)),
        cur=torch.from_numpy(cur.astype(np.float32)),
        target=torch.from_numpy(target.astype(np.float32)),
        force_ctx=torch.from_numpy(f_ctx.astype(np.float32)),
        force_fut=torch.from_numpy(f_fut.astype(np.float32)),
    )


class WindowDataset(Dataset):
    def __init__(self, path: str, mcfg: ModelConfig, samples_per_episode: int = 32, seed: int = 0, subset: int = 0):
        """`path` may be a comma-separated list of .npz files; episodes are concatenated. `subset` keeps
        only the first N episodes of that concatenation (nested subsets for the scaling study)."""
        parts, n = [], 0
        for f in path.split(","):
            d = np.load(f)
            take = len(d["material"]) if not subset else min(len(d["material"]), subset - n)
            parts.append({k: d[k][:take] for k in ("pos", "pose", "action", "material")}); n += take
            if subset and n >= subset:
                break
        assert not subset or n == subset, f"only {n} episodes available, {subset} requested"
        self.pos, self.pose, self.action = (np.concatenate([d[k] for d in parts]) for k in ("pos", "pose", "action"))
        self.material = np.concatenate([pad_material(d["material"]) for d in parts])
        self.K, self.H = mcfg.context_len, mcfg.horizon
        self.E, self.T = self.pos.shape[:2]
        self.spe = samples_per_episode
        self.rng = np.random.default_rng(seed)
        assert self.T >= self.K + self.H

    def __len__(self):
        return self.E * self.spe

    def __getitem__(self, idx):
        e = idx // self.spe
        t0 = int(self.rng.integers(self.K - 1, self.T - self.H))
        return make_sample(self.pos[e], self.pose[e], self.action[e], self.material[e], t0, self.K, self.H)


class TwoCableWindowDataset(Dataset):
    """Windows from two-cable episodes: every cable of every episode is one record with its own
    contact-force arrays. `contact_frac` of the windows are drawn so that the horizon contains
    contact (the other windows are drawn uniformly), because only ~10-20 % of the steps are in
    contact. `paths` may be comma-separated."""

    def __init__(self, path: str, mcfg: ModelConfig, samples_per_episode: int = 32, seed: int = 0,
                 contact_frac: float = 0.5, cables=(0, 1)):
        parts = [np.load(f) for f in path.split(",")]
        cat = lambda k: np.concatenate([d[k] for d in parts])
        pos, pose, action, material, force = cat("pos"), cat("pose"), cat("action"), cat("material"), cat("force")
        E = pos.shape[0]
        self.episode = np.repeat(np.arange(E), len(cables))                      # record -> episode
        self.cable = np.tile(np.array(cables), E)                               # record -> cable
        self.pos = pos[self.episode, self.cable]                                 # (R, T, N, 3)
        self.pose, self.action = pose[self.episode, self.cable], action[self.episode, self.cable]
        self.material = pad_material(material[self.episode, self.cable])
        self.force = force[self.episode, self.cable]                             # (R, T, N, 3) float16
        # contact sampling follows cable-cable contact; u2 data also carry floor forces in `force`
        fc = cat("force_cable")[self.episode, self.cable] if all("force_cable" in d for d in parts) else self.force
        self.touch = (np.abs(fc).sum(-1).sum(-1) > 0)                           # (R, T) node in contact at step t
        self.K, self.H = mcfg.context_len, mcfg.horizon
        self.R, self.T = self.pos.shape[:2]
        self.spe = samples_per_episode
        self.contact_frac = contact_frac
        self.rng = np.random.default_rng(seed)
        # per record: the t0 values whose horizon (t0+1 .. t0+H) contains at least one contact step
        cs = np.cumsum(np.pad(self.touch, ((0, 0), (1, 0))), axis=1)             # (R, T+1)
        t0s = np.arange(self.K - 1, self.T - self.H)
        n_in = cs[:, t0s + self.H + 1] - cs[:, t0s + 1]                          # contact steps inside the horizon
        self.contact_t0 = [t0s[n_in[r] > 0] for r in range(self.R)]
        self.frac_touch = float(self.touch.mean())

    def __len__(self):
        return self.R * self.spe

    def __getitem__(self, idx):
        r = idx // self.spe
        ct = self.contact_t0[r]
        if len(ct) and self.rng.uniform() < self.contact_frac:
            t0 = int(ct[self.rng.integers(len(ct))])
        else:
            t0 = int(self.rng.integers(self.K - 1, self.T - self.H))
        return make_sample(self.pos[r], self.pose[r], self.action[r], self.material[r], t0, self.K, self.H,
                           force=self.force[r].astype(np.float32))
