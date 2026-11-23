"""DLO-Anything: configuration, token construction and the physics transformer.

Every cable is predicted by one PhysicsTransformer. Its input tokens are
    41 node tokens         positions and velocities of one cable node over the last K = 10 steps (0.2 s)
     6 material tokens     bending stiffness, torsional stiffness, damping, density, length, gripper height
    25 action tokens       planned gripper motion per predicted step (own gripper + gripper of the other cable)
     1 global token        boundary condition (cable fixed at both ends or not)
    41 rigid-contact tokens  closest point of the floor per node
    41 soft-contact tokens   closest point on the other cable per node
and it predicts H = 25 steps (0.5 s) in one pass. Longer forecasts chain passes; the contact tokens are rebuilt from
the predicted state before every pass (`predict`, `rollout`).

This file is imported by the other scripts and has no command line."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from torch.utils.data import Dataset
from simulate import SimConfig, normalize_material


@dataclass
class ModelConfig:
    context_len: int = 10      # past control steps given to the model
    horizon: int = 25          # future control steps per prediction (0.5 s)
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    dropout: float = 0.0
    n_nodes: int = 41             # cable nodes (SimConfig.n_segments + 1)
    n_material: int = 6
    contact_in: int = 0           # feature size of the soft-contact tokens (0 = no contact tokens), one per cable node
    rigid_in: int = 0             # feature size of the rigid-contact tokens (0 = none): one floor patch below every node


POS_SCALE = 1.0 / 0.5            # positions in hand frame are O(0.5 m) (1.3 m cable)
_cfg = SimConfig()
ACT_SCALE = np.array([1 / (_cfg.max_speed * _cfg.control_dt),
                      1 / (_cfg.max_speed * _cfg.control_dt),
                      1 / (_cfg.max_yaw_rate * _cfg.control_dt)], np.float32)
VEL_SCALE = 1 / (_cfg.max_speed * _cfg.control_dt)


def rot2(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]])


def action_material_tokens(pose, action, material, t0, H):
    """Action and material tokens of one cable for the prediction that starts after step t0.

    pose (T, 3), action (T, 3), material (6,); actions = action[t0 .. t0+H-1]
    Everything is expressed in the gripper frame at t0.
    """
    pose0 = pose[t0]
    R = rot2(pose0[2])
    act = np.array(action[t0:t0 + H], dtype=np.float32, copy=True)     # (H, 3)
    act[:, :2] = act[:, :2] @ R                                         # rotate dx,dy into frame
    fut_pose = pose[t0 + 1:t0 + H + 1]                                  # (H, 3)
    rel_xy = (fut_pose[:, :2] - pose0[:2]) @ R
    rel_yaw = fut_pose[:, 2] - pose0[2]
    act_feat = np.concatenate([act * ACT_SCALE, rel_xy * POS_SCALE,
                               np.sin(rel_yaw)[:, None], np.cos(rel_yaw)[:, None]], -1)  # (H, 7)

    return dict(
        act=torch.from_numpy(act_feat.astype(np.float32)),
        mat=torch.from_numpy(normalize_material(material)),
    )


class PhysicsTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig, node_in: int, act_in: int = 7, global_in: int = 3):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.node_proj = nn.Linear(node_in, d)
        if cfg.contact_in:
            self.contact_proj = nn.Linear(cfg.contact_in, d)
            self.contact_emb = nn.Parameter(torch.randn(cfg.n_nodes, d) * 0.02)   # index of the cable node
            self.contact_type = nn.Parameter(torch.randn(d) * 0.02)
            self.act_partner_proj = nn.Linear(act_in, d)
            nn.init.zeros_(self.act_partner_proj.weight); nn.init.zeros_(self.act_partner_proj.bias)
        if cfg.rigid_in:
            self.rigid_proj = nn.Linear(cfg.rigid_in, d)
            self.rigid_emb = nn.Parameter(torch.randn(cfg.n_nodes, d) * 0.02)     # index of the node above the patch
            self.rigid_type = nn.Parameter(torch.randn(d) * 0.02)
        self.node_emb = nn.Parameter(torch.randn(cfg.n_nodes, d) * 0.02)
        self.mat_proj = nn.Linear(1, d)
        self.mat_emb = nn.Parameter(torch.randn(cfg.n_material, d) * 0.02)
        self.act_proj = nn.Linear(act_in, d)
        self.time_emb = nn.Parameter(torch.randn(cfg.horizon, d) * 0.02)
        self.global_proj = nn.Linear(global_in, d)
        self.type_emb = nn.Parameter(torch.randn(4, d) * 0.02)  # node / material / action / global
        # inverse head and "material unknown" embedding: part of the architecture, but not trained or used by train.py
        self.mat_unknown = nn.Parameter(torch.randn(cfg.n_material, d) * 0.02)
        # inverse head: pooled node tokens + global token -> the 6 normalised material parameters
        self.inv_head = nn.Sequential(nn.Linear(2 * d, 256), nn.GELU(), nn.Linear(256, cfg.n_material))

        layer = nn.TransformerEncoderLayer(d, cfg.n_heads, 4 * d, cfg.dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, cfg.n_layers)
        self.norm = nn.LayerNorm(d)

        # factorised decoder: (time token k, node token i) -> displacement of node i at step k
        dd = 128
        self.dec_t = nn.Linear(d, dd)
        self.dec_n = nn.Linear(d, dd)
        self.dec_mlp = nn.Sequential(nn.Linear(3 * dd, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(),
                                     nn.Linear(256, 3))
        nn.init.zeros_(self.dec_mlp[-1].weight)
        nn.init.zeros_(self.dec_mlp[-1].bias)

    def forward(self, node, act, mat, cur, glob=None, mat_mask=None, return_inv=False, contact=None, act_partner=None, rigid=None):
        """node (B,N,F)  act (B,H,7)  mat (B,M)  cur (B,N,3) -> pred positions (B,H,N,3)

        mat_mask (B,) bool: samples whose material tokens are replaced by the "unknown" embedding.
        return_inv: also return the inverse head's material estimate (B,M) from the context.
        contact (B,P,cfg.contact_in): soft-contact tokens, one per cable node; None = no partner.
        act_partner (B,H,7): planned hand motion of the partner in the own frame.
        rigid (B,R,cfg.rigid_in): rigid-contact tokens, one per patch of a rigid surface (the floor).
        """
        B, N, _ = node.shape
        H = act.shape[1]
        if glob is None:
            glob = torch.zeros(B, 3, device=node.device, dtype=node.dtype)
        tn = self.node_proj(node) + self.node_emb[None] + self.type_emb[0]
        tm = self.mat_proj(mat[..., None]) + self.mat_emb[None] + self.type_emb[1]
        if mat_mask is not None:
            tm = torch.where(mat_mask[:, None, None], (self.mat_unknown + self.type_emb[1])[None].expand_as(tm), tm)
        ta = self.act_proj(act) + self.time_emb[None, :H] + self.type_emb[2]
        tg = (self.global_proj(glob) + self.type_emb[3])[:, None]
        toks = [tn, tm, ta]
        if self.cfg.contact_in and contact is not None:
            if act_partner is not None:
                toks[2] = ta + self.act_partner_proj(act_partner)
            tc = self.contact_proj(contact) + self.contact_emb[None, :contact.shape[1]] + self.contact_type
            toks.append(tc)
        if self.cfg.rigid_in and rigid is not None:
            toks.append(self.rigid_proj(rigid) + self.rigid_emb[None, :rigid.shape[1]] + self.rigid_type)
        x = torch.cat(toks + [tg], dim=1)
        x = self.norm(self.encoder(x))
        hn = x[:, :N]
        ha = x[:, N + mat.shape[1]:N + mat.shape[1] + H]
        a = self.dec_t(ha)[:, :, None, :].expand(B, H, N, -1)
        n = self.dec_n(hn)[:, None, :, :].expand(B, H, N, -1)
        an = a * n
        disp = self.dec_mlp(torch.cat([an, a, n], dim=-1))
        pred = cur[:, None] + disp
        if return_inv:
            inv = self.inv_head(torch.cat([hn.mean(1), x[:, -1]], dim=-1))
            return pred, inv
        return pred

    @torch.no_grad()
    def estimate_material(self, node, act, cur):
        """Direct material estimate (B,M, normalised) from the context alone (material masked)."""
        B = node.shape[0]
        mat = torch.zeros(B, self.cfg.n_material, device=node.device, dtype=node.dtype)
        return self(node, act, mat, cur, mat_mask=torch.ones(B, dtype=torch.bool, device=node.device), return_inv=True)[1]

    @staticmethod
    def n_params(m):
        return sum(p.numel() for p in m.parameters())


def to_frame(v, yaw):
    """World vectors (..., 3) -> frame rotated by yaw; yaw broadcasts against v[..., 0]."""
    c, s = torch.cos(yaw), torch.sin(yaw)
    return torch.stack([c * v[..., 0] + s * v[..., 1], -s * v[..., 0] + c * v[..., 1], v[..., 2]], -1)


def from_frame(v, yaw):
    c, s = torch.cos(yaw), torch.sin(yaw)
    return torch.stack([c * v[..., 0] - s * v[..., 1], s * v[..., 0] + c * v[..., 1], v[..., 2]], -1)


SOFT_SCALES = (0.02, 0.10, 0.40)       # [m] length scales of the soft contact activation
RIGID_SCALES = (0.005, 0.02, 0.10)     # [m] length scales of the proximity to a rigid surface
FLOOR_REST = 0.004                     # [m] height of a node centre that lies on the floor
FLOOR_FRICTION = 1.0                   # MuJoCo default sliding friction of the plane


def own_in(mcfg: ModelConfig) -> int:
    """Soft-contact tokens seen from the own cable: one token per own node (see own_contact_tokens)."""
    return 3 + 3 + mcfg.context_len + len(SOFT_SCALES) + 3 + 1 + mcfg.n_material + 1


def own_contact_tokens(ctx, pose0, mat, fixed):
    """Soft-contact tokens, one per node of the own cable: every own node observes its closest point on the other cable.
    Same layout as the rigid-contact tokens: contact point in the own frame (3), unit direction from the contact point to the
    node (3), distance over the context steps (K), proximity exp(-d / s) for s = 2, 10, 40 cm (3), velocity of the node
    relative to the contact point (3); then arc position of the contact point on the other cable (1), material of the other
    cable (M) and whether it is fixed at both ends (1).   ctx (B,2,K,N,3) world -> (B,2,N,own_in)."""
    B, C, K, N, _ = ctx.shape
    own, par = frames(ctx.float(), pose0.float())
    q, arc, gap, i, ti = closest_on_cable(own, par)                          # per context step: (B,2,K,N,...)
    q, arc, i, ti = q[:, :, -1], arc[:, :, -1], i[:, :, -1], ti[:, :, -1]
    now = own[:, :, -1]
    direction = (now - q) / gap[:, :, -1, :, None].clamp_min(1e-6)
    v_par = par[:, :, -1] - par[:, :, -2]
    take = lambda idx: v_par.gather(2, idx[..., None].expand(-1, -1, -1, 3))
    v_q = take(i) * (1 - ti[..., None]) + take(i + 1) * ti[..., None]        # velocity of the contact point
    prox = torch.stack([torch.exp(-gap[:, :, -1] / s) for s in SOFT_SCALES], -1)
    return torch.cat([q * POS_SCALE, direction, gap.permute(0, 1, 3, 2) * POS_SCALE, prox, ((now - own[:, :, -2]) - v_q) * VEL_SCALE,
                      (2 * arc - 1)[..., None], mat.flip(1)[:, :, None].expand(-1, -1, N, -1),
                      fixed.flip(1)[:, :, None, None].expand(-1, -1, N, 1).to(own.dtype)], -1)


def rigid_in(mcfg: ModelConfig) -> int:
    return 3 + 3 + mcfg.context_len + len(RIGID_SCALES) + 3 + 1


def rigid_tokens(ctx, pose0):
    """Rigid-contact tokens for the floor: one patch below every node of each cable, in the cable's own frame.
    patch position (3), surface normal (3), distance of the node to the surface over the context steps (K),
    proximity exp(-d / s) for s = 0.5, 2, 10 cm (3), velocity of the node relative to the surface (3), friction (1).
    ctx (B,2,K,N,3) world -> (B,2,N,rigid_in)."""
    own, _ = frames(ctx.float(), pose0.float())
    now = own[:, :, -1]
    patch = now * torch.tensor([1.0, 1.0, 0.0], device=ctx.device)
    normal = torch.zeros_like(now); normal[..., 2] = 1.0
    d = (own[..., 2] - FLOOR_REST).permute(0, 1, 3, 2)                      # (B,2,N,K)
    prox = torch.stack([torch.exp(-d[..., -1].clamp_min(0) / s) for s in RIGID_SCALES], -1)
    vel = now - own[:, :, -2]
    return torch.cat([patch * POS_SCALE, normal, d * POS_SCALE, prox, vel * VEL_SCALE, torch.full_like(now[..., :1], FLOOR_FRICTION)], -1)


def closest_on_cable(p, cable):
    """p (..., P, 3) points, cable (..., N, 3) polyline -> closest point on the polyline per p, its arc
    position in [0, 1], gap, and (segment index, segment parameter) for interpolating other node values."""
    a, b = cable[..., None, :-1, :], cable[..., None, 1:, :]              # (..., 1, N-1, 3)
    ab = b - a
    t = (((p[..., :, None, :] - a) * ab).sum(-1) / (ab * ab).sum(-1).clamp_min(1e-10)).clamp(0, 1)   # (..., P, N-1)
    q = a + t[..., None] * ab
    d = (q - p[..., :, None, :]).norm(dim=-1)
    gap, i = d.min(-1)                                                     # (..., P)
    ti = t.gather(-1, i[..., None])[..., 0]
    qi = q.gather(-2, i[..., None, None].expand(*i.shape, 1, 3))[..., 0, :]
    return qi, (i + ti) / (cable.shape[-2] - 1), gap, i, ti


def frames(ctx, pose0):
    """ctx (B,2,K,N,3) world -> own cable and partner cable, both in the own frame of each cable."""
    t = torch.cat([pose0[..., :2], torch.zeros_like(pose0[..., :1])], -1)[:, :, None, None]
    yaw = pose0[..., 2][:, :, None, None]
    return to_frame(ctx - t, yaw), to_frame(ctx.flip(1) - t, yaw)


def partner_actions(pose0, pose_fut):
    """Planned hand motion of the partner in the own frame, same 7 features as the own action tokens."""
    p0, pf = pose0.flip(1), pose_fut.flip(1)                                # partner (B,2,3), (B,2,H,3)
    yaw = pose0[..., 2][:, :, None]
    step = torch.diff(pf, dim=2, prepend=p0[:, :, None])
    z = torch.zeros_like(step[..., :1])
    d_xy = to_frame(torch.cat([step[..., :2], z], -1), yaw)[..., :2]
    rel = to_frame(torch.cat([pf[..., :2] - p0[:, :, None, :2], z], -1), yaw)[..., :2]
    ryaw = pf[..., 2] - p0[:, :, None, 2]
    sc = torch.as_tensor(ACT_SCALE, device=pose0.device)
    return torch.cat([d_xy * sc[:2], step[..., 2:] * sc[2], rel * POS_SCALE, torch.sin(ryaw)[..., None], torch.cos(ryaw)[..., None]], -1)


def node_inputs(ctx, pose0):
    own, _ = frames(ctx.float(), pose0.float())
    vel = torch.diff(own, dim=2, prepend=own[:, :, :1])
    B, C, K, N, _ = own.shape
    node = torch.cat([own * POS_SCALE, vel * VEL_SCALE], -1).permute(0, 1, 3, 2, 4).reshape(B, C, N, K * 6)
    return node, own[:, :, -1] * POS_SCALE


def predict(models, ctx, b, use_contact=True, cables=(0, 1), amp=None):
    """models: (model of cable 0, model of cable 1). ctx (B,2,K,N,3) world; b: act, mat, pose0, pose_fut, fixed
    -> world positions (B,len(cables),H,N,3) of the requested cables."""
    node, cur = node_inputs(ctx, b["pose0"])
    glob = torch.zeros(*b["fixed"].shape, 3, device=ctx.device); glob[..., 0] = b["fixed"]
    tok = actp = None
    if use_contact and any(models[c].cfg.contact_in for c in cables):
        tok = own_contact_tokens(ctx, b["pose0"], b["mat"], b["fixed"]); actp = partner_actions(b["pose0"], b["pose_fut"])
    out, rig = [], None
    for c in cables:
        kw = dict(contact=tok[:, c], act_partner=actp[:, c]) if tok is not None and models[c].cfg.contact_in else {}
        if models[c].cfg.rigid_in and use_contact:
            rig = rigid_tokens(ctx, b["pose0"]) if rig is None else rig
            kw["rigid"] = rig[:, c]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=models[c].cfg.horizon >= 100 if amp is None else amp):   # short horizons need float32
            y = models[c](node[:, c], b["act"][:, c], b["mat"][:, c], cur[:, c], glob=glob[:, c], **kw)
        y = y.float() / POS_SCALE
        t = torch.cat([b["pose0"][:, c, :2], torch.zeros_like(b["pose0"][:, c, :1])], -1)[:, None, None]
        y = from_frame(y, b["pose0"][:, c, 2][:, None, None]) + t
        out.append(y)
    return torch.stack(out, 1)


class SceneWindows(Dataset):
    """Chains of W windows (stride H) of both cables of the U scenario; W = 1 is the plain window."""

    def __init__(self, path, mcfg, spe=8, seed=0, contact_frac=0.5, n_windows=1, subset=0):
        d = np.load(path)
        n = subset or len(d["mode"])
        self.pos, self.pose, self.action = d["pos"][:n], d["pose"][:n], d["action"][:n]
        self.material = d["material"][:n]
        fc = d["force_cable"][:n]
        self.touch = np.abs(fc).sum((1, 3, 4)) > 0                           # (E,T) cable-cable contact
        self.mode = d["mode"][:n]
        self.K, self.H, self.W = mcfg.context_len, mcfg.horizon, n_windows
        self.E, self.T = self.pos.shape[0], self.pos.shape[2]
        self.spe, self.frac = spe, contact_frac
        self.rng = np.random.default_rng(seed)
        self.hi = self.T - 1 - self.H * n_windows                           # last possible first t0
        t0s = np.arange(self.K - 1, self.hi + 1)
        cs = np.cumsum(np.pad(self.touch, ((0, 0), (1, 0))), axis=1)
        n_in = cs[:, t0s + self.H * n_windows + 1] - cs[:, t0s + 1]
        self.contact_t0 = [t0s[n_in[e] > 0] for e in range(self.E)]

    def __len__(self):
        return self.E * self.spe

    def item(self, e, t0):
        ss = [action_material_tokens(self.pose[e, c], self.action[e, c], self.material[e, c], t0, self.H) for c in (0, 1)]
        sl = slice(t0 + 1, t0 + self.H + 1)
        return dict(act=torch.stack([s["act"] for s in ss]), mat=torch.stack([s["mat"] for s in ss]),
                    pose0=torch.from_numpy(self.pose[e, :, t0]), pose_fut=torch.from_numpy(self.pose[e, :, sl]),
                    gt=torch.from_numpy(self.pos[e, :, sl]), fixed=torch.tensor([0.0, 1.0]))

    def __getitem__(self, idx):
        e = idx // self.spe
        ct = self.contact_t0[e]
        if len(ct) and self.rng.uniform() < self.frac:
            t0 = int(ct[self.rng.integers(len(ct))])
        else:
            t0 = int(self.rng.integers(self.K - 1, self.hi + 1))
        its = [self.item(e, t0 + w * self.H) for w in range(self.W)]
        out = {k: torch.stack([it[k] for it in its]) for k in its[0]}
        out["ctx"] = torch.from_numpy(self.pos[e, :, t0 - self.K + 1:t0 + 1])
        return out


def load(path, device):
    ck = torch.load(path, map_location=device)
    mcfg = ModelConfig(**{k: v for k, v in ck["cfg"].items() if k in ModelConfig.__dataclass_fields__})
    model = PhysicsTransformer(mcfg, node_in=mcfg.context_len * 6).to(device)
    model.load_state_dict(ck["model"]); model.eval()
    return model, mcfg


def window_starts(K, H, T):
    t0s = list(range(K - 1, T - H, H))
    if t0s[-1] + H < T - 1:
        t0s.append(T - 1 - H)
    return t0s


@torch.no_grad()
def rollout(models, ds, episodes, device, use_contact=True, chain=True, batch=48):
    """Forecast over whole episodes -> (E,2,T,N,3) metres, NaN before the first predicted step.
    chain: every window starts from the model's own prediction of the window before."""
    K, H, T = ds.K, ds.H, ds.T
    out = np.full((len(episodes),) + ds.pos.shape[1:], np.nan, np.float32)
    for i in range(0, len(episodes), batch):
        es = episodes[i:i + batch]
        track = torch.from_numpy(np.stack([ds.pos[e] for e in es])).to(device)       # (B,2,T,N,3)
        true = track.clone()
        done = torch.zeros(T, dtype=torch.bool)
        for t0 in window_starts(K, H, T):
            its = [ds.item(e, t0) for e in es]
            b = {k: torch.stack([it[k] for it in its]).to(device) for k in its[0]}
            src = track if chain else true
            p = predict(models, src[:, :, t0 - K + 1:t0 + 1], b, use_contact)
            sl = slice(t0 + 1, t0 + H + 1); new = ~done[sl]; new_d = new.to(device)
            seg = track[:, :, sl]; seg[:, :, new_d] = p[:, :, new_d]
            out[i:i + len(es), :, sl][:, :, new.numpy()] = p[:, :, new_d].cpu().numpy()
            done[sl] = True
    return out
