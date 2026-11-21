"""Contact correction on top of two frozen DLO-Anything predictions.

Both cables are predicted independently. The contact point follows from the closest approach of
the two predictions. Only around that point, and only once contact is expected, a small network
adds a learned correction to the prediction. Everywhere else the DLO-Anything prediction stays
exactly as it is.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dataset import POS_SCALE, VEL_SCALE

GATE = 0.03          # [m] predicted gap below which a contact is assumed
K_NODES = 12         # correction reaches this many nodes to each side of the contact point
BACK = 15            # [steps] correction may start this early (predicted contact time is uncertain)
N_FEATURES = 39


def to_frame(v, yaw):
    """World vectors (..., 3) -> frame rotated by yaw; yaw broadcasts against v[..., 0]."""
    c, s = torch.cos(yaw), torch.sin(yaw)
    return torch.stack([c * v[..., 0] + s * v[..., 1], -s * v[..., 0] + c * v[..., 1], v[..., 2]], -1)


def from_frame(v, yaw):
    c, s = torch.cos(yaw), torch.sin(yaw)
    return torch.stack([c * v[..., 0] - s * v[..., 1], s * v[..., 0] + c * v[..., 1], v[..., 2]], -1)


def seg_seg(P0, P1, Q0, Q1, eps=1e-12):
    """Closest points between segments, broadcasting; returns s, t, point on P, point on Q, distance."""
    d1, d2, r = P1 - P0, Q1 - Q0, P0 - Q0
    a, e = (d1 * d1).sum(-1), (d2 * d2).sum(-1)
    f, c, b = (d2 * r).sum(-1), (d1 * r).sum(-1), (d1 * d2).sum(-1)
    den = a * e - b * b
    s = torch.where(den > eps, ((b * f - c * e) / den.clamp_min(eps)).clamp(0, 1), torch.zeros_like(den))
    t = (b * s + f) / e.clamp_min(eps)
    s = torch.where(t < 0, (-c / a.clamp_min(eps)).clamp(0, 1), torch.where(t > 1, ((b - c) / a.clamp_min(eps)).clamp(0, 1), s))
    t = t.clamp(0, 1)
    Pc, Qc = P0 + s[..., None] * d1, Q0 + t[..., None] * d2
    return s, t, Pc, Qc, (Pc - Qc).norm(dim=-1)


def closest_contact(A, B):
    """A, B (..., N, 3) -> distance, continuous node index on A and on B, point on A, point on B."""
    s, t, Pc, Qc, dist = seg_seg(A[..., :-1, None, :], A[..., 1:, None, :], B[..., None, :-1, :], B[..., None, 1:, :])
    S = dist.shape[-1]
    fl = dist.flatten(-2)
    d, idx = fl.min(-1)
    i = idx // S; j = idx % S
    take = lambda x: x.flatten(-2).gather(-1, idx[..., None])[..., 0]
    take3 = lambda x: x.flatten(-3, -2).gather(-2, idx[..., None, None].expand(*idx.shape, 1, 3))[..., 0, :]
    return d, i + take(s), j + take(t), take3(Pc), take3(Qc)


def contact_reference(base, q0, gate=GATE, back=BACK):
    """Contact state per step. base (B,2,H,N,3), q0 (B,2,N,3) observed positions at t0.

    Once a contact was predicted, the last contact point stays the reference because the
    deformation outlives the touch. Up to `back` steps before a predicted contact, the coming
    contact point is the reference. A contact observed at t0 counts from the first step on.
    Returns ref (B,2,H) node index, has_ref (B,H), active (B,H), dmin (B,H), normal (B,H,3) from
    cable 2 to cable 1, point (B,H,3)."""
    Bn, _, H, N, _ = base.shape
    dmin, ni, nj, Pc, Qc = closest_contact(base[:, 0], base[:, 1])
    d0, ni0, nj0, _, _ = closest_contact(q0[:, 0], q0[:, 1])
    active = dmin < gate
    normal = (Pc - Qc) / dmin[..., None].clamp_min(1e-6)
    nodes = torch.stack([ni, nj], 1)                                     # (B,2,H)
    ref = torch.zeros_like(nodes); has = torch.zeros((Bn, H), dtype=torch.bool, device=base.device)
    last = torch.stack([ni0, nj0], 1); have = d0 < gate
    for h in range(H):
        a = active[:, h]
        last = torch.where(a[:, None], nodes[:, :, h], last); have = have | a
        ref[:, :, h] = last; has[:, h] = have
    nxt = nodes[:, :, -1].clone(); nxt_h = torch.where(active[:, -1], torch.full((Bn,), H - 1, device=base.device), torch.full((Bn,), 10 ** 6, device=base.device))
    for h in range(H - 1, -1, -1):
        a = active[:, h]
        nxt = torch.where(a[:, None], nodes[:, :, h], nxt)
        nxt_h = torch.where(a, torch.full((Bn,), h, device=base.device), nxt_h)
        early = ~has[:, h] & (nxt_h - h <= back)
        ref[:, :, h] = torch.where(early[:, None], nxt, ref[:, :, h]); has[:, h] = has[:, h] | early
    return dict(ref=ref, has=has, active=active, dmin=dmin, normal=normal, point=0.5 * (Pc + Qc))


def node_weight(ref, has, N, k=K_NODES):
    """(B,2,H,N): 1 near the contact point, linear falloff to 0 at k nodes distance, 0 without contact."""
    n = torch.arange(N, device=ref.device, dtype=ref.dtype)
    w = ((k + 1 - (n - ref[..., None]).abs()) / (k / 2)).clamp(0, 1)
    return w * has[:, None, :, None]


def features(base, q0, pose0, mat, geo, w):
    """Network input for both cables as a batch of grids (2B, C, H, N). Cable c sees its own hand frame."""
    Bn, _, H, N, _ = base.shape
    out = []
    n_idx = torch.arange(N, device=base.device, dtype=base.dtype)
    for c in (0, 1):
        o = 1 - c
        yaw = pose0[:, c, 2][:, None, None]
        t = torch.cat([pose0[:, c, :2], torch.zeros(Bn, 1, device=base.device)], -1)[:, None, None]
        P, Q = base[:, c], base[:, o]
        Pv = P - torch.cat([q0[:, c][:, None], P[:, :-1]], 1)
        Qv = Q - torch.cat([q0[:, o][:, None], Q[:, :-1]], 1)
        d2 = ((P[:, :, :, None] - Q[:, :, None]) ** 2).sum(-1)
        dn, idx = d2.min(-1)
        dn = dn.sqrt()
        ix = idx[..., None].expand(-1, -1, -1, 3)
        r = Q.gather(2, ix) - P
        dv = Qv.gather(2, ix) - Pv
        d20 = ((q0[:, c][:, :, None] - q0[:, o][:, None]) ** 2).sum(-1)
        dn0, idx0 = d20.min(-1)
        r0 = q0[:, o].gather(1, idx0[..., None].expand(-1, -1, 3)) - q0[:, c]
        sgn = 1.0 if c == 0 else -1.0
        nrm = to_frame(sgn * geo["normal"], yaw[:, 0])[:, :, None].expand(-1, -1, N, -1)
        h_grid = (torch.arange(H, device=base.device, dtype=base.dtype) / H)[None, :, None].expand(Bn, H, N)
        n_grid = (n_idx / N)[None, None].expand(Bn, H, N)
        arc = (n_idx - geo["ref"][:, c][..., None]) / N * geo["has"][..., None]
        chans = [to_frame(P - t, yaw) * POS_SCALE, to_frame(Pv, yaw) * VEL_SCALE,
                 (to_frame(r, yaw) / 0.05).clamp(-4, 4), torch.exp(-dn / 0.02)[..., None],
                 to_frame(dv, yaw) * VEL_SCALE,
                 w[:, c][..., None], arc[..., None], geo["has"][..., None].expand(Bn, H, N)[..., None].to(base.dtype),
                 torch.exp(-geo["dmin"] / 0.02)[..., None].expand(Bn, H, N)[..., None], nrm,
                 geo["active"][..., None].expand(Bn, H, N)[..., None].to(base.dtype),
                 h_grid[..., None], n_grid[..., None],
                 (to_frame(r0, yaw[:, 0]) / 0.05).clamp(-4, 4)[:, None].expand(-1, H, -1, -1),
                 torch.exp(-dn0 / 0.02)[:, None, :, None].expand(-1, H, -1, -1),
                 mat[:, c][:, None, None].expand(-1, H, N, -1), mat[:, o][:, None, None].expand(-1, H, N, -1)]
        out.append(torch.cat(chans, -1).permute(0, 3, 1, 2))
    return torch.cat(out, 0)


class Block(nn.Module):
    def __init__(self, ch, dt, dn, drop=0.0):
        super().__init__()
        self.drop = nn.Dropout2d(drop)
        self.c1 = nn.Conv2d(ch, ch, 3, padding=(dt, dn), dilation=(dt, dn), padding_mode="replicate")
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, padding_mode="replicate")
        self.n1, self.n2 = nn.GroupNorm(8, ch), nn.GroupNorm(8, ch)

    def forward(self, x):
        y = self.c1(F.gelu(self.n1(x)))
        y = self.c2(self.drop(F.gelu(self.n2(y))))
        return x + y


class ContactNet(nn.Module):
    """Convolutions over the time-node grid; the last layer starts at zero, so training starts from
    the unchanged DLO-Anything prediction."""

    def __init__(self, ch=96, drop=0.0):
        super().__init__()
        self.inp = nn.Conv2d(N_FEATURES, ch, 1)
        dil = [(1, 1), (2, 1), (4, 2), (8, 2), (16, 4), (1, 1), (4, 2), (16, 4)]
        self.blocks = nn.Sequential(*[Block(ch, *d, drop) for d in dil])
        self.out = nn.Conv2d(ch, 3, 1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, x):
        return self.out(F.gelu(self.blocks(self.inp(x))))


def load_net(path, device):
    """Checkpoints store their network size; older ones used the defaults."""
    ck = torch.load(path, map_location=device, weights_only=False)
    net = ContactNet(ch=ck.get("ch", 96)).to(device)
    net.load_state_dict(ck["net"]); net.eval()
    return net


def correct(net, base, q0, pose0, mat, return_geo=False, truth=None, truth_gate=0.008):
    """base (B,2,H,N,3) world -> corrected prediction (B,2,H,N,3) world. mat is normalised (B,2,6).
    truth: true future positions; the contact state then comes from them instead of the predictions.
    This is only a reference for what perfect contact knowledge would give."""
    Bn, _, H, N, _ = base.shape
    geo = contact_reference(base, q0) if truth is None else contact_reference(truth, q0, gate=truth_gate)
    w = node_weight(geo["ref"], geo["has"], N)
    with torch.no_grad():
        x = features(base, q0, pose0, mat, geo, w)
    dx = net(x).permute(0, 2, 3, 1).view(2, Bn, H, N, 3).permute(1, 0, 2, 3, 4) / POS_SCALE
    yaw = pose0[:, :, 2][:, :, None, None]
    dx = from_frame(dx, yaw) * w[..., None]
    pred = base + dx
    return (pred, geo, w) if return_geo else pred
