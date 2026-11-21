"""Contact force model and the loop between it and the two DLO-Anything instances.

    predicted course of both cables -> contact forces at the nodes -> both DLO-Anything instances
    -> changed course -> changed forces -> ...

The DLO-Anything instances stay frozen and take the forces through their force channel. All forces
inside the loop are in the instance's own frame at t0 and in force-channel units, see force_to_frame.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from .config import ModelConfig, pad_material
from .contact_model import Block, N_FEATURES, contact_reference, features, from_frame, node_weight, to_frame
from .couple import load_model
from .dataset import FORCE_CLIP, FORCE_SCALE, POS_SCALE, force_to_frame, make_sample


class ForceNet(nn.Module):
    """Convolutions over the time-node grid. Output per node and step: contact logit and force; with
    `floor` a second head for the floor contact (input gets the height above the floor)."""

    def __init__(self, ch=64, drop=0.1, floor=False):
        super().__init__()
        self.floor = floor
        self.inp = nn.Conv2d(N_FEATURES + (1 if floor else 0) + 3, ch, 1)
        dil = [(1, 1), (2, 1), (4, 2), (8, 2), (16, 4), (1, 1), (4, 2), (16, 4)]
        self.blocks = nn.Sequential(*[Block(ch, *d, drop) for d in dil])
        self.out = nn.Conv2d(ch, 8 if floor else 4, 1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
        self.out.bias.data[0] = -3.0
        if floor:
            self.out.bias.data[4] = -3.0

    def forward(self, x):
        return self.out(F.gelu(self.blocks(self.inp(x))))

    def load_cable_only(self, sd):
        """Initialise a floor net from a cable-only net: new input / output channels start at zero."""
        own = self.state_dict()
        for k, v in sd.items():
            if k == "inp.weight":
                own[k].zero_(); own[k][:, :N_FEATURES] = v[:, :N_FEATURES]; own[k][:, -3:] = v[:, -3:]
            elif k in ("out.weight", "out.bias"):
                own[k][:4] = v
            else:
                own[k] = v
        self.load_state_dict(own)


def force_step(net, pred, q0, pose0, mat, f_prev, parts_out=None, gate=None):
    """pred (B,2,H,N,3) world, f_prev (B,2,H,N,3) own frame -> total forces and cable-contact logits per
    cable. A floor net appends dict(f_cable, f_floor, lg_floor) to parts_out."""
    Bn, _, H, N, _ = pred.shape
    with torch.no_grad():
        geo = contact_reference(pred, q0)
        w = node_weight(geo["ref"], geo["has"], N)
        x = features(pred, q0, pose0, mat, geo, w)
        prev = f_prev.permute(1, 0, 4, 2, 3).flatten(0, 1) / 4.0        # (2B,3,H,N): order cable 0 batch, cable 1 batch
        if getattr(net, "floor", False):
            z = pred[..., 2].permute(1, 0, 2, 3).flatten(0, 1)[:, None]  # (2B,1,H,N) height above the floor
            x = torch.cat([x, torch.exp(-z.clamp_min(0) / 0.01), prev], 1)
        else:
            x = torch.cat([x, prev], 1)
    k = 8 if getattr(net, "floor", False) else 4
    y = net(x).view(2, Bn, k, H, N).permute(1, 0, 3, 4, 2)              # (B,2,H,N,k)
    logit, f = y[..., 0], y[..., 1:4]
    fc = (torch.sigmoid(logit)[..., None] * f).clamp(-FORCE_CLIP, FORCE_CLIP)
    if gate is not None:
        # geometric gate: cable force only where the two predicted cables are closer than `gate` metres
        # (node-node distance, widened by 2 steps and 2 nodes)
        with torch.no_grad():
            d = torch.cdist(pred[:, 0].flatten(0, 1).float(), pred[:, 1].flatten(0, 1).float()).view(Bn, H, N, N)
            near = torch.stack([d.min(-1).values, d.min(-2).values], 1) < gate          # (B,2,H,N)
            near = F.max_pool2d(near.float(), (5, 5), 1, (2, 2)) > 0
        fc = fc * near[..., None]
    if k == 4:
        return fc, logit
    lgf, ffl = y[..., 4], y[..., 5:8]
    ffl = (torch.sigmoid(lgf)[..., None] * ffl).clamp(-FORCE_CLIP, FORCE_CLIP)
    if parts_out is not None:
        parts_out.append(dict(f_cable=fc, f_floor=ffl, lg_floor=lgf))
    return (fc + ffl).clamp(-FORCE_CLIP, FORCE_CLIP), logit


class Instances(nn.Module):
    """The two frozen DLO-Anything instances: cable 1 on the hand, cable 2 as fixed U."""

    def __init__(self, move_ckpt, fixed_ckpt, device):
        super().__init__()
        self.move, self.mcfg = load_model(move_ckpt, device)
        self.fixed, _ = load_model(fixed_ckpt, device)
        for m in (self.move, self.fixed):
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)

    def forward(self, b, f):
        """b holds (B,2,...) model inputs; f (B,2,H,N,3) in force-channel units -> world positions."""
        out = []
        for c, m in enumerate((self.move, self.fixed)):
            y = m(b["node"][:, c], b["act"][:, c], b["mat"][:, c], b["cur"][:, c],
                  force_ctx=b["force_ctx"][:, c], force_fut=f[:, c]).float() / POS_SCALE
            yaw = b["pose0"][:, c, 2][:, None, None]
            t = torch.cat([b["pose0"][:, c, :2], torch.zeros(len(y), 1, device=y.device)], -1)[:, None, None]
            out.append(from_frame(y, yaw) + t)
        return torch.stack(out, 1)


def run_loop(net, inst, b, n_iter, autocast=True, grad_through_model=True, relax=1.0, parts_out=None):
    """Returns the predictions per iteration (index 0: no contact force) and forces and logits per iteration."""
    H, N = b["force_fut"].shape[2:4]
    f = torch.zeros_like(b["force_fut"])
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast), torch.no_grad():
        p0 = inst(b, f)
    preds, forces, logits = [p0], [], []
    for _ in range(n_iter):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            ff, lg = force_step(net, preds[-1].detach(), b["q0"], b["pose0"], b["mat"], f.detach(), parts_out)
            ff, lg = ff.float(), lg.float()
            if relax < 1.0:
                ff = relax * ff + (1 - relax) * f.detach()     # damped update against oscillation
            if grad_through_model:
                preds.append(inst(b, ff))
            else:
                with torch.no_grad():
                    preds.append(inst(b, ff))
        forces.append(ff); logits.append(lg); f = ff
    return preds, forces, logits


def force_world(f, pose0):
    """Force-channel units in the own frame -> newton in world coordinates; (B,2,H,N,3)."""
    return from_frame(f / FORCE_SCALE, pose0[:, :, 2][:, :, None, None])


class UWindows(Dataset):
    """Windows of both cables of the U scenario. Past forces are zero: the loop knows no measured force."""

    def __init__(self, path, mcfg: ModelConfig, samples_per_episode=8, seed=0, contact_frac=0.5):
        d = np.load(path)
        self.pos, self.pose, self.action = d["pos"], d["pose"], d["action"]
        self.material = pad_material(d["material"]); self.force = d["force"]
        # u2 data: force = cable + floor; the force net learns both parts separately
        self.force_floor = d["force_floor"] if "force_floor" in d else np.zeros_like(self.force)
        self.force_cable = d["force_cable"] if "force_cable" in d else self.force
        self.K, self.H = mcfg.context_len, mcfg.horizon
        self.E, self.T = self.pos.shape[0], self.pos.shape[2]
        self.spe, self.frac = samples_per_episode, contact_frac
        self.rng = np.random.default_rng(seed)
        t0s = np.arange(self.K - 1, self.T - self.H)
        cs = np.cumsum(np.pad(d["n_contact"] > 0, ((0, 0), (1, 0))), axis=1)
        n_in = cs[:, t0s + self.H + 1] - cs[:, t0s + 1]
        self.contact_t0 = [t0s[n_in[e] > 0] for e in range(self.E)]

    def __len__(self):
        return self.E * self.spe

    def item(self, e, t0, pos=None):
        """`pos` (2, T, N, 3) replaces the true positions as context, e.g. for a chained rollout."""
        P = self.pos[e] if pos is None else pos
        ss = [make_sample(P[c], self.pose[e, c], self.action[e, c], self.material[e, c], t0, self.K, self.H,
                          force=self.force[e, c].astype(np.float32)) for c in (0, 1)]
        b = {k: torch.stack([s[k] for s in ss]) for k in ("node", "act", "mat", "cur", "force_fut")}
        b["force_ctx"] = torch.zeros_like(torch.stack([s["force_ctx"] for s in ss]))
        b["force_fut_floor"] = torch.stack([torch.from_numpy(force_to_frame(self.force_floor[e, c, t0 + 1:t0 + self.H + 1].astype(np.float32),
                                                                            self.pose[e, c, t0]).astype(np.float32)) for c in (0, 1)])
        b["force_fut_cable"] = b["force_fut"] - b["force_fut_floor"]
        b["gt"] = torch.from_numpy(self.pos[e, :, t0 + 1:t0 + self.H + 1].astype(np.float32))
        b["q0"] = torch.from_numpy(P[:, t0].astype(np.float32))
        b["pose0"] = torch.from_numpy(self.pose[e, :, t0].astype(np.float32))
        return b

    def __getitem__(self, idx):
        e = idx // self.spe
        ct = self.contact_t0[e]
        if len(ct) and self.rng.uniform() < self.frac:
            t0 = int(ct[self.rng.integers(len(ct))])
        else:
            t0 = int(self.rng.integers(self.K - 1, self.T - self.H))
        return self.item(e, t0)


def _blur_kernel(sig_t=2.0, sig_n=1.5, rt=6, rn=4, device=None):
    t = torch.arange(-rt, rt + 1, device=device, dtype=torch.float32); n = torch.arange(-rn, rn + 1, device=device, dtype=torch.float32)
    k = torch.exp(-t[:, None] ** 2 / (2 * sig_t ** 2) - n[None] ** 2 / (2 * sig_n ** 2))
    return (k / k.sum())[None, None]


def blurred_targets(force_fut):
    """The contact point predicted from positions is uncertain by a few nodes and steps, and the
    DLO-Anything force channel tolerates such shifts. The target is therefore the simulator force
    blurred over nodes and steps; the blur keeps the total force. Returns target (B,2,H,N,3) and
    the contact zone (B,2,H,N) where the blurred force is not negligible."""
    B, C, H, N, _ = force_fut.shape
    k = _blur_kernel(device=force_fut.device)
    x = force_fut.permute(0, 1, 4, 2, 3).reshape(B * C * 3, 1, H, N)
    y = F.conv2d(x, k, padding=(k.shape[2] // 2, k.shape[3] // 2))
    tgt = y.view(B, C, 3, H, N).permute(0, 1, 3, 4, 2)
    touch = (force_fut.abs().sum(-1) > 0).float().view(B * C, 1, H, N)
    zone = F.conv2d(touch, k, padding=(k.shape[2] // 2, k.shape[3] // 2)).view(B, C, H, N) > 0.01
    return tgt, zone


def floor_loss(parts, b):
    """Floor head: contact zone (BCE) and blurred floor force, like the cable head."""
    tgt, zone = blurred_targets(b["force_fut_floor"])
    z = zone.float(); total = 0.0
    for p in parts:
        bce = F.binary_cross_entropy_with_logits(p["lg_floor"].float(), z)
        h = F.huber_loss(p["f_floor"].float(), tgt, delta=1.0, reduction="none").sum(-1)
        total = total + bce + (h * z).sum() / z.sum().clamp_min(1.0) + (h * (1 - z)).mean()
    return total / max(len(parts), 1)


def loop_loss_blur(preds, forces, logits, b, newton_weight=0.1, pos_weight=0.0, parts=None):
    """With `parts` (floor net) the cable head is trained on the cable-cable force only and the floor
    head on the floor force; otherwise on b["force_fut"]."""
    if parts:
        forces = [p["f_cable"].float() for p in parts]
        b = dict(b, force_fut=b["force_fut_cable"])
    """Supervised loss on blurred simulator forces for every loop iteration. The contact logit learns
    the contact zone, the force itself only where contact is."""
    tgt, zone = blurred_targets(b["force_fut"])
    z = zone.float()
    total = 0.0
    parts = dict(bce=0.0, force=0.0, pos=0.0, newton=0.0)
    for k, (f, lg) in enumerate(zip(forces, logits)):
        bce = F.binary_cross_entropy_with_logits(lg, z, pos_weight=torch.tensor(3.0, device=lg.device))
        h = F.huber_loss(f, tgt, delta=1.0, reduction="none").sum(-1)
        force = (h * z).sum() / z.sum().clamp_min(1.0) + (h * (1 - z)).mean()
        w = force_world(f, b["pose0"]).sum(3)
        newton = ((w[:, 0] + w[:, 1]) * FORCE_SCALE).pow(2).mean()
        pos = ((preds[k + 1] - b["gt"]) ** 2).mean() * POS_SCALE ** 2 / 1e-3 if pos_weight > 0 else torch.zeros((), device=f.device)
        total = total + bce + force + newton_weight * newton + pos_weight * pos
        for n, v in zip(parts, (bce, force, pos, newton)):
            parts[n] += v.item() / len(forces)
    return total / len(forces), parts


def loop_loss_floor(preds, forces, logits, b, parts):
    """Cable head (blurred cable force) + floor head (blurred floor force)."""
    lc, pc = loop_loss_blur(preds, forces, logits, b, parts=parts)
    lf = floor_loss(parts, b)
    pc["floor"] = lf.item()
    return lc + lf, pc


def loop_loss(preds, forces, logits, b, pos_weight=0.5, newton_weight=0.1):
    tgt = b["force_fut"]
    touch = (tgt.abs().sum(-1) > 0).float()
    total = 0.0
    parts = dict(bce=0.0, force=0.0, pos=0.0, newton=0.0)
    for k, (f, lg) in enumerate(zip(forces, logits)):
        bce = F.binary_cross_entropy_with_logits(lg, touch, pos_weight=torch.tensor(5.0, device=lg.device))
        h = F.huber_loss(f, tgt, delta=1.0, reduction="none").sum(-1)
        force = (h * touch).sum() / touch.sum().clamp_min(1.0) + 0.1 * (h * (1 - touch)).mean()
        pos = ((preds[k + 1] - b["gt"]) ** 2).mean() * POS_SCALE ** 2 / 1e-3
        w = force_world(f, b["pose0"]).sum(3)
        newton = ((w[:, 0] + w[:, 1]) * FORCE_SCALE).pow(2).mean()
        total = total + bce + force + pos_weight * pos + newton_weight * newton
        for n, v in zip(parts, (bce, force, pos, newton)):
            parts[n] += v.item() / len(forces)
    return total / len(forces), parts


def load_force_net(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    net = ForceNet(ch=ck.get("ch", 64), drop=0.0, floor=ck.get("floor", False)).to(device)
    net.load_state_dict(ck["net"]); net.eval()
    return net
