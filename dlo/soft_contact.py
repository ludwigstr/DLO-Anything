"""Soft-contact tokens: contact between two cables inside one DLO-Anything model.

Two models instead of three: the earlier arrangement had one DLO-Anything instance per cable with a
force channel plus a separate force network and a loop between them. Now there are only the two
DLO-Anything models (pulled cable, fixed U); the force network, the force channel and the loop are
gone. In the model of cable c the other cable enters as contact tokens, one per partner node j:

    partner node j over the context steps, in the frame of cable c     K x 3 positions, K x 3 velocities
    closest point of cable c (on its segments) to partner node j now   offset vector (3), arc position (1)
    soft contact activation exp(-gap / s) for s = 2, 10, 40 cm         (3)  1 = touching, -> 0 with distance
    relative velocity of that closest point and node j                 (3)
    material of the partner (normalised)                               (M)
    partner is fixed at both ends                                      (1)

"Soft" because there is no contact flag and no force: every partner node is always a token, and how much
it matters fades continuously with the gap. The transformer resolves the contact by attention between
node tokens and contact tokens. The planned hand motion of the partner goes onto the action tokens,
and the global token says whether the predicted cable itself is fixed at both ends.

    python -m dlo.soft_contact --cable 0 --out runs/soft/token_move       # model of the pulled cable
    python -m dlo.soft_contact --cable 1 --out runs/soft/token_fixed      # model of the fixed U
    python -m dlo.soft_contact --cable 0 --no-contact --out runs/soft/ohne_move     # reference without tokens
    python -m dlo.soft_contact --eval runs/soft/token_move/best.pt,runs/soft/token_fixed/best.pt --out stats/soft/token.json
"""
from __future__ import annotations
import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .config import ModelConfig, asdict_safe, pad_material
from .contact_model import from_frame, to_frame
from .dataset import ACT_SCALE, POS_SCALE, VEL_SCALE, make_sample
from .model import PhysicsTransformer

SOFT_SCALES = (0.02, 0.10, 0.40)       # [m] length scales of the soft contact activation
RIGID_SCALES = (0.005, 0.02, 0.10)     # [m] length scales of the proximity to a rigid surface
FLOOR_REST = 0.004                     # [m] height of a node centre that lies on the floor
FLOOR_FRICTION = 1.0                   # MuJoCo default sliding friction of the plane
GROUPS = ("frei", "nur Boden", "einfach", "schwer")


def contact_in(mcfg: ModelConfig) -> int:
    return mcfg.context_len * 6 + 3 + 1 + len(SOFT_SCALES) + 3 + mcfg.n_material + 1


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


def contact_tokens(ctx, pose0, mat, fixed):
    """ctx (B,2,K,N,3) world, pose0 (B,2,3), mat (B,2,M) normalised, fixed (B,2) -> (B,2,N,contact_in)."""
    B, C, K, N, _ = ctx.shape
    own, par = frames(ctx.float(), pose0.float())
    vel = torch.diff(par, dim=2, prepend=par[:, :, :1])
    q, arc, gap, i, ti = closest_on_cable(par[:, :, -1], own[:, :, -1])
    v_own = own[:, :, -1] - own[:, :, -2]                                   # (B,2,N,3)
    take = lambda idx: v_own.gather(2, idx[..., None].expand(-1, -1, -1, 3))
    v_q = take(i) * (1 - ti[..., None]) + take(i + 1) * ti[..., None]
    soft = torch.stack([torch.exp(-gap / s) for s in SOFT_SCALES], -1)
    hist = lambda x: x.permute(0, 1, 3, 2, 4).reshape(B, C, N, K * 3)
    return torch.cat([hist(par) * POS_SCALE, hist(vel) * VEL_SCALE, (q - par[:, :, -1]) * POS_SCALE, (2 * arc - 1)[..., None], soft,
                      (v_q - vel[:, :, -1]) * VEL_SCALE, mat.flip(1)[:, :, None].expand(-1, -1, N, -1),
                      fixed.flip(1)[:, :, None, None].expand(-1, -1, N, 1).to(par.dtype)], -1)


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
        first = next(models[c] for c in cables if models[c].cfg.contact_in)
        make = own_contact_tokens if first.cfg.contact_in == own_in(first.cfg) else contact_tokens
        tok = make(ctx, b["pose0"], b["mat"], b["fixed"]); actp = partner_actions(b["pose0"], b["pose_fut"])
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
        if getattr(models[c].cfg, "residual_vel", False):
            y = y + (ctx[:, c, -1] - ctx[:, c, -2])[:, None]
        out.append(y)
    return torch.stack(out, 1)


class SceneWindows(Dataset):
    """Chains of W windows (stride H) of both cables of the U scenario; W = 1 is the plain window."""

    def __init__(self, path, mcfg, spe=8, seed=0, contact_frac=0.5, n_windows=1, subset=0):
        d = np.load(path)
        n = subset or len(d["mode"])
        self.pos, self.pose, self.action = d["pos"][:n], d["pose"][:n], d["action"][:n]
        self.material = pad_material(d["material"][:n])
        fc = d["force_cable"][:n] if "force_cable" in d else d["force"][:n]
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

    def groups(self):
        """frei / nur Boden / einfach (one direction) / schwer (reversal), as dlo.rollout.groups."""
        any_touch = self.touch.any(1)
        return np.array(["frei" if m == "free" else "nur Boden" if not c else "schwer" if m == "there_back" else "einfach"
                         for m, c in zip(self.mode, any_touch)])

    def __len__(self):
        return self.E * self.spe

    def item(self, e, t0):
        ss = [make_sample(self.pos[e, c], self.pose[e, c], self.action[e, c], self.material[e, c], t0, self.K, self.H) for c in (0, 1)]
        sl = slice(t0 + 1, t0 + self.H + 1)
        return dict(act=torch.stack([s["act"] for s in ss]), mat=torch.stack([s["mat"] for s in ss]),
                    pose0=torch.from_numpy(self.pose[e, :, t0]), pose_fut=torch.from_numpy(self.pose[e, :, sl]),
                    gt=torch.from_numpy(self.pos[e, :, sl]), fixed=torch.tensor([0.0, 1.0]),
                    touch=torch.tensor(bool(self.touch[e, sl].any())))

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
    mcfg = ModelConfig(**ck["cfg"])
    model = PhysicsTransformer(mcfg, node_in=mcfg.context_len * 6).to(device)
    model.load_state_dict(ck["model"]); model.eval()
    return model, mcfg


@torch.no_grad()
def validate(model, loader, device, cable):
    """Single windows of one cable from the true state: mean node error [cm], all windows / windows with contact."""
    model.eval(); acc = dict(all=[0.0, 0], contact=[0.0, 0]); models = {cable: model}
    for b in loader:
        b = {k: v.to(device) for k, v in b.items()}
        w = {k: v[:, 0] for k, v in b.items() if k != "ctx"}
        err = (predict(models, b["ctx"], w, cables=(cable,)) - w["gt"][:, [cable]]).norm(dim=-1).mean((1, 2, 3)) * 100
        for k, m in (("all", torch.ones_like(w["touch"])), ("contact", w["touch"])):
            acc[k][0] += err[m].sum().item(); acc[k][1] += int(m.sum())
    model.train()
    return {k: v[0] / max(v[1], 1) for k, v in acc.items()}


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


def study_metric(models, ds, device, modes=(True,)):
    """Same metric and layout as dlo.rollout.study_metric: mean node error [cm] per difficulty group."""
    g = ds.groups(); eps = list(range(ds.E)); res = {}
    for chain in (True, False):
        for use in modes:
            p = rollout(models, ds, eps, device, use, chain)
            err = np.linalg.norm(p - ds.pos, axis=-1) * 100
            per = np.nanmean(err, axis=(1, 2, 3))
            r = {k: float(per[g == k].mean()) for k in GROUPS if (g == k).any()}
            r["alle"] = float(per.mean()); r["gezogen"] = float(np.nanmean(err[:, 0])); r["U"] = float(np.nanmean(err[:, 1]))
            if chain:
                r["pro_sekunde"] = [float(np.nanmean(err[:, :, s * 50 - 25:s * 50 + 25])) for s in range(1, 8)]
            res[("durchgehend" if chain else "neustart") + ("_token" if use and (models[0].cfg.contact_in or models[0].cfg.rigid_in) else "_ohne")] = r
    res["n"] = {k: int((g == k).sum()) for k in GROUPS}
    return res


def fmt(res):
    lines = [f"{'':24s}" + "".join(f"{k:>11s}" for k in GROUPS + ("alle",))]
    for k, r in res.items():
        if k != "n":
            lines.append(f"{k:24s}" + "".join(f"{r.get(c, float('nan')):11.2f}" for c in GROUPS + ("alle",)))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/train_u2.npz"); ap.add_argument("--val", default="data/val_u2.npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--eval", default=None, help="two checkpoints 'pulled,fixed': only compute the 8 s study metric and write it to --out (json)")
    ap.add_argument("--cable", type=int, default=0, help="which cable this model predicts: 0 = pulled cable, 1 = fixed U")
    ap.add_argument("--no-contact", action="store_true", help="same model without soft-contact tokens (the cable alone)")
    ap.add_argument("--own", action="store_true", help="soft-contact tokens per own node: closest point on the other cable")
    ap.add_argument("--rigid", action="store_true", help="rigid-contact tokens: one floor patch below every node")
    ap.add_argument("--init", default=None, help="checkpoint to start from, e.g. a single-cable DLO-Anything model")
    ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--spe", type=int, default=32)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--subset", type=int, default=0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d_model", type=int, default=256); ap.add_argument("--layers", type=int, default=6); ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--workers", type=int, default=6); ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit-steps", type=int, default=0)
    ap.add_argument("--horizon", type=int, default=100, help="control steps predicted per pass (100 = 2 s)")
    args = ap.parse_args()
    device = torch.device(args.device)

    if args.eval:
        models = [load(p, device)[0] for p in args.eval.split(",")]; mcfg = models[0].cfg
        res = study_metric(models, SceneWindows(args.val, mcfg), device)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(res, open(args.out, "w"), indent=1); print(fmt(res), flush=True)
        return

    torch.manual_seed(0)
    mcfg = ModelConfig(d_model=args.d_model, n_heads=args.heads, n_layers=args.layers, horizon=args.horizon)
    mcfg.contact_in = 0 if args.no_contact else own_in(mcfg) if args.own else contact_in(mcfg)
    mcfg.rigid_in = rigid_in(mcfg) if args.rigid else 0
    os.makedirs(args.out, exist_ok=True)
    json.dump(vars(args), open(os.path.join(args.out, "args.json"), "w"), indent=1)
    model = PhysicsTransformer(mcfg, node_in=mcfg.context_len * 6).to(device)
    if args.init:
        own = model.state_dict(); src = torch.load(args.init, map_location=device)["model"]
        hit = [k for k, v in src.items() if k in own and own[k].shape == v.shape]
        own.update({k: src[k] for k in hit}); model.load_state_dict(own)
        print(f"initialised from {args.init}: {len(hit)} of {len(own)} tensors", flush=True)
    tr = SceneWindows(args.train, mcfg, args.spe, subset=args.subset)
    va = SceneWindows(args.val, mcfg, 8, seed=1)
    trl = DataLoader(tr, args.batch, shuffle=True, num_workers=args.workers, drop_last=True, persistent_workers=True, pin_memory=True)
    val = DataLoader(va, args.batch, num_workers=4)
    print(f"params {PhysicsTransformer.n_params(model) / 1e6:.2f} M, soft-contact tokens {mcfg.contact_in or 'off'}, rigid-contact tokens {mcfg.rigid_in or 'off'}, "
          f"episodes {tr.E}, steps/epoch {len(trl)}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))
    total = args.epochs * (args.limit_steps or len(trl)); warm = max(1, min(1000, total // 10))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    log, best, t0 = [], float("inf"), time.time()
    for ep in range(args.epochs):
        run, nb = 0.0, 0
        for bb in trl:
            bb = {k: v.to(device, non_blocking=True) for k, v in bb.items()}
            b = {k: v[:, 0] for k, v in bb.items() if k != "ctx"}
            p = predict({args.cable: model}, bb["ctx"], b, cables=(args.cable,))
            loss = F.mse_loss(p * POS_SCALE, b["gt"][:, [args.cable]] * POS_SCALE)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); run += loss.item(); nb += 1
            if args.limit_steps and nb >= args.limit_steps:
                break
        v = validate(model, val, device, args.cable)
        log.append(dict(epoch=ep, train_mse=run / nb, val_all_cm=v["all"], val_contact_cm=v["contact"], time=time.time() - t0))
        print(f"ep {ep:3d} train_mse {run / nb:.5f} | val all {v['all']:.2f} cm, contact windows {v['contact']:.2f} cm | {time.time() - t0:.0f}s", flush=True)
        json.dump(log, open(os.path.join(args.out, "log.json"), "w"), indent=1)
        ck = {"model": model.state_dict(), "cfg": asdict_safe(mcfg)}
        torch.save(ck, os.path.join(args.out, "last.pt"))
        if v["all"] < best:
            best = v["all"]; torch.save(ck, os.path.join(args.out, "best.pt"))
    print("best score", best, flush=True)


if __name__ == "__main__":
    main()
