"""Coupling of two DLO-Anything model instances through contact forces.

Two cables, one model instance each (same weights, different material tokens / states). The
model has an external-force channel (per node, context + future). The coupling algorithm is a
projected Gauss-Seidel iteration on the contact complementarity, with the learned models in the
role of the dynamics:

    lambda = 0                                   # one non-negative contact force per (step k, segment i of cable A, segment j of cable B)
    repeat n_iter times:
        P_A, P_B = model(cable A | forces from lambda), model(cable B | forces from lambda)   (one batched call)
        for every step k and segment pair (i, j):
            d, n, s, t = closest points of the two predicted segments       (n: unit normal from B to A)
            penetration delta = d_contact - d                                 (d_contact = 2 r + margin)
            lambda <- max(0, lambda + k_c * delta)                            (penalty accumulation, projected to >= 0)
        node forces: +lambda n on cable A split to the segment's end nodes by s, -lambda n on cable B split by t
    return P_A, P_B, node forces, contact points

With lambda >= 0 and delta -> 0 at convergence this is the usual contact LCP, solved with the
model's own force response instead of a stiffness matrix. The result is a joint prediction of
both cables that is consistent with the contact forces the models were fed.

Usage (evaluation on a two-cable test set, windows whose horizon contains contact):
    python -m dlo.couple --ckpt runs/force1/best.pt --data data/test_two.npz --out stats/couple
"""
from __future__ import annotations
import argparse, json, os, time
import numpy as np
import torch

from .config import SimConfig, ModelConfig, pad_material
from .dataset import make_sample, from_gripper_frame, force_to_frame, POS_SCALE
from .model import PhysicsTransformer


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device)
    mcfg = ModelConfig(**ck["cfg"])
    model = PhysicsTransformer(mcfg, node_in=mcfg.context_len * 6).to(device)
    model.load_state_dict(ck["model"], strict=False)
    model.eval()
    return model, mcfg


def seg_seg_closest(P0, P1, Q0, Q1, eps=1e-12):
    """Closest points of segments P0P1 and Q0Q1 (broadcastable (..., 3)).
    Returns s, t in [0,1] (parameters on P and Q), the closest points and their distance."""
    u, v, w0 = P1 - P0, Q1 - Q0, P0 - Q0
    a = (u * u).sum(-1); b = (u * v).sum(-1); c = (v * v).sum(-1)
    d = (u * w0).sum(-1); e = (v * w0).sum(-1)
    D = a * c - b * b
    s = np.where(D > eps, np.clip((b * e - c * d) / np.where(D > eps, D, 1.0), 0.0, 1.0), 0.0)
    t = (b * s + e) / np.maximum(c, eps)
    s_lo = np.clip(-d / np.maximum(a, eps), 0.0, 1.0)
    s_hi = np.clip((b - d) / np.maximum(a, eps), 0.0, 1.0)
    s = np.where(t < 0, s_lo, np.where(t > 1, s_hi, s))
    t = np.clip(t, 0.0, 1.0)
    Pc = P0 + s[..., None] * u
    Qc = Q0 + t[..., None] * v
    dist = np.linalg.norm(Pc - Qc, axis=-1)
    return s, t, Pc, Qc, dist


class CoupledPredictor:
    """Joint prediction of two cables with one model instance per cable, coupled by contact forces."""

    def __init__(self, model, mcfg: ModelConfig, device, radius=0.003, margin=0.002, k_c=2.0, n_iter=6,
                 relax=1.0):
        self.model, self.mcfg, self.device = model, mcfg, device
        self.d_contact = 2 * radius + margin
        self.k_c, self.n_iter, self.relax = k_c, n_iter, relax   # k_c [N/m]: penalty gain per iteration

    @torch.no_grad()
    def _forward(self, batch, force_fut_world, poses0):
        """batch: stacked samples of both cables; force_fut_world (2,H,N,3) -> predictions in world (2,H,N,3)."""
        ff = np.stack([force_to_frame(force_fut_world[c], poses0[c]) for c in range(2)])
        b = dict(batch)
        b["force_fut"] = torch.from_numpy(ff.astype(np.float32)).to(self.device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            pred = self.model(b["node"], b["act"], b["mat"], b["cur"],
                              force_ctx=b["force_ctx"], force_fut=b["force_fut"]).float().cpu().numpy()
        return np.stack([from_gripper_frame(pred[c] / POS_SCALE, poses0[c]) for c in range(2)])

    def contact_step(self, pred, lam):
        """pred (2,H,N,3) world -> updated lam (H,S,S), node forces (2,H,N,3), contact mask (H,S,S), points."""
        A0, A1 = pred[0, :, :-1, None, :], pred[0, :, 1:, None, :]        # (H,S,1,3)
        B0, B1 = pred[1, :, None, :-1, :], pred[1, :, None, 1:, :]        # (H,1,S,3)
        s, t, Pc, Qc, dist = seg_seg_closest(A0, A1, B0, B1)              # (H,S,S)
        delta = self.d_contact - dist
        lam = np.maximum(0.0, lam + self.relax * self.k_c * delta)
        n = (Pc - Qc) / np.maximum(dist, 1e-6)[..., None]                 # normal from B to A
        f = lam[..., None] * n                                             # force on A per pair (H,S,S,3)
        H, S = lam.shape[:2]
        N = S + 1
        force = np.zeros((2, H, N, 3))
        wA0, wA1 = (1 - s)[..., None] * f, s[..., None] * f               # split to the end nodes of segment i (cable A)
        force[0, :, :-1] += wA0.sum(2); force[0, :, 1:] += wA1.sum(2)
        wB0, wB1 = (1 - t)[..., None] * (-f), t[..., None] * (-f)         # cable B gets the opposite force
        force[1, :, :-1] += wB0.sum(1); force[1, :, 1:] += wB1.sum(1)
        active = lam > 0
        return lam, force, active, 0.5 * (Pc + Qc)

    def predict(self, pos, pose, action, material, force_hist, t0, return_iters=False):
        """pos/pose/action/force_hist: arrays (2, T, ...) of both cables; t0: window start.
        force_hist (2,T,N,3) world forces up to t0 (context channel); future forces are estimated.
        Returns dict(pred (2,H,N,3) world, force (2,H,N,3) world, active (H,S,S), points (H,S,S,3),
        pred_uncoupled (2,H,N,3))."""
        K, H = self.mcfg.context_len, self.mcfg.horizon
        samples = [make_sample(pos[c], pose[c], action[c], material[c], t0, K, H, force=force_hist[c]) for c in range(2)]
        batch = {k: torch.stack([smp[k] for smp in samples]).to(self.device) for k in samples[0]}
        poses0 = [pose[c][t0] for c in range(2)]
        N = pos.shape[2]; S = N - 1
        force = np.zeros((2, H, N, 3))
        lam = np.zeros((H, S, S))
        iters = []
        pred0 = self._forward(batch, force, poses0)
        pred = pred0
        for it in range(self.n_iter):
            lam, force, active, points = self.contact_step(pred, lam)
            pred = self._forward(batch, force, poses0)
            if return_iters:
                iters.append(dict(pred=pred.copy(), force=force.copy(), n_active=int(active.sum())))
        lam_f, force_f, active, points = self.contact_step(pred, lam)      # contacts of the final prediction
        out = dict(pred=pred, force=force, active=active, points=points, pred_uncoupled=pred0, lam=lam)
        if return_iters:
            out["iters"] = iters
        return out

    @torch.no_grad()
    def predict_with_forces(self, pos, pose, action, material, force_hist, force_fut, t0):
        """Reference: prediction with GIVEN future forces (oracle / simulator forces)."""
        K, H = self.mcfg.context_len, self.mcfg.horizon
        samples = [make_sample(pos[c], pose[c], action[c], material[c], t0, K, H, force=force_hist[c]) for c in range(2)]
        batch = {k: torch.stack([smp[k] for smp in samples]).to(self.device) for k in samples[0]}
        poses0 = [pose[c][t0] for c in range(2)]
        return self._forward(batch, force_fut[:, t0 + 1:t0 + H + 1].astype(np.float64), poses0)


# ---- evaluation ------------------------------------------------------------------------------
def contact_windows(n_contact, K, H, per_episode, rng):
    """t0 values per episode whose horizon contains contact (at most per_episode, spread over the episode)."""
    T = len(n_contact)
    t0s = np.arange(K - 1, T - H)
    cs = np.cumsum(np.concatenate([[0], n_contact > 0]))
    ok = t0s[(cs[t0s + H + 1] - cs[t0s + 1]) > 0]
    if len(ok) == 0:
        return []
    if len(ok) <= per_episode:
        return list(ok)
    return sorted(set(int(ok[i]) for i in np.linspace(0, len(ok) - 1, per_episode).round().astype(int)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="data/test_two.npz")
    ap.add_argument("--out", default="stats/couple")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--per-episode", type=int, default=3)
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--kc", type=float, default=2.0, help="penalty gain [N/m] per iteration")
    ap.add_argument("--margin", type=float, default=0.002)
    ap.add_argument("--ctx-forces", default="true", choices=["true", "zero"],
                    help="context-force channel: simulator forces (measured past) or zero")
    ap.add_argument("--baseline", default=None, help="force-channel checkpoint with zero force layers (runs/force0 = single-cable wide4): its prediction is the reference without any contact knowledge")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device)
    model, mcfg = load_model(args.ckpt, device)
    assert mcfg.force, "checkpoint has no force channel"
    cp = CoupledPredictor(model, mcfg, device, k_c=args.kc, n_iter=args.iters, margin=args.margin)
    cp_base = None
    if args.baseline:
        cp_base = CoupledPredictor(load_model(args.baseline, device)[0], mcfg, device)
    d = np.load(args.data)
    K, H = mcfg.context_len, mcfg.horizon
    rng = np.random.default_rng(0)
    rows = []
    t_start = time.time()
    E = min(args.episodes, d["pos"].shape[0])
    for e in range(E):
        pos, pose, action = d["pos"][e].astype(np.float64), d["pose"][e], d["action"][e]
        material = pad_material(d["material"][e]); force = d["force"][e].astype(np.float32)
        force_hist = force if args.ctx_forces == "true" else np.zeros_like(force)
        for t0 in contact_windows(d["n_contact"][e], K, H, args.per_episode, rng):
            true = pos[:, t0 + 1:t0 + H + 1]
            f_true = force[:, t0 + 1:t0 + H + 1]
            t1 = time.time()
            r = cp.predict(pos, pose, action, material, force_hist, t0)
            dt = time.time() - t1
            oracle = cp.predict_with_forces(pos, pose, action, material, force_hist, force, t0)
            err = lambda p: np.linalg.norm(p - true, axis=-1)            # (2,H,N)
            e_un, e_cp, e_or = err(r["pred_uncoupled"]), err(r["pred"]), err(oracle)
            e_ba = err(cp_base.predict_with_forces(pos, pose, action, material, np.zeros_like(force), np.zeros_like(force), t0)) if cp_base else e_un
            touch_true = np.linalg.norm(f_true, axis=-1) > 0                # (2,H,N)
            touch_est = np.linalg.norm(r["force"], axis=-1) > 0
            tp = (touch_true & touch_est).sum(); fp = (~touch_true & touch_est).sum(); fn = (touch_true & ~touch_est).sum()
            mag_t, mag_e = np.linalg.norm(f_true, axis=-1), np.linalg.norm(r["force"], axis=-1)
            rows.append(dict(episode=e, t0=int(t0), mode=str(d["mode"][e]),
                             err_baseline=float(e_ba.mean()), err_h100_baseline=float(e_ba[:, -1].mean()),
                             err_touch_baseline=float(e_ba[touch_true].mean()) if touch_true.any() else np.nan,
                             err_moving_baseline=float(e_ba[0].mean()), err_moving_coupled=float(e_cp[0].mean()), err_moving_oracle=float(e_or[0].mean()),
                             err_resting_baseline=float(e_ba[1].mean()), err_resting_coupled=float(e_cp[1].mean()), err_resting_oracle=float(e_or[1].mean()),
                             err_uncoupled=float(e_un.mean()), err_coupled=float(e_cp.mean()), err_oracle=float(e_or.mean()),
                             err_h100_uncoupled=float(e_un[:, -1].mean()), err_h100_coupled=float(e_cp[:, -1].mean()), err_h100_oracle=float(e_or[:, -1].mean()),
                             err_touch_uncoupled=float(e_un[touch_true].mean()) if touch_true.any() else np.nan,
                             err_touch_coupled=float(e_cp[touch_true].mean()) if touch_true.any() else np.nan,
                             err_touch_oracle=float(e_or[touch_true].mean()) if touch_true.any() else np.nan,
                             contact_steps_true=int(touch_true.any((0, 2)).sum()), contact_steps_est=int(touch_est.any((0, 2)).sum()),
                             tp=int(tp), fp=int(fp), fn=int(fn),
                             force_true_sum=float(mag_t.sum()), force_est_sum=float(mag_e.sum()),
                             force_corr=float(np.corrcoef(mag_t.ravel(), mag_e.ravel())[0, 1]) if mag_e.std() > 0 and mag_t.std() > 0 else np.nan,
                             n_iter=args.iters, time_s=dt))
        if (e + 1) % 20 == 0:
            print(f"{e + 1}/{E} episodes, {len(rows)} windows, {time.time() - t_start:.0f}s", flush=True)
    R = {k: np.array([r[k] for r in rows], dtype=float) for k in rows[0] if k != "mode"}
    tp, fp, fn = R["tp"].sum(), R["fp"].sum(), R["fn"].sum()
    lines = [f"coupled prediction on {args.data}: {len(rows)} windows with contact in the horizon from {E} episodes "
             f"(ckpt {args.ckpt}, iters {args.iters}, k_c {args.kc} N/m, margin {args.margin * 1000:.0f} mm, context forces: {args.ctx_forces})",
             "", f"{'':36s} {'wide4':>8s} {'zero force':>11s} {'coupled':>9s} {'oracle forces':>14s}",
             f"{'mean node error, 2 s window [cm]':36s} {R['err_baseline'].mean() * 100:8.2f} {R['err_uncoupled'].mean() * 100:11.2f} {R['err_coupled'].mean() * 100:9.2f} {R['err_oracle'].mean() * 100:14.2f}",
             f"{'error at h = 100 [cm]':36s} {R['err_h100_baseline'].mean() * 100:8.2f} {R['err_h100_uncoupled'].mean() * 100:11.2f} {R['err_h100_coupled'].mean() * 100:9.2f} {R['err_h100_oracle'].mean() * 100:14.2f}",
             f"{'error of node-steps in contact [cm]':36s} {np.nanmean(R['err_touch_baseline']) * 100:8.2f} {np.nanmean(R['err_touch_uncoupled']) * 100:11.2f} {np.nanmean(R['err_touch_coupled']) * 100:9.2f} {np.nanmean(R['err_touch_oracle']) * 100:14.2f}",
             f"{'moving cable (hand 1) [cm]':36s} {R['err_moving_baseline'].mean() * 100:8.2f} {'':11s} {R['err_moving_coupled'].mean() * 100:9.2f} {R['err_moving_oracle'].mean() * 100:14.2f}",
             f"{'resting cable (hand 2) [cm]':36s} {R['err_resting_baseline'].mean() * 100:8.2f} {'':11s} {R['err_resting_coupled'].mean() * 100:9.2f} {R['err_resting_oracle'].mean() * 100:14.2f}",
             "",
             f"contact detection (node-step in contact): precision {tp / max(tp + fp, 1):.2f}  recall {tp / max(tp + fn, 1):.2f}",
             f"contact steps per window: true {R['contact_steps_true'].mean():.1f}, estimated {R['contact_steps_est'].mean():.1f}",
             f"force magnitude: sum true {R['force_true_sum'].mean():.2f} N*steps, estimated {R['force_est_sum'].mean():.2f}; "
             f"per-window corr(|F|) median {np.nanmedian(R['force_corr']):.2f}",
             f"time per window (both cables, {args.iters} iterations): {R['time_s'].mean() * 1000:.0f} ms",
             "", "per mode (mean node error [cm], uncoupled -> coupled | oracle):"]
    for mode in sorted(set(r["mode"] for r in rows)):
        idx = [i for i, r in enumerate(rows) if r["mode"] == mode]
        lines.append(f"  {mode:7s} n={len(idx):3d}  {R['err_uncoupled'][idx].mean() * 100:5.2f} -> {R['err_coupled'][idx].mean() * 100:5.2f} | {R['err_oracle'][idx].mean() * 100:5.2f}")
    txt = "\n".join(lines); print(txt)
    open(os.path.join(args.out, "couple.txt"), "w").write(txt)
    json.dump(dict(args=vars(args), rows=rows), open(os.path.join(args.out, "couple.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
