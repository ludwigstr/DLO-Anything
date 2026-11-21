"""One-step model: the transformer predicts only the next 20 ms step of its cable. A forecast is a rollout:

    step t:   every node gets its soft- and rigid-contact token (closest contact partner in the current state)
              -> both cable models predict the state at t+1
    step t+1: the contact tokens are rebuilt from the predicted state -> prediction of t+2 ...   (300 steps = 6 s)

Tokens per pass: 41 node tokens (last K = 10 states), 6 material, 1 action (next gripper step), 41 soft-contact,
41 rigid-contact, 1 global.  Training unrolls --unroll steps from a simulated state with the gradient running through
the fed-back predictions; the loss is the position error at every unrolled step.

Training recipe (see runs/u3/pipeline_onestep.sh):
  * --warm W: before the unrolled steps the models roll forward 0..W steps on their own without gradient, so the trained
    steps start from the models' own, slightly wrong states anywhere in a 6-s forecast (not only from simulated states).
  * the checkpoint is selected on the motion validation set (data/val_motion.npz: circle, sine and jerk robot motions).
    Motions of these kinds are never part of the training data; the trainer refuses training files that contain them.
  * scenes in which the pulled cable wraps around the U (<train>.wrap.npy > 360 degrees, runs/u3/wrap.py) are left out.

    python -m dlo.onestep --init a.pt,b.pt --train data/u3/B_24000.npz --out runs/u3/one_B24000
    python -m dlo.onestep --eval runs/u3/one_B24000 --out stats/u3/one_B24000.json
"""
from __future__ import annotations
import argparse, json, math, os, time
import numpy as np, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from .config import ModelConfig, asdict_safe
from .dataset import POS_SCALE
from .model import PhysicsTransformer
from . import soft_contact as sc

F6 = 300
PATHS = {"val": "data/val_u2.npz", "test": "data/test_u2.npz", "unseen": "data/val_u2_ood.npz", "motion": "data/val_motion.npz", "motion_test": "data/test_motion.npz"}
HELD_OUT = ("circle", "sine", "jerk")          # robot motions reserved for validation and test


class RolloutBatches(Dataset):
    """Each item is a whole batch: B scenes, w warm-up steps (no gradient) followed by n trained steps; w is drawn per batch."""
    def __init__(self, ds, batch, n, warm, length):
        self.ds, self.B, self.n, self.warm, self.length, self.rng = ds, batch, n, warm, length, None

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        if self.rng is None:
            self.rng = np.random.default_rng([os.getpid(), i])
        ds = self.ds; w = int(self.rng.integers(0, self.warm + 1)); m = w + self.n; hi = ds.T - 1 - m
        rows = []
        for _ in range(self.B):
            e = int(self.rng.integers(ds.E)); ct = ds.contact_t0[e]; ct = ct[(ct <= hi) & (ct >= ds.K - 1)]
            t0 = int(ct[self.rng.integers(len(ct))]) if len(ct) and self.rng.uniform() < 0.5 else int(self.rng.integers(ds.K - 1, hi + 1))
            its = [ds.item(e, t0 + j) for j in range(m)]
            row = {k: torch.stack([it[k] for it in its]) for k in its[0]}
            row["ctx"] = torch.from_numpy(ds.pos[e, :, t0 - ds.K + 1:t0 + 1]); rows.append(row)
        out = {k: torch.stack([x[k] for x in rows]) for k in rows[0]}; out["warm"] = torch.tensor(w)
        return out


def training_scenes(path, mcfg, subset=0):
    """Training scenes without wraps; refuses files that contain the held-out robot motions."""
    ds = sc.SceneWindows(path, mcfg, subset=subset)
    assert not np.isin(ds.mode, HELD_OUT).any(), f"{path} contains held-out validation motions {HELD_OUT}: never train on them"
    if os.path.exists(path + ".wrap.npy"):
        keep = np.where(np.load(path + ".wrap.npy")[:ds.E] <= 360)[0]
        ds.pos, ds.pose, ds.action, ds.material, ds.touch, ds.mode = ds.pos[keep], ds.pose[keep], ds.action[keep], ds.material[keep], ds.touch[keep], ds.mode[keep]
        ds.contact_t0 = [ds.contact_t0[e] for e in keep]; print(f"training scenes: {len(keep)} of {ds.E} (wraps removed)", flush=True); ds.E = len(keep)
    else:
        print(f"training scenes: {ds.E} (no wrap file next to {path}: nothing removed)", flush=True)
    return ds


def to_onestep(path, dev):
    """Load a checkpoint as one-step model (horizon 1); a 2-s model keeps everything but the time embedding of step 1."""
    ck = torch.load(path, map_location=dev); cfg = ModelConfig(**ck["cfg"]); src = ck["model"]
    if cfg.horizon != 1:
        cfg.horizon = 1; src = dict(src); src["time_emb"] = src["time_emb"][:1]
    cfg.contact_fut_in = 0
    m = PhysicsTransformer(cfg, node_in=cfg.context_len * 6).to(dev)
    own = m.state_dict(); own.update({k: v for k, v in src.items() if k in own and own[k].shape == v.shape}); m.load_state_dict(own)
    return m


def unroll(models, bb, noise=0.0, warm=0):
    """bb: ctx (B,2,K,N,3) and per-step items stacked on dim 1 -> predicted states (B,2,n,N,3), gradient through the rollout.
    warm: the first `warm` steps run without gradient and are not returned (the models' own states become the start)."""
    ctx, K, out = bb["ctx"], bb["ctx"].shape[2], []
    if noise:
        ctx = ctx + torch.cumsum(torch.randn_like(ctx) * noise / math.sqrt(K), 2)
    for j in range(bb["act"].shape[1]):
        b = {k: v[:, j] for k, v in bb.items() if k not in ("ctx", "stride", "warm")}
        if j < warm:
            with torch.no_grad():
                p = sc.predict(models, ctx, b, amp=False)
        else:
            p = sc.predict(models, ctx, b, amp=False)                                # (B,2,1,N,3)
            out.append(p[:, :, 0])
        ctx = torch.cat([ctx[:, :, 1:], p], 2)
    return torch.stack(out, 2)


@torch.no_grad()
def forecast6(models, ds, s, dev, batch=100):
    """6 s rollout from the simulated context ending at step s -> (E,2,300,N,3)."""
    K = ds.K; out = np.zeros((ds.E, 2, F6) + ds.pos.shape[3:], np.float32)
    for i in range(0, ds.E, batch):
        es = list(range(i, min(i + batch, ds.E)))
        its = [[ds.item(e, s + j) for j in range(F6)] for e in es]
        bb = {k: torch.stack([torch.stack([it[k] for it in row]) for row in its]).to(dev) for k in its[0][0]}
        bb["ctx"] = torch.from_numpy(np.stack([ds.pos[e, :, s - K + 1:s + 1] for e in es])).to(dev)
        out[i:i + len(es)] = unroll(models, bb).cpu().numpy()
    return out


def evaluate(models, dev, sets=("motion", "motion_test", "val", "test", "unseen"), sub=1):
    """6-s forecast error per set; sub: use every sub-th scene (int, or dict per set)."""
    [m.eval() for m in models]; res = {}
    for dn in sets:
        if not os.path.exists(PATHS[dn]): continue
        ds = sc.SceneWindows(PATHS[dn], models[0].cfg); g = ds.groups() if dn in ("val", "test") else ds.mode
        k = sub.get(dn, 1) if isinstance(sub, dict) else sub
        if k > 1:
            keep = np.arange(0, ds.E, k); g = g[keep]
            ds.pos, ds.pose, ds.action, ds.material, ds.E = ds.pos[keep], ds.pose[keep], ds.action[keep], ds.material[keep], len(keep)
        err = np.stack([np.linalg.norm(forecast6(models, ds, s, dev) - ds.pos[:, :, s + 1:s + F6 + 1], axis=-1) * 100
                        for s in (ds.K - 1, ds.T - 1 - F6)], 1)                      # (E,2 starts,2 cables,300,N)
        per = err.mean((1, 2, 3, 4)); q = {str(k): float(per[g == k].mean()) for k in sorted(set(g))}
        q.update(alle=float(per.mean()), sem=float(per.std() / np.sqrt(len(per))), gezogen=float(err[:, :, 0].mean()), U=float(err[:, :, 1].mean()),
                 pro_sekunde=[float(err[:, :, :, k * 50:(k + 1) * 50].mean()) for k in range(6)], n=int(len(per)))
        res[dn] = q
        print(f"{dn:11s} 6 s: {q['alle']:.2f} cm  per second {[round(x, 2) for x in q['pro_sekunde']]}", flush=True)
    [m.train() for m in models]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default=None); ap.add_argument("--eval", default=None); ap.add_argument("--train", default="data/u3/B_24000.npz")
    ap.add_argument("--out", required=True); ap.add_argument("--steps", type=int, default=6000); ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--unroll", type=int, default=32); ap.add_argument("--noise", type=float, default=0.0, help="random-walk noise on the start context [m]")
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--workers", type=int, default=8); ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--resvel", action="store_true", help="predict the change relative to constant velocity"); ap.add_argument("--keep", action="store_true", help="--init is already a one-step model: keep it as it is")
    ap.add_argument("--subset", type=int, default=0); ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--warm", type=int, default=0, help="up to this many own rollout steps without gradient before the trained steps")
    ap.add_argument("--warmup", type=int, default=200, help="learning-rate warm-up steps"); ap.add_argument("--select", default="motion", help="set used to select the best checkpoint")
    args = ap.parse_args(); dev = torch.device(args.device)
    if args.eval:
        models = [sc.load(f"{args.eval}/best_{c}.pt", dev)[0] for c in (0, 1)]
        json.dump(evaluate(models, dev), open(args.out, "w"), indent=1); return
    os.makedirs(args.out, exist_ok=True); json.dump(vars(args), open(os.path.join(args.out, "args.json"), "w"), indent=1)
    models = [to_onestep(p, dev) for p in args.init.split(",")]; [m.train() for m in models]
    if args.resvel and not args.keep:
        for m in models:
            m.cfg.residual_vel = True; torch.nn.init.zeros_(m.dec_mlp[-1].weight); torch.nn.init.zeros_(m.dec_mlp[-1].bias)
    mcfg = models[0].cfg
    ds = training_scenes(args.train, mcfg, subset=args.subset)
    dl = DataLoader(RolloutBatches(ds, args.batch, args.unroll, args.warm, args.steps), batch_size=None, shuffle=False, num_workers=args.workers, pin_memory=True)
    params = [p for m in models for p in m.parameters()]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps))))
    t0, run, log, best, last = time.time(), [], [], float("inf"), float("nan")
    for step, bb in enumerate(dl, 1):
        w = int(bb.pop("warm")); bb = {k: v.to(dev, non_blocking=True) for k, v in bb.items()}
        p = unroll(models, bb, args.noise, warm=w); gt = bb["gt"][:, w:, :, 0].transpose(1, 2)   # (B,2,n,N,3)
        loss = F.mse_loss(p * 100, gt * 100)                                                # cm^2
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step()
        run.append((p.detach()[:, :, -1] - gt[:, :, -1]).norm(dim=-1).mean().item() * 100)
        if step % 250 == 0:
            print(f"step {step} | {time.time() - t0:.0f}s | error after {args.unroll} steps: {np.mean(run):.3f} cm", flush=True); last = float(np.mean(run)); run = []
        if step % args.val_every == 0 or step == args.steps:
            r = evaluate(models, dev, ("motion", "val"), sub={"motion": 2, "val": 4}); log.append(dict(step=step, time=time.time() - t0, train_cm=last, **{k: v["alle"] for k, v in r.items()}))
            json.dump(log, open(os.path.join(args.out, "log.json"), "w"), indent=1)
            for c, m in enumerate(models):
                ck = {"model": m.state_dict(), "cfg": asdict_safe(mcfg)}
                ck["cfg"] = asdict_safe(m.cfg); torch.save(ck, os.path.join(args.out, f"last_{c}.pt"))
                if r[args.select]["alle"] < best:
                    torch.save(ck, os.path.join(args.out, f"best_{c}.pt"))
            best = min(best, r[args.select]["alle"])
    print(f"training finished, best {args.select}", best, flush=True)
    models = [sc.load(os.path.join(args.out, f"best_{c}.pt"), dev)[0] for c in (0, 1)]
    json.dump(evaluate(models, dev), open(os.path.join(args.out, "eval_6s.json"), "w"), indent=1)
    print("evaluation finished", flush=True)


if __name__ == "__main__":
    main()
