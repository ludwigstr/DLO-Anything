"""Train both cable models on the chained 6 s forecast: three 2 s passes, every pass starts from the models' own
prediction of the pass before, contact tokens recomputed from the predicted states of both cables.

    python -m dlo.train_chain --init runs/u3/B_24000_0/best.pt,runs/u3/B_24000_1/best.pt --train data/u3/B_24000.npz --out runs/u3/chain_B24000
"""
from __future__ import annotations
import argparse, json, math, os, time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .config import asdict_safe
from .dataset import POS_SCALE
from .soft_contact import SceneWindows, load, predict


def chain(models, bb, W, detach):
    """-> list of W predictions (B,2,H,N,3); pass w+1 gets the last K predicted steps of pass w as context."""
    ctx, K, out = bb["ctx"], bb["ctx"].shape[2], []
    for w in range(W):
        b = {k: v[:, w] for k, v in bb.items() if k != "ctx"}
        p = predict(models, ctx, b); out.append(p)
        ctx = torch.cat([ctx, p.detach() if detach else p], 2)[:, :, -K:]          # horizons shorter than K keep older context steps
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True); ap.add_argument("--train", default="data/u3/B_24000.npz"); ap.add_argument("--val", default="data/val_u2.npz")
    ap.add_argument("--out", required=True); ap.add_argument("--windows", type=int, default=3)
    ap.add_argument("--steps", type=int, default=3000); ap.add_argument("--batch", type=int, default=64); ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--subset", type=int, default=0)
    ap.add_argument("--detach", action="store_true", help="no gradient through the predicted context")
    ap.add_argument("--workers", type=int, default=6); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--val-every", type=int, default=500)
    args = ap.parse_args(); dev = torch.device(args.device); W = args.windows
    os.makedirs(args.out, exist_ok=True); json.dump(vars(args), open(os.path.join(args.out, "args.json"), "w"), indent=1)
    models = [load(p, dev)[0] for p in args.init.split(",")]; mcfg = models[0].cfg
    [m.train() for m in models]
    tr = SceneWindows(args.train, mcfg, spe=1, n_windows=W, subset=args.subset)
    tr.spe = math.ceil(args.steps * args.batch / tr.E) + 1
    va = SceneWindows(args.val, mcfg, 4, seed=1, n_windows=W)
    trl = DataLoader(tr, args.batch, shuffle=True, num_workers=args.workers, drop_last=True, pin_memory=True)
    val = DataLoader(va, args.batch, num_workers=4)
    opt = torch.optim.AdamW([p for m in models for p in m.parameters()], lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 200) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps))))

    @torch.no_grad()
    def validate():
        [m.eval() for m in models]; tot, n = 0.0, 0
        for bb in val:
            bb = {k: v.to(dev) for k, v in bb.items()}
            ps = chain(models, bb, W, True)
            e = torch.stack([(p - bb["gt"][:, w]).norm(dim=-1).mean((1, 2, 3)) for w, p in enumerate(ps)], 1).mean(1) * 100
            tot += e.sum().item(); n += len(e)
        [m.train() for m in models]
        return tot / n

    def save(name):
        for c, m in enumerate(models):
            torch.save({"model": m.state_dict(), "cfg": asdict_safe(mcfg)}, os.path.join(args.out, f"{name}_{c}.pt"))

    log, best, t0, step, run = [], validate(), time.time(), 0, 0.0
    print(f"start: 6 s chain validation {best:.2f} cm", flush=True); save("best")
    while step < args.steps:
        for bb in trl:
            bb = {k: v.to(dev, non_blocking=True) for k, v in bb.items()}
            ps = chain(models, bb, W, args.detach)
            loss = sum(F.mse_loss(p * POS_SCALE, bb["gt"][:, w] * POS_SCALE) for w, p in enumerate(ps)) / W
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_([p for m in models for p in m.parameters()], 1.0)
            opt.step(); sched.step(); step += 1; run += loss.item()
            if step % args.val_every == 0 or step == args.steps:
                v = validate(); log.append(dict(step=step, train_mse=run / args.val_every, val_cm=v, time=time.time() - t0)); run = 0.0
                print(f"step {step} train_mse {log[-1]['train_mse']:.5f} | 6 s chain validation {v:.2f} cm | {time.time() - t0:.0f}s", flush=True)
                json.dump(log, open(os.path.join(args.out, "log.json"), "w"), indent=1)
                if v < best:
                    best = v; save("best")
            if step >= args.steps:
                break
    print("best score", best, flush=True)


if __name__ == "__main__":
    main()
