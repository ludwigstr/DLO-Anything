"""Wrap detection: how far the pulled cable winds around the U cable (degrees), per episode.
For every frame, the nodes of the pulled cable that are within 3 cm of the U are taken; their position relative to the closest point
on the U is projected onto the plane normal to the U at that point, and the angle swept along the run of consecutive close nodes is
the winding. Lying over the U gives up to about 180 degrees; a wrap gives clearly more.   usage: wrap.py device file [file ...]
Writes <file>.wrap.npy (max winding in degrees per episode)."""
import sys, numpy as np, torch
from dlo import soft_contact as sc
dev = torch.device(sys.argv[1]); NEAR = 0.03

@torch.no_grad()
def winding(pos):                                          # (B,2,T,N,3) -> (B,) degrees
    p0, p1 = pos[:, 0], pos[:, 1]
    q, arc, gap, i, ti = sc.closest_on_cable(p0, p1)       # (B,T,N,...)
    seg = p1[:, :, 1:] - p1[:, :, :-1]
    tan = torch.gather(seg, 2, i[..., None].expand(-1, -1, -1, 3)); tan = tan / tan.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    near = gap < NEAR
    r = p0 - q; r = r - (r * tan).sum(-1, keepdim=True) * tan
    # angle between consecutive relative vectors, signed around the U tangent
    a, b = r[:, :, :-1], r[:, :, 1:]
    d = torch.atan2((torch.cross(a, b, dim=-1) * tan[:, :, :-1]).sum(-1), (a * b).sum(-1))
    d = d * (near[:, :, :-1] & near[:, :, 1:])
    # largest absolute sum over a run of consecutive close nodes
    best = torch.zeros(d.shape[:2], device=d.device); run = torch.zeros_like(best)
    for k in range(d.shape[-1]):
        run = torch.where(d[..., k] != 0, run + d[..., k], torch.zeros_like(run)); best = torch.maximum(best, run.abs())
    return torch.rad2deg(best.max(-1)[0])

for f in sys.argv[2:]:
    d = np.load(f, mmap_mode=None); pos = d["pos"]; out = []
    for i in range(0, len(pos), 64): out.append(winding(torch.from_numpy(pos[i:i + 64]).to(dev).float()).cpu().numpy())
    w = np.concatenate(out); np.save(f + ".wrap.npy", w)
    print(f, len(w), "episodes | winding > 180: %d, > 270: %d, > 360: %d | quantiles 50/90/99: %s" % ((w > 180).sum(), (w > 270).sum(), (w > 360).sum(), np.quantile(w, [.5, .9, .99]).round(0)), flush=True)
