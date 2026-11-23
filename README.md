# DLO-Anything: Towards a Foundation Model for Deformable Linear Objects

Project page with videos: https://ludwigstr.github.io/DLO-Anything/

**Authors:** Ludwig Streloke (a, corresponding author, <ludwig.streloke@faps.fau.de>), Matthias Lang (a),
Alexander Weilacher (a), Christopher Straub (b), Annalena Hartmann (a), Jörg Franke (a)

(a) Friedrich-Alexander-Universität Erlangen-Nürnberg, Institute for Factory Automation and Production Systems (FAPS)\
(b) Fraunhofer Institute for Integrated Systems and Device Technology (IISB)

A physics-conditioned transformer that forecasts the motion of cables (deformable linear objects, DLOs) in contact
with each other and with the floor. A robot pulls one cable across a second cable that is fixed at both ends.
Each cable has its own model with 5.2 million parameters. The inputs are tokens for the cable nodes, the material,
the planned gripper motion, and one rigid-contact and one soft-contact token per node (closest point on the floor
and on the other cable). One pass predicts 0.5 s; a 6-s forecast chains twelve passes and never looks at the
simulation again after its 0.2 s of context.

## Files

| File | Purpose |
|---|---|
| `simulate.py` | MuJoCo scenes: training sets A to F and the held-out validation / test scenes |
| `model.py` | model configuration, material normalization, token construction, transformer (imported by the other scripts) |
| `train.py` | two-stage training of the two cable models |
| `evaluate.py` | 6-s forecast error of a model pair; fit of the scaling law |
| `make_videos.py` | videos of simulation and forecast, grid video |

The project page is in `docs/`.

## Setup

Python 3.11, PyTorch 2 with CUDA, MuJoCo 3.

```bash
pip install -r requirements.txt
```

Rendering is headless (`MUJOCO_GL=egl` is set by the scripts if the variable is not set). The videos use the
DejaVu Sans Mono fonts from `/usr/share/fonts/truetype/dejavu/`.

## Usage

```bash
# 1. data: a training set, a small set of trained motions for checkpoint selection, held-out validation and test
python simulate.py --set F --scenes 12000 --slim --out data/F_12000.npz --seed 1000000
python simulate.py --set A --scenes 300 --out data/val_designed.npz --seed 2000000
python simulate.py --held-out --scenes 300 --out data/val.npz  --seed 3000000
python simulate.py --held-out --scenes 340 --out data/test.npz --seed 4000000

# 2. training (stage 1: each cable model alone; stage 2: both on the chained 6-s forecast)
python train.py --train data/F_12000.npz --val data/val_designed.npz --out runs/F_12000

# 3. evaluation on robot motions that never occur in training
python evaluate.py --models runs/F_12000/model_cable0.pt,runs/F_12000/model_cable1.pt --data data/test.npz

# 4. video of one scene
python make_videos.py --models runs/F_12000/model_cable0.pt,runs/F_12000/model_cable1.pt --data data/test.npz --index 3
```

Every script explains its options with `--help`. A set of 12,000 scenes needs about 7 GB with `--slim`
(without the floor forces, which only the videos use); training one set takes several hours on one GPU.

## Training sets and held-out motions

Every set contains its robot motion kinds in equal shares.

| Set | Motion kinds | Robot motions |
|---|---|---|
| A | 1 | designed patterns (straight, angled, arc, S-curve, there and back, free swing) |
| B | 2 | A + random waypoints |
| C | 3 | B + spline curves |
| D | 4 | C + varying speed |
| E | 5 | D + retraced paths |
| F | 8 | E + loops, weaving, bursts |

Circular, sinusoidal and jerky robot motions are never training data: `simulate.py --set` cannot produce them, and
`--held-out` produces only them. Checkpoints are selected on trained motions (`--val`), so the held-out validation
and test scenes stay untouched during training. Unstable scenes and scenes in which the pulled cable winds around
the fixed cable by more than one turn are dropped.

## Results

Mean node error of the 6-s forecast in cm on the held-out test motions (341 scenes), one training run per entry:

| Training scenes | 371 | 743 | 1,486 | 2,971 | 6,000 | 12,000 | 24,000 |
|---|---|---|---|---|---|---|---|
| A (1 kind) | 5.26 | 4.53 | 4.10 | 3.67 | 3.48 | 3.32 | 3.42 |
| B (2 kinds) | 4.65 | 3.94 | 3.43 | 3.18 | 2.86 | 2.77 | 2.75 |
| C (3 kinds) | | | | | | 2.68 | |
| D (4 kinds) | | | | | | 2.52 | |
| E (5 kinds) | | | | | | 2.50 | |
| F (8 kinds) | 3.80 | 3.47 | 3.16 | 2.66 | 2.47 | 2.33 | 2.22 |

Scaling law for N training scenes with k motion kinds, fitted to these 24 runs (`python evaluate.py --fit`):

```
E(N, k) = a k^-beta + c (N / 1000)^-gamma      a = 2.93 cm, beta = 0.21, c = 1.29 cm, gamma = 0.50      R2 = 0.98
```

More scenes of the same kinds reduce the error only down to a floor, and this floor falls with every motion kind
that is added.

## Notes on reproduction

* This repository is a compact version of the research code. The simulation, the model and the training
  procedure are the same; the command line and the random seeds of the scenes differ, so regenerated data sets
  contain other scenes than the ones behind the table above.
* The models of the table started stage 1 from a transformer pretrained on single cables. That checkpoint and its
  training code are not part of this repository; `train.py` starts from random weights unless `--init` is given.
  Results from random initialization have not been measured.
* All results are from simulation.

## License

MIT License, see [LICENSE](LICENSE).
