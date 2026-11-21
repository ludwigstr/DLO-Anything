# DLO-Anything

Project page with videos: https://ludwigstr.github.io/DLO-Anything/

A material-conditioned physics transformer that forecasts the motion of cables (deformable linear objects, DLOs)
in contact with each other and with the floor.

**Authors:** Ludwig Streloke (a, corresponding author, <ludwig.streloke@faps.fau.de>), Matthias Lang (a),
Christopher Straub (b), Annalena Hartmann (a), Jörg Franke (a)

(a) Friedrich-Alexander-Universität Erlangen-Nürnberg, Institute for Factory Automation and Production Systems (FAPS)\
(b) Fraunhofer Institute for Integrated Systems and Device Technology (IISB)

**Status:** work in progress. The code and the results are still changing.

## What this is

* **Simulation.** A MuJoCo scene with two cables of 40 segments each. One cable is pulled by a robot gripper across
  a second cable that hangs as a fixed "U". The cables touch each other and the floor.
* **Model.** A physics transformer with about 5.2 M parameters, one model per cable, conditioned on the material
  parameters of its cable. The one-step model predicts the next 20 ms; a forecast is a rollout of this step.
  Before every step each cable node gets one *soft-contact token* (the closest point on the other cable) and one
  *rigid-contact token* (the floor patch below the node). Both are rebuilt from the predicted state, so the
  forecast never looks at the simulation again after its start.
  Tokens per pass: 41 node, 6 material, 1 action, 41 soft-contact, 41 rigid-contact, 1 global.
* **Training data.** 24,000 simulated scenes with straight, angled, arc, S-curve and there-and-back crossings and
  random-waypoint motions of the gripper.
* **Validation.** Robot motions of kinds that never occur in training: circles, sine oscillations and jerky
  start-stop motions (`data/val_motion.npz`, 303 scenes; a separate test set `data/test_motion.npz`, 341 scenes).
* **Metric.** Mean node position error of a 6-s forecast that starts from 0.2 s of simulated context.

Project page with videos: see `docs/index.html` (served with GitHub Pages from the `/docs` folder).

## Repository layout

```
dlo/                 Python package
  config.py            simulation and model configuration, material parameter ranges
  cable_env.py         single-cable MuJoCo environment
  two_cable_env.py     two-cable MuJoCo environment
  u_env.py             the "U" scenario: one cable pulled across a fixed, hanging cable
  generate_data.py     single-cable data generation (pretraining)
  generate_u.py        two-cable scenes with crossing motions
  generate_u_ood.py    two-cable scenes with unseen layouts, random-waypoint motions,
                       and the held-out circle / sine / jerk motions
  dataset.py           windowed datasets
  model.py             the physics transformer
  train.py             single-cable training
  soft_contact.py      soft- and rigid-contact tokens, per-cable training with contact tokens
  onestep.py           one-step model: rollout training and 6-s evaluation
  train_chain.py       chained multi-pass training (longer-horizon model variants)
  contact_model.py, contact_data.py, force_model.py, couple.py
                       earlier contact / force modules; still imported by the video code
  make_videos*.py      video rendering (make_videos_soft.py renders the videos of the project page)
scripts/
  gen.sh               generates the two-cable training scenes
  make_u3.py           assembles the nested training files data/u3/{A,B}_<n>.npz
  wrap.py              winding of the pulled cable around the U, per scene
  pipeline_onestep.sh  complete training pipeline of the one-step model
  eval_h.py, eval_motion.py                      evaluation scripts used during development
  diag_contact.py, diag_memo.py, diag_noise.py   diagnostic scripts used during development
docs/                project page (static HTML) and videos
sync.sh              refreshes dlo/ and scripts/ from the development tree
```

The `eval_*.py` and `diag_*.py` scripts contain checkpoint paths of specific experiments
(for example `runs/u3/hz/H25_chain2`, `runs/u3/one_long64`) and write to `stats/u3/`. These paths are examples:
replace them with your own run directories and create the output directory before running them.
Some labels and result keys in the code are in German.

## Setup

* Python 3.11
* MuJoCo 3.x (Python bindings)
* PyTorch 2.x with CUDA
* numpy, imageio with ffmpeg (`imageio-ffmpeg`), Pillow

```bash
pip install -r requirements.txt
```

Rendering is headless via EGL. The package sets `MUJOCO_GL=egl` when `dlo` is imported before `mujoco`
and the variable is not set; you can also export it yourself:

```bash
export MUJOCO_GL=egl
```

The video code uses the DejaVu Sans Mono fonts from `/usr/share/fonts/truetype/dejavu/`.

All commands below are run from the repository root. Data go to `data/`, runs to `runs/`, videos to `videos/`;
these directories are not part of the repository.

## Reproduce

### (a) Single-cable pretraining

Data: single cables carried by the gripper along random trajectories, in chunks of 10,000 episodes.

```bash
python -m dlo.generate_data --episodes 10000 --seed 3600000 --style mix --out data/train_wide_6.npz
python -m dlo.generate_data --episodes 200   --seed 500000  --style mix --out data/val_wide.npz
```

Training (`--train` takes a comma-separated list of files):

```bash
python -m dlo.train --train data/train_wide_1.npz,data/train_wide_2.npz,... --val data/val_wide.npz \
    --out runs/scale/M_n150k --lr 2e-4 --total-steps 120000 --device cuda:0
```

The pretrained model used by the pipeline (`runs/scale/M_n150k/best.pt`) was trained on 15 such chunks
(`train_wide_1` … `train_wide_15`, about 150,000 episodes) with the default model size
(`--d_model 256 --layers 6 --heads 8`). The seeds of chunks 6–15 are `3000000 + i * 100000`; the seeds of the
first five chunks and of the validation file are not recorded here, the ones above are placeholders.

### (b) Two-cable scenes

In-distribution training, validation and test scenes (crossing motions), and validation scenes with unseen layouts:

```bash
python -m dlo.generate_u --episodes 3000 --seed 510000 --out data/train_u2.npz
python -m dlo.generate_u --episodes 300  --seed 520000 --out data/val_u2.npz
python -m dlo.generate_u --episodes 300  --seed 530000 --out data/test_u2.npz
python -m dlo.generate_u_ood --per-family 80 --seed 990000 --out data/val_u2_ood.npz
```

The large training set:

```bash
bash scripts/gen.sh
```

`scripts/gen.sh` generates further crossing scenes (`dlo.generate_u`, files `data/u3/narrow_*.npz`) and
random-waypoint scenes (`dlo.generate_u_ood --families random --slim`, files `data/u3/broad_*.npz`). After every
chunk it calls `scripts/make_u3.py`, which writes nested training files of 371 to 24,000 scenes:
`data/u3/A_<n>.npz` (crossing scenes only: `data/train_u2.npz` followed by the `narrow` chunks) and
`data/u3/B_<n>.npz` (every second scene is a random-waypoint scene). The model is trained on
`data/u3/B_24000.npz`. The number of worker processes is set at the top of `scripts/gen.sh` (`W=36`).

### (c) Validation and test sets with held-out robot motions

```bash
python -m dlo.generate_u_ood --families circle,sine,jerk --per-family 110 --out data/val_motion_raw.npz --seed 995000
python -m dlo.generate_u_ood --families circle,sine,jerk --per-family 125 --out data/test_motion_raw.npz --seed 996000
```

Wrap filtering: `scripts/wrap.py` computes, for every scene, how far the pulled cable winds around the U (in
degrees) and writes the result to `<file>.wrap.npy`. Lying over the U gives up to about 180 degrees; a wrap gives
clearly more. Scenes above 360 degrees are removed by `scripts/make_motion_sets.py`.

```bash
PYTHONPATH=. python scripts/wrap.py cuda:0 data/val_motion_raw.npz data/test_motion_raw.npz
python scripts/make_motion_sets.py data/val_motion_raw.npz  data/val_motion.npz     # 303 scenes
python scripts/make_motion_sets.py data/test_motion_raw.npz data/test_motion.npz    # 341 scenes
```

### (d) Training pipeline of the one-step model

```bash
bash scripts/pipeline_onestep.sh <train.npz> <out_dir> <gpu>
# e.g.
bash scripts/pipeline_onestep.sh data/u3/B_24000.npz runs/u3/one_final 0
```

The script expects the pretrained single-cable model at `runs/scale/M_n150k/best.pt` (variable `PRE`) and the
validation files from (b) and (c). Stages that already have a `best` checkpoint are skipped. Logs go to
`<out_dir>/logs/`.

0. **Wrap scenes.** `scripts/wrap.py` writes `<train.npz>.wrap.npy` if it does not exist; the trainer leaves out
   scenes above 360 degrees.
1. **Per-cable models with contact tokens** (`dlo.soft_contact --own --rigid`, one run per cable, started from the
   pretrained model): single 2-s passes from simulated states, with one soft-contact token and one rigid-contact
   token per node. Output: `<out_dir>/s1_0`, `<out_dir>/s1_1`.
2. **One-step model** (`dlo.onestep --init s1_0/best.pt,s1_1/best.pt`): conversion to the model that predicts the
   next 20 ms, then rollout training with the gradient running through 64 fed-back steps (`--unroll 64`).
   Output: `<out_dir>/s2`.
3. **Long rollouts** (`dlo.onestep --keep --warm 236`): the 64 trained steps start from the models' own forecast,
   up to 236 steps (4.7 s) after the simulated state. Output: `<out_dir>/s3`.
4. **Evaluation** (`dlo.onestep --eval <out_dir>/s3`): 6-s forecast error on the motion validation set, the motion
   test set, the in-distribution validation and test sets and the unseen scene layouts, written to
   `<out_dir>/eval_6s.json`.

The checkpoint of stages 2 and 3 is selected on `data/val_motion.npz`; `data/test_motion.npz` is only used for the
final numbers.

### (e) Evaluation and videos

Evaluation of a run directory that contains `best_0.pt` and `best_1.pt`:

```bash
python -m dlo.onestep --eval runs/u3/one_final/s3 --out runs/u3/one_final/eval_6s.json --device cuda:0
```

Videos (simulation as a dark band, model forecast on top; one continuous forecast over the whole episode):

```bash
python -m dlo.make_videos_soft --token a.pt,b.pt --ohne a.pt,b.pt --solo --name-b "DLO-Anything" \
    --data data/val_motion.npz --index 154 --out videos/val_motion
```

`--token` takes the checkpoints of the pulled cable and of the U (`best_0.pt,best_1.pt`). `--ohne` takes a second
model pair for comparison (a model without contact tokens); with `--solo` only the `--token` pair is shown, and the
same pair can be passed to both options. `--index` is the scene index in the data file. Render one episode per
process: call the command once per index.

The videos of the project page are the scenes 154, 184, 177, 4 and 36 of `data/val_motion.npz`.

## Data rules

* **Held-out motions are never in training.** Circle, sine and jerk motions of the gripper are used for
  validation and test only. `dlo.onestep` aborts if a training file contains scenes of these kinds.
* **Wrap scenes are excluded.** Scenes in which the pulled cable wraps around the U (winding above 360 degrees)
  are removed from the training file by the trainer (via `<train.npz>.wrap.npy`) and are not part of
  `data/val_motion.npz` and `data/test_motion.npz`.
* Checkpoints are selected on the motion validation set; the motion test set is used for final numbers only.

## Refreshing the code

`./sync.sh [PROJECT_DIR]` copies the released file list from the development tree again and re-applies the path
adaptations of the scripts (default `PROJECT_DIR`: two levels up).

## License

MIT License, see [LICENSE](LICENSE).
