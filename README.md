# DLO-Anything: Towards a Foundation Model for Deformable Linear Objects

Project page with abstract and videos: https://ludwigstr.github.io/DLO-Anything/

## Setup

Python 3.11, PyTorch 2 with CUDA, MuJoCo 3.

```bash
pip install -r requirements.txt
```

## Scripts

Run the scripts from the repository root, in this order. Each one lists its options with `--help`.

### 1. `simulate.py`: create the scenes

```bash
python simulate.py --set F --scenes 12000 --slim --out data/train.npz --seed 1000000    # training set
python simulate.py --set A --scenes 300 --out data/val_designed.npz --seed 2000000      # for checkpoint selection
python simulate.py --held-out --scenes 300 --out data/val.npz  --seed 3000000           # validation
python simulate.py --held-out --scenes 340 --out data/test.npz --seed 4000000           # test
```

* `--set A` to `--set F` chooses the robot motion kinds of a training set (A: one kind, F: eight kinds).
* `--held-out` creates scenes with circular, sinusoidal and jerky robot motions. They are for validation and test only.
* Use a different `--seed` for every file.
* `--slim` leaves out the floor forces, which only the videos use. 12,000 scenes then take about 7 GB.

### 2. `train.py`: train the two cable models

```bash
python train.py --train data/train.npz --val data/val_designed.npz --out runs/F_12000
```

Writes `runs/F_12000/model_cable0.pt` (pulled cable) and `model_cable1.pt` (fixed cable). `--val` selects the
checkpoints and must not be a held-out file. Training starts from random weights unless `--init` names a checkpoint.

### 3. `evaluate.py`: measure the forecast error

```bash
python evaluate.py --models runs/F_12000/model_cable0.pt,runs/F_12000/model_cable1.pt --data data/test.npz
python evaluate.py --fit
```

The first command prints the mean node error of the 6-s forecast in cm, in total and per robot motion.
`--fit` fits the scaling law to the results of the paper.

### 4. `make_videos.py`: render videos

```bash
python make_videos.py --models runs/F_12000/model_cable0.pt,runs/F_12000/model_cable1.pt --data data/test.npz --index 3 --out videos
python make_videos.py --grid videos/tiles --rows 5 --cols 5 --out videos/grid.mp4
```

The first command renders scene 3 of the file (simulation dark, forecast orange). For a file created with `--slim`
the floor contact of the simulation is not drawn. The second command arranges the videos of a folder in a grid;
render those with `--plain --margin 0.12`.

### `model.py`

The model itself. It is imported by the other scripts and is not run directly.

## License

MIT License, see [LICENSE](LICENSE).
