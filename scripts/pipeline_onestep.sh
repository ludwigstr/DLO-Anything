#!/bin/bash
# Training pipeline of the one-step model (the transformer predicts the next 20 ms; a forecast is a rollout in which the
# contact tokens are rebuilt from the predicted state before every step).
#
#   usage: scripts/pipeline_onestep.sh <train.npz> <out_dir> <gpu>        e.g.  data/u3/B_24000.npz runs/u3/one_final 0
#
# Data rules (enforced by dlo.onestep):
#   * circle, sine and jerk robot motions are validation / test only (data/val_motion.npz, data/test_motion.npz) and are
#     never part of a training file; the trainer aborts if a training file contains them
#   * scenes in which the pulled cable wraps around the U are removed from training (scripts/wrap.py writes <train>.wrap.npy)
#   * the checkpoint of every stage is selected on data/val_motion.npz; data/test_motion.npz is only used for the final numbers
set -e
cd "$(dirname "$0")/.."; export PYTHONPATH=.
TRAIN=${1:-data/u3/B_24000.npz}; OUT=${2:-runs/u3/one_final}; GPU=cuda:${3:-0}; L=$OUT/logs; mkdir -p $L
PRE=runs/scale/M_n150k/best.pt          # single-cable pretraining (dlo.train)

# 0. wrap scenes of the training file
[ -f $TRAIN.wrap.npy ] || python scripts/wrap.py $GPU $TRAIN

# 1. both cable models on single 2-s passes from simulated states (contact tokens per own node, floor tokens)
for C in 0 1; do
  [ -f $OUT/s1_$C/best.pt ] || python -m dlo.soft_contact --train $TRAIN --cable $C --own --rigid --init $PRE --epochs 40 --spe 4 --limit-steps 743 \
      --workers 4 --device $GPU --out $OUT/s1_$C > $L/s1_$C.log 2>&1 &
done; wait

# 2. conversion to the one-step model and rollout training with the gradient through 64 fed-back steps
[ -f $OUT/s2/best_0.pt ] || python -m dlo.onestep --init $OUT/s1_0/best.pt,$OUT/s1_1/best.pt --train $TRAIN --out $OUT/s2 \
    --steps 26000 --batch 16 --unroll 64 --lr 1e-4 --val-every 2000 --device $GPU > $L/s2.log 2>&1

# 3. long rollouts: the trained 64 steps start from the models' own forecast, up to 236 steps (4.7 s) after the simulated state
[ -f $OUT/s3/best_0.pt ] || python -m dlo.onestep --init $OUT/s2/best_0.pt,$OUT/s2/best_1.pt --keep --train $TRAIN --out $OUT/s3 \
    --steps 20000 --batch 16 --unroll 64 --warm 236 --lr 3e-5 --warmup 1000 --val-every 2000 --workers 12 --device $GPU > $L/s3.log 2>&1

# 4. final numbers: motion validation, motion test, in-distribution validation / test, unseen scene layouts
python -m dlo.onestep --eval $OUT/s3 --out $OUT/eval_6s.json --device $GPU | tee $L/eval.log
