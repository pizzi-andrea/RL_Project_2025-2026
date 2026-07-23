#!/usr/bin/env bash

RL_ROOT="/content/drive/MyDrive/RL"

python3 ../src/doomEval.py \
  -e VizdoomTakeCover-v1 \
  -n 30 \
  --skip_frame 4 \
  --save_frames \
  --dreamer_ckpt "$RL_ROOT/weights/VizdoomTakeCover/rssm.ckpt" \
  --policy_ckpt "$RL_ROOT/weights/VizdoomTakeCover/policy.ckpt" \
  --out_dir "$RL_ROOT/results/VizdoomTakeCover/evalresults"

  #--save_frames salva video (facoltativo)