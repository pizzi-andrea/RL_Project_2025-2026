#!/usr/bin/env bash

doom_scenarios=("VizdoomBasic-v1")
episodes=(15000)
skip_frame=(4)

sticky=("--sticky")

dataset_name="Doom4RSSMs"

for idx in "${!episodes[@]}"; do
  
  CURRENT_STICKY=${sticky[$idx]}

  echo "Generating for: ${doom_scenarios[$idx]} (Skip: ${skip_frame[$idx]}, Sticky: '${CURRENT_STICKY}')"

  python ../src/doomGen.py \
    -d "$dataset_name" \
    -n "${episodes[$idx]}" \
    -s train \
    -e "${doom_scenarios[$idx]}" \
    --skip_frame "${skip_frame[$idx]}" \
    --resolution_h 64 \
    --resolution_w 64 \
    --crop \
    $CURRENT_STICKY \
    -t 1200
done
