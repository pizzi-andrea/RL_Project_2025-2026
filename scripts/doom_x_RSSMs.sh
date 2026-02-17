#!/usr/bin/env bash

doom_scenarios=("VizdoomCorridor-v0" "VizdoomBasic-v0")
episodes=(8000 2000)
skip_frame=(1 2)

sticky=("--sticky" "")

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
    $CURRENT_STICKY \
    -t 2100
done
