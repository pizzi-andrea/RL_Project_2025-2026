#!/usr/bin/env bash

doom_scenarios=("VizdoomDeadlyCorridor-v1")
episodes=(100)
skip_frame=(4)

sticky=("")

dataset_name="DC"

for idx in "${!episodes[@]}"; do
  
  CURRENT_STICKY=${sticky[$idx]}

  echo "Generating for: ${doom_scenarios[$idx]} (Skip: ${skip_frame[$idx]}, Sticky: '${CURRENT_STICKY}')"

  python3 ../src/doomGen.py \
    -d "$dataset_name" \
    -n "${episodes[$idx]}" \
    -s train \
    -e "${doom_scenarios[$idx]}" \
    --skip_frame "${skip_frame[$idx]}" \
    --resolution_h 64 \
    --resolution_w 64 \
    --crop \
    $CURRENT_STICKY \
    -t 2100
done
