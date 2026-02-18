#!/usr/bin/env bash

doom_scenarios=("VizdoomDeadlyCorridor-v1" "VizdoomBasicNotifications-v1" "VizdoomDefendLine-v1")
episodes=(7000 1000 2000)
skip_frame=(2 4 4)

sticky=("--sticky" "--sticky" "--sticky")

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
    -t 1200
done
