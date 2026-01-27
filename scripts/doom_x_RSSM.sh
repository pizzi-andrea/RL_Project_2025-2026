#!/usr/bin/env bash

doom_scenarios=(
  "VizdoomBasic-v0"
  "VizdoomCorridor-v0"
  "VizdoomDefendCenter-v0"
  "VizdoomDefendLine-v0"
  "VizdoomHealthGatheringSupreme-v0"
  "VizdoomMyWayHome-v0"
)

episodes=(80 240 80 80 400 80)


skip_frame=(1 4 1 1 4 4)

sticky=("--sticky" "" "" "" "--sticky" "--sticky")
dataset_name="Doom4RSSM"

for idx in "${!episodes[@]}"; do
  echo "Generating for: ${doom_scenarios[$idx]} (Skip: ${skip_frame[$idx]}, Sticky: '${sticky[$idx]}')"

  # Generazione Training
  python src/doomGen.py \
    -d "$dataset_name" \
    -n "${episodes[$idx]}" \
    -s train \
    --reload \
    -e "${doom_scenarios[$idx]}" \
    --episodic \
    --skip_frame "${skip_frame[$idx]}" \
    ${sticky[$idx]} \
    -m
    
  
  python src/doomGen.py \
    -d "$dataset_name" \
    -n 1 \
    -s validation \
    --reload \
    -e "${doom_scenarios[$idx]}" \
    --episodic \
    --skip_frame "${skip_frame[$idx]}" \
    ${sticky[$idx]} \
    -m
    
done
