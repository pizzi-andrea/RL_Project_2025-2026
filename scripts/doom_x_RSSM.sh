#!/usr/bin/env bash

doom_scenarios=(
  "VizdoomDefendLine-v0"
)

episodes=(500)

dataset_name="Doom4RSSM"

for idx in "${!episodes[@]}"; do
  python src/doomGen.py \
    -d "$dataset_name" \
    -n "${episodes[$idx]}" \
    -s train \
    -e "${doom_scenarios[$idx]}" \
    --episodic True
done
