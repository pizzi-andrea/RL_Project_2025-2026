#!/usr/bin/env bash

doom_scenarios=(
  "VizdoomDefendLine-v0"
)

episodes=(10)

dataset_name="Doom4RSSM"

for idx in "${!episodes[@]}"; do
  python src/doomGen.py \
    -d "$dataset_name" \
    -n "${episodes[$idx]}" \
    -s train \
    --reload \
    -e "${doom_scenarios[$idx]}" \
    --episodic \
    -m
    
done
