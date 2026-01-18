#!/usr/bin/env bash

doom_scenarios=(
  "VizdoomBasic-v0"
  "VizdoomCorridor-v0"
  "VizdoomDeathmatch-v0"
  "VizdoomDefendCenter-v0"
  "VizdoomDefendLine-v0"
  "VizdoomHealthGatheringSupreme-v0"
  "VizdoomMyWayHome-v0"
)

episodes=(8 24 40 8 8 40 8)

dataset_name="Doom4VAE"

for idx in "${!episodes[@]}"; do
  python doomGen.py \
    -d "$dataset_name" \
    -n "${episodes[$idx]}" \
    -s train \
    -e "${doom_scenarios[$idx]}" \
    -r True \
    -m False

  python doomGen.py \
    -d "$dataset_name" \
    -n 1 \
    -s validation \
    -e "${doom_scenarios[$idx]}" \
    -r True \
    -m False

  python doomGen.py \
    -d "$dataset_name" \
    -n 1 \
    -s test \
    -e "${doom_scenarios[$idx]}" \
    -r True \
    -m False
done
