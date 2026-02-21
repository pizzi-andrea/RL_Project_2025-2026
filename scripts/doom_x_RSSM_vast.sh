#!/usr/bin/env bash
set -e

doom_scenarios=(
  "VizdoomFreedoom2MAP01-S5-v0"
  "VizdoomFreedoom2MAP02-S5-v0"
  "VizdoomFreedoom2MAP03-S5-v0"
  "VizdoomFreedoom2MAP04-S5-v0"
  "VizdoomFreedoom2MAP05-S5-v0"
)

episodes=(80 240 80 80 80)
skip_frame=(1 4 1 1 4)
sticky=("" "" "" "" "")

dataset_name="Doom4RSSM"

for idx in "${!episodes[@]}"; do
  echo "Generating for: ${doom_scenarios[$idx]} (Skip: ${skip_frame[$idx]})"

  python ../src/doomGen.py \
    -d "$dataset_name" \
    -n "${episodes[$idx]}" \
    -s train \
    -e "${doom_scenarios[$idx]}" \
    --skip_frame "${skip_frame[$idx]}" \
    ${sticky[$idx]} \
    -t 300

  python ../src/doomGen.py \
    -d "$dataset_name" \
    -n 1 \
    -s validation \
    -e "${doom_scenarios[$idx]}" \
    --skip_frame "${skip_frame[$idx]}" \
    ${sticky[$idx]} \
    -t 300
done

chmod +x scripts/doom_x_RSSM_vast.sh
