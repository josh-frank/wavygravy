#!/usr/bin/env bash
# usage: WAVYGRAVY_PASSWORD=... ./samples.sh out.wav
set -euo pipefail
IN="${1:?usage: $0 encoded.wav}"

DUR=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$IN")

# random cut of `len` seconds -> $2 ; prints the start offset used
random_cut() {
  local len="$1" out="$2" start
  start=$(awk -v d="$DUR" -v l="$len" -v s="$RANDOM" 'BEGIN{srand(s); printf "%.6f", rand()*(d-l)}')
  ffmpeg -v error -y -i "$IN" -ss "$start" -t "$len" -c:a pcm_s16le "$out"
  echo "kept ${len}s from ${start}s -> $out"
}

HALF=$(awk -v d="$DUR" 'BEGIN{printf "%.6f", d/2}')
random_cut "$HALF" half.wav      # random half-length portion
random_cut 10      sample10.wav  # random ten-second sample

for f in half.wav sample10.wav; do
  printf '%s: ' "$f"
  python wavygravy.py decode "$f" 2>&1 || true
  echo
done