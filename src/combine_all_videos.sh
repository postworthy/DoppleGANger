#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

# REQUIREMENTS: ffmpeg, ffprobe

# 1) Gather all .mp4 files
files=( *.mp4 )
N=${#files[@]}
if (( N < 1 )); then
  echo "ERROR: No .mp4 files found in $(pwd)." >&2
  exit 1
fi
echo "Found $N .mp4 file(s)."

# 2) Probe height, width, and integer duration (for longest determination)
declare -a heights widths durations_int
for i in "${!files[@]}"; do
  f="${files[i]}"
  heights[i]=$(ffprobe -v error -select_streams v:0 \
    -show_entries stream=height -of csv=p=0 "$f")
  widths[i]=$(ffprobe -v error -select_streams v:0 \
    -show_entries stream=width  -of csv=p=0 "$f")
  durations_int[i]=$(ffprobe -v error \
    -show_entries format=duration -of csv=p=0 "$f" \
    | awk '{printf "%d",$1}')
done

# 3) Find the minimum height
min_h=${heights[0]}
for h in "${heights[@]}"; do
  (( h < min_h )) && min_h=$h
done

# 4) Compute scaled widths at that height, maintain aspect ratio
declare -a scaled_w
for i in "${!files[@]}"; do
  scaled_w[i]=$(awk "BEGIN { printf \"%d\", (${widths[i]} * $min_h / ${heights[i]}) }")
done

# 5) Find the minimum scaled width
min_w=${scaled_w[0]}
for w in "${scaled_w[@]}"; do
  (( w < min_w )) && min_w=$w
done

echo "→ Target dimensions for all videos: ${min_w}×${min_h}"

# 6) Identify the longest video (by integer seconds)
max_d=0; max_idx=0
for i in "${!files[@]}"; do
  d=${durations_int[i]}
  if (( d > max_d )); then
    max_d=$d
    max_idx=$i
  fi
done
echo "→ Longest video: '${files[max_idx]}' (${max_d}s) → will be center"

# 7) Re-encode each clip: scale to min_h, center-crop to min_w×min_h, drop audio
declare -a fixed
for i in "${!files[@]}"; do
  f="${files[i]}"
  out="${f%.*}_fixed.mp4"
  ffmpeg -y -i "$f" -vf "scale=-2:${min_h},crop=${min_w}:${min_h}:(in_w-${min_w})/2:0,setsar=1:1" \
         -c:v libx264 -crf 18 -preset veryfast -an \
         "$out"
  fixed[i]="$out"
done

# 8) Build final input order: left_part, middle (longest), right_part
mid_pos=$(( N / 2 ))
order=()
for i in "${!files[@]}"; do
  (( i == max_idx )) || order+=( "$i" )
done
left_part=( "${order[@]:0:mid_pos}" )
right_part=( "${order[@]:mid_pos}" )
final_order=( "${left_part[@]}" "$max_idx" "${right_part[@]}" )

# 9) Get the exact (decimal) duration of the middle clip
mid_fixed="${fixed[max_idx]}"
mid_dur=$(ffprobe -v error -select_streams v:0 \
           -show_entries format=duration \
           -of default=noprint_wrappers=1:nokey=1 \
           "$mid_fixed")
echo "→ Middle clip exact duration: ${mid_dur}s"

# 10) Build the filter_complex string for N-way hstack
inputs_labels=$(printf "[%d:v]" $(seq 0 $((N-1))))
filter_complex="${inputs_labels}hstack=inputs=${N},format=yuv420p"

# 11) Assemble and run the combine command, looping shorter clips
cmd=( ffmpeg -y )
for idx in "${final_order[@]}"; do
  if (( idx == max_idx )); then
    cmd+=( -i "${fixed[idx]}" )
  else
    cmd+=( -stream_loop -1 -i "${fixed[idx]}" )
  fi
done
cmd+=( -filter_complex "$filter_complex" \
       -c:v libx264 -crf 18 -preset veryfast \
       -t "$mid_dur" \
       combined.mp4 )

# Execute
"${cmd[@]}"

echo "✅ Done: combined.mp4 (${min_w}×${min_h}, ${mid_dur}s)"
