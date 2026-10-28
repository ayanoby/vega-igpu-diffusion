#!/bin/bash
# usage: ./upscale.sh [--x2] image.png [more.png ...]   -> image_x4.png (or image_x2.png) next to the original
# realesr-animevideov3 via realesrgan-ncnn-vulkan (Vulkan, ~2 s per 512x768 image); --x2 runs the native x2 model
# the old sd.cpp path (RealESRGAN Anime 6B, ~26 s): ./bin-opt-sd-cli -M upscale -i IN --upscale-model upscalers/RealESRGAN_x4plus_anime_6B.pth -o OUT
RE="$(dirname "$(realpath "$0")")/../deps/realesrgan"
SCALE=4
[ "$1" = "--x2" ] && { SCALE=2; shift; }
[ $# -gt 0 ] || { echo "usage: $0 [--x2] image.png [more.png ...]"; exit 1; }
for f in "$@"; do
  [ -f "$f" ] || { echo "no such file: $f"; continue; }
  out="${f%.*}_x$SCALE.png"
  s=$(date +%s.%N)
  "$RE/realesrgan-ncnn-vulkan" -i "$f" -o "$out" -n realesr-animevideov3 -s $SCALE -m "$RE/models" >/dev/null 2>&1 \
    || { echo "error: upscale failed for $f"; continue; }
  echo "upscaled, taking $(python3 -c "import time; print(round(time.time() - $s, 2))")s"
  echo "$out"
done
