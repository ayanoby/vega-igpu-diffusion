#!/bin/bash
# usage: ./gen-anima.sh "prompt" [steps=4] [seed] [WxH=512x768]   -> out/<timestamp>.png (or $OUT)
# Anima Turbo v1.1 (official circlestone-labs/Anima, f16) or any Anima model with a turbo LoRA baked in: CFG 1, 4 steps
# MODEL=file.safetensors in anima/models picks another one; on a Vega 7 iGPU 512x768 takes ~23 s once loaded
# the f16 model is made by ./merge_lora.py (Vega has no bf16, and baking skips the per-run LoRA apply); text encoder runs on the CPU
# decoding uses the tiny VAE taew2_1 on the GPU in 128px tiles (~3x faster than the full VAE, nearly identical, 108 MB buffer);
# untiled it needs ~2.6 GB at 512x768, which together with the weights exhausted the shared memory and froze the laptop.
# VAE=full ./gen-anima.sh ... uses the full Qwen-Image VAE (512x384 tiles) instead
cd "$(dirname "$0")"
mkdir -p out
# two copies of the model do not fit in the shared GPU memory (GTT): the kernel starts evicting and the laptop can hang
if [ -z "$FORCE" ] && ps -eo stat=,comm= | awk '$1 !~ /Z/ && $2 ~ /^(bin-opt-sd-cli|bin-opt-sd-serv|sd-cli|sd-server)$/ {f=1} END {exit !f}'; then
  echo "another generation process holds the model (GUI backend or sd-cli); unload it first (GUI: 'выгрузить'), or FORCE=1"; exit 1
fi
S=${4:-512x768}
if [ "$VAE" = full ]; then
  DECODER=(--vae models/qwen_image_vae.safetensors --vae-tiling --vae-tile-size 512x384 --vae-tile-overlap 0.125)
elif [ $(( ${S%x*} * ${S#*x} )) -le $((640 * 960)) ] && \
     [ "$(cat /sys/class/drm/card*/device/mem_info_gtt_total 2>/dev/null | sort -n | tail -1)" -ge $((10 << 30)) ] 2>/dev/null; then
  # in one piece: ~0.7 s faster up to 640x960 (peak 7.0 GB GPU memory), slower above; needs the GTT limit raised to ~11 GiB
  DECODER=(--taesd models/taew2_1.safetensors)
else
  DECODER=(--taesd models/taew2_1.safetensors --vae-tiling --vae-tile-size 128x128 --vae-tile-overlap 0.25)
fi
../tools/bin-opt-sd-cli \
  --diffusion-model "models/${MODEL:-anima-turbo-v1.1_f16.safetensors}" \
  --llm models/qwen_3_06b_base.safetensors \
  -p "best quality, score_7, score_9, very aesthetic, ultra detailed, $1" \
  --sampling-method euler --cfg-scale 1 --steps "${2:-4}" -W "${S%x*}" -H "${S#*x}" \
  --backend clip=cpu --params-backend clip=cpu \
  "${DECODER[@]}" \
  --diffusion-fa --eager-load \
  -s "${3:--1}" -o "${OUT:-out/$(date +%s).png}"
