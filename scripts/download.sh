#!/bin/bash
# usage: scripts/download.sh [anima] [sdxl] [upscaler]   (no args = anima upscaler)
# Every file is checked against its SHA256; files already present and verified are skipped.
#   anima    official circlestone-labs/Anima turbo v1.1 (converted to f16, Vega has no bf16), Qwen3 0.6B text encoder,
#            Qwen-Image VAE and the tiny decoder taew2_1
#   sdxl     4-step DMD2 LoRA (tianweiy/DMD2) and the tiny decoder taesdxl; the checkpoint itself is yours:
#            put any SDXL / Illustrious-class model at sdxl/models/sdxl.safetensors
#   upscaler realesrgan-ncnn-vulkan (realesr-animevideov3)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WHAT="${*:-anima upscaler}"
HF=https://huggingface.co

get() {  # dest url sha256
  local dest="$ROOT/$1" url="$2" sha="$3"
  if [ -f "$dest" ] && echo "$sha  $dest" | sha256sum -c --quiet 2>/dev/null; then echo "ok (present) $1"; return; fi
  mkdir -p "$(dirname "$dest")"
  curl -fSL -C - -o "$dest.part" "$url"
  echo "$sha  $dest.part" | sha256sum -c --quiet || { echo "checksum mismatch: $1"; exit 1; }
  mv "$dest.part" "$dest"
  echo "ok $1"
}

if [[ " $WHAT " == *" anima "* ]]; then
  A=$HF/circlestone-labs/Anima/resolve/main/split_files
  get anima/models/anima-turbo-v1.1.safetensors "$A/diffusion_models/anima-turbo-v1.1.safetensors" \
    fba11953276b57edf59d1dc4f1857ac05aa079c56f982b4d7c20298d57d3f7eb
  get anima/models/qwen_3_06b_base.safetensors "$A/text_encoders/qwen_3_06b_base.safetensors" \
    cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba
  get anima/models/qwen_image_vae.safetensors "$A/vae/qwen_image_vae.safetensors" \
    a70580f0213e67967ee9c95f05bb400e8fb08307e017a924bf3441223e023d1f
  get anima/models/taew2_1.safetensors "https://github.com/madebyollin/taehv/raw/refs/heads/main/safetensors/taew2_1.safetensors" \
    04766eac0221b5390b985ae3fdcca652cbb4b1e8b82b28ea7ff89dfad1b1a93f
  M="$ROOT/anima/models/anima-turbo-v1.1_f16.safetensors"
  [ -f "$M" ] || python3 "$ROOT/anima/merge_lora.py" "$ROOT/anima/models/anima-turbo-v1.1.safetensors" none "$M"
fi
if [[ " $WHAT " == *" sdxl "* ]]; then
  get sdxl/models/loras/dmd2_sdxl_4step_lora_fp16.safetensors "$HF/tianweiy/DMD2/resolve/main/dmd2_sdxl_4step_lora_fp16.safetensors" \
    b3d9173815a4b595991c3a7a0e0e63ad821080f314a0b2a3cc31ecd7fcf2cbb8
  get sdxl/models/taesdxl.safetensors "$HF/madebyollin/taesdxl/resolve/main/diffusion_pytorch_model.safetensors" \
    ff4824aca94dd6111e0340fa749347fb74101060d9712cb5ef1ca8f1cf17502f
  [ -f "$ROOT/sdxl/models/sdxl.safetensors" ] || echo "note: put your SDXL checkpoint at sdxl/models/sdxl.safetensors"
fi
if [[ " $WHAT " == *" upscaler "* ]]; then
  get deps/realesrgan-ncnn-vulkan.zip "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-ubuntu.zip" \
    e5aa6eb131234b87c0c51f82b89390f5e3e642b7b70f2b9bbe95b6a285a40c96
  if [ ! -x "$ROOT/deps/realesrgan/realesrgan-ncnn-vulkan" ]; then
    mkdir -p "$ROOT/deps/realesrgan"
    unzip -q -o "$ROOT/deps/realesrgan-ncnn-vulkan.zip" -d "$ROOT/deps/realesrgan"
    chmod +x "$ROOT/deps/realesrgan/realesrgan-ncnn-vulkan"
  fi
  echo "ok deps/realesrgan"
fi
