#!/bin/bash
# Builds stable-diffusion.cpp (Vulkan) with this repo's patches and installs tools/bin-opt-sd-cli and tools/bin-opt-sd-server.
# usage: scripts/build.sh [work dir, default ./build-src]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="${1:-$ROOT/build-src}"
SD_COMMIT=4c870cc60cd466427e4802d35a15b681ba9e4e90
GGML_COMMIT=a5915eb0e47adef21c3f25b889a008925ea838c2

mkdir -p "$WORK"
cd "$WORK"
if [ ! -f prefix/share/cmake/SPIRV-Headers/SPIRV-HeadersConfig.cmake ]; then
  [ -d SPIRV-Headers ] || git clone --depth 1 https://github.com/KhronosGroup/SPIRV-Headers.git
  cmake -S SPIRV-Headers -B SPIRV-Headers/build -DCMAKE_INSTALL_PREFIX="$WORK/prefix" -DSPIRV_HEADERS_ENABLE_TESTS=OFF
  cmake --install SPIRV-Headers/build
fi
if [ ! -d stable-diffusion.cpp ]; then
  git clone https://github.com/leejet/stable-diffusion.cpp.git
  git -C stable-diffusion.cpp checkout -q "$SD_COMMIT"
  git -C stable-diffusion.cpp submodule update --init --recursive
fi
cd stable-diffusion.cpp
[ "$(git -C ggml rev-parse HEAD)" = "$GGML_COMMIT" ] || { echo "ggml is not at $GGML_COMMIT"; exit 1; }
apply() {  # repo dir, patch
  if git -C "$1" apply --check "$2" 2>/dev/null; then git -C "$1" apply "$2"
  elif git -C "$1" apply --reverse --check "$2" 2>/dev/null; then echo "already applied: $(basename "$2")"
  else echo "patch does not apply: $2"; exit 1; fi
}
apply ggml "$ROOT/patches/ggml-vega.patch"
apply . "$ROOT/patches/sd-cpp.patch"
cmake -S . -B build -DSD_VULKAN=ON -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$WORK/prefix" -DCMAKE_CXX_FLAGS="-I$WORK/prefix/include"
cmake --build build -j"$(nproc)" --target sd-cli sd-server
cp build/bin/sd-cli "$ROOT/tools/bin-opt-sd-cli"
cp build/bin/sd-server "$ROOT/tools/bin-opt-sd-server"
echo "installed $ROOT/tools/bin-opt-sd-cli and bin-opt-sd-server"
