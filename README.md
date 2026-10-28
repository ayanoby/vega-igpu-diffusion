# Stable Diffusion on an integrated Vega 7: Anima, SDXL and other models with stable-diffusion.cpp

[Русский](README.ru.md) · **English**

Local image generation on a laptop without a discrete GPU, using [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) (sd.cpp) on Vulkan. The repository contains:

- **Patches** `patches/ggml-vega.patch` (ggml-vulkan speedups for Vega/GCN, with tests) and `patches/sd-cpp.patch` (sd.cpp changes: Anima, Wan VAE, the UNet FFN output).
- **Scripts** `scripts/build.sh` (builds `sd-cli` and `sd-server`) and `scripts/download.sh` (official models, checked by SHA256).
- **Anima** `anima/gen-anima.sh` (command line) and `anima/merge_lora.py` (f16 conversion and LoRA baking).
- **UI** `gui/`: a GTK 4 window or a web page with a queue, live preview, history, img2img, sampler/scheduler choice and LoRAs. The UI text is in Russian.
- **Upscaling** `tools/upscale.sh` (realesr-animevideov3, ×2 in ~1.6 s) and a "living photo" (cinemagraph) from a Wan clip, `wan/cinemagraph.py`.
- **Power** `power/`: a RyzenAdj timer for power limits.
- A detailed **measurement log**: what helped, what did not, and why.

## Hardware

| | |
|---|---|
| Laptop | ThinkPad T14s Gen 1 |
| CPU | Ryzen 7 PRO 4750U |
| GPU | Radeon Vega 7 (Renoir, gfx90c): 7 CUs, up to 1600 MHz, no matrix cores, no int8 dot, no bf16 |
| RAM | 16 GB DDR4, soldered (the OS sees 14.3 GiB); the GPU gets 0.5 GB dedicated (BIOS) + shared GTT memory (~7.2 GiB by default, can be raised to 11 GiB, see "Shared memory and hangs") |
| OS | Ubuntu 26.04, kernel 7.0, Mesa RADV 26.0 |

The theoretical GPU peak is 2.87 TFLOPS fp16. In practice matrix multiplication reaches ~1.0–1.3 TFLOPS: the shader uses 128 registers, which means 2 waves per SIMD.

## Results

Time per image with the model already in memory (UI or `sd-server`), RyzenAdj power limits at 25 W. Without a cooling pad the laptop reaches 90 °C during a long series and steps get up to ~17% slower. From the command line, add ~5–10 s for loading the weights.

| Model and mode | Resolution | Time |
|---|---|---|
| Anima 2B turbo, 4 steps, CFG 1, TAE | 384×576 | ~12 s |
| Anima 2B turbo | 512×768 | **~23 s** (steps ~20 s, decoder ~1.6 s) |
| Anima 2B turbo | 920×1536 (native) | ~2:05 |
| Anima 2.9B + Turbo-ANIMA V4-29B, 4 steps, CFG 1, TAE | 512×768 | ~33 s |
| SDXL (Illustrious class) + DMD2, 4 steps, CFG 1, taesdxl | 832×1216 | **~38 s** (7.6 s per step) |
| SDXL + DMD2 | 1024×1536 | ~1:02 |
| Qwen-Image 2.1 Turbo (Q5_K_S), 8 steps | 512×768 | ~3.5 min |
| Kromoss (MOSS-Image 5B, Q6_K) + DMD2 LoRA, 4 steps | 512×768 | ~2 min |
| Wan 2.2 TI2V 5B Turbo (Q4_K_M), 4 steps, 33 frames | 512×768 | ~4:15 |
| Upscale ×2 (realesr-animevideov3) | 512×768 → 1024×1536 | ~1.6 s |

The patch was first tuned on SD1.5 (an anime checkpoint, 20 steps, CFG 7): 512×768 took ~80–85 s. For comparison, the only public measurement found on similar hardware is a desktop Ryzen 5 5600G (Vega 7, 65 W) running SD 1.4 via ROCm: ~3 s per step at 512×512 ([blog](https://www.gabriel.urdhr.fr/2022/08/28/trying-to-run-stable-diffusion-on-amd-ryzen-5-5600g/)); here the same amount of work takes ~2.5 s.

## Gallery

The same four SFW scenes on every model: a knight, a mountain lake, a night city, a cat. Seed 42, the first result with no cherry-picking or retouching, CFG 1 everywhere. Time per image with the model already loaded (prompt encoding and decoding included); the laptop ran without a cooling pad at Tctl 84–98 °C, Anima and SDXL match the results table, while the long Qwen-Image and Kromoss runs heated the chip and came out 20–25% slower than there. Anima and SDXL went through the UI's `sd-server`, Qwen-Image and Kromoss through `sd-cli` (`generate_image` time, without loading).

| Model and settings | Knight | Lake | City | Cat |
|---|---|---|---|---|
| **MiaoMiao 3D Harem (Anima 2B)**<br>+ Anima Turbo v1.1, 4 steps, euler / discrete, 512×768 | <img src="docs/gallery/anima-portrait.jpg" width="150"><br>24.2 s | <img src="docs/gallery/anima-landscape.jpg" width="150"><br>23.7 s | <img src="docs/gallery/anima-city.jpg" width="150"><br>24.2 s | <img src="docs/gallery/anima-cat.jpg" width="150"><br>23.9 s |
| **MiaoMiao Anime Reality (Anima 2B)**<br>+ Anima Turbo v1.1, 4 steps, euler / discrete, 512×768 | <img src="docs/gallery/animereal-portrait.jpg" width="150"><br>24.5 s | <img src="docs/gallery/animereal-landscape.jpg" width="150"><br>23.9 s | <img src="docs/gallery/animereal-city.jpg" width="150"><br>24.2 s | <img src="docs/gallery/animereal-cat.jpg" width="150"><br>24.4 s |
| **MiaoMiao RealSkin 1.3 (Anima 2B)**<br>+ Anima Turbo v1.1, 4 steps, euler / discrete, 512×768 | <img src="docs/gallery/realskin-portrait.jpg" width="150"><br>24.2 s | <img src="docs/gallery/realskin-landscape.jpg" width="150"><br>23.9 s | <img src="docs/gallery/realskin-city.jpg" width="150"><br>24.2 s | <img src="docs/gallery/realskin-cat.jpg" width="150"><br>23.9 s |
| **MiaoMiao Harem 2.9B β1.1**<br>+ Turbo-ANIMA V4-29B, 4 steps, euler / sgm_uniform, 512×768 | <img src="docs/gallery/harem29-portrait.jpg" width="150"><br>37.3 s | <img src="docs/gallery/harem29-landscape.jpg" width="150"><br>38.8 s | <img src="docs/gallery/harem29-city.jpg" width="150"><br>39.0 s | <img src="docs/gallery/harem29-cat.jpg" width="150"><br>38.3 s |
| **MiaoMiao RealSkin 2.9B 1.0**<br>+ Turbo-ANIMA V4-29B, 4 steps, euler / sgm_uniform, 512×768 | <img src="docs/gallery/realskin29-portrait.jpg" width="150"><br>39.1 s | <img src="docs/gallery/realskin29-landscape.jpg" width="150"><br>38.8 s | <img src="docs/gallery/realskin29-city.jpg" width="150"><br>39.0 s | <img src="docs/gallery/realskin29-cat.jpg" width="150"><br>38.3 s |
| **JANKU v7.77 (SDXL, NoobAI)**<br>+ DMD2 4-step LoRA (A1), 4 steps, euler_a / gits, clip skip 2, 832×1216 | <img src="docs/gallery/janku-portrait.jpg" width="150"><br>44.6 s | <img src="docs/gallery/janku-landscape.jpg" width="150"><br>38.8 s | <img src="docs/gallery/janku-city.jpg" width="150"><br>38.3 s | <img src="docs/gallery/janku-cat.jpg" width="150"><br>38.6 s |
| **Qwen-Image 2.1 Turbo Q5_K_S**<br>8 steps, euler, turbo sigmas, taeqi2_1, 512×768 | <img src="docs/gallery/qwen-portrait.jpg" width="150"><br>4:24 | <img src="docs/gallery/qwen-landscape.jpg" width="150"><br>3:51 | <img src="docs/gallery/qwen-city.jpg" width="150"><br>4:26 | <img src="docs/gallery/qwen-cat.jpg" width="150"><br>4:24 |
| **Kromoss (MOSS-Image 5B) Q6_K**<br>+ 8-steps DMD2 LoRA, 4 steps, euler, full FLUX.2 VAE, 512×768 | <img src="docs/gallery/kromoss-portrait.jpg" width="150"><br>2:27 | <img src="docs/gallery/kromoss-landscape.jpg" width="150"><br>2:27 | <img src="docs/gallery/kromoss-city.jpg" width="150"><br>2:26 | <img src="docs/gallery/kromoss-cat.jpg" width="150"><br>2:21 |

Anima finetunes have the turbo LoRA baked in with `anima/merge_lora.py`; this repo's `download.sh` fetches the official `anima-turbo-v1.1` instead, which is in the same class. The ~24 s and ~38 s of Anima 2B and 2.9B and ~40 s of SDXL are the same for any finetune of the same size.

<details><summary>Prompts</summary>

Anima / SDXL (booru tags; the quality prefix differs per model, see `gui/server.py`):

```
best quality, score_7, score_9, very aesthetic, ultra detailed, [photorealistic, raw photo, photo background,]
knight:  1girl, adult, solo, knight, silver armor, long red hair, holding sword, castle courtyard, banners, upper body, looking at viewer, determined
lake:    no humans, scenery, mountain lake, sunrise, pine forest, mist, wooden pier, reflection, clouds
city:    1boy, adult, solo, from behind, hooded jacket, transparent umbrella, cyberpunk city, night, rain, neon lights, neon sign, reflections, street
cat:     no humans, orange cat, sleeping, curled up, windowsill, sunlight, potted plant, cozy room
SDXL prefix: lazypos, masterpiece, best quality, general,
```

Qwen-Image / Kromoss (natural language):

```
knight:  An adult woman knight with long red hair in polished silver armor holding a sword, standing in a castle courtyard with banners, upper body, looking at the viewer with a determined expression
lake:    A calm mountain lake at sunrise surrounded by pine forest, morning mist, a wooden pier, clouds reflected in the water, no people
city:    An adult man in a hooded jacket with a transparent umbrella seen from behind on a cyberpunk city street at night, rain, neon lights reflecting on the wet road, a large neon sign that says "VEGA 7"
cat:     An orange cat sleeping curled up on a sunlit windowsill next to a potted plant in a cozy room
```

</details>

## Quick start

### 1. Dependencies

```bash
sudo apt install build-essential cmake git libvulkan-dev glslc curl unzip python3-numpy
```

The UI window needs `python3-gi` with GTK 4 and libadwaita (preinstalled on Ubuntu with GNOME).

### 2. Build

```bash
scripts/build.sh
```

The script clones stable-diffusion.cpp at commit `4c870cc` (ggml `a5915eb`) into `build-src/`, applies both patches and builds static `sd-cli` and `sd-server` with Vulkan. The result goes to `tools/bin-opt-sd-cli` and `tools/bin-opt-sd-server`, which the scripts and the UI use. SPIR-V headers are installed into `build-src/prefix`; nothing is installed system-wide. Building the shaders takes a few minutes.

### 3. Models

```bash
scripts/download.sh            # anima + upscaler
scripts/download.sh sdxl       # DMD2 LoRA and taesdxl for SDXL
```

| Group | Files | Source |
|---|---|---|
| `anima` | `anima-turbo-v1.1` (4.2 GB, bf16 → converted to f16), `qwen_3_06b_base` (text encoder, 1.2 GB), `qwen_image_vae` (254 MB), `taew2_1` (22 MB) | [circlestone-labs/Anima](https://huggingface.co/circlestone-labs/Anima), [madebyollin/taehv](https://github.com/madebyollin/taehv) |
| `sdxl` | `dmd2_sdxl_4step_lora_fp16` (376 MB), `taesdxl` (9.8 MB). Put the SDXL checkpoint itself (any Illustrious/NoobAI-class model) at `sdxl/models/sdxl.safetensors` | [tianweiy/DMD2](https://huggingface.co/tianweiy/DMD2), [madebyollin/taesdxl](https://huggingface.co/madebyollin/taesdxl) |
| `upscaler` | realesrgan-ncnn-vulkan 20220424 (47 MB) | [Real-ESRGAN v0.2.5.0](https://github.com/xinntao/Real-ESRGAN/releases/tag/v0.2.5.0) |

For Anima 2.9B: bake a 40-block model (for example [Anima-2.9B](https://huggingface.co/Gazingstars123/Anima-2.9B) or a finetune of it) and the turbo LoRA [Turbo-ANIMA V4-29B](https://civitai.com/models/2619830) into one file: `anima/merge_lora.py anima-2.9b.safetensors Turbo-ANIMA-v4-29B.safetensors anima/models/anima-2.9b_turbo4_f16.safetensors`. This LoRA has alpha = rank/2; the script takes that into account.

### 4. Shared memory limit (recommended)

By default the driver gives the GPU half of the RAM (~7.2 GiB). With 11 GiB, SDXL with the text encoders on the GPU, Anima 2.9B, Wan video and untiled decoding all fit. Add `ttm.pages_limit=2883584 ttm.page_pool_size=2883584` to `GRUB_CMDLINE_LINUX_DEFAULT` in `/etc/default/grub`, then:

```bash
sudo update-grub
```

After a reboot `cat /sys/class/drm/card*/device/mem_info_gtt_total` should print 11811160064. Do not go above 11–11.5 GiB: the system needs ~3 GB left. The scripts and the UI enable the heavy modes only when the limit is ≥ 10 GiB.

### 5. UI

```bash
gui/start.sh
```

The `gui/app.py` window uses GTK 4 (~110 MB of RAM versus ~1–1.4 GB for a browser tab); `gui/start.sh --browser` opens the web page instead (http://127.0.0.1:7860). The `gui/server.py` server uses only the Python standard library; it keeps one `sd-server` with the selected model in memory. Features:

- models: Anima 2B turbo, Anima 2.9B turbo, SDXL + DMD2 (only models whose files exist are listed);
- resolution with an estimated time, steps, seed, count; sampler and scheduler (recommended ones are marked ★); your own LoRAs with weights (`anima/loras/`, `sdxl/models/loras/`, the ⟳ button rereads the folder);
- queue, live preview, history with settings, favourites, deleting to the trash (`gio trash`), automatic ×2 upscale;
- img2img: a source image (file, drag and drop, Ctrl+V, an image from the history) and a denoising strength. The strength is passed as a noise level (`strength_as_noise_level`, `force_first_sigma`): by default sd.cpp gives only 3 levels at 4 steps, and at 0.75 it starts from pure noise.

The window renders on the CPU (`GSK_RENDERER=cairo`): with GTK's default Vulkan renderer an open window slowed the steps by ~7% even while idle (5.39 vs 5.05 s, A/B with three runs each). For the same reason, minimise browsers and Electron apps while generating: minimised windows do not render.

### 6. Command line

```bash
anima/gen-anima.sh "1girl, solo, red kimono, sitting by a river, smile" 4 -1 512x768
tools/upscale.sh --x2 anima/out/1793000000.png
```

`gen-anima.sh` arguments: `"prompt" [steps=4] [seed] [WxH=512x768]`; `MODEL=` picks another file in `anima/models/`, `VAE=full` uses the full Qwen-Image VAE instead of TAE.

SDXL with DMD2 (what the UI does):

```bash
tools/bin-opt-sd-cli -m sdxl/models/sdxl.safetensors --lora-model-dir sdxl/models/loras --taesd sdxl/models/taesdxl.safetensors \
  -p "masterpiece, best quality, 1girl, solo, ...<lora:dmd2_sdxl_4step_lora_fp16:1>" --cfg-scale 1 --sampling-method euler_a \
  --scheduler gits --steps 4 --clip-skip 2 -W 832 -H 1216 --diffusion-fa --eager-load -o out.png
```

CLIP-L/G text encoders on the GPU (sd.cpp's default) take 0.5 s per prompt instead of 2.5 s on the CPU, but use 1.6 GB more video memory (peak 9.5 GB at 1024×1536); with the default GTT limit add `--backend clip=cpu --params-backend clip=cpu`. The full SDXL VAE decodes in 30–76 s here, `taesdxl` in 2.4–4.8 s.

## Anima: why these settings

| Setting | Why |
|---|---|
| Turbo model in f16 | Vega has no bf16, so the weights are converted to f16 once (`merge_lora.py … none …`). For Anima finetunes a turbo LoRA is baked into the weights in advance (`merge_lora.py`: W += alpha/rank·B·A) so it is not applied on every run |
| 4 steps, euler, CFG 1 | this is how the turbo was trained; at 6 steps the difference is minimal, at 2–3 the image falls apart. CFG 1 means no negative pass |
| `--backend clip=cpu` | the Qwen3 0.6B text encoder runs on the CPU: ~1 s, and it saves GPU memory for the DiT |
| `--diffusion-fa` | flash attention; the patch has its own parameters for head size 128 (see below) |
| TAE `taew2_1` | tiny decoder: ~3× faster than the full VAE, nearly identical by eye. With the default GTT limit it uses 128×128 tiles (108 MB buffer; untiled, ~2.6 GB at 512×768 together with the weights **froze the laptop**). With a limit ≥ 10 GiB it decodes up to 640×960 in one piece, 0.6–0.8 s faster |
| Full VAE (`VAE=full`): 512×384 tiles | when maximum decoding accuracy is needed |
| 512×768 by default | the native 920×1536 has more detail but takes ~3 min per image; below 384×576 quality is too low, so those sizes are not offered in the UI (still available from the command line) |

**What tiles are.** The decoder processes the image in overlapping pieces (tiles) and stitches them. Peak memory depends on the tile size, not on the image size. An overlap of 0.25 hides the seams.

## What the ggml-vulkan patch does

A patch to `ggml/` (`patches/ggml-vega.patch`: ggml-vulkan plus one change in `ggml.c`), about 1200 lines including tests.

1. **Matmul tiles for GCN.** A 64×64 tile on one wave with 4×4 values per thread. The large 128×128 tile is used only when it covers the matrix exactly. On average 740 → 1000 GFLOPS on UNet shapes. AMD GCN only.
2. **Flash attention for head size 40 (SD1.x).** A 16-row block in 4 groups without splitting D: self-attention 54 → 43 ms, cross-attention 6.3 → 2.9 ms. Also fixes a shader bug: with `row_split > 1`, subgroup shuffles crossed row-group boundaries.
3. **Fused normalisations.** `group_norm → ×γ → +β [→ SiLU]` and `layer_norm → ×γ → +β` run in one pass (`group_norm_mul_add.comp`, `norm_mul_add.comp`). The standard ggml check rejected sd.cpp's in-place chains, so a custom matcher was added.
4. **Bias fused into matmul.** `mul_mat → [reshape] → add(bias)` for linear layers and convolutions: the bias is added when the result is written. Separate ADDs per step 270 → 142.
5. **Convolution without a needless copy.** With batch 1, `ggml_conv_2d` no longer does a useless CONT after the matmul.
6. **Tiled im2col.** A 64×64 transpose through shared memory: reads go along image rows, writes along CHW. 2.6× faster on VAE convolutions (4.97 → 1.87 s per image). Roughly unchanged on the UNet (+0.3 s per image).

7. **Flash attention for head size 128 (Anima / Cosmos DiT).** A one-wave workgroup (64 threads), 16×16 blocks, `row_split=4`, `d_split=4`. Faster than the defaults by ~11% for self-attention over 1.5k tokens, ~22% for 5.5k tokens and ~34% for attention to 512 text tokens. Found by a sweep; the measurements are the Anima cases in `SD_UNET_PERF`.
8. **GQA fix.** Merging heads under GQA checked the maximum ratio using parameters for 512 rows; with small blocks this gave wrong results. It now checks `block_rows` for the real row count.
9. **Large tile for large k (GCN).** Anima's MLP-down layer (k=8192) ran at only 0.55 TFLOPS on the medium 64×64 tile: as k grows, the medium tile with twice the memory traffic thrashes the L2. With k ≥ 4096 the 128×128 tile is now chosen: 0.55 → 0.92 TFLOPS, sampling at 512×768 34.9 → 29.5 s (−15%), 920×1536 −12%, bit-identical image. For this the tile selector now receives k (it used to get 0). Swizzled tile order and split-K were tested and do not help.
10. **Fusions for the DiT (Anima), −8.5% at 512×768, same image within rounding:**
    - adaLN `norm(x)·(1+scale)+shift` in one pass via the existing NORM_AFFINE fusion. For this `patches/sd-cpp.patch` writes the modulation as norm → mul → add and adds scale/shift/gate to the graph in advance (otherwise depth-first traversal puts the timestep MLP between norm and mul).
    - RMSNorm of short rows (per-head q/k norms, 128 columns): a separate 64-thread shader variant instead of 512 — 3.5 → 0.85 ms per call.
    - Projection outputs `residual + h·gate`: the gate multiply and residual add happen in the `mul_mm` epilogue (binding 4, `epi` flags). GELU in the epilogue was tried: MLP-up slowed from 41 to 61 ms, so GELU stays separate.
    - The fused NORM_AFFINE keeps the row (up to 2048 values) in registers between the statistics and the output and uses 256 threads per row: 1.53 → 0.78 ms per call.
12. **Cross-attention over real text tokens only** (`patches/sd-cpp.patch`, `anima.hpp`): the context is zero-padded to 512 tokens, and the K/V projections have no bias while the key RMS norm keeps zero rows zero, so K/V are computed only for the real prompt tokens and zero-padded afterwards. Bit-identical (CPU and GPU), sampling at 512×768 23.0 → 22.9 s (−0.6%).
11. **TAE (taew2_1) without needless frames** (`patches/sd-cpp.patch`, `tae.hpp`): the video decoder doubles the frame count twice and then keeps the last frame; after the memory blocks of the last layer everything is per-frame, so the unneeded frames are dropped before the upsample — full-resolution convolutions compute 1 frame instead of 4. Decoding 512×768 2.81 → 2.47 s, 920×1536 9.55 → 8.24 s, bit-identical. (Untiled at 512×768 it is 1.66 s, but a 1.5 GB buffer next to 4 GB of weights is too close to the default GTT limit; with the limit raised to 11 GiB the UI and `gen-anima.sh` decode untiled up to 640×960, see below.)
13. **FA without bounds checks on full blocks** (`flash_attn.comp`, `flash_attn_block.glsl`). When the KV length is not a multiple of the block (16 for Anima), the whole shader ran in the bounds-checked variant, ~60% slower: at 448×672, 576×864, 704×1056 and 832×1248 the token count `(W/16)·(H/16)` is not a multiple of 16. The KV block loop body moved into a file included twice: full blocks without checks, only the last one with checks. Attention over 2904 tokens 196 → 122 ms, over 4056 tokens 354 → 200 ms; through the UI 576×864 40.9 → 36.3 s, 704×1056 69.3 → 60.0 s, 832×1248 103.4 → 86.6 s. Same image within rounding. Checking only in the last block via a runtime variable was slower (210 ms): the compiler needs a constant. A block of 8 (which divides all such lengths) takes 135 ms, ~28% slower than a block of 16 on its own.
14. **Fused RMSNorm → ×w → RoPE (+ cast to f16)** (`rms_norm_mul_rope_apply.comp`). The self-attention q/k norms and sd.cpp's `ROPE_APPLY` run in one pass: one workgroup per row (head, token), the sum of squares and arithmetic as in the unfused shaders — bit-identical image. For k the fused shader writes f16 directly for FA. Previously the overlap check cancelled this fusion: the RoPE output is permuted (heads ↔ tokens), and the allocator placed it over the input. Now `anima.hpp` keeps the original q/k in `src[2]` of RoPE (and of the k cast to f16), so they stay alive until the fused op ends. Along the way: `ggml_can_fuse` requires the same shape along the chain (RoPE's differs) and rejects an in-place MUL (a view of the RMS_NORM output), so the check is custom; the overlap check skips the `ggml_cast` self-reference in `src[1]`. Per step at 512×768: 125 → 50 ms.
15. **Matmul inputs in f16, fused into NORM_AFFINE and GELU** (`norm_mul_add.comp` with f16 output, `gelu_f32_f16.comp`). The matmul shader rounds input B to f16 anyway (f16 weights), so `anima.hpp` casts the outputs of the three adaLNs and of the MLP GELU to f16 — bit-identical result (also on the CPU), while matmuls with f16 input are 5–12% faster (less traffic). The casts are fused into the normalisation/GELU, with no separate copies; GELU with f16 output writes 4 values per thread (the generic `unary.comp` at 2 bytes per thread was slower than f32). 8 steps at 512×768 with cooldown before each run: 46.0 → 44.5 s (three pairs out of three).
16. **FA for head size 64 (SDXL)** (`get_fa_tuning_params_scalar`). Flash-attention parameters had only been tuned for head sizes 40 and 128; the SDXL UNet uses 64, and attention ran at 370–460 GFLOPS (22% of a step). A sweep of ~30 combinations on SDXL 832×1216 shapes (`SD_UNET_PERF=1`): `Br=16, Bc=32, WG=128, row_split=4, d_split=4`. Self-attention over 3952 tokens 104 → 62 ms, over 988 tokens 9.9 → 7.3 ms, cross-attention to 77 tokens 1.74 → 0.98 ms; attention per step 1.77 → 1.15 s. Full generation A/B (old and new binaries alternating): step 8.42 → 7.85 s, 4 steps 38.3 → 36.0 s. Same image within rounding (different summation order). All 5332 FLASH_ATTN_EXT tests pass.
17. **FFN output without f32 accumulation** (`patches/sd-cpp.patch`, `block.hpp`). On Vulkan sd.cpp enabled f32 accumulation for `ff.net.2` in the UNet — a guard against f16 overflow, but it is a separate shader, 1.5× slower (SDXL: 736 vs 1150 GFLOPS). Now on Vulkan the input is scaled by 1/128 and the matmul runs in f16 (as sd.cpp does for CUDA with k-quants, "precision_fix"). SDXL 832×1216 step −2…2.3% (five prompts, A/B in each pair, including 896×1152 and 1024×1536, three different 4-step DMD2 LoRAs); no NaNs, same image within rounding (mean difference 5–14 out of 255).

Changes 1, 2, 7 and 9 apply only to AMD GCN. The others work on any Vulkan GPU.

### sd.cpp patch (`patches/sd-cpp.patch`)

- **Cross-attention through a sink** (`anima.hpp`, `ggml_extend.cpp`): the text context is zero-padded to 512 tokens. A zero key gives a score of 0, i.e. it adds e⁰ = 1 to the softmax denominator and nothing to the numerator. That is exactly what an FA "sink" (`ggml_flash_attn_ext_add_sinks`) with the value ln(number of zero tokens) per head does, so attention runs over the real tokens only (~40–110 instead of 512). Without FA the same is done via `ggml_soft_max_add_sinks`. Same image within rounding (CPU: at most 2/255). Cross-attention FA per step at 512×768: 271 → 53 ms, the whole image −1.3 s.
- **Text K/V cache between steps** (`anima.hpp`): the adapter and the cross-attention K/V depend only on the prompt. The first step stores K (after the norm) and V of all 28 blocks in the runner cache in f16 (FA casts them to f16 anyway); later steps take them from there and do not build the adapter or the context projections at all. The key is a hash of the context, t5_ids and weights, so with CFG the positive and negative prompts get separate entries; the cache is cleared in `end_runners()` on every generation. Bit-identical (GPU, CPU, CFG 2); 8 steps with cooldown: 44.1 → 43.8 s (~0.13 s at 4 steps).
- **No extra copies before FA** (`ggml_extend.cpp`): with batch 1, q and k go into FA as permuted views (FA works with strides), and V is permuted while being cast to f16 — one copy instead of two. −45 ms per step, bit-identical.
- **kohya-format LoRAs for Anima** (`name_conversion.cpp`): in names like `lora_unet_llm_adapter_…` the underscore in `llm_adapter` became a dot, and the text-adapter part of the LoRA was silently not applied (for one style LoRA, 840 of 948 tensors were applied). `llm_adapter` was added to the protected tokens.
- Wan VAE (Anima uses it as the Qwen-Image VAE) is a video VAE with causal 3D convolutions. For a single frame without a cache the temporal padding is zero, and only the last temporal slice of the kernel matters. In that case the patch makes `CausalConv3d::forward` run a regular 2D convolution with that slice instead of a 3D one. Same result, faster and lighter on memory.

### Verification

`tests/test-backend-ops.cpp` gains `GROUP_NORM_AFFINE`, `NORM_AFFINE` and `MUL_MAT_BIAS` tests (including a convolution case and split-K), plus a set of UNet shape benchmarks behind `SD_UNET_PERF=1`. All tests pass:

| Test | Passed |
|---|---|
| MUL_MAT | 2279 / 2279 |
| MUL_MAT_ID | 939 / 939 |
| FLASH_ATTN_EXT | 5332 / 5332 (after changes 7–8) |
| CONV_2D | 1966 / 1966 |
| IM2COL | 102 / 102 |
| MUL_MAT_BIAS | 11 / 11 |
| NORM / GROUP_NORM / *_AFFINE / *_MUL_ADD | all |
| ADD / MUL / SILU | all |
| `test-conv2d`, `test-conv1d` | PASSED |

Running the tests in the build tree:

```bash
build-src/stable-diffusion.cpp/build/bin/test-backend-ops test -b Vulkan0 -o MUL_MAT_BIAS
```

`test-backend-ops` is built separately (`cmake --build build --target test-backend-ops`).

## Other models and video (measured, not in the UI)

All on this laptop, GTT 11 GiB, the same patched ggml. There are no launch scripts for these models in the repo (except `wan/cinemagraph.py`); the recipes are given as flags together with the results.

| Model | Files | Time | Findings |
|---|---|---|---|
| **Qwen-Image 2.1 Turbo** (7B, 8 steps, CFG 1, `--sigmas qwen-image-2.1-turbo`) | `qwen-image-2.1-turbo-Q5_K_S.gguf` (unsloth, 4.5 GB), `Qwen3VL-8B-Instruct-Q4_K_M.gguf` on the CPU, `taeqi2_1` | 512×768 — 212 s (26 s per step, encoder 1.9 s, decoder 0.24 s) | Text in the image is rendered correctly. Step profile: matmul 84% (Q4_K/Q5_K at 0.75–1.1 TFLOPS — the same as f16), attention 6.8%. Q5_K layers are slower than Q4_K (875 vs 1078 GFLOPS) — a pure Q4_K_M would save ~3%. Nothing left to speed up in the VAE |
| **Kromoss** (MOSS-Image 5B, lodestones) + `8-steps-dmd2-lora` (CFG 1) | `kromoss-Q6_K.gguf` (molbal), `Qwen3.5-9B-Q4_K_M.gguf` on the CPU + `tokenizer.json`, `flux2-vae_small` | 512×768: 8 steps — 3.6 min, 4 steps — ~2 min (almost the same quality) | sd.cpp maps the LoRA's separate q/k/v onto the fused `qkv` by itself (756/756 tensors). On a quantised model the LoRA is applied at runtime: 25 s per step vs 21 s without it (+17%); baking it in requires bf16/f32 and requantising. The full VAE takes 10.5 s; sd.cpp has no tiny decoder for the FLUX.2 VAE. It sometimes draws a watermark-like caption in a corner |
| **Wan 2.2 TI2V 5B Turbo** (4 steps, CFG 1) | `Wan2_2-TI2V-5B-Turbo-Q4_K_M.gguf` (hum-ma, 3.4 GB), `umt5-xxl-encoder-Q4_K_M.gguf` on the CPU, `taew2_2` | 33 frames (1.4 s of video): 480×832 and 512×768 — ~4:15 (~57 s per step, ~15 ms per latent token), 49 frames — ~5 min | At 352×640 and below the output is broken (32 px blocks, also in the stock build): the model was trained at ~704p. Image-to-video from an Anima picture keeps the style and face. 49 frames without `--vae-tiling --temporal-tiling` used up all the memory while decoding and hung the laptop. K-quants ran almost as in the stock build (123 vs 129 s) |
| Wan: loop and "living photo" | `--end-img` = `-i` (the same image at the start and the end) | as above | The last latent frame (4 video frames) decodes washed out, so it is dropped. The "living photo" (`wan/cinemagraph.py`): Wan frames are used only inside a mask of the character, the background is the source pixel for pixel; Wan did not blink on request, so a blink is blended in from the source with the eyes inpainted closed (Anima inpainting with an eye mask, 7 frames). The automatic motion mask does not always work — a hand-drawn mask is better for quiet loops |
| LTX-2.5 (22B) | — | not run | The smallest quant is 7.3 GB (Q2_K); with the Gemma 4 12B encoder (~7 GB at Q4) it does not fit in 14 GB of RAM; with weights streamed from disk the estimate is 10–30 min for a 2-second clip |

## Experiment log: what did not help

Rows about CFG, schedulers, step caches, embeddings and negatives refer to SD1.5 (an anime checkpoint, 20 steps, CFG 7).

| Idea | Result |
|---|---|
| ucache in quality mode | −42% time, but it breaks fine detail and anatomy, extra hands appear |
| spectrum cache | −30% time, rough colour artefacts |
| AYS scheduler, 14 steps | −25% time, nearly the same quality but finer textures are worse; Karras kept |
| AYS, 10 steps | noticeably flatter and poorer in detail |
| CFG 1 on the last 10 steps | paler and flatter; the optimum is 6 steps |
| Hires fix (latent, ×1.5) | quality as on Civitai, but 325 s instead of 107; without hires the difference is small |
| Tile prefetch in the matmul shader | 9% slower: 128 → 176 registers, 2 → 1 wave per SIMD |
| ROCm (TheRock gfx900 nightly + `HSA_OVERRIDE_GFX_VERSION=9.0.0`) | worked only with Ubuntu's HSA runtime; 3.5× slower than Vulkan, hung the system |
| Weight quantisation (q8/q4) | will not help: matmul is compute-bound, and Vega has no int8 dot (`int dot: 0`) |
| Negative embeddings EasyNegative and bad-hands-5 | same speed, but no consistent improvement of hands and anatomy: the image changes about as much as with another seed (4 prompts, A/B) |
| `performance` power profile and DPM `high` | GPU and memory clocks at maximum (1600/1333 MHz), yet generation is ~10% slower than with `balanced` + `auto`; `performance` + `auto` is the worst |
| `GGML_VK_ALLOW_GRAPHICS_QUEUE=1` | about 2%, may make the desktop stutter |
| Upscaler: tiles 64 / 256 / 384, direct convolutions | 32.5 / 38.5 / 44.5 / 31.6 s vs 29.6 s with the default tile of 128 |
| ROCm for Anima | same as above: Vulkan is faster, ROCm removed |
| Decoding Anima with the full VAE using direct convolutions | garbage output; `--offload-to-cpu` crashes |
| TAE `taew2_1` untiled on the GPU (512×768) | a 2.6 GB buffer + the weights — out of memory, the laptop hung. Tiles only |
| TAE with tiles 256 / 512 / 64 | 128 is the best trade-off: 64 is slower with more visible seams, 256+ use too much memory |
| Cosmos LoRA instead of Anima Turbo | worse at 4 steps, removed |
| Turbo v0.2 → v1.1 (4 and 8 steps, 4 prompts) | roughly equal quality: v1.1 is cleaner with sharper outlines, v0.2 more saturated and varied; v1.1 chosen. 8 steps are more detailed for both, but twice as slow |
| Turbo LoRA at 1–2 steps | the image falls apart |
| MIGraphX / rocBLAS (ROCm 7.2 unpacked without sudo, gfx900 kernels from 6.3.4, a hipblaslt stub) | ROCm ran; rocBLAS fp16 up to 1.51 TFLOPS on large shapes, but at 512×768 ~37% slower overall than our Vulkan; moving to MIGraphX is not worth porting |
| Grouped tile order (swizzle) in `mul_mm`, split-K for k=8192 | no gain or slower |
| Speeding up `mul_mm`: tile sweeps, register prefetch, A in row pairs, B in fp16 | 0–4%: the shader is bound by memory latency (2 waves per SIMD), not compute; at 512×768 it is already ~2× faster than rocBLAS |
| IREE 3.13 (Vulkan, `cdna1`/`rdna1` targets; no GCN targets) | 0.03–0.40 TFLOPS vs 1.0–1.43 for ggml |
| TVM (MLC 0.26): dlight 0.14–0.31 TFLOPS; MetaSchedule autotuning | 0.64 TFLOPS after 32 trials, and random kernels longer than 2 s **reset the GPU** (the screen went black, the session crashed) — do not run without a raised timeout |
| ToDo: 2×2 K/V downsampling in self-attention (sampling, averaging, averaging before RoPE with cell centres and renormalisation) | −7% at 512×768, −24% at 920×1536, but blocks/blur/broken images in every variant: a DiT has no convolutions, local detail flows only through attention |
| ToMA (simplified: 4×4 tiles, every other token as anchors, soft merge τ=0.01) | in all or the last blocks — pixelation; only middle blocks 8–17: −15%, 4–19: −24…26%, but softer, lower contrast, small details lost — rejected |
| AMDVLK 2023.Q3.3 (AMD's last official Vulkan driver with Vega; LLPC/LLVM compiler; loaded via `VK_ICD_FILENAMES` without installing) | matmul 0.12–0.22 TFLOPS vs 1.0–1.43 with RADV (6–8× slower), 32 KB shared memory instead of 64, the flash-attention test crashed the GPU (`DeviceLost`) |
| Mesa 26.2.5 (RADV built locally with libdrm 2.4.133, via `VK_ICD_FILENAMES`) instead of the system 26.0.8 | tests pass; MLP-up/FA microbenchmarks +3–6%, but full generation is **2–4% slower** (512×768: 26.4 → 27.1 s; 920×1536: 133 → 139 s) |
| ACO compiler changes (Mesa 26.0.8, local build): scheduler reordering windows/limits for LDS, VMEM, clauses; a VGPR cap of 84 for 3 waves | the defaults are best; without the scheduler −35%; at 84 VGPRs the medium tile still gets 2 waves (bound by 8.7 KB of LDS), the large one gets 3 waves but is slower with register spilling (1.08 → 1.01 TFLOPS) |
| Attention without FA at 512×768 (plain matmuls + softmax) | 28.9 s vs 24.2 s — slower |
| Flash attention with f16 accumulation (`GGML_PREC_DEFAULT`) | FA +14%, step ~−2%, images look the same, but a risk of f16 overflow on rare prompts — f32 kept |
| TAE untiled with the default 7 GiB GTT | decoding 2.42 → 1.66 s at 512×768, but together with 4 GB of weights too close to the limit (the laptop hung); enabled automatically only with GTT ≥ 10 GiB |
| TAE untiled at 768×1152 and above / 256 and 384 px tiles | 768×1152: 5.45 → 6.29 s (slower), peak 8.5 GB; larger tiles are within noise |
| TAE: 192/256 tiles, direct convolutions | 192: −5% decoding; 256 and direct convolutions are slower |
| Batch of images (`-b 2`), text encoder cache | sd.cpp computes a batch in a loop; prompt encoding takes 0.12 s — no gain |
| Fusing RMSNorm(q/k) → ×w → RoPE into one shader (first attempt) | tests pass, but in the Anima graph the allocator puts the RoPE output into the input buffer, and the RoPE output is permuted (heads ↔ tokens) — the fused shader would overwrite unread rows, and ggml's overlap check cancels the fusion (together with RMSNorm+MUL). Reverted, later solved in change 14 |
| Packed fp16 FMA in `mul_mm` (llama.cpp PR #29877) | already in our ggml: the machine code has only `v_pk_fma_f16` |
| Blocking GPU wait instead of busy polling at the end of the graph | CPU use during steps 1.3 cores → ~0.5, but the same time: the GPU is not limited by the shared power budget |
| FA parameter sweep for cross-attention (KV ~107) | the current Br16 Bc16 WG64 are best; the others are 1.1–9× slower |
| V in FA as f32 (without the cast to f16) | FA +2.9 ms vs 0.8 ms for the copy — worse |
| Cross-attention as P·(V·W_o) (k = 16·text length instead of 2048) | a gain only for short prompts; with typical ~90–110 tokens ~0.15 s — not done |
| Decoder tile overhead | 40 tiles of ~58 ms each are almost entirely GPU time; graph preparation is ~0.1 s for the whole decode |
| FA with f16 output (fused cast for the output projection) | projection −0.77 ms, but self-attention +1 ms per call: ~−13 ms per step in total, not visible overall — reverted |
| "Gate + residual" epilogue without integer index divisions | same time: the epilogue costs ~0.5 ms, and the difference from a plain matmul came from the f32 input |
| Tiles for f16 input | the large tile for k=8192 is still better than the medium one (50–59 vs 79–86 ms), the choice was not changed |
| SDXL: matmul tile for the FFN output (1280×988, k=5120; 640×3952, k=2560) | 736–810 GFLOPS in the profile vs 1150–1320 in an isolated test, but the tile is already the best in the graph too (medium is slower: 21 vs 17.5 ms). The cause is the f32 accumulation sd.cpp enabled on Vulkan for `ff.net.2`; replaced by the 1/128 input scale (see change 17). f16 without any guard is −3.5% but risks NaNs, not enabled |
| Disabling the live preview in the UI (`preview: none`) or doing it once instead of at steps 2 and 4 | 23.2–24.6 s at 512×768 in every variant, within noise: the preview (`proj`) costs nothing |
| Qwen-Image 2.1 Turbo: 6 and 4 steps instead of 8 (a thinned turbo schedule) | 157 and 105 s instead of 209 s, the scene is almost the same, but text in the image falls apart — and text is the main reason to use this model |
| A/B without a cooling pad | ±1 s spread from heat; compare only after cooling to <55 °C before each run and with 8 steps, or with the op profiler (but it inserts barriers and can mislead) |

## Power: power limits (RyzenAdj)

The GPU is limited not by temperature but by the power limit of the whole chip: the CPU and GPU share one budget. Telemetry (`/sys/class/drm/card1/device/gpu_metrics`) while generating 512×768 on a cooling pad:

| Setting | Steps | Power | GPU | Memory (fclk) | Temp. |
|---|---|---|---|---|---|
| "Balanced", Lenovo limits (STAPM 20, PPT slow **15**, fast 20 W) | 28.2 s | 15.3 W | 1416 MHz | 800 MHz | 47 °C |
| "Performance" | 32.9 s | 10.6 W | 1118 MHz | 800 MHz | 43 °C |
| "Power saver" | 32.9 s | 10.4 W | 1114 MHz | 800 MHz | 42 °C |
| **RyzenAdj 25/25/30 W, Tctl 90** | **25.7–26.0 s** | 19.5 W | 1588 MHz | 800 MHz | 50–54 °C |
| RyzenAdj + `power_dpm_force_performance_level=high` | 25.6 s | 20.5 W | 1593 MHz | 1333 MHz | 54–59 °C |
| RyzenAdj + `manual` (memory stuck at 400 MHz) | 35.0 s | 16.6 W | 1595 MHz | 400 MHz | 50 °C |

- The "Performance" profile on the T14s Gen 1 AMD behaves like power saving (firmware). Use "Balanced".
- The GPU does not go above 1600 MHz (`pp_od_clk_voltage`: range 200–1600), so limits above 25 W are not needed.
- Memory at 1333 instead of 800 MHz gives ~1.7%, and `high` keeps everything at maximum even when idle — not used.

Installation (needs sudo; RyzenAdj is built from [FlyGoat/RyzenAdj](https://github.com/FlyGoat/RyzenAdj) v0.19.0, needs `libpci-dev`):

```bash
sudo install -m 755 ryzenadj /usr/local/bin/ryzenadj
sudo install -m 644 power/ryzenadj-limits.service power/ryzenadj-limits.timer /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now ryzenadj-limits.timer
```

The timer applies the limits 15 s after boot and then every minute (the firmware may reset them after sleep or a profile change). To disable: `sudo systemctl disable --now ryzenadj-limits.timer`.

## Pitfalls

### amdgpu driver timeout (2 seconds)

The driver resets a GPU job that runs longer than 2 seconds (`lockup_timeout`, 2000 ms by default). At 896×1344 and above a single attention computation takes longer, and generation fails with `ErrorDeviceLost`. This also affects the stock build. To raise the timeout for compute only (needs sudo and a reboot):

```bash
echo 'options amdgpu lockup_timeout=0,30000,0,0' | sudo tee /etc/modprobe.d/amdgpu-timeout.conf
```

```bash
sudo update-initramfs -u
```

### Shared memory and hangs

The GPU and CPU share 16 GB. If the model weights (Anima f16 — 4.2 GB) and the compute buffers together exceed the GTT limit, the system does not fail with an error but **hangs hard**; only a reboot helps. This happened with untiled TAE at 512×768 under the default limit, with ROCm, with two models at once (the UI and `sd-cli`) and with 49 video frames decoded in one piece. So: do not run two generations at the same time (the UI keeps one `sd-server`, and `gen-anima.sh` and the server refuse to start next to another generation process), and try new resolutions at a small size first. GPU memory is not limited by `systemd-run -p MemoryMax`. A hung `sd-cli`/`sd-server` holds gigabytes of GTT: check `pgrep -a 'sd-(cli|server)'` before starting.

The GTT limit is not hardware but a driver setting: half of the RAM by default (`ttm.pages_limit`). It can be raised to 11 GiB by adding `ttm.pages_limit=2883584 ttm.page_pool_size=2883584` to `GRUB_CMDLINE_LINUX_DEFAULT` in `/etc/default/grub`, then `sudo update-grub` and a reboot; check with `cat /sys/class/drm/card1/device/mem_info_gtt_total` (11811160064). It does not make anything faster by itself, but with GTT ≥ 10 GiB the UI and `gen-anima.sh` decode untiled up to 640×960, 0.6–0.8 s faster (peak 7.0 GB at 640×960; at 768×1152 untiled is already slower). GPU memory is not limited by systemd's `MemoryMax`, so leave ~3 GB for the system, the browser and the text encoder: do not go above 11–11.5 GiB.

### Heat

After an hour of continuous generation sampling slows down by about 8% (71 → 77 s). Good cooling helps.

### Hires fix

sd.cpp does it itself: `--hires --hires-upscaler "Latent (nearest-exact)" --hires-denoising-strength 0.55`. No separate upscaler is needed; this is how image authors on Civitai do it. Above 768×1152 it hits the timeout above, and the VAE has to decode in tiles at that size (`--vae-tiling`).

## Profiling

Time of every GPU op per step:

```bash
GGML_VK_PERF_LOGGER=1 tools/bin-opt-sd-cli ... 2> perf.txt
```

UNet shape benchmarks:

```bash
SD_UNET_PERF=1 build-src/stable-diffusion.cpp/build/bin/test-backend-ops perf -b Vulkan0 -o MUL_MAT
```

Register and wave statistics for the shaders:

```bash
RADV_DEBUG=shaderstats,nocache ...
```

A typical breakdown of one UNet pass at 512×768: matmul 57%, attention 20%, im2col 6%, the rest is small. Anima is even more bound by matmul, i.e. by the hardware ceiling.

## Open directions

- Direct convolutions (`conv2d direct`) fail sd.cpp's probe on Vega and are disabled, so everything goes through im2col: in the taesdxl decoder im2col is 60% of the time, in an SDXL step ~3.5%.
- Distilling Anima to 2 steps, token merging with acceptable quality, adapters for a larger text encoder for Anima (only available for ComfyUI/Forge so far).
- For any direction: profile first, then change, then correctness tests (`test-backend-ops`, zero FAILs), A/B alternating with the old binary, and comparing the images by eye.

## License

Code and patches — MIT (like stable-diffusion.cpp and ggml). Models — under their authors' licenses.
