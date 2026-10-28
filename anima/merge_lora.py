#!/usr/bin/env python3
"""Bakes a LoRA (lora_A/lora_B or lora_down/lora_up pairs, optional .alpha) into an Anima diffusion model and writes it as f16 safetensors.

usage: ./merge_lora.py model.safetensors lora.safetensors|none out.safetensors [strength=1.0]
"none" only converts the model to f16 (e.g. the official bf16 Anima Turbo).
LoRA keys "diffusion_model.X.lora_{A,B}.weight" patch model key "net.X.weight" (or "model.diffusion_model.X.weight"):
W += strength * alpha/rank * B @ A
(alpha/rank = 1 without an .alpha key).
Streams tensor by tensor, so RAM use stays around the size of the largest tensor.
"""
import json
import struct
import sys

import numpy as np


def read_header(path):
    f = open(path, "rb")
    n = struct.unpack("<Q", f.read(8))[0]
    return f, json.loads(f.read(n)), 8 + n


def load(f, base, info):
    start, end = info["data_offsets"]
    f.seek(base + start)
    raw = np.frombuffer(f.read(end - start), dtype={"BF16": np.uint16, "F16": np.float16, "F32": np.float32}[info["dtype"]])
    if info["dtype"] == "BF16":
        raw = (raw.astype(np.uint32) << 16).view(np.float32)
    return raw.astype(np.float32).reshape(info["shape"])


def main():
    model_path, lora_path, out_path = sys.argv[1:4]
    strength = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
    mf, mh, mbase = read_header(model_path)
    lf, lh, lbase = read_header(lora_path) if lora_path != "none" else (None, {}, 0)
    names = [k for k in mh if k != "__metadata__"]

    header, offset = {}, 0
    for k in names:
        size = int(np.prod(mh[k]["shape"])) * 2
        header[k] = {"dtype": "F16", "shape": mh[k]["shape"], "data_offsets": [offset, offset + size]}
        offset += size
    blob = json.dumps(header).encode()
    blob += b" " * (-len(blob) % 8)

    merged = 0
    with open(out_path, "wb") as out:
        out.write(struct.pack("<Q", len(blob)) + blob)
        for k in names:
            w = load(mf, mbase, mh[k])
            # model keys are "net.X" or, in ComfyUI-style files, "model.diffusion_model.X"
            pre = next((p for p in ("net.", "model.diffusion_model.") if k.startswith(p)), None)
            lk = "diffusion_model." + k[len(pre):-len(".weight")] if pre and k.endswith(".weight") else None
            down, up = (".lora_A.weight", ".lora_B.weight") if lk and lk + ".lora_A.weight" in lh else (".lora_down.weight", ".lora_up.weight")
            if lk and lk + down in lh:
                a = load(lf, lbase, lh[lk + down])
                b = load(lf, lbase, lh[lk + up])
                scale = float(load(lf, lbase, lh[lk + ".alpha"]).item()) / a.shape[0] if lk + ".alpha" in lh else 1.0
                w = w + strength * scale * (b @ a).reshape(w.shape)
                merged += 1
            if np.abs(w).max() > 65504:
                print(f"warning: {k} exceeds f16 range, clipped")
                w = np.clip(w, -65504, 65504)
            out.write(w.astype(np.float16).tobytes())
    lora_pairs = sum(1 for k in lh if k.endswith(("lora_A.weight", "lora_down.weight")))
    print(f"merged {merged}/{lora_pairs} LoRA layers into {len(names)} tensors -> {out_path}")
    assert merged == lora_pairs


if __name__ == "__main__":
    main()
