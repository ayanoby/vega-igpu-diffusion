#!/usr/bin/env python3
"""Cinemagraph: keeps the source image pixel-exact except where the character moves.

usage: cinemagraph.py loop.avi source.png out_basename [--frames N] [--mask mask.png]
                     [--blink closed_eyes.png --blink-mask eyes_mask.png] [--repeat N]
The loop comes from Wan with the source as both first and last frame (--end-img); its last latent frame
(4 video frames) decodes washed out, so only the first N frames are used (default: all but 4).
Without --mask the mask is the largest moving area (the character), closed, hole-filled, widened and feathered;
a hand-drawn mask (white = animate) replaces it. Writes out_basename.mp4, .gif and _mask.png.
Wan does not blink on request, so a blink is blended in from the source with the eyes inpainted closed (Anima,
--mask on the eyes); --repeat plays the motion loop N times with the blink in the middle of the first pass only.
"""
import argparse
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def read_frames(path, w, h):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)


def morph(img, size, grow):
    # ponytail: repeated 9px PIL max/min filters, O(size) passes; fine at 512x768
    f = ImageFilter.MaxFilter(9) if grow else ImageFilter.MinFilter(9)
    for _ in range(max(1, size // 8)):
        img = img.filter(f)
    return img


def largest_component(img):
    """Boolean array of the largest white 8-connected area, labelled with PIL flood fills."""
    work = img.copy()
    best, best_area = None, 0
    while True:
        ys, xs = np.nonzero(np.asarray(work) == 255)
        if not len(ys):
            return best
        ImageDraw.floodfill(work, (int(xs[0]), int(ys[0])), 100)
        comp = np.asarray(work) == 100
        if comp.sum() > best_area:
            best, best_area = comp, comp.sum()
        ImageDraw.floodfill(work, (int(xs[0]), int(ys[0])), 0)


def auto_mask(frames, src):
    motion = np.percentile(np.abs(frames.astype(np.float32) - src).mean(-1), 90, axis=0)
    # the whole figure sways a little, so its outline lights up as a closed contour; the background moves less
    m = Image.fromarray(((motion > 45) * 255).astype(np.uint8))
    m = morph(morph(m, 16, True), 16, False)          # close gaps in the outline
    m = largest_component(m)                          # drop specks (flickering lights, passers-by)
    # ponytail: fill each row between its outermost outline pixels; holds for one character, not for several
    span = np.zeros_like(m)
    for y, row in enumerate(m):
        xs = np.flatnonzero(row)
        if len(xs) > 1:
            span[y, xs[0]:xs[-1] + 1] = True
    m = Image.fromarray((span * 255).astype(np.uint8)).filter(ImageFilter.MedianFilter(15))
    return morph(m, 16, True).filter(ImageFilter.GaussianBlur(8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("source")
    ap.add_argument("out")
    ap.add_argument("--frames", type=int)
    ap.add_argument("--mask")
    ap.add_argument("--blink")
    ap.add_argument("--blink-mask")
    ap.add_argument("--repeat", type=int, default=1)
    a = ap.parse_args()
    src_img = Image.open(a.source).convert("RGB")
    frames = read_frames(a.video, *src_img.size)
    frames = frames[:a.frames or len(frames) - 4]
    src = np.asarray(src_img, np.float32)
    mask = Image.open(a.mask).convert("L").resize(src_img.size).filter(ImageFilter.GaussianBlur(4)) if a.mask \
        else auto_mask(frames, src)
    mask.save(a.out + "_mask.png")
    alpha = np.asarray(mask, np.float32)[..., None] / 255
    out = frames * alpha + src * (1 - alpha)
    out = np.concatenate([out] * a.repeat)
    if a.blink:
        closed = np.asarray(Image.open(a.blink).convert("RGB"), np.float32)
        eyes = np.asarray(Image.open(a.blink_mask).convert("L"), np.float32)[..., None] / 255
        start = len(frames) // 2
        for i, k in enumerate((0.35, 0.8, 1, 1, 0.7, 0.35, 0.1)):   # eyelids close fast, open a bit slower
            out[start + i] = out[start + i] * (1 - eyes * k) + closed * eyes * k
    out = out.round().astype(np.uint8)
    h, w = out.shape[1:3]
    enc = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-framerate", "24", "-i", "-"]
    subprocess.run(enc + ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", a.out + ".mp4"], input=out.tobytes(), check=True)
    subprocess.run(enc + ["-vf", "split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4",
                          "-loop", "0", a.out + ".gif"], input=out.tobytes(), check=True)
    print(f"{len(out)} frames -> {a.out}.mp4, {a.out}.gif")


if __name__ == "__main__":
    main()
