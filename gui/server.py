#!/usr/bin/env python3
"""Local web UI for image generation.

usage: ./server.py [port=7860]   then open http://127.0.0.1:7860
The selected model stays loaded in an sd-server process (port 7861), so only the first image after a model switch
pays for loading. Its settings mirror anima/gen-anima.sh; keep them in sync when changing those.
Images and history live in gui/data/; deleted images go to the desktop trash (gio trash).
Standard library only. Listens on 127.0.0.1, so it is not reachable from the network.
"""
import atexit
import base64
import glob
import hashlib
import json
import queue as queue_lib
import os
import random
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import urllib.request
from urllib.parse import parse_qs, unquote, urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(HERE, "data")
IMAGES = os.path.join(DATA, "images")
INPUTS = os.path.join(DATA, "inputs")  # img2img source images, named by content hash
HISTORY = os.path.join(DATA, "history.json")

ANIMA = os.path.join(ROOT, "anima")
SDXL = os.path.join(ROOT, "sdxl")
TOOLS = os.path.join(ROOT, "tools")
SD_SERVER = os.path.join(TOOLS, "bin-opt-sd-server")
BACKEND_PORT = 7861
# tiny VAE taew2_1 in 128px tiles (fastest and smallest buffer; 64px is slower and shows seams): untiled it needs ~2.6 GB at 512x768, which froze the laptop next to the weights
TILES = {"enabled": True, "tile_size_w": 128, "tile_size_h": 128, "target_overlap": 0.25}


def gtt_gib():
    try:
        return max(int(open(f).read()) for f in glob.glob("/sys/class/drm/card*/device/mem_info_gtt_total")) / 2 ** 30
    except (OSError, ValueError):
        return 0


# decoding in one piece is ~0.6-0.8 s faster up to 640x960 (peak 7.0 GB of GPU memory with the weights) and slower
# above; it needs the GTT limit raised to ~11 GiB (ttm.pages_limit), with the default half of RAM it would hang
UNTILED_MAX = 640 * 960 if gtt_gib() >= 10 else 0

ANIMA_TIMES = {"920x1536": "~2:05", "832x1248": "~1:25", "768x1152": "~1:03", "704x1056": "~51 с", "640x960": "~40 с",
               "576x864": "~31 с", "512x768": "~23 с", "448x672": "~17 с", "384x576": "~12 с"}


# (name, why it is recommended or ""): the window marks recommended ones with a star
ANIMA_SAMPLERS = [("euler", "так обучена турбо-LoRA"), ("euler_a", ""), ("dpm++2m", ""), ("res_multistep", ""), ("ipndm", "")]
ANIMA_SCHEDULERS = [("discrete", "по умолчанию в sd.cpp"), ("simple", ""), ("sgm_uniform", ""), ("karras", ""), ("exponential", "")]
# Anima 2.9B (40 blocks) with Turbo-ANIMA V4-29B baked in: authors recommend Euler + sgm_uniform, a step is ~1.5x of 2B
ANIMA29_SCHEDULERS = [("sgm_uniform", "рекомендация для Anima 2.9B"), ("simple", ""), ("discrete", ""), ("karras", ""), ("exponential", "")]
ANIMA29_TIMES = {"920x1536": "~3:00", "832x1248": "~2:05", "768x1152": "~1:32", "704x1056": "~1:14", "640x960": "~58 с",
                 "576x864": "~45 с", "512x768": "~33 с", "448x672": "~25 с", "384x576": "~18 с"}
SDXL_SAMPLERS = [("euler_a", "проверено с DMD2 на 4 шагах"), ("lcm", "рекомендация авторов DMD2"), ("dpm++2s_a", "от 5 шагов"),
                 ("euler", ""), ("dpm++2m", ""), ("dpm++2m_sde", "")]
SDXL_SCHEDULERS = [("gits", "проверено с DMD2 на 4 шагах"), ("simple", ""), ("sgm_uniform", ""), ("karras", ""),
                   ("exponential", ""), ("discrete", "")]


def anima_mode(label, model, prefix, schedulers=ANIMA_SCHEDULERS, times=ANIMA_TIMES):
    """Anima models in f16 (anima/merge_lora.py): the official Anima Turbo or any Anima finetune with a turbo LoRA baked in.
    Own LoRAs go to anima/loras (not the turbo one, it is already in the weights)."""
    return {
        "label": label, "steps": 4, "size": "512x768",
        "sizes": ["920x1536", "832x1248", "768x1152", "704x1056", "640x960", "576x864", "512x768", "448x672", "384x576"],
        "times": times,
        "samplers": ANIMA_SAMPLERS, "schedulers": schedulers, "loras": [],
        "cwd": ANIMA,
        "lora_dir": os.path.join(ANIMA, "loras"),
        "model": os.path.join(ANIMA, "models", model),
        "args": ["--diffusion-model", "models/" + model,
                 "--llm", "models/qwen_3_06b_base.safetensors",
                 "--taesd", "models/taew2_1.safetensors", "--lora-model-dir", "loras",
                 "--backend", "clip=cpu", "--params-backend", "clip=cpu", "--diffusion-fa", "--eager-load"],
        "request": lambda p: {
            "prompt": prefix + p,
            "sample_params": {"guidance": {"txt_cfg": 1.0}},
            "vae_tiling_params": TILES},
    }


def sdxl_mode(label, model, loras, prefix):
    """SDXL checkpoint with a 4-step LoRA (CFG 1) and the tiny decoder taesdxl: the full SDXL VAE takes 30-76 s here.
    LoRAs, the turbo ones included, are picked per image from sdxl/models/loras."""
    return {
        "label": label, "steps": 4, "size": "832x1216",
        # the author's recommended sizes plus 896x1152, a standard SDXL one
        "sizes": ["1024x1536", "896x1152", "832x1216", "768x1280", "768x1344", "704x1408"],
        "times": {"1024x1536": "~1:02", "896x1152": "~40 с", "832x1216": "~38 с", "768x1280": "~37 с", "768x1344": "~39 с",
                  "704x1408": "~38 с"},
        "samplers": SDXL_SAMPLERS, "schedulers": SDXL_SCHEDULERS, "loras": loras,
        "cwd": SDXL,
        "lora_dir": os.path.join(SDXL, "models", "loras"),
        "model": os.path.join(SDXL, "models", model),
        # embeddings by file name in the prompt (sdxl/models/embeddings); CLIP-L/G stay on the GPU: 0.5 s per prompt instead
        # of 2.4-2.7 s on the CPU, +1.6 GB of GPU memory (peak 9.5 GB at 1024x1536), same quality
        "args": ["-m", "models/" + model, "--lora-model-dir", "models/loras", "--embd-dir", "models/embeddings",
                 "--taesd", "models/taesdxl.safetensors", "--diffusion-fa", "--eager-load"]
                + ([] if gtt_gib() >= 10 else ["--backend", "clip=cpu", "--params-backend", "clip=cpu"]),
        # taesdxl in one piece: 2.4 s at 832x1216 (peak 7.9 GB); at 1024x1536 tiles are faster (4.8 vs 8.6 s) and stay out of 9.4 GB
        "untiled_max": 896 * 1152 if gtt_gib() >= 10 else 0,
        "request": lambda p: {
            "prompt": prefix + p, "clip_skip": 2,
            "sample_params": {"guidance": {"txt_cfg": 1.0}},
            "vae_tiling_params": TILES},
    }


# Prompt prefixes are quality tags only; CFG is 1 with the turbo models, so negative prompts have no effect.
_QUALITY = "best quality, score_7, score_9, very aesthetic, ultra detailed, "
MODES = {k: v for k, v in {
    # official circlestone-labs/Anima turbo v1.1, converted to f16 (scripts/download.sh anima)
    "anima": anima_mode("Anima 2B · turbo", "anima-turbo-v1.1_f16.safetensors", _QUALITY),
    # Anima 2.9B (40 blocks) with Turbo-ANIMA V4-29B baked in by anima/merge_lora.py (see README)
    "anima29": anima_mode("Anima 2.9B · turbo", "anima-2.9b_turbo4_f16.safetensors", _QUALITY, ANIMA29_SCHEDULERS, ANIMA29_TIMES),
    # any Illustrious/NoobAI-class SDXL checkpoint saved as sdxl/models/sdxl.safetensors, with the 4-step DMD2 LoRA (CFG 1)
    "sdxl": sdxl_mode("SDXL · DMD2 4 шага", "sdxl.safetensors",
                      [{"file": "dmd2_sdxl_4step_lora_fp16.safetensors", "weight": 1.0}], "masterpiece, best quality, newest, absurdres, "),
}.items() if os.path.exists(v["model"])}


def lora_files(mode):
    d = MODES[mode]["lora_dir"]
    return sorted(f for f in os.listdir(d) if f.endswith(".safetensors")) if os.path.isdir(d) else []


UPSCALE = os.path.join(TOOLS, "upscale.sh")

lock = threading.Lock()
wake = threading.Event()
upscale_jobs = queue_lib.Queue()  # (entry, path): x2 runs here so the next generation does not wait for it
queue = []          # jobs waiting or running
current = {}        # progress of the image being made
history = []        # newest first
version = 0         # bumped on every history change so the page knows to refetch
proc = None
cancel_current = False
last_error = {}     # tail of the output of the last failed run, shown in the page
preview_png = b""   # latest step preview of the image being made


def backend_call(path, body=None, timeout=30):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{BACKEND_PORT}{path}", data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


class Backend:
    """One sd-server with the current mode's model loaded; only one model fits in the GPU memory at a time."""

    def __init__(self):
        self.mode, self.proc, self.log = None, None, os.path.join(DATA, "backend.log")
        self.lock = threading.Lock()

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def ensure(self, mode):
        with self.lock:
            if self.mode == mode and self.alive():
                return
            self._stop()
            # a second copy of the model does not fit next to this one in the shared GPU memory (GTT): the kernel
            # starts evicting buffers, which hung the laptop and once left an unkillable process behind
            ps = subprocess.run(["ps", "-eo", "pid=,stat=,comm="], capture_output=True, text=True).stdout.split("\n")
            other = [f[0] for f in (l.split() for l in ps) if len(f) == 3 and "Z" not in f[1] and
                     f[2] in ("bin-opt-sd-cli", "bin-opt-sd-serv", "sd-cli", "sd-server")]
            if other:
                raise RuntimeError("another generation process is running (pid " + ", ".join(other) + "); "
                                   "wait for it or stop it, two models do not fit in GPU memory")
            m = MODES[mode]
            with open(self.log, "wb") as log:
                self.proc = subprocess.Popen([SD_SERVER, "--listen-port", str(BACKEND_PORT)] + m["args"], cwd=m["cwd"],
                                             stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            self.mode = mode
            deadline = time.time() + 300
            while time.time() < deadline:
                if not self.alive():
                    self.mode = None
                    raise RuntimeError("sd-server exited while loading:\n" + self.tail())
                try:
                    backend_call("/sdcpp/v1/capabilities", timeout=2)
                    return
                except OSError:
                    time.sleep(0.5)
            self._stop()
            raise RuntimeError("sd-server did not start in 300 s:\n" + self.tail())

    def tail(self, n=15):
        try:
            with open(self.log, "rb") as f:
                lines = [l.decode("utf-8", "replace") for l in re.split(rb"[\r\n]", f.read()[-20000:]) if l.strip()]
            return "\n".join(lines[-n:])
        except OSError:
            return ""

    def _stop(self):
        if self.alive():
            os.killpg(self.proc.pid, signal.SIGTERM)
            try:
                self.proc.wait(15)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
        self.proc, self.mode = None, None

    def stop(self):
        with self.lock:
            self._stop()


backend = Backend()
atexit.register(backend.stop)


def save_history():
    tmp = HISTORY + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=1)
    os.replace(tmp, HISTORY)


def load_history():
    global history
    if os.path.exists(HISTORY):
        with open(HISTORY, encoding="utf-8") as f:
            history = [e for e in json.load(f) if os.path.exists(os.path.join(IMAGES, e["file"]))]


def add_entry(entry):
    global version
    with lock:
        history.insert(0, entry)
        version += 1
        save_history()


def run(cmd, env, on_line):
    """Runs a generation command, feeding progress lines (split on \\r and \\n) to on_line.
    Returns (exit code, last output lines)."""
    global proc
    p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
    with lock:
        proc = p
    buf = b""
    tail = []
    while True:
        chunk = p.stdout.read1(4096)
        if not chunk:
            break
        buf += chunk
        *lines, buf = re.split(rb"[\r\n]", buf)
        for line in lines:
            text = line.decode("utf-8", "replace")
            if text.strip():
                tail = (tail + [text])[-15:]
            on_line(text)
    code = p.wait()
    with lock:
        proc = None
    return code, tail


def report_failure(code, tail):
    global last_error
    if cancel_current:
        return
    msg = "\n".join(tail)
    print(f"generation failed (exit {code}):\n{msg}", file=sys.stderr, flush=True)
    with lock:
        last_error = {"time": time.time(), "code": code, "text": msg}


def progress_parser():
    def on_line(line):
        if "upscaled" in line:
            with lock:
                current.update(phase="done")
    return on_line


def generate_one(job, index):
    global preview_png
    seed = job["seed"] + index if job["seed"] >= 0 else random.randrange(2 ** 31)
    name = time.strftime("%Y%m%d-%H%M%S") + f"-{uuid.uuid4().hex[:6]}.png"
    with lock:
        current.clear()
        current.update(job=job["id"], index=index, count=job["count"], step=0, total=job["steps"],
                       started=time.time(), seed=seed,
                       phase="generating" if backend.mode == job["mode"] and backend.alive() else "loading")
        preview_png = b""
    t0 = time.time()
    try:
        backend.ensure(job["mode"])
        with lock:
            current["phase"] = "generating"
        w, h = map(int, job["size"].split("x"))
        body = MODES[job["mode"]]["request"](job["prompt"])
        body.update(width=w, height=h, seed=seed, preview="proj", preview_interval=2)  # every step costs ~0.8 s at 512x768 (sync readback + png)
        body["sample_params"].update(sample_steps=job["steps"], sample_method=job["sampler"], scheduler=job["scheduler"])
        body["lora"] = [{"path": l["file"], "multiplier": l["weight"]} for l in job["loras"]]
        if w * h <= MODES[job["mode"]].get("untiled_max", UNTILED_MAX):
            body["vae_tiling_params"] = {"enabled": False}
        if job.get("init"):
            with open(os.path.join(INPUTS, job["init"]), "rb") as f:
                body["init_image"] = base64.b64encode(f.read()).decode()
            body["strength"] = job["strength"]
            # by default sd.cpp turns strength into a whole number of steps: at 4 steps only 3 levels, and >= 0.75 starts
            # from pure noise (the source is ignored); as a noise level it is continuous
            body["sample_params"]["extra_sample_args"] = "strength_as_noise_level=true,force_first_sigma=true"
        bj = backend_call("/sdcpp/v1/img_gen", body)
        with lock:
            current["backend_job"] = bj["id"]
        while True:
            st = backend_call(f"/sdcpp/v1/jobs/{bj['id']}")
            pv = st.get("preview")
            if pv:
                with lock:
                    if pv["step"] != current.get("step"):
                        preview_png = base64.b64decode(pv["b64_json"])
                    current.update(phase="decoding" if pv["step"] >= pv["total_steps"] else "sampling",
                                   step=pv["step"], total=pv["total_steps"])
            if st["status"] in ("completed", "failed", "cancelled"):
                break
            time.sleep(0.25)
        if st["status"] != "completed":
            if st["status"] == "failed":
                report_failure(1, [str(st.get("error"))])
            return False
        with open(os.path.join(IMAGES, name), "wb") as f:
            f.write(base64.b64decode(st["result"]["images"][0]["b64_json"]))
    except (OSError, RuntimeError, ValueError, KeyError) as e:
        report_failure(1, [f"{type(e).__name__}: {e}", backend.tail()])
        return False
    entry = {"id": uuid.uuid4().hex[:12], "file": name, "mode": job["mode"], "prompt": job["prompt"],
             "steps": job["steps"], "seed": seed, "size": job["size"], "created": time.time(),
             "duration": round(time.time() - t0, 1), "fav": False, "out_size": job["size"],
             "sampler": job["sampler"], "scheduler": job["scheduler"], "loras": job["loras"]}
    if job.get("init"):
        entry.update(init=job["init"], strength=job["strength"])
    if job.get("upscale"):
        upscale_jobs.put((entry, os.path.join(IMAGES, name)))
    else:
        add_entry(entry)
    return True


def upscale_worker():
    """Upscales finished images x2 one by one; the entry shows the x2 result and keeps the base image as orig_file."""
    while True:
        entry, path = upscale_jobs.get()
        out = path[:-4] + "_x2.png"
        r = subprocess.run([UPSCALE, "--x2", path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if r.returncode == 0 and os.path.exists(out):
            entry.update(orig_file=entry["file"], file=os.path.basename(out), out_size=double(entry["size"]))
        else:
            report_failure(r.returncode, r.stdout.splitlines()[-15:])
        add_entry(entry)


def double(size):
    w, h = (int(v) * 2 for v in size.split("x"))
    return f"{w}x{h}"


def upscale_file(path):
    """x2 next to path via upscale.sh; returns the new path or None."""
    code, tail = run([UPSCALE, "--x2", path], os.environ.copy(), progress_parser())
    out = path[:-4] + "_x2.png"
    if code == 0 and os.path.exists(out):
        return out
    report_failure(code, tail)
    return None


def upscale_one(job):
    src = next((e for e in history if e["id"] == job["source"]), None)
    if not src:
        return False
    with lock:
        current.clear()
        current.update(job=job["id"], index=0, count=1, phase="upscaling", step=0, total=0, started=time.time())
    t0 = time.time()
    out = upscale_file(os.path.join(IMAGES, src["file"]))
    if not out:
        return False
    entry = {**src, "id": uuid.uuid4().hex[:12], "file": os.path.basename(out),
             "out_size": double(src.get("out_size", src["size"])), "created": time.time(),
             "duration": round(time.time() - t0, 1), "fav": False, "upscaled_from": src["id"]}
    entry.pop("orig_file", None)
    add_entry(entry)
    return True


def worker():
    global cancel_current
    while True:
        wake.wait()
        with lock:
            job = queue[0] if queue else None
            if not job:
                wake.clear()
                continue
            job["status"] = "running"
        if job["kind"] == "upscale":
            upscale_one(job)
            job["done"] = 1
        else:
            while job["done"] < job["count"]:
                with lock:
                    if cancel_current:
                        break
                ok = generate_one(job, job["done"])
                job["done"] += 1
                if not ok:
                    with lock:
                        job["failed"] = job.get("failed", 0) + 1
        with lock:
            cancel_current = False
            current.clear()
            if queue and queue[0] is job:
                queue.pop(0)


def trash(ids):
    global version
    with lock:
        keep, gone = [], []
        for e in history:
            (gone if e["id"] in ids else keep).append(e)
        history[:] = keep
        version += 1
        save_history()
    used = {f for e in history for f in (e["file"], e.get("orig_file")) if f}
    for e in gone:
        for f in (e["file"], e.get("orig_file")):
            if f and f not in used:
                subprocess.run(["gio", "trash", os.path.join(IMAGES, f)], check=False)
    return len(gone)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code, body, ctype="application/json"):
        if not isinstance(body, bytes):
            body = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "max-age=86400" if ctype == "image/png" else "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/":
            with open(os.path.join(HERE, "index.html"), "rb") as f:
                return self.send(200, f.read(), "text/html; charset=utf-8")
        if url.path.startswith(("/images/", "/inputs/")):
            folder, name = url.path[1:].split("/", 1)
            path = os.path.join(IMAGES if folder == "images" else INPUTS, os.path.basename(unquote(name)))
            if os.path.isfile(path):
                with open(path, "rb") as f:
                    return self.send(200, f.read(), "image/png")
            return self.send(404, {"error": "not found"})
        if url.path == "/api/config":
            return self.send(200, {k: {**{kk: v[kk] for kk in ("label", "steps", "sizes", "size", "times", "samplers", "schedulers",
                                                               "loras")}, "lora_folder": os.path.relpath(v["lora_dir"], ROOT)}
                                   for k, v in MODES.items()})
        if url.path == "/api/loras":
            mode = parse_qs(url.query).get("mode", [""])[0]
            return self.send(200, {"files": lora_files(mode) if mode in MODES else []})
        if url.path == "/api/state":
            with lock:
                cur = dict(current)
                if cur:
                    cur["elapsed"] = round(time.time() - cur["started"], 1)
                return self.send(200, {"version": version, "current": cur, "error": last_error,
                                       "loaded": backend.mode if backend.alive() else None,
                                       "queue": [{k: j.get(k) for k in ("id", "kind", "mode", "prompt", "count", "done", "status", "size")} for j in queue]})
        if url.path == "/api/preview":
            with lock:
                return self.send(200, preview_png or b"", "image/png")
        if url.path == "/api/history":
            with lock:
                return self.send(200, {"version": version, "items": history})
        return self.send(404, {"error": "not found"})

    def do_POST(self):
        global cancel_current, version
        url = urlparse(self.path)
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        if url.path == "/api/generate":
            mode = body.get("mode")
            if mode not in MODES or not str(body.get("prompt", "")).strip():
                return self.send(400, {"error": "нужен режим и промпт"})
            size = str(body.get("size") or MODES[mode]["size"])
            if not re.fullmatch(r"\d{2,4}x\d{2,4}", size):
                return self.send(400, {"error": "размер вида 512x768"})
            job = {"id": uuid.uuid4().hex[:8], "kind": "gen", "mode": mode, "prompt": body["prompt"].strip(),
                   "steps": max(1, min(100, int(body.get("steps") or MODES[mode]["steps"]))),
                   "seed": int(body.get("seed", -1)), "size": size,
                   "count": max(1, min(100, int(body.get("count") or 1))), "done": 0, "status": "queued",
                   "upscale": bool(body.get("upscale", True))}
            m = MODES[mode]
            names = lambda opts: [n for n, _ in opts]  # noqa: E731
            job["sampler"] = body.get("sampler") if body.get("sampler") in names(m["samplers"]) else m["samplers"][0][0]
            job["scheduler"] = body.get("scheduler") if body.get("scheduler") in names(m["schedulers"]) else m["schedulers"][0][0]
            # a request without the field (the web page) gets the mode's default LoRAs, e.g. the turbo one for SDXL
            loras = body["loras"] if isinstance(body.get("loras"), list) else m["loras"]
            files = lora_files(mode)
            job["loras"] = [{"file": l["file"], "weight": max(-2.0, min(2.0, float(l.get("weight", 1))))}
                            for l in loras[:8] if isinstance(l, dict) and l.get("file") in files]
            if body.get("init"):
                init = os.path.basename(str(body["init"]))
                if not os.path.isfile(os.path.join(INPUTS, init)):
                    return self.send(400, {"error": "исходная картинка не найдена, загрузите её заново"})
                job.update(init=init, strength=max(0.05, min(1.0, float(body.get("strength", 0.75)))))
            with lock:
                queue.append(job)
            wake.set()
            return self.send(200, {"id": job["id"]})
        if url.path == "/api/init":
            # source for img2img: an uploaded file (data URL) or an image from the history, copied into inputs/
            if body.get("image"):
                src = os.path.join(IMAGES, os.path.basename(str(body["image"])))
                if not os.path.isfile(src):
                    return self.send(404, {"error": "картинка не найдена"})
                with open(src, "rb") as f:
                    raw, ext = f.read(), "png"
            else:
                m = re.fullmatch(r"data:image/(png|jpeg|webp);base64,(.+)", str(body.get("data", "")), re.S)
                if not m:
                    return self.send(400, {"error": "нужна картинка PNG, JPEG или WebP"})
                raw, ext = base64.b64decode(m[2]), m[1].replace("jpeg", "jpg")
            name = hashlib.sha1(raw).hexdigest()[:16] + "." + ext
            if not os.path.exists(os.path.join(INPUTS, name)):
                with open(os.path.join(INPUTS, name), "wb") as f:
                    f.write(raw)
            return self.send(200, {"init": name})
        if url.path == "/api/upscale":
            job = {"id": uuid.uuid4().hex[:8], "kind": "upscale", "mode": "upscale", "source": body.get("id"),
                   "prompt": "апскейл ×2", "count": 1, "done": 0, "status": "queued"}
            with lock:
                queue.append(job)
            wake.set()
            return self.send(200, {"id": job["id"]})
        if url.path == "/api/cancel":
            target = body.get("id")
            with lock:
                for j in list(queue):
                    if target in (None, j["id"]):
                        if j["status"] == "running":
                            cancel_current = True
                            if proc:
                                os.killpg(proc.pid, signal.SIGTERM)
                            if current.get("backend_job"):
                                threading.Thread(target=backend_call, daemon=True,
                                                 args=(f"/sdcpp/v1/jobs/{current['backend_job']}/cancel", {})).start()
                        else:
                            queue.remove(j)
            return self.send(200, {"ok": True})
        if url.path == "/api/fav":
            with lock:
                for e in history:
                    if e["id"] in body.get("ids", []):
                        e["fav"] = bool(body.get("fav"))
                version += 1
                save_history()
            return self.send(200, {"ok": True})
        if url.path == "/api/trash":
            return self.send(200, {"trashed": trash(set(body.get("ids", [])))})
        if url.path == "/api/preload":
            # warm the model the page switched to, unless a job is going to load its own anyway
            if body.get("mode") in MODES and not queue:
                threading.Thread(target=self.preload, args=(body["mode"],), daemon=True).start()
            return self.send(200, {"ok": True})
        if url.path == "/api/unload":
            if not queue:
                backend.stop()
            return self.send(200, {"ok": not queue})
        if url.path == "/api/open_folder":
            subprocess.Popen(["xdg-open", IMAGES], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return self.send(200, {"ok": True})
        return self.send(404, {"error": "not found"})


    @staticmethod
    def preload(mode):
        try:
            backend.ensure(mode)
        except RuntimeError as e:
            report_failure(1, [str(e)])


def main():
    os.makedirs(IMAGES, exist_ok=True)
    os.makedirs(INPUTS, exist_ok=True)
    load_history()
    threading.Thread(target=worker, daemon=True).start()
    threading.Thread(target=upscale_worker, daemon=True).start()
    signal.signal(signal.SIGTERM, lambda *a: sys.exit(0))
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 7860
    print(f"http://127.0.0.1:{port}")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
