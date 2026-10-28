#!/usr/bin/env python3
"""Native GTK 4 window for the generation server (instead of the browser page, which costs ~1 GB of RAM).

usage: ./app.py   (starts ./server.py if it is not running; stops it on exit if it started it, freeing the model)
Everything (queue, history, model loading, img2img) stays in server.py; this window only shows it and sends requests.
Images are read straight from gui/data/images. Needs python3-gi with GTK 4 and libadwaita (stock on Ubuntu GNOME).
"""
import base64
import json
import os
import subprocess
import threading
import time
import urllib.request

import gi

# GTK's default Vulkan renderer keeps a context on the same GPU and slowed the generation steps by ~7% (5.39 vs 5.05 s at
# 512x768) even with the window idle; this simple window draws fine on the CPU
os.environ.setdefault("GSK_RENDERER", "cairo")
gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gdk, GdkPixbuf, Gio, GLib, Gtk, Pango  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
IMAGES = os.path.join(HERE, "data", "images")
FORM = os.path.join(HERE, "data", "app-form.json")
URL = "http://127.0.0.1:7860"
PHASES = {"loading": "загрузка модели", "generating": "генерация", "sampling": "шаг", "decoding": "декодирование",
          "upscaling": "апскейл", "done": "готово"}
CSS = b"""
.thumb { padding: 0; border-radius: 6px; }
.thumb.active { outline: 2px solid alpha(currentColor, .6); outline-offset: -2px; }
.star { color: #ffd166; font-size: 15px; margin: 2px 5px; text-shadow: 0 0 3px black; }
.side { padding: 12px; }
.error { color: #ff6b6b; font-family: monospace; font-size: 11px; }
"""


def api(path, body=None, raw=False, timeout=30):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(URL + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read() if raw else json.load(r)


def background(fn, done=None):
    """Runs fn in a thread and hands its result (or the exception) to done on the GTK main loop."""
    def run():
        try:
            res = fn()
        except Exception as e:  # noqa: BLE001 — shown to the user
            res = e
        if done:
            GLib.idle_add(done, res)
    threading.Thread(target=run, daemon=True).start()


def thumb_texture(path, height=110):
    pb = GdkPixbuf.Pixbuf.new_from_file_at_scale(path, -1, height, True)
    return Gdk.Texture.new_for_pixbuf(pb)


class Window(Adw.ApplicationWindow):
    def __init__(self, app, config):
        super().__init__(application=app, title="Генерация", default_width=1200, default_height=860)
        self.config, self.items, self.version, self.shown = config, [], -1, None
        self.only_fav, self.landscape, self.init_name = False, False, None
        self.thumbs, self.thumb_cache, self.preview_key, self.err_time, self.cards = {}, {}, None, None, {}

        side = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6, css_classes=["side"], width_request=330)
        self.mode = Gtk.DropDown.new_from_strings([v["label"] for v in config.values()])
        self.mode.connect("notify::selected", self.on_mode)
        side.append(self.label("Модель"))
        side.append(self.mode)

        side.append(self.label("Промпт"))
        self.prompt = Gtk.TextView(wrap_mode=Gtk.WrapMode.WORD_CHAR, top_margin=6, bottom_margin=6, left_margin=6,
                                   right_margin=6)
        side.append(Gtk.ScrolledWindow(child=self.prompt, min_content_height=130, has_frame=True))

        side.append(self.label("Исходная картинка (img2img) — необязательно"))
        init_row = Gtk.Box(spacing=6)
        self.init_pic = Gtk.Picture(content_fit=Gtk.ContentFit.CONTAIN, height_request=64, width_request=48, visible=False)
        self.init_btn = Gtk.Button(label="выбрать файл…", hexpand=True,
                                   tooltip_text="Или перетащите файл сюда, или Ctrl+V с картинкой в буфере")
        self.init_btn.connect("clicked", self.on_pick_init)
        self.init_clear = Gtk.Button(icon_name="window-close-symbolic", visible=False, tooltip_text="Убрать")
        self.init_clear.connect("clicked", lambda *_: self.set_init(None))
        for w in (self.init_pic, self.init_btn, self.init_clear):
            init_row.append(w)
        drop = Gtk.DropTarget.new(Gio.File, Gdk.DragAction.COPY)
        drop.connect("drop", lambda _t, f, *_: self.upload_init_file(f.get_path()) or True)
        init_row.add_controller(drop)
        side.append(init_row)
        self.strength_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, visible=False)
        self.strength_box.append(self.label("Сила изменения: меньше — ближе к исходнику"))
        self.strength = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, 0.2, 0.95, 0.05)
        self.strength.set_digits(2)
        self.strength.set_draw_value(True)
        self.strength.set_value(0.75)
        self.strength_box.append(self.strength)
        side.append(self.strength_box)

        side.append(self.label("Разрешение"))
        row = Gtk.Box(spacing=6)
        self.size = Gtk.DropDown(hexpand=True)
        flip = Gtk.Button(label="⇄", tooltip_text="Вертикально ↔ горизонтально")
        flip.connect("clicked", self.on_flip)
        row.append(self.size)
        row.append(flip)
        side.append(row)

        grid = Gtk.Grid(column_spacing=6, row_spacing=4, column_homogeneous=True)
        self.steps = Gtk.SpinButton.new_with_range(1, 100, 1)
        self.count = Gtk.SpinButton.new_with_range(1, 100, 1)
        self.seed = Gtk.SpinButton.new_with_range(-1, 2 ** 31 - 1, 1)
        self.seed.set_value(-1)
        grid.attach(self.label("Шаги"), 0, 0, 1, 1)
        grid.attach(self.label("Картинок"), 1, 0, 1, 1)
        grid.attach(self.steps, 0, 1, 1, 1)
        grid.attach(self.count, 1, 1, 1, 1)
        self.sampler, self.scheduler = Gtk.DropDown(), Gtk.DropDown()
        grid.attach(self.label("Сэмплер"), 0, 2, 1, 1)
        grid.attach(self.label("Планировщик"), 1, 2, 1, 1)
        grid.attach(self.sampler, 0, 3, 1, 1)
        grid.attach(self.scheduler, 1, 3, 1, 1)
        side.append(grid)
        # why the selected sampler/scheduler is recommended (★), under the dropdowns
        self.sampling_note = Gtk.Label(xalign=0, wrap=True, css_classes=["dim-label", "caption"])
        for d in (self.sampler, self.scheduler):
            d.connect("notify::selected", lambda *_: self.update_sampling_note())
        side.append(self.sampling_note)

        self.lora_title = self.label("LoRA")
        side.append(self.lora_title)
        self.lora_rows = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        side.append(self.lora_rows)
        row = Gtk.Box(spacing=6)
        self.lora_pick = Gtk.DropDown(hexpand=True)
        add = Gtk.Button(icon_name="list-add-symbolic", tooltip_text="Добавить выбранную LoRA")
        add.connect("clicked", lambda *_: self.add_lora())
        reload = Gtk.Button(icon_name="view-refresh-symbolic", tooltip_text="Перечитать папку с LoRA")
        reload.connect("clicked", lambda *_: self.refresh_lora_files())
        for w in (self.lora_pick, add, reload):
            row.append(w)
        side.append(row)
        side.append(self.label("Seed (−1 — случайный)"))
        row = Gtk.Box(spacing=6)
        self.seed.set_hexpand(True)
        reset = Gtk.Button(label="−1")
        reset.connect("clicked", lambda *_: self.seed.set_value(-1))
        row.append(self.seed)
        row.append(reset)
        side.append(row)
        self.upscale = Gtk.CheckButton(label="апскейл ×2 сразу", active=True)
        side.append(self.upscale)

        self.go = Gtk.Button(label="Сгенерировать", css_classes=["suggested-action", "pill"], margin_top=8)
        self.go.connect("clicked", lambda *_: self.generate())
        side.append(self.go)

        row = Gtk.Box(spacing=6, margin_top=6)
        self.loaded = Gtk.Label(xalign=0, hexpand=True, css_classes=["dim-label"], ellipsize=Pango.EllipsizeMode.END)
        self.unload = Gtk.Button(label="выгрузить", css_classes=["flat"], tooltip_text="Освободить память GPU")
        self.unload.connect("clicked", lambda *_: background(lambda: api("/api/unload", {})))
        row.append(self.loaded)
        row.append(self.unload)
        side.append(row)
        self.queue = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        side.append(self.queue)
        self.error = Gtk.Label(xalign=0, wrap=True, selectable=True, css_classes=["error"], visible=False)
        side.append(self.error)
        side.append(Gtk.Label(label="Ctrl+Enter — генерация · ← → — листать · F — избранное · Del — в корзину · "
                                    "Ctrl+V — исходная картинка", wrap=True, xalign=0, vexpand=True, valign=Gtk.Align.END,
                              css_classes=["dim-label", "caption"]))

        # main view: picture, info bar, history strip
        self.picture = Gtk.Picture(content_fit=Gtk.ContentFit.CONTAIN, vexpand=True, hexpand=True)
        self.info = Gtk.Label(xalign=0, ellipsize=Pango.EllipsizeMode.END, hexpand=True, selectable=True)
        self.params = Gtk.Label(xalign=0, ellipsize=Pango.EllipsizeMode.END, css_classes=["dim-label", "caption"])
        info_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, hexpand=True)
        info_box.append(self.info)
        info_box.append(self.params)
        bar = Gtk.Box(spacing=2, margin_start=10, margin_end=10, margin_top=4, margin_bottom=4)
        bar.append(info_box)
        self.fav_btn = Gtk.Button(label="☆", css_classes=["flat"], tooltip_text="В избранное (F)")
        self.fav_btn.connect("clicked", lambda *_: self.toggle_fav())
        bar.append(self.fav_btn)
        for text, tip, fn in (("↺ настройки", "Подставить все настройки", self.reuse),
                              ("seed", "Подставить этот seed", self.reuse_seed),
                              ("копировать", "Скопировать промпт", self.copy_prompt),
                              ("→ img2img", "Сделать исходной картинкой", self.to_init),
                              ("×2", "Апскейл ×2", self.upscale_one),
                              ("🗑", "В корзину (Del)", self.trash_one)):
            b = Gtk.Button(label=text, css_classes=["flat"], tooltip_text=tip)
            b.connect("clicked", lambda _b, f=fn: f())
            bar.append(b)

        tools = Gtk.Box(spacing=6, margin_start=10, margin_end=10, margin_top=4)
        self.fav_filter = Gtk.ToggleButton(label="★ Избранное")
        self.fav_filter.connect("toggled", lambda b: self.set_filter(b.get_active()))
        self.search = Gtk.SearchEntry(placeholder_text="Поиск по промпту", width_request=240)
        self.search.connect("search-changed", lambda *_: self.render_strip())
        folder = Gtk.Button(label="папка", css_classes=["flat"])
        folder.connect("clicked", lambda *_: background(lambda: api("/api/open_folder", {})))
        self.count_label = Gtk.Label(css_classes=["dim-label"], hexpand=True, xalign=1)
        for w in (self.fav_filter, self.search, folder, self.count_label):
            tools.append(w)
        self.strip = Gtk.Box(spacing=6, margin_start=10, margin_end=10, margin_bottom=8, margin_top=4)
        self.strip_scroll = Gtk.ScrolledWindow(child=self.strip, vscrollbar_policy=Gtk.PolicyType.NEVER,
                                               min_content_height=124)

        main = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        for w in (self.picture, bar, Gtk.Separator(), tools, self.strip_scroll):
            main.append(w)
        paned = Gtk.Box()
        paned.append(Gtk.ScrolledWindow(child=side, hscrollbar_policy=Gtk.PolicyType.NEVER))
        paned.append(Gtk.Separator(orientation=Gtk.Orientation.VERTICAL))
        paned.append(main)
        view = Adw.ToolbarView(content=paned)
        view.add_top_bar(Adw.HeaderBar())
        self.toasts = Adw.ToastOverlay(child=view)
        self.set_content(self.toasts)

        keys = Gtk.EventControllerKey()
        keys.connect("key-pressed", self.on_key)
        self.add_controller(keys)
        self.load_form()
        # while minimized nothing is redrawn; catch up as soon as the window is shown again
        self.connect("notify::suspended", lambda *_: self.is_suspended() or self.poll())
        self.poll()

    @staticmethod
    def label(text):
        return Gtk.Label(label=text, xalign=0, css_classes=["dim-label", "caption"], margin_top=4)

    def toast(self, text):
        self.toasts.add_toast(Adw.Toast(title=text, timeout=2))

    # ---------- form ----------
    def mode_key(self):
        return list(self.config)[self.mode.get_selected()]

    def fill_sizes(self, keep=None):
        m = self.config[self.mode_key()]
        rev = lambda s: "x".join(reversed(s.split("x")))  # noqa: E731
        if keep and keep not in m["sizes"] and rev(keep) in m["sizes"]:
            self.landscape = True
        elif keep in m["sizes"] and keep != rev(keep):
            self.landscape = False
        self.size_values = [rev(s) if self.landscape else s for s in m["sizes"]]
        labels = [v.replace("x", " × ") + (f"  · {m['times'][s]}" if s in m.get("times", {}) else "")
                  for v, s in zip(self.size_values, m["sizes"])]
        self.size.set_model(Gtk.StringList.new(labels))
        target = keep if keep in self.size_values else (rev(m["size"]) if self.landscape else m["size"])
        self.size.set_selected(self.size_values.index(target) if target in self.size_values else 0)

    def set_mode_options(self, sampler=None, scheduler=None, loras=None):
        """Sampler, scheduler and LoRA list of the current mode; None means the mode's default."""
        m = self.config[self.mode_key()]
        for drop, opts, want in ((self.sampler, m["samplers"], sampler), (self.scheduler, m["schedulers"], scheduler)):
            names = [n for n, _ in opts]
            drop.set_model(Gtk.StringList.new([n + (" ★" if why else "") for n, why in opts]))
            drop.set_selected(names.index(want) if want in names else 0)
        self.loras = [dict(l) for l in (m["loras"] if loras is None else loras)]
        self.lora_title.set_label(f"LoRA — свои кладите в {m['lora_folder']}/")
        self.render_loras()
        self.refresh_lora_files()

    def update_sampling_note(self):
        m = self.config[self.mode_key()]
        notes = []
        for drop, opts in ((self.sampler, m["samplers"]), (self.scheduler, m["schedulers"])):
            i = drop.get_selected()
            if i < len(opts) and opts[i][1]:
                notes.append(f"★ {opts[i][0]}: {opts[i][1]}")
        self.sampling_note.set_label("\n".join(notes))
        self.sampling_note.set_visible(bool(notes))

    def picked(self, drop, key):
        opts = self.config[self.mode_key()][key]
        i = drop.get_selected()
        return opts[i][0] if i < len(opts) else opts[0][0]

    def render_loras(self):
        child = self.lora_rows.get_first_child()
        while child:
            self.lora_rows.remove(child)
            child = self.lora_rows.get_first_child()
        for l in self.loras:
            row = Gtk.Box(spacing=6)
            row.append(Gtk.Label(label=l["file"].removesuffix(".safetensors"), xalign=0, hexpand=True,
                                 ellipsize=Pango.EllipsizeMode.MIDDLE, tooltip_text=l["file"]))
            w = Gtk.SpinButton.new_with_range(-2, 2, 0.05)
            w.set_digits(2)
            w.set_value(l["weight"])
            w.set_tooltip_text("Вес LoRA")
            w.connect("value-changed", lambda b, l=l: l.update(weight=round(b.get_value(), 2)))
            x = Gtk.Button(icon_name="window-close-symbolic", css_classes=["flat"], tooltip_text="Убрать")
            x.connect("clicked", lambda _b, l=l: (self.loras.remove(l), self.render_loras()))
            row.append(w)
            row.append(x)
            self.lora_rows.append(row)
        self.lora_rows.set_visible(bool(self.loras))

    def refresh_lora_files(self):
        mode = self.mode_key()

        def done(r):
            if isinstance(r, Exception) or mode != self.mode_key():
                return
            self.lora_files = r["files"]
            self.lora_pick.set_model(Gtk.StringList.new(
                [f.removesuffix(".safetensors") for f in self.lora_files] or ["(папка пуста)"]))
        background(lambda: api(f"/api/loras?mode={mode}"), done)

    def add_lora(self):
        files = getattr(self, "lora_files", [])
        i = self.lora_pick.get_selected()
        if i >= len(files):
            return
        if all(l["file"] != files[i] for l in self.loras):
            self.loras.append({"file": files[i], "weight": 1.0})
            self.render_loras()

    def on_mode(self, *_):
        self.fill_sizes()
        self.steps.set_value(self.config[self.mode_key()]["steps"])
        self.set_mode_options()
        background(lambda: api("/api/preload", {"mode": self.mode_key()}))

    def on_flip(self, *_):
        i = self.size.get_selected()
        self.landscape = not self.landscape
        self.fill_sizes()
        self.size.set_selected(i)

    def prompt_text(self):
        b = self.prompt.get_buffer()
        return b.get_text(b.get_start_iter(), b.get_end_iter(), False)

    def form(self):
        return {"mode": self.mode_key(), "prompt": self.prompt_text(),
                "size": self.size_values[self.size.get_selected()], "steps": int(self.steps.get_value()),
                "count": int(self.count.get_value()), "seed": int(self.seed.get_value()),
                "upscale": self.upscale.get_active(), "init": self.init_name, "strength": round(self.strength.get_value(), 2),
                "sampler": self.picked(self.sampler, "samplers"), "scheduler": self.picked(self.scheduler, "schedulers"),
                "loras": [dict(l) for l in self.loras]}

    def save_form(self):
        try:
            with open(FORM, "w", encoding="utf-8") as f:
                json.dump(self.form(), f, ensure_ascii=False)
        except OSError:
            pass

    def load_form(self):
        try:
            with open(FORM, encoding="utf-8") as f:
                f = json.load(f)
        except (OSError, ValueError):
            f = {}
        keys = list(self.config)
        mode = f.get("mode") if f.get("mode") in self.config else keys[0]
        self.mode.handler_block_by_func(self.on_mode)
        self.mode.set_selected(keys.index(mode))
        self.mode.handler_unblock_by_func(self.on_mode)
        self.fill_sizes(f.get("size"))
        self.set_mode_options(f.get("sampler"), f.get("scheduler"), f.get("loras"))
        self.prompt.get_buffer().set_text(f.get("prompt", ""))
        self.steps.set_value(f.get("steps") or self.config[mode]["steps"])
        self.count.set_value(f.get("count") or 1)
        self.seed.set_value(f.get("seed", -1))
        self.upscale.set_active(f.get("upscale", True))
        self.strength.set_value(f.get("strength") or 0.75)
        init = f.get("init")
        self.set_init(init if init and os.path.exists(os.path.join(HERE, "data", "inputs", init)) else None)
        background(lambda: api("/api/preload", {"mode": mode}))

    def generate(self):
        body = self.form()
        if not body["prompt"].strip():
            return self.toast("Нужен промпт")
        self.save_form()

        def done(r):
            if isinstance(r, Exception) or r.get("error"):
                self.toast(str(r) if isinstance(r, Exception) else r["error"])
            else:
                self.toast(f"В очереди: {body['count']}" if body["count"] > 1 else "В очереди")
                self.poll()
        background(lambda: api("/api/generate", body), done)

    # ---------- img2img source ----------
    def set_init(self, name):
        self.init_name = name
        for w in (self.init_pic, self.init_clear, self.strength_box):
            w.set_visible(bool(name))
        self.init_btn.set_label("заменить…" if name else "выбрать файл…")
        if name:
            self.init_pic.set_filename(os.path.join(HERE, "data", "inputs", name))
        self.go.set_label("Сгенерировать из картинки" if name else "Сгенерировать")

    def init_uploaded(self, r):
        if isinstance(r, Exception) or r.get("error"):
            return self.toast("Не удалось: " + (str(r) if isinstance(r, Exception) else r["error"]))
        self.set_init(r["init"])
        self.toast("Исходная картинка задана")
        # the size list holds one aspect (2:3); turn it to the source's orientation, the rest is center-cropped
        tex = Gdk.Texture.new_from_filename(os.path.join(HERE, "data", "inputs", r["init"]))
        w, h = tex.get_width(), tex.get_height()
        if w != h and (w > h) != self.landscape:
            self.on_flip()

    def upload_init_bytes(self, data, mime):
        body = {"data": f"data:{mime};base64," + base64.b64encode(data).decode()}
        background(lambda: api("/api/init", body), self.init_uploaded)

    def upload_init_file(self, path):
        ext = os.path.splitext(path or "")[1].lower()
        mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}.get(ext)
        if not mime:
            return self.toast("Нужна картинка PNG, JPEG или WebP")
        with open(path, "rb") as f:
            self.upload_init_bytes(f.read(), mime)

    def on_pick_init(self, *_):
        filt = Gtk.FileFilter(name="Картинки")
        for m in ("image/png", "image/jpeg", "image/webp"):
            filt.add_mime_type(m)
        dlg = Gtk.FileDialog(title="Исходная картинка", default_filter=filt)

        def done(d, res):
            try:
                self.upload_init_file(d.open_finish(res).get_path())
            except GLib.Error:
                pass  # cancelled
        dlg.open(self, None, done)

    def paste_init(self):
        def done(cb, res):
            try:
                tex = cb.read_texture_finish(res)
            except GLib.Error:
                tex = None
            if tex:
                self.upload_init_bytes(tex.save_to_png_bytes().get_data(), "image/png")
            else:
                self.toast("В буфере нет картинки")
        self.get_clipboard().read_texture_async(None, done)

    # ---------- queue / progress ----------
    def poll(self):
        GLib.source_remove(self.timer) if getattr(self, "timer", None) else None
        self.timer = None
        background(lambda: api("/api/state", timeout=5), self.on_state)
        return False

    def on_state(self, s):
        if isinstance(s, Exception):
            self.loaded.set_label("сервер не отвечает")
            self.timer = GLib.timeout_add(3000, self.poll)
            return
        if self.is_suspended():
            self.timer = GLib.timeout_add(3000, self.poll)
            return
        cur = s["current"]
        # every redraw of this window makes the compositor share the GPU with the generation (~5% slower steps when
        # the cards were rebuilt every 0.7 s), so cards are built once per job and only their text and bar change
        ids = [j["id"] for j in s["queue"]]
        if ids != list(self.cards):
            child = self.queue.get_first_child()
            while child:
                self.queue.remove(child)
                child = self.queue.get_first_child()
            self.cards = {j["id"]: self.make_card(j) for j in s["queue"]}
        for j in s["queue"]:
            c = cur if cur.get("job") == j["id"] else None
            pct, line = 0, "в очереди" if j["status"] != "running" else ""
            if c:
                ph = c.get("phase")
                per = (0.1 + 0.8 * c["step"] / c["total"] if ph == "sampling" else 0.93 if ph == "decoding" else
                       0.97 if ph == "upscaling" else 0.05 if ph == "generating" else 0)
                pct = (c["index"] + per) / c["count"]
                line = (f"{c['index'] + 1}/{c['count']} · " if c["count"] > 1 else "") + PHASES.get(ph, ph or "") + \
                       (f" {c['step']}/{c['total']}" if ph == "sampling" else "") + f" · {round(c.get('elapsed', 0))} с"
            label, bar = self.cards[j["id"]]
            if label.get_label() != line:
                label.set_label(line)
            if abs(bar.get_fraction() - pct) > 1e-3:
                bar.set_fraction(pct)
        err = s.get("error") or {}
        if err.get("time") and err["time"] != self.err_time:
            if self.err_time is not None:
                self.error.set_label(f"Ошибка генерации (код {err['code']}):\n" + "\n".join(err["text"].split("\n")[-6:]))
                self.error.set_visible(True)
            self.err_time = err["time"]
        elif self.err_time is None:
            self.err_time = err.get("time", 0)
        text = (f"в памяти: {self.config[s['loaded']]['label']}" if s.get("loaded") else
                "загружается…" if cur.get("phase") == "loading" else "модель не загружена")
        if self.loaded.get_label() != text:
            self.loaded.set_label(text)
        self.unload.set_visible(bool(s.get("loaded")) and not s["queue"])
        if s["version"] != self.version:
            background(lambda: api("/api/history"), self.on_history)
        self.show_preview(cur)
        self.timer = GLib.timeout_add(700 if s["queue"] else 3000, self.poll)

    def make_card(self, j):
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2, css_classes=["card"])
        head = Gtk.Box(spacing=4, margin_start=8, margin_end=4, margin_top=4)
        head.append(Gtk.Label(label=f"{self.config.get(j['mode'], {}).get('label', 'Апскейл')} · {j.get('size') or ''}",
                              xalign=0, hexpand=True))
        x = Gtk.Button(icon_name="window-close-symbolic", css_classes=["flat"], tooltip_text="Отменить")
        x.connect("clicked", lambda _b, i=j["id"]: background(lambda: api("/api/cancel", {"id": i}), lambda _r: self.poll()))
        head.append(x)
        box.append(head)
        box.append(Gtk.Label(label=j["prompt"], xalign=0, ellipsize=Pango.EllipsizeMode.END, css_classes=["dim-label", "caption"],
                             margin_start=8))
        label = Gtk.Label(xalign=0, css_classes=["caption"], margin_start=8)
        bar = Gtk.ProgressBar(margin_start=8, margin_end=8, margin_bottom=6)
        box.append(label)
        box.append(bar)
        self.queue.append(box)
        return label, bar

    def show_preview(self, cur):
        key = f"{cur['job']}-{cur['index']}-{cur['step']}" if cur.get("step") else None
        if key == self.preview_key:
            return
        self.preview_key = key
        following = not self.shown or (self.items and self.shown == self.items[0]["id"])
        if not following:
            return
        if not key:
            return self.render_view()

        def done(png):
            if isinstance(png, bytes) and png and self.preview_key == key:
                self.picture.set_paintable(Gdk.Texture.new_from_bytes(GLib.Bytes.new(png)))
        background(lambda: api("/api/preview", raw=True, timeout=5), done)

    # ---------- history ----------
    def on_history(self, r):
        if isinstance(r, Exception):
            return
        prev_top = self.items[0]["id"] if self.items else None
        self.items, self.version = r["items"], r["version"]
        if self.items and prev_top and self.items[0]["id"] != prev_top:
            self.shown = self.items[0]["id"]
        if not any(e["id"] == self.shown for e in self.items):
            self.shown = self.items[0]["id"] if self.items else None
        self.render_strip()
        self.render_view()

    def visible(self):
        q = self.search.get_text().strip().lower()
        return [e for e in self.items if (not self.only_fav or e["fav"]) and (not q or q in e["prompt"].lower())]

    def set_filter(self, fav):
        self.only_fav = fav
        self.render_strip()

    def render_strip(self):
        child = self.strip.get_first_child()
        while child:
            self.strip.remove(child)
            child = self.strip.get_first_child()
        self.thumbs = {}
        items = self.visible()
        for e in items:
            path = os.path.join(IMAGES, e.get("orig_file") or e["file"])  # the base image is enough for a thumbnail
            tex = self.thumb_cache.get(path)
            if tex is None:
                try:
                    tex = self.thumb_cache[path] = thumb_texture(path)
                except GLib.Error:
                    continue
            over = Gtk.Overlay(child=Gtk.Picture(paintable=tex, can_shrink=False))
            if e["fav"]:
                over.add_overlay(Gtk.Label(label="★", css_classes=["star"], halign=Gtk.Align.END, valign=Gtk.Align.START))
            b = Gtk.Button(child=over, css_classes=["flat", "thumb"] + (["active"] if e["id"] == self.shown else []),
                           tooltip_text=e["prompt"][:300])
            b.connect("clicked", lambda _b, i=e["id"]: self.show(i))
            self.strip.append(b)
            self.thumbs[e["id"]] = b
        self.count_label.set_label(f"{len(items)} шт.")

    def current(self):
        return next((e for e in self.items if e["id"] == self.shown), None)

    def show(self, item_id):
        old = self.thumbs.get(self.shown)
        if old:
            old.remove_css_class("active")
        self.shown = item_id
        new = self.thumbs.get(item_id)
        if new:
            new.add_css_class("active")
        self.render_view()

    def render_view(self):
        e = self.current()
        if not e:
            self.picture.set_paintable(None)
            self.info.set_label("Здесь появится картинка")
            self.params.set_label("")
            return
        self.picture.set_filename(os.path.join(IMAGES, e["file"]))
        self.info.set_label(e["prompt"])
        size = f"{e['size']} → {e['out_size']}" if e.get("out_size", e["size"]) != e["size"] else e["size"]
        when = time.strftime("%d.%m.%y %H:%M", time.localtime(e["created"]))
        parts = [self.config.get(e["mode"], {}).get("label", e["mode"]), size.replace("x", "×"), f"{e['steps']} шаг.",
                 f"seed {e['seed']}", f"{e['duration']} с", when]
        if e.get("upscaled_from"):
            parts.append("апскейл")
        if e.get("init"):
            parts.append(f"img2img {e['strength']}")
        if e.get("sampler"):
            parts.append(f"{e['sampler']} + {e['scheduler']}")
        parts += [f"{l['file'].removesuffix('.safetensors')} {l['weight']:g}" for l in e.get("loras", [])]
        self.params.set_label(" · ".join(parts))
        self.fav_btn.set_label("★" if e["fav"] else "☆")

    def step(self, d):
        items = self.visible()
        ids = [e["id"] for e in items]
        if not ids:
            return
        i = ids.index(self.shown) if self.shown in ids else 0
        self.show(ids[min(len(ids) - 1, max(0, i + d))])
        b = self.thumbs.get(self.shown)
        if b:
            b.grab_focus()

    def refresh(self, _r=None):
        background(lambda: api("/api/history"), self.on_history)

    def toggle_fav(self):
        e = self.current()
        if e:
            background(lambda: api("/api/fav", {"ids": [e["id"]], "fav": not e["fav"]}), self.refresh)

    def trash_one(self):
        e = self.current()
        if not e:
            return
        ids = [x["id"] for x in self.visible()]
        i = ids.index(e["id"]) if e["id"] in ids else 0
        rest = [x for x in ids if x != e["id"]]
        self.shown = rest[max(0, i - 1)] if rest else None
        background(lambda: api("/api/trash", {"ids": [e["id"]]}), lambda r: (self.toast("В корзине"), self.refresh()))

    def reuse(self):
        e = self.current()
        if not e or e["mode"] not in self.config:
            return
        self.mode.handler_block_by_func(self.on_mode)
        self.mode.set_selected(list(self.config).index(e["mode"]))
        self.mode.handler_unblock_by_func(self.on_mode)
        self.fill_sizes(e["size"])
        # entries made before LoRAs were selectable used the mode's defaults
        self.set_mode_options(e.get("sampler"), e.get("scheduler"), e.get("loras"))
        self.prompt.get_buffer().set_text(e["prompt"])
        self.steps.set_value(e["steps"])
        self.seed.set_value(e["seed"])
        if e.get("init"):
            self.strength.set_value(e["strength"])
        self.set_init(e.get("init"))
        self.toast("Настройки подставлены")

    def reuse_seed(self):
        e = self.current()
        if e:
            self.seed.set_value(e["seed"])
            self.toast(f"Seed {e['seed']}")

    def copy_prompt(self):
        e = self.current()
        if e:
            self.get_clipboard().set_content(Gdk.ContentProvider.new_for_bytes(
                "text/plain;charset=utf-8", GLib.Bytes.new(e["prompt"].encode())))
            self.toast("Промпт скопирован")

    def to_init(self):
        e = self.current()
        if e:
            background(lambda: api("/api/init", {"image": e["file"]}), self.init_uploaded)

    def upscale_one(self):
        e = self.current()
        if e:
            background(lambda: api("/api/upscale", {"id": e["id"]}), lambda _r: (self.toast("Апскейл в очереди"), self.poll()))

    def on_key(self, _c, keyval, _code, state):
        ctrl = state & Gdk.ModifierType.CONTROL_MASK
        if ctrl and keyval in (Gdk.KEY_Return, Gdk.KEY_KP_Enter):
            self.generate()
            return True
        focus = self.get_focus()
        typing = isinstance(focus, (Gtk.TextView, Gtk.Text, Gtk.SpinButton))
        if ctrl and keyval in (Gdk.KEY_v, Gdk.KEY_Cyrillic_em) and not typing:
            self.paste_init()
            return True
        if typing or ctrl:
            return False
        if keyval == Gdk.KEY_Left:
            self.step(-1)
        elif keyval == Gdk.KEY_Right:
            self.step(1)
        elif keyval in (Gdk.KEY_f, Gdk.KEY_Cyrillic_a):
            self.toggle_fav()
        elif keyval == Gdk.KEY_Delete:
            self.trash_one()
        else:
            return False
        return True


class App(Adw.Application):
    def __init__(self):
        super().__init__(application_id="local.vega.imagegen")
        self.server = None

    def do_activate(self):
        if self.get_active_window():
            return self.get_active_window().present()
        css = Gtk.CssProvider()
        css.load_from_data(CSS, len(CSS))
        Gtk.StyleContext.add_provider_for_display(Gdk.Display.get_default(), css, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)
        config = self.connect_server()
        if config is None:
            dlg = Adw.AlertDialog(heading="Сервер не запустился", body="См. gui/data/server.log")
            win = Adw.ApplicationWindow(application=self)
            win.present()
            dlg.add_response("ok", "Закрыть")
            dlg.connect("response", lambda *_: self.quit())
            return dlg.present(win)
        Window(self, config).present()

    def connect_server(self):
        try:
            return api("/api/config", timeout=2)
        except OSError:
            pass
        log = open(os.path.join(HERE, "data", "server.log"), "ab")
        self.server = subprocess.Popen([os.path.join(HERE, "server.py")], cwd=HERE, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        for _ in range(50):
            time.sleep(0.1)
            try:
                return api("/api/config", timeout=2)
            except OSError:
                pass
        return None

    def do_shutdown(self):
        # stop the server only if this window started it: its exit unloads the model and frees GPU memory
        if self.server and self.server.poll() is None:
            self.server.terminate()
            try:
                self.server.wait(20)
            except subprocess.TimeoutExpired:
                self.server.kill()
        Adw.Application.do_shutdown(self)


if __name__ == "__main__":
    os.makedirs(IMAGES, exist_ok=True)
    App().run(None)
