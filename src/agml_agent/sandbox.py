"""Sandbox for the `run_python` benchmark tool: the model's code runs on the item's image in a separate process.

What the code gets: `img`, a PIL image of the picture the model was shown (same size it saw; `imgs` lists all of
them when an item has several; also saved as input_file_0.png, input_file_1.png, ... in the working folder), `W`, `H`
(of `img`), `np`
(numpy), `Image` (PIL), and `show(image)` (or PIL's `img.show()`) to send an image back (a PIL image or a numpy array; up to MAX_IMAGES per
call). `print()` output comes back as text (up to MAX_OUTPUT characters).

What it cannot do (each call is a fresh `python -I` process in a temp folder holding only the image):
  - import anything outside ALLOWED (no torch/transformers/ultralytics etc., so no pretrained models; no os,
    subprocess, socket, urllib);
  - open files other than those in its temp folder (builtins.open, PIL.Image.open, cv2.imread are wrapped);
  - connect to the network (socket connections raise);
  - run longer than TIMEOUT seconds or use more than MEMORY_MB of memory.
The checks on imports and files apply to the model's own code; the libraries it calls import and read what they
need (OpenCV reads its own config files on import).
Not an OS-level jail (no containers on the cluster): the limits above are enforced inside Python, and code that
reaches for the usual ways around them (DENY: __import__, builtins, sys.modules, subprocess, os, eval/exec, ...) is
refused before it runs. That guards against a model's accidents, not against deliberate escape.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

TIMEOUT = 60  # running the code, incl. Python start and imports (shared filesystem on farm); not writing it
MEMORY_MB = 3072
MAX_OUTPUT = 2000
MAX_IMAGES = 2
ALLOWED = {"numpy", "PIL", "cv2", "skimage", "scipy", "math", "cmath", "statistics", "collections", "itertools",
           "functools", "operator", "json", "re", "random", "heapq", "bisect", "string", "textwrap", "dataclasses",
           "typing", "fractions", "decimal", "copy", "time"}

DENY = re.compile(r"__import__|__builtins__|__subclasses__|__globals__|__code__|\bbuiltins\b|sys\.modules|importlib|"
                  r"\bsubprocess\b|\bos\b|\bshutil\b|\bsocket\b|\beval\s*\(|\bexec\s*\(|\bcompile\s*\(|\bglobals\s*\(|"
                  r"\bgetattr\s*\(|\bsetattr\s*\(|\b_pil_open\b|\b_open\b|\b_import\b")

# Runs inside the child process: argv = [code file, JSON list of image files, out dir, ...]. Writes result.json there.
CHILD = r'''
import builtins, io, json, resource, sys, traceback
code_path, image_paths, out_dir, allowed, mem_mb, max_images = sys.argv[1], json.loads(sys.argv[2]), sys.argv[3], set(sys.argv[4].split(",")), int(sys.argv[5]), int(sys.argv[6])
resource.setrlimit(resource.RLIMIT_AS, (mem_mb << 20, mem_mb << 20))
import socket
def _no_net(*a, **k):
    raise PermissionError("network access is not allowed")
socket.socket.connect = socket.socket.connect_ex = _no_net
socket.create_connection = _no_net
import os
import numpy as np
from PIL import Image
MODEL = "<model code>"
def _from_model():
    f = sys._getframe(2)
    while f is not None:
        if f.f_code.co_filename == MODEL:
            return True
        if not f.f_code.co_filename.startswith("<frozen"):
            return False
        f = f.f_back
    return False
_import = builtins.__import__
def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0 and _from_model() and name.split(".")[0] not in allowed:
        raise ImportError(f"import of {name!r} is not allowed here; available: {', '.join(sorted(allowed))}")
    return _import(name, globals, locals, fromlist, level)
builtins.__import__ = _guarded_import
root = os.path.realpath(out_dir)
def _inside(p):
    try:
        return os.path.realpath(os.fspath(p)).startswith(root + os.sep)
    except TypeError:
        return True  # file-like objects
_open = builtins.open
def _guarded_open(file, *a, **k):
    if isinstance(file, (str, bytes, os.PathLike)) and _from_model() and not _inside(file):
        raise PermissionError("only files in the working folder can be opened")
    return _open(file, *a, **k)
builtins.open = _guarded_open
_pil_open = Image.open
def _guarded_pil_open(fp, *a, **k):
    if isinstance(fp, (str, bytes, os.PathLike)) and _from_model() and not _inside(fp):
        raise PermissionError("only files in the working folder can be opened")
    return _pil_open(fp, *a, **k)
Image.open = _guarded_pil_open
try:
    import cv2
    _imread = cv2.imread
    def _guarded_imread(p, *a, **k):
        if _from_model() and not _inside(p):
            raise PermissionError("only files in the working folder can be opened")
        return _imread(p, *a, **k)
    cv2.imread = _guarded_imread
except ImportError:
    pass
shown = []
def show(image, *_):
    """Send an image back to the model (PIL image or numpy array, HxW or HxWx3/4)."""
    if len(shown) >= max_images:
        print(f"[show: only {max_images} images per call are returned]")
        return
    if not isinstance(image, Image.Image):
        a = np.asarray(image)
        if a.dtype == bool:
            a = a.astype(np.uint8) * 255
        elif a.dtype != np.uint8:
            a = a.astype(float)
            lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
            a = ((a - lo) / (hi - lo) * 255 if hi > lo else a * 0).astype(np.uint8)
        image = Image.fromarray(a)
    path = os.path.join(out_dir, f"show_{len(shown)}.png")
    image.convert("RGB").save(path)
    shown.append(path)
imgs = [_pil_open(p).convert("RGB") for p in image_paths]
img = imgs[0]
Image.Image.show = lambda self, *a, **k: show(self)  # PIL's own show() also sends the image back
W, H = img.size
env = {"__name__": "__main__", "img": img, "imgs": imgs, "W": W, "H": H, "np": np, "Image": Image, "show": show}
buf, error = io.StringIO(), None
sys.stdout = sys.stderr = buf
try:
    exec(compile(_open(code_path).read(), MODEL, "exec"), env)
except MemoryError:
    error = "MemoryError: the code used more memory than allowed"
except BaseException as e:
    tb = [l for l in traceback.format_exception(type(e), e, e.__traceback__) if MODEL in l or not l.startswith("  File")]
    error = "".join(tb[-6:]).strip()
sys.stdout = sys.__stdout__
with _open(os.path.join(out_dir, "result.json"), "w") as f:
    json.dump({"output": buf.getvalue(), "error": error, "images": shown}, f)
'''


def run(code: str, images, out_dir: Path | None = None, timeout: int = TIMEOUT) -> tuple[str, list[str]]:
    """Run the model's `code` on `images` (a PIL image or a list of them, as the model saw them). Returns (text for the
    model, paths of the images it passed to show(), moved to `out_dir`, a new temp folder when not given)."""
    if m := DENY.search(code):
        return f"error: {m.group(0).strip('(').strip()!r} is not allowed in this tool.", []
    if out_dir is None:
        out_dir = Path(tempfile.mkdtemp(prefix="agml_code_images_"))
    with tempfile.TemporaryDirectory(prefix="agml_code_") as tmp:
        work = Path(tmp)
        paths = []
        for i, im in enumerate(images if isinstance(images, (list, tuple)) else [images]):
            paths.append(str(work / f"input_file_{i}.png"))  # the names gemma expects (2026-11-09 trial: every call opened input_file_0.png)
            im.convert("RGB").save(paths[-1])
        (work / "code.py").write_text(code)
        env = {"PATH": os.environ.get("PATH", ""), "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
               "MKL_NUM_THREADS": "1", "HOME": tmp}
        try:
            subprocess.run([sys.executable, "-I", "-c", CHILD, str(work / "code.py"), json.dumps(paths), str(work),
                            ",".join(sorted(ALLOWED)), str(MEMORY_MB), str(MAX_IMAGES)],
                           cwd=work, env=env, capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return f"error: the code ran longer than {timeout} s and was stopped.", []
        try:
            res = json.loads((work / "result.json").read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return "error: the code crashed the Python process (for example by using too much memory).", []
        images = []
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, p in enumerate(res["images"]):
            dest = out_dir / f"{next(_counter)}_{i}.png"
            Path(p).replace(dest)
            images.append(str(dest))
        text = res["output"][:MAX_OUTPUT] + ("\n[output cut]" if len(res["output"]) > MAX_OUTPUT else "")
        if res["error"]:
            text = (text + "\n" if text else "") + "error: " + res["error"]
        if res["images"]:
            text = (text + "\n" if text else "") + f"[{len(res['images'])} image(s) from show() follow]"
        return text or "(no output; use print() or show() to see results)", images


def _counter_gen():
    n = 0
    while True:
        n += 1
        yield f"{os.getpid()}_{n}"


_counter = _counter_gen()
