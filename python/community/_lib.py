"""ctypes access to the compiled Louvain local-moving kernel."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.path.join(ROOT, "dist", "libmojo-louvain.so")
I = ctypes.c_int64
F = ctypes.c_double


class BuildError(RuntimeError):
    pass


def _mojo_command() -> list[str]:
    override = os.environ.get("MOJO_LOUVAIN_MOJO")
    if override:
        return override.split()
    found = shutil.which("mojo")
    if found:
        return [found]
    pixi = shutil.which("pixi") or os.path.expanduser("~/.pixi/bin/pixi")
    if os.path.exists(pixi):
        return [pixi, "run", "--manifest-path", os.path.join(ROOT, "pixi.toml"), "mojo"]
    raise BuildError("mojo not found; set MOJO_LOUVAIN_MOJO=/path/to/mojo")


def build(force: bool = False) -> str:
    source = os.path.join(ROOT, "src", "capi.mojo")
    if not force and os.path.exists(LIB) and os.path.getmtime(LIB) >= os.path.getmtime(source):
        return LIB
    os.makedirs(os.path.dirname(LIB), exist_ok=True)
    proc = subprocess.run(
        _mojo_command() + ["build", "--emit", "shared-lib", source, "-o", LIB],
        capture_output=True, text=True, timeout=1800,
    )
    if proc.returncode or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_lib: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL(build())
        fn = _lib.mlj_one_level
        fn.argtypes = [I] * 12 + [F, F, I, I, F]
        fn.restype = F
    return _lib
