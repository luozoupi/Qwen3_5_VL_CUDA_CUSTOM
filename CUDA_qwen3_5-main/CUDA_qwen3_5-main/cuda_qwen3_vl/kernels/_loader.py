"""Shared JIT loader for CUDA extensions.

All ops are registered in the `cuda_qwen3_vl` torch.ops namespace via TORCH_LIBRARY_FRAGMENT,
so loading a given op makes it callable as torch.ops.cuda_qwen3_vl.<op_name>.

Fallback tracking: any time a CUDA kernel fails to build/load/run and we fall back to
PyTorch, the event is recorded in `FALLBACK_LOG` and emitted as a clearly-tagged warning
to stderr. Call `summarize_fallbacks()` to print a final report.
"""
from __future__ import annotations

import ctypes
import os
import sys
import warnings
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

_PKG_ROOT = Path(__file__).resolve().parents[1]
_CSRC = _PKG_ROOT / "csrc"
_CACHE_DIR = _PKG_ROOT.parents[0] / ".cache" / "torch_extensions"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

_LOADED: dict[str, bool] = {}
_LOAD_ERROR: dict[str, Exception] = {}

# Registry of every fallback event. Each entry: (op_name, reason, exception_repr)
FALLBACK_LOG: list[tuple[str, str, str]] = []
_FALLBACK_WARNED: dict[str, bool] = {}

_FALLBACK_TAG = "[CUDA-QWEN3-VL-FALLBACK]"


_REQUIRED_CXXABI = b"CXXABI_1.3.13"  # System gcc-11 / conda py310_2 libstdc++ must provide this


def _find_newer_libstdcxx_dir() -> str | None:
    """Find a lib dir containing libstdc++.so.6 with the ABI our compiled kernels need.

    We prefer system libstdc++ (`/usr/lib/x86_64-linux-gnu/libstdc++.so.6`) because it is
    ABI-compatible with the py310_2 env's glibc. py310's libstdc++ works too but mixes
    compiler toolchains; system's is the safer default.
    """
    candidates = [
        "/usr/lib/x86_64-linux-gnu",
        "/home/luo00466/miniconda3/envs/py310/lib",
        "/opt/miniconda3/pkgs/libstdcxx-15.2.0-h39759b7_7/lib",
    ]
    for d in candidates:
        sofile = os.path.join(d, "libstdc++.so.6")
        if not os.path.exists(sofile):
            continue
        try:
            with open(sofile, "rb") as f:
                if _REQUIRED_CXXABI in f.read():
                    return d
        except Exception:
            continue
    return None


_NEWER_LIBSTDCXX_DIR = _find_newer_libstdcxx_dir()


def _system_gcc11() -> tuple[str, str] | None:
    """Return (CC, CXX) paths to system gcc-11/g++-11 if available, else None.

    Using system's older compiler avoids needing the newer libstdc++ that ships with
    py310's GCC 14 toolchain — keeping everything ABI-compatible with py310_2.
    """
    cc = "/usr/bin/gcc-11"
    cxx = "/usr/bin/g++-11"
    if os.path.exists(cc) and os.path.exists(cxx):
        return cc, cxx
    return None


def _check_libstdcxx_ok() -> tuple[bool, str | None]:
    """Return (ok, needed_preload_path). ok=True means current libstdc++ has all needed ABI.

    Using system gcc-11, our compiled kernels need CXXABI up to 1.3.13. py310_2's own
    libstdc++ only reaches 1.3.9, so we'll preload the system one (1.3.13).
    """
    try:
        with open("/proc/self/maps", "r") as f:
            maps = f.read()
    except Exception:
        return True, None

    loaded_libstdcxx = None
    for line in maps.splitlines():
        if "libstdc++.so" in line:
            parts = line.strip().split()
            if parts:
                loaded_libstdcxx = parts[-1]
                break

    if loaded_libstdcxx is None:
        return True, None

    try:
        with open(loaded_libstdcxx, "rb") as f:
            if _REQUIRED_CXXABI in f.read():
                return True, None
    except Exception:
        return True, None

    if _NEWER_LIBSTDCXX_DIR is not None:
        candidate = os.path.join(_NEWER_LIBSTDCXX_DIR, "libstdc++.so.6")
        return False, candidate
    return False, None


def _maybe_reexec_with_preload() -> None:
    """If libstdc++ is too old and we have a newer one available, re-exec with LD_PRELOAD.

    Controlled by env var CUDA_QWEN3_VL_AUTO_PRELOAD (default: on). Set =0 to disable.
    """
    if os.environ.get("CUDA_QWEN3_VL_AUTO_PRELOAD") == "0":
        return
    if os.environ.get("_CUDA_QWEN3_VL_PRELOAD_DONE") == "1":
        return  # Already re-exec'd once; don't loop
    ok, preload_path = _check_libstdcxx_ok()
    if ok or preload_path is None:
        return
    # Read the FULL original cmdline from /proc so we preserve -c, -m, etc.
    try:
        with open("/proc/self/cmdline", "rb") as f:
            raw_argv = f.read().split(b"\x00")
        full_argv = [a.decode("utf-8", "replace") for a in raw_argv if a]
    except Exception:
        full_argv = None

    if not full_argv:
        # Can't reliably re-exec; emit a clear warning and let fallbacks take over
        sys.stderr.write(
            f"{_FALLBACK_TAG} libstdc++ too old (needs CXXABI_1.3.15); "
            f"please run with LD_PRELOAD={preload_path}\n"
        )
        sys.stderr.flush()
        return

    existing = os.environ.get("LD_PRELOAD", "")
    new_preload = preload_path if not existing else f"{preload_path}:{existing}"
    new_env = os.environ.copy()
    new_env["LD_PRELOAD"] = new_preload
    new_env["_CUDA_QWEN3_VL_PRELOAD_DONE"] = "1"
    sys.stderr.write(
        f"{_FALLBACK_TAG} re-exec with LD_PRELOAD={preload_path} to pick up CXXABI_1.3.15 "
        f"(set CUDA_QWEN3_VL_AUTO_PRELOAD=0 to disable)\n"
    )
    sys.stderr.flush()
    os.execve(full_argv[0], full_argv, new_env)


_maybe_reexec_with_preload()


def _record_fallback(op_name: str, reason: str, exc: Exception | None = None) -> None:
    """Register a fallback event. Emits a single warning per op_name, but logs every event."""
    entry = (op_name, reason, f"{type(exc).__name__}: {exc}" if exc else "")
    FALLBACK_LOG.append(entry)
    if not _FALLBACK_WARNED.get(op_name):
        _FALLBACK_WARNED[op_name] = True
        msg = f"{_FALLBACK_TAG} op={op_name} reason={reason}"
        if exc is not None:
            msg += f" exc={type(exc).__name__}: {exc}"
        print(msg, file=sys.stderr, flush=True)
        warnings.warn(msg, RuntimeWarning, stacklevel=3)


def summarize_fallbacks() -> None:
    """Print a summary of every op that ever fell back during this process."""
    if not FALLBACK_LOG:
        print(f"{_FALLBACK_TAG} no fallbacks recorded — all CUDA kernels executed.", file=sys.stderr)
        return
    print(f"{_FALLBACK_TAG} === FALLBACK SUMMARY ({len(FALLBACK_LOG)} events) ===", file=sys.stderr)
    by_op: dict[str, list[tuple[str, str]]] = {}
    for op, reason, exc in FALLBACK_LOG:
        by_op.setdefault(op, []).append((reason, exc))
    for op, events in by_op.items():
        print(f"  {op}: {len(events)} fallback(s)", file=sys.stderr)
        reasons = set(ev[0] for ev in events)
        for reason in reasons:
            print(f"    - {reason}", file=sys.stderr)


def load_op(name: str, sources: list[str] | None = None, verbose: bool = False) -> bool:
    """Load a CUDA extension. Returns True on success.

    `sources` is a list of paths relative to the csrc directory. If None, defaults to
    [name + "_op.cpp", name + "_op.cu"].
    """
    if _LOADED.get(name):
        return True
    if name in _LOAD_ERROR:
        return False

    if sources is None:
        sources = [f"{name}_op.cpp", f"{name}_op.cu"]
    source_paths = [str(_CSRC / s) for s in sources]

    ext_name = f"cuda_qwen3_vl_{name}"
    build_dir = _CACHE_DIR / ext_name
    build_dir.mkdir(parents=True, exist_ok=True)

    extra_ldflags: list[str] = []
    if _NEWER_LIBSTDCXX_DIR is not None:
        # Bake RPATH into .so so it prefers the system libstdc++ over py310_2's older one
        extra_ldflags.append(f"-Wl,-rpath,{_NEWER_LIBSTDCXX_DIR}")

    # Prefer system gcc-11 over whatever is first on PATH — py310's GCC 14 would need
    # CXXABI_1.3.15 which is not shipped with py310_2's libstdc++.
    gcc = _system_gcc11()
    saved_env: dict[str, str | None] = {}
    if gcc is not None:
        for key, val in (("CC", gcc[0]), ("CXX", gcc[1])):
            saved_env[key] = os.environ.get(key)
            os.environ[key] = val

    try:
        load(
            name=ext_name,
            sources=source_paths,
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "--use_fast_math", "-std=c++17",
                               "-ccbin", gcc[0]] if gcc is not None else ["-O3", "--use_fast_math", "-std=c++17"],
            extra_ldflags=extra_ldflags,
            build_directory=str(build_dir),
            with_cuda=True,
            is_python_module=False,
            verbose=verbose,
        )
        _LOADED[name] = True
        return True
    except Exception as exc:  # pragma: no cover - build error path
        _LOAD_ERROR[name] = exc
        _record_fallback(name, "build_or_load_failed", exc)
        return False
    finally:
        for key, val in saved_env.items():
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val


def get_load_error(name: str) -> Exception | None:
    return _LOAD_ERROR.get(name)


def strict_mode() -> bool:
    return os.environ.get("CUDA_QWEN3_VL_STRICT") == "1"


def maybe_strict_raise(op_name: str, exc: Exception) -> None:
    """Raise if strict mode, otherwise warn once and record fallback."""
    if strict_mode():
        raise RuntimeError(
            f"CUDA_QWEN3_VL_STRICT=1 but {op_name} kernel failed: {type(exc).__name__}: {exc}"
        ) from exc
    _record_fallback(op_name, "runtime_exception", exc)


def record_fallback(op_name: str, reason: str = "not_implemented") -> None:
    """Public helper: call when a kernel is intentionally not-yet-implemented and torch path is taken."""
    _record_fallback(op_name, reason)
