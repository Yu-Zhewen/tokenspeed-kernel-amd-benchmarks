"""Bracket TokenSpeed's /start_profile and /stop_profile with roctx.

Imported from a .pth file in a disposable container and inert unless
EMULATE_PROFILER_HOOK=roctx. Each profile window becomes a roctx range named
after its output directory, with roctxProfilerResume/Pause around it, for
``rocprofv3 --selected-regions --marker-trace``.
"""

import importlib.abc
import importlib.util
import os
import sys

_TARGET = "tokenspeed.runtime.engine.request_handler"
PID_FILE = "/tmp/emulate-scheduler.pid"
_ROCTX_CANDIDATES = (
    "/opt/venv/lib/python3.12/site-packages/_rocm_sdk_core/lib/librocprofiler-sdk-roctx.so.1",
    "/opt/rocm/lib/librocprofiler-sdk-roctx.so.1",
)


def _roctx_lib():
    paths = [os.environ.get("EMULATE_ROCTX_LIB", ""), *_ROCTX_CANDIDATES]
    for path in paths:
        if path and os.path.exists(path):
            return path
    raise FileNotFoundError(f"no roctx library among {paths}")


def _patch(cls):
    import ctypes

    lib = ctypes.CDLL(_roctx_lib())
    lib.roctxProfilerResume.argtypes = [ctypes.c_uint64]
    lib.roctxProfilerPause.argtypes = [ctypes.c_uint64]
    lib.roctxRangePushA.argtypes = [ctypes.c_char_p]
    start, stop = cls.start_profile, cls.stop_profile

    def start_profile(self, *args, **kwargs):
        result = start(self, *args, **kwargs)
        with open(PID_FILE, "w") as f:
            f.write(str(os.getpid()))
        lib.roctxProfilerResume(0)
        lib.roctxRangePushA(str(self.profiler_output_dir).encode())
        return result

    def stop_profile(self, *args, **kwargs):
        if self.profile_in_progress:
            lib.roctxRangePop()
            lib.roctxProfilerPause(0)
        return stop(self, *args, **kwargs)

    cls.start_profile = start_profile
    cls.stop_profile = stop_profile
    print(f"emulate profiler hook installed in pid {os.getpid()}", file=sys.stderr, flush=True)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname != _TARGET:
            return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(fullname)
        if spec is None or spec.loader is None:
            return spec
        exec_module = spec.loader.exec_module

        def patched_exec(module):
            exec_module(module)
            _patch(module.RequestHandler)

        spec.loader.exec_module = patched_exec
        return spec


if os.environ.get("EMULATE_PROFILER_HOOK") == "roctx":
    sys.meta_path.insert(0, _Finder())
