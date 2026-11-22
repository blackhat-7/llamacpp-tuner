"""The engines lct serves, one module each.

Every backend module offers:
- serve(model, options, env): run the server in the foreground;
- setup(force, cmake_args, torch_index): install the engine, returning what to print;
- address(line): the URL in its ready log line, else "".

Shared types and helpers are in base.py. A new engine is a module here plus a
line in BACKENDS.
"""

from types import ModuleType

from llamacpp_tuner.backends import comfyui, jeeves, llama, loom, strata

BACKENDS: dict[str, ModuleType] = {
    "llama": llama,
    "jeeves": jeeves,
    "strata": strata,
    "loom": loom,
    "comfyui": comfyui,
}
