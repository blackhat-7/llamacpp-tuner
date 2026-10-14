# RTX 5090 + i9-14900KS, 2026-10-14, rerun

Same machine, build, model and config as `../5090-14900ks-20261014/` (see its `notes.md`), run again right after it to
show the run-to-run spread. Same `LD_LIBRARY_PATH=/run/opengl-driver/lib`. Strata server stopped. Nsight Systems was not
on PATH this time, so there is no `nsys_code` run.

Decode tok/s, first run / rerun: speed_code 116.2 / 122.5, speed_edit 81.2 / 85.2, speed_prose 138.8 / 138.1,
prof_code 117.7 / 118.9, prof_prose 125.8 / 132.3, noskip_code 126.0 / 125.4, noskip_prose 139.1 / 140.0.
Prefill tok/s: prompt_15k 4372 / 4580, prompt_35k 5051 / 5048.
