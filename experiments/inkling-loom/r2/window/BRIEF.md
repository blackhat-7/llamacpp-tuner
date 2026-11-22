# Shared brief: speeding up Inkling-2-Mini on Loom (7900 XTX)

You are one of five agents speeding up Inkling-2-Mini (a 26B-A3B looped MoE) served by AMD's Loom engine on a
Radeon RX 7900 XTX (gfx1100, 24 GB, ~960 GB/s, 96 MB infinity cache, Linux). The user wants maximum decode speed
(code, prose, code-edit) and prefill speed with output quality unchanged. You own one workstream (your prompt).
Others own other kernels in parallel: stay inside your scope.

## Read as needed
- Worktree /home/illusion/Documents/projects/llamacpp-tuner-inkling (branch feat/inkling-speed):
  `docs/inkling-loom-research.md` (model facts, what is done, measurements), `engines/loom/` (README, build.sh,
  tools/transform.py, tools/dsp_window.py, kernels/: how the served package is built from upstream),
  `experiments/inkling-loom/` (README: kernel generators, microbenchmark harness gen.py/run.sh/run2.sh,
  e2e tool session/session.py).
- Loom's knowledge base: /home/illusion/Documents/projects/llamacpp-tuner/tmp/loom/loom-skills/skills/
  (loom-kb/topic/{decode,prefill,speculative,spec-tuning,roofline,optimize-loop,kernels-ir,long-context,benchmark}.md;
  skills loom/, loom-optimize/, loom-roofline/ for the .loom language and the optimize loop).
- Tools: /home/illusion/Documents/projects/llamacpp-tuner/tmp/loom/venv/bin/loom-{serve,benchmark,compile,
  compile-report,check,test}; each has `--agents_md` and `--help`.
- Model config: $LCT_HOME/loom/kv8w4096/llm_config.json (hidden 2304, 48 layer visits = 24 physical layers with
  layers 6-17 run 3 times, 24 q heads / 8 KV heads / head_dim 128, 8 global-attention visits, 40 SWA visits with
  a 1,024 window, 128 experts top-8, expert ffn 1152, vocab 200,058, NVFP4 experts + 4 Q8_0 expert layers, Q8_0
  attention, DSpark/DFlash drafter with block 8).

## Paths and rules
- LCT_HOME=/home/illusion/Documents/projects/llamacpp-tuner/tmp. Served package: $LCT_HOME/loom/kv8w4096
  (program/llm.loom ~80K lines; program/control.loom drives drafting), preset `xtx-kv8-196k` (196,608 context,
  prefill_capacity 512, draft_depth 1, prefix cache, max_tokens 32768). `$LCT_HOME/loom/kv8w4096-d` is the same
  package with extra presets d2, d3, d4, d5, d7 (draft_depth N).
- NEVER modify kv8w4096, kv8w4096-d or any existing package in tmp/loom. Make your own variants as hard-link
  copies: `cp -al $LCT_HOME/loom/kv8w4096 $LCT_HOME/loom/<ws>-<name>`, then `rm` each file you will change before
  writing it (otherwise you write through the hard link into the original).
- Scratch: $LCT_HOME/loom/<ws>/ and $LCT_HOME/experiments/inkling-loom/<ws>/.
- GPU: every GPU command (loom-benchmark, loom-serve, session.py) runs under `flock /tmp/xtx-gpu.lock ...`.
  Others share the GPU: hold the lock for one run at a time, never while thinking or editing. One loom-serve at a
  time fits in VRAM. Keep each run to a few minutes. After runs check
  `journalctl -k --since "-10min" | grep -i amdgpu` for faults or timeouts; stop and report if any.
- Never `pkill -f` (it kills your own shell); kill by PID. No sudo. Do not touch the main checkout
  /home/illusion/Documents/projects/llamacpp-tuner (another session owns its working tree; reading its tmp/ is
  fine). Do not commit or push. Do not edit engines/loom/build.sh, presets.json or docs: I integrate.

## Measuring
- End to end, from the worktree:
  `cd /home/illusion/Documents/projects/llamacpp-tuner-inkling && export LCT_HOME=/home/illusion/Documents/projects/llamacpp-tuner/tmp &&
  flock /tmp/xtx-gpu.lock $LCT_HOME/loom/venv/bin/python -I experiments/inkling-loom/session/session.py --model PKG
  --preset P --jobs JOBS --out OUT.jsonl --label L [--serve-arg=...] [--port N]`.
  Jobs: $LCT_HOME/experiments/inkling-loom/opt/jobs_base.json (warm, short_code 1500 tokens, short_prose 700,
  edit 4000). Make your own job files for 64K/195K ($LCT_HOME/model-hunt/prompt_inkling_{64000,100000,195000}.txt).
  Each result line has decode_tps, ttft_s and metrics_delta with `draft_proposed_tokens` and
  `draft_accepted_inputs` (steps = proposed / depth; tokens per step = (accepted + steps) / steps).
- Per-kernel device time in a live server: add
  `--serve-arg=--device_profiling_mode=dispatch-events,executable-metadata
  --serve-arg=--device_profiling_flush_interval_ms=5 --serve-arg=--print_device_statistics`; the server log
  $LCT_HOME/experiments/inkling-loom/serve-kv8-<label>.log ends with `dispatch NAME count=.. total=..` lines.
  Profiling slows decode ~11%.
- Kernel microbenchmarks: loom-benchmark (experiments/inkling-loom/gen.py + run.sh is a working harness).
- Noise: e2e t/s moves 2-3% run to run. Repeat (or interleave A/B) before claiming a gain under 5%.

## Baseline (kv8w4096, xtx-kv8-196k, depth 1), measured 2026-11-22 06:17
- Short (1.3K prompt): code 139.6, prose 119.6, edit 131.2 t/s. 64K: 114 / 104; 195K: 90 / 75 (code / prose).
  Drafter off ~110 t/s short. Prefill: fresh 195K read 2.3 min (~1,420 t/s).
- Step costs (from draft metrics): span 2 (depth 1) ~13.2 ms; span 3 and 4 (depth 2, 3) ~17.7 ms; span 5
  (depth 4) ~26.2 ms. Depth 3: code 167.9, prose 117.6, edit 147.4. Depth 4: 125 / 77 / 109.
- Short profile, depth 1, per verify step (~12.9 ms device; $LCT_HOME/experiments/inkling-loom/opt/prof-short.txt:
  name, calls, total us over 922 steps): MoE gate/up nvfp4 2.18 ms; q8_0 matvec k2304 t4 (5 per layer visit)
  1.58 (~614 GB/s); MoE down nvfp4 1.12; SWA span attention 0.75 (18.6 us a call, ~215 GB/s); LM head t4 0.60;
  g_quant_q8 0.58 (215 calls); O-proj k3072 0.58; attn_combine 0.55 (11.4 us a call); global attention v2 0.30
  (37 us a call at ~2.5K context); drafter ~0.85; ~2.8 ms of 2-4 us Inkling-specific small kernels.

## Correctness bar
- A kernel change matches the original kernel's outputs on the same inputs within float rounding: a
  loom-benchmark/loom-test check like gen.py's test.loom, with a negative control that fails.
- Greedy e2e text stays coherent. Exact match is not required; report the first differing token
  (session/compare.py). No change to sampling, weights or quantization unless your prompt says so.

## Deliverable (your final message: short, plain words)
- What you changed and why; e2e numbers vs baseline measured in the same conditions; correctness evidence.
- Files under $LCT_HOME/loom/<ws>/deliver/: a deterministic tool `<name>.py` that edits a package's
  program/*.loom in place (applied after engines/loom/build.sh's existing steps, i.e. to a copy of kv8w4096),
  plus kernel files and any preset change. List the kernels and launch sites you touched so I can merge with the
  other workstreams.
- Negative results too: what you tried and the numbers.

## GPU hold (added 13:12)
The user restarted their own inkling server on the XTX at 12:05 and is using it (claude-local). Never stop it and
never touch its process. While `/tmp/xtx-gpu.hold` exists, the GPU is theirs: run no GPU command at all. Wrap every
GPU command in `/home/illusion/Documents/projects/llamacpp-tuner/tmp/loom/notes/gpu-run CMD ARGS...` instead of
`flock /tmp/xtx-gpu.lock CMD...`: it waits while the hold exists, then takes the lock.
