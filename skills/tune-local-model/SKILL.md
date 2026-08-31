---
name: tune-local-model
description: Given a specific local LLM or VLM, choose its best quantized artifact and runtime arguments, then prove them with staged benchmarks. Use for GGUF/quant selection, context sizing, offload, KV cache, batching, multimodal placement, tokens-per-second tuning, or OOM diagnosis. If the model itself is not chosen, use select-local-model first.
---

# Tune a Local Model

Optimize for required quality, speed, context, and concurrency—not maximum allocation or one short benchmark.

## 1. Fix the target

Collect the exact model, artifact family, accelerator/backend and dedicated or shared memory, topology, RAM, CPU, runtime/build, target context, minimum generation speed, concurrency, modalities, and whether host-memory offload is acceptable.

Inspect current memory use. Confirm that a pinned runtime build supports the exact architecture, quant format, template, reasoning/tool parser, and projector. Do not upgrade blindly; recent builds can regress. Inspect that build's help because flags and defaults change.

## 2. Choose candidate artifacts

Inspect repository metadata and actual files. Never invent a model ID, filename, quant, projector, or download command. Verify provenance, license, quantizer, split files, and checksums when supplied.

Test a small ladder:

- a widely supported balanced quant such as Q4_K_M
- one higher-quality candidate such as Q5_K_M or Q6_K
- one smaller candidate only when memory, context, or speed requires it

Names and trade-offs vary by format and runtime. Importance-matrix quants can beat similarly sized plain quants when their provenance is sound. Q8 or floating-point weights often cost substantial memory for small practical gains; measure them before choosing.

Memory must cover total weights—not active MoE parameters—plus KV/recurrent state, runtime and batch buffers, projector, request slots, and other users. Prefer the highest-quality quant that passes peak-load tests with real headroom. A configuration that merely initializes does not fit.

If the requested downloader cannot select the winning artifact, use the model host's official CLI/API or add explicit support; never silently fetch another file.

## 3. Establish a boring baseline

Pin settings needed for a reproducible baseline:

- one request slot and the required total context
- documented/default offload and projector placement
- Flash Attention when supported
- runtime-default batches
- F16/BF16 KV as the quality baseline when it fits; test Q8 and lower precision as memory-saving candidates
- the embedded/documented chat template

Avoid speculative decoding, custom tensor splits, forced locking, or many overrides until this works. Keep sampling fixed during end-to-end comparisons because it changes output length and latency, even though raw model-evaluation benchmarks exclude it. Do not load a projector for text-only use.

## 4. Benchmark in stages

Change one variable at a time. Record the exact build, artifact, command, mean/variance, and peak accelerator and host memory. Use the runtime's native benchmark; pass only arguments that benchmark tool supports.

### A. Prove single-slot memory fit

Start the real server at the target context with one slot. Measure idle memory, then peak memory during a representative long prompt and generation. Initialization alone is insufficient.

### B. Measure prefill and decode

For llama.cpp, choose a representative prompt length `P` within the target context and keep all benchmark-supported arguments fixed:

```bash
llama-bench -m MODEL -p P -n 0 -r 5 [BENCH_ARGS]
llama-bench -m MODEL -p 0 -n 256 -r 5 [BENCH_ARGS]
```

`pp` measures prompt processing; `tg` measures generation. Documents and tool results can be prefill-heavy. These microbenchmarks exclude tokenization, sampling, networking, templates, and client overhead.

### C. Measure late-context decode

Short-context `tg` is optimistic. Use `-d` to prefill untimed tokens before decode:

```bash
llama-bench -m MODEL -p 0 -n 128 -d 0,D1,D2,D3 -r 5 [BENCH_ARGS]
```

Choose depths near 25%, 50%, 75%, and the realistic worst case, with each depth no greater than approximately `target_context - n_gen`. `llama-bench` allocates for prompt + generation + depth. The speed requirement must hold at realistic depth.

### D. Tune KV precision

Compare supported KV types with everything else fixed. Lower precision can save substantial context memory but may alter long-context quality and sometimes speed. Measure both throughput and retrieval/summary quality; tokens per second cannot detect quality loss.

### E. Tune prompt batches

Test a few paired logical/physical batch sizes from conservative to runtime default. Run pairs separately: multiple multi-value options may create a Cartesian product. Larger batches mainly improve prefill, consume more peak memory, and usually do little for single-stream decode. Stop when gains flatten.

### F. Test deployment features

Only after the baseline, compare full versus automatic/partial offload, CPU-MoE, projector placement, and required concurrency. Runtime slot/KV semantics differ: verify whether context is shared, divided, or replicated. Re-run peak-memory and throughput tests with the real server after each change.

For vision, compare image-ingestion latency and peak memory with the projector on host versus accelerator. For hybrid inference, measure on the target machine; CPU, memory bandwidth, interconnect, architecture, and runtime all matter.

### G. Validate real workloads

Run representative chats, tools, structured output, long documents, and images. Record time to first token, prefill/decode speed at real depth, peak memory, correctness, tool-call validity, and failures. Exercise several multi-turn sessions or a short soak test to expose cache and OOM problems.

## 5. Select and report

Choose the simplest configuration that passes quality checks, meets speed at realistic depth, survives peak memory and concurrency, and retains safe system headroom. Keep only flags required for correctness, reproducibility, or a measured benefit.

Return:

1. exact download command when a download is needed
2. exact recommended serve command
3. a useful text-only or low-memory variant, if any
4. measured prefill/decode speeds, depths, and peak memory
5. why the quant, KV, context, batches, and offload won
6. one fallback for OOM and one for insufficient speed

Never present estimates as measurements. Remove flags that merely repeat defaults unless pinning them is valuable for reproducibility.
