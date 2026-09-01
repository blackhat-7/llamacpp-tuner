---
name: tune-local-model
description: Given a specific local LLM or VLM, choose its best quantized artifact and runtime arguments, then prove them with staged benchmarks. Use for GGUF/quant selection, context sizing, offload, KV cache, batching, multimodal placement, tokens-per-second tuning, or OOM diagnosis. If the model itself is not chosen, use select-local-model first.
---

# Tune a Local Model

Optimize for required quality, speed, context, and concurrency—not maximum allocation or one short benchmark.

## 1. Fix the target

Collect the exact model, artifact family, accelerator/backend and dedicated or shared memory, topology, RAM, CPU, runtime/build, modalities, concurrency, and whether host-memory offload is acceptable. Define context capacity, typical initial prompt size, acceptable cold-prefill latency, and minimum decode speed at a realistic filled depth separately.

Inspect current memory use. Verify the active client/harness context and output limits, then align the server; do not infer runtime behavior from an inactive configuration file. Confirm that a pinned runtime build supports the exact architecture, quant format, template, reasoning/tool parser, and projector. Do not upgrade blindly; inspect that build's help because flags and defaults change.

## 2. Choose candidate artifacts

Inspect repository metadata and actual files. Never invent a model ID, filename, quant, projector, or download command. Verify provenance, license, quantizer, split files, and checksums when supplied.

Test a small ladder:

- a widely supported balanced quant such as Q4_K_M
- one higher-quality candidate such as Q5_K_M or Q6_K
- one smaller candidate only when memory, context, or speed requires it

Names and trade-offs vary by format and runtime. Importance-matrix quants can beat similarly sized plain quants when their provenance is sound. Q8 or floating-point weights often cost substantial memory for small practical gains. Compare every viable artifact on the same representative quality tasks and runtime settings; speed or perplexity alone cannot choose the winner.

Memory must cover total weights—not active MoE parameters—plus KV/recurrent state, runtime and batch buffers, projector, request slots, and other users. Include host RAM and memory-mapping behavior for CPU or hybrid offload. Prefer the highest-quality quant that passes peak-load tests with real headroom. A configuration that merely initializes does not fit.

If the requested downloader cannot select the winning artifact, use the model host's official CLI/API or add explicit support; never silently fetch another file. A structurally equivalent artifact may estimate fit and speed, but only the final artifact can validate quality.

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

Change one variable at a time. Record the exact build, artifact, command, mean/variance, and peak accelerator and host memory. Use the runtime's native benchmark; pass only arguments that benchmark tool supports. Estimate long-run duration from a short prefill first. Run expensive depths one at a time with a timeout and report each result before continuing.

### A. Prove single-slot memory fit

Start the real server at the target context with one slot. Measure idle memory, then peak memory during a representative long prompt and generation. Initialization alone is insufficient.

### B. Measure prefill and decode

For llama.cpp, choose a representative prompt length `P` within the target context and keep all benchmark-supported arguments fixed:

```bash
llama-bench -m MODEL -p P -n 0 -r 5 [BENCH_ARGS]
llama-bench -m MODEL -p 0 -n 256 -r 5 [BENCH_ARGS]
```

`pp` measures prompt processing; `tg` measures generation. Use a prompt length representative of the real harness, including system instructions and tool schemas. These microbenchmarks exclude tokenization, sampling, networking, templates, client overhead, and cache reuse.

### C. Measure late-context decode

Short-context `tg` is optimistic. Use `-d` to prefill tokens before decode, one depth per run:

```bash
llama-bench -m MODEL -p 0 -n 128 -d DEPTH -r 3 [BENCH_ARGS]
```

Start near a typical depth, then approach the realistic worst case only if needed. Keep `DEPTH` below approximately `target_context - n_gen`. Filling a large depth still takes time even when excluded from the reported generation timing.

### D. Tune KV precision

Compare supported KV types with everything else fixed. Lower precision can save substantial context memory but may alter long-context quality and sometimes speed. Measure both throughput and retrieval/summary quality; tokens per second cannot detect quality loss.

### E. Tune prefill independently

Use a bounded representative prompt to sweep paired logical/physical batch sizes. Run pairs separately because multi-value options may create a Cartesian product. Tune generation threads and batch/prefill threads independently when the runtime supports both. Larger batches can improve prefill without changing quality, but gains are not monotonic: stop at the plateau or first failure. Re-run the winner in the real server at target context because benchmark fit does not prove deployment fit.

### F. Test deployment features

Only after the baseline, compare full versus automatic/partial offload, CPU-MoE, projector placement, and required concurrency. Runtime slot/KV semantics differ: verify whether context is shared, divided, or replicated. Re-run peak-memory and throughput tests with the real server after each change.

Offload can affect prefill and decode differently. Sparse CPU-MoE may decode quickly while cold prefill remains limited by CPU or RAM bandwidth, so report both. For vision, compare image-ingestion latency and peak memory with the projector on host versus accelerator. Measure hybrid inference on the target machine; CPU, memory bandwidth, interconnect, architecture, and runtime all matter.

### G. Validate real workloads

Run representative chats, tools, structured output, long documents, and images. Record time to first token, prefill/decode speed at real depth, peak memory, correctness, tool-call validity, and failures. Compare a cold first request with append-only follow-ups to verify prompt-cache reuse; client-side message changes can defeat it. Exercise several multi-turn sessions or a short soak test to expose cache and OOM problems.

## 5. Select and report

Choose the simplest configuration that passes quality checks, meets speed at realistic depth, survives peak memory and concurrency, and retains safe system headroom. Keep only flags required for correctness, reproducibility, or a measured benefit.

Return:

1. exact download command when a download is needed
2. exact recommended serve command
3. a useful text-only or low-memory variant, if any
4. measured cold/cached prefill and decode speeds, depths, and peak accelerator/host memory
5. why the quant, KV, context, batches, and offload won
6. one fallback for OOM and one for insufficient speed

Never present estimates as measurements. Remove flags that merely repeat defaults unless pinning them is valuable for reproducibility.
