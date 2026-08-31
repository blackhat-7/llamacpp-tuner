# Development Guide

## General Principles
- Prefer the simplest correct solution. If two approaches achieve the same result, choose the simpler one.
- Keep code short, clear, readable, and maintainable.
- Avoid unnecessary abstractions, indirection, dependencies, configuration, or defensive complexity.
- Do not add code, tests, documentation, comments, or files unless they provide meaningful value.
- Follow existing repository patterns and conventions before introducing new ones.
- Make focused changes. Avoid unrelated refactors or cleanup.
- Optimize for correctness and clarity first, cleverness last.

## Scope

`lct` does only four things:

1. find or build `llama-server`
2. download an exact GGUF quantization and optional projector
3. start `llama-server` with explicit pass-through arguments
4. list downloaded GGUF files

Model recommendations and tuning methodology belong in `skills/`, not Python heuristics.

## Rules

- Keep quantization names open-ended strings. Never add a fixed quant allowlist.
- Verify exact Hugging Face filenames; never silently fall back to another quant.
- Prefer `huggingface_hub` and llama.cpp behavior over custom download, cache, or benchmark implementations.
- Keep runtime flags explicit. Do not guess hardware-specific “optimal” arguments.
- Library code raises exceptions; Click commands turn them into actionable user errors.
- Public functions have type hints.
- Keep user-writable paths in `cache.py`; package directories may be read-only.
- Do not add dependencies unless the standard library or an existing dependency cannot do the job.

## Verification

```bash
uv run pytest
uv run ruff check .
uv run pyright
```
