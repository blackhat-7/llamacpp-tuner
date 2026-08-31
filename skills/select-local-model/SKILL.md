---
name: select-local-model
description: Compare and choose the best current local LLM or VLM for a user's workloads and system before a model is fixed. Use for model recommendations or comparisons involving chat, research, documents, coding, agents, tools, vision, multilingual, privacy, or uncensored use. Once a specific model is chosen, use tune-local-model for its artifact and runtime settings.
---

# Select a Local Model

Give an evidence-backed shortlist, not a model dump. Releases change quickly, so search current sources instead of relying on memory.

## 1. Define “best”

Collect only missing information:

- weighted use cases and examples of real tasks
- native modalities versus acceptable preprocessing such as PDF extraction, OCR, or video frame sampling
- tool/function calling, structured output, and agent-loop needs
- accelerator/backend and dedicated or shared memory, multi-device topology, RAM, CPU, and OS
- runtime or serving format, if fixed
- minimum generation speed, latency, context, and concurrency
- privacy, license, language, censorship, and power constraints

Separate hard requirements from preferences. “Best” means best for these constraints, not the highest headline benchmark.

## 2. Build a small current candidate set

Search official model cards, release notes, runtime support, and model-host metadata. Prefer official checkpoints or established fine-tuners and quantizers with clear provenance.

For every candidate verify the exact:

- generation, parameter variant, instruct/reasoning mode, and release
- dense or MoE architecture; total parameters govern storage while active parameters mainly govern compute
- artifact format, file size, license, context, modalities, and chat template
- runtime support for the architecture, projector, reasoning, tools, and structured output
- quantized artifact provenance and available files

Do not substitute evidence from a nearby size or family member.

## 3. Check practical fit

Use actual artifact sizes where possible. Account for weights, KV/recurrent state at the required context, runtime and batch buffers, projector, parallel requests, and memory used by the display or other processes.

Reject configurations that only fit on paper or require unrequested host-memory offload. Distinguish full accelerator execution, hybrid offload, and CPU/RAM inference. Treat advertised maximum context as a capability, not a sensible default.

## 4. Compare the relevant capabilities

Use exact-model evaluations closest to the workload:

- knowledge, reasoning, and instruction following
- tool choice, argument correctness, multi-step agents, and search
- long-context retrieval and document understanding
- OCR, charts, scientific figures, and vision
- coding, math, multilingual, or domain tasks when relevant

Label vendor-reported, third-party, and anecdotal evidence. Check reasoning mode, token budget, prompts, and whether scores are comparable. For altered or “uncensored” models, require provenance and capability-retention evidence; fewer refusals do not imply greater intelligence.

Tool reliability also depends on the runtime, template, parser, and harness. A function-calling claim alone is insufficient.

## 5. Recommend and validate

Return no more than:

1. **Default:** best overall match
2. **Quality alternative:** only if its extra cost is worthwhile
3. **Speed/simplicity alternative:** only if meaningfully different

State each exact model ID, evidence, deployment class, important caveat, and confidence. Include source links, fit uncertainties, and say plainly when the user's current model remains best.

When published evidence cannot decide, propose the smallest representative A/B set that can. Keep tools, templates, sampling, context, and comparable quantization fixed; score task success, tool-call validity, source fidelity, hallucinations, latency, and user preference.

Never invent memory or throughput figures. Measure them on the target system when they matter. Hand the chosen model to `tune-local-model`.
