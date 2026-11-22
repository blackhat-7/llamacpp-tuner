# KV8 format for Inkling-2-Mini global attention (tmp/loom/Inkling-2-Mini-kv8)

Scope: the 16 global-attention planes in the arena (K and V of the 8 global layer visits, fslot 0-7).
The SWA rings and the drafter's own K/V (allocation 9) stay f16 and are unchanged.

## Planes and pages

- 16 planes, same order as before: plane = fslot * 2 (K), fslot * 2 + 1 (V).
- Plane bytes = pool * 1088 (was pool * 2048). Plane p starts at arena offset p * pool * 1088.
- A position (token slot) is 1088 bytes. Slot = physical_page * 16 + (position % 16), from the same page table.
- Page (block) bytes = 16 * 1088 = 17408 (was 32768). storage.loom geometry[12] = 17408, geometry[11] = pool * 1088.
- Byte offset of a position: `(plane * pool + slot) * 1088` = `@g_kv8_offset` (copy of `@g_kv_offset` with 1088).
- V offset = K offset + pool * 1088 (`%plane_stride`).

## One position (1088 bytes)

| bytes | content |
|---|---|
| 0 .. 1023 | int8 codes, `[8 kv heads][128 dims]`: code of (head h, dim d) at `h * 128 + d` |
| 1024 .. 1087 | 32 f16 scales, `[8 kv heads][4 blocks]`: scale of (h, block b = d / 32) at `1024 + (h * 4 + b) * 2` |

Value = float(code) * float(scale). Symmetric, no zero point (llama.cpp q8_0 style, block of 32 dims).

## How the store kernels write it

`@g_kv8_store4(arena, at, head, lane, x: vector<4xf32>)`; one wave = one head, lane holds dims lane*4 .. lane*4+3
(so block b = lanes 8b .. 8b+7). Input x is the same f32 value the f16 path wrote (after the K norm clip / V RMS norm).

1. amax = max |x| over the 8 lanes of the block (shuffle xor 1, 2, 4).
2. d = amax / 127, rounded to f16 (d16). The rounded value is used to quantize, so the stored scale is exact.
3. code = fptosi(roundf(x / d16)) as int8; if d16 == 0 all codes are 0.
4. Lane 8b stores d16 at scale index head * 4 + b.

Callers: `@g_ink_sconv1_dyn_kv_prep_store`, `@g_ink_kv_prep_store`, `@g_ink_kv_prep_store_lv`.

## Readers (all converted)

- `@g_kv8_load4` (lane x 4 dims -> vector<4xf16>), `@g_kv8_load16` (16 dims from `dim`, which must not cross a 32-dim block
  -> vector<16xf16>), `@g_kv8_scale(arena, at, index)`.
- Decode: `@g_ink_attn_split` (rewritten, paged butterfly), `@g_ink_attn_split_span`, `@g16_ink_attn_split_listed`, `@g_ink_attn_direct` (unused).
- Prefill: `@g_ink_attention_flash2_ts2`, `@g_ink_attention_flash2_partial` (dequantize to f16 while staging LDS tiles).
- Keys past the visible range in the last page hold stale codes (finite) and stale scales (may be NaN/Inf):
  a reader must select (not multiply) them away, for both K scores and V weights.

## Generator

`tmp/loom/kv8_tools/transform.py` rewrites the original `llm.loom` into the kv8 one (helpers, kernel edits);
`tmp/loom/kv8_tools/split_kernel.py` emits the decode kernel. storage.loom is edited by hand (1088, 17408).
