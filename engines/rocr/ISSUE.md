# ROCr: BlitSdma::PendingBytes() reads one past bytes_written_ when the SDMA read pointer is in the ring's last 7 bytes

**Component:** rocr-runtime (`runtime/hsa-runtime/core/inc/amd_blit_sdma.h`), seen on ROCm 7.2.4; the same
`resize`/`convert` code is on `develop`.

**Symptom:** with libstdc++ assertions on (Arch Linux builds every package with `-D_GLIBCXX_ASSERTIONS`), a
process doing many async copies aborts after millions of them:

```
/usr/include/c++/16.1.1/bits/stl_vector.h:1253: std::vector<...>::operator[](size_type) [with _Tp = long unsigned int ...]: Assertion '__n < this->size()' failed.
```

Backtrace (offsets resolved against the disassembly labels of `libhsa-runtime64.so.1` 7.2.4):
`BlitSdma<true>::PendingBytes()` ← `GpuAgent::DmaEngineIsFree()` ← `GpuAgent::DmaCopyStatus()` ← HIP's copy-engine
selection in `hipMemcpyAsync`. Without assertions it is a silent 8-byte heap over-read. Hit in practice by an
LLM inference engine (~1 crash per 90 requests on an RX 7900 XTX).

**Cause:** `bytes_written_` maps byte offsets to 8-byte slots with `convert(i) = (i + 7) / 8` (rounds up) and is sized
`convert(kQueueSize) = kQueueSize / 8`. `PendingBytes()` reads `bytes_written_[WrapIntoRing(hw_read_index)]`. When the
engine's read pointer sits in the last 7 bytes of the ring (for example inside end-of-ring NOP padding),
`convert()` returns `kQueueSize / 8`, one past the end.

**Fix (one line):**

```diff
-    void resize(size_t size) { data_.resize(convert(size)); }
+    void resize(size_t size) { data_.resize(convert(size) + 1); }
```

Related: `PadRingToEnd()` calls `bytes_written_.fill(WrapIntoRing(curr_index), WrapIntoRing(new_index), ...)`; at the
ring end `WrapIntoRing(new_index)` is 0, so the fill never runs and the padding's entries keep stale values.

**Reproducer:** `sdma_stress2.cpp` (attached): 4 streams, async H2D/D2H copies of 4 KB–1 MB with random cross-stream
event waits so command sizes vary. `hipcc -O2 sdma_stress2.cpp -o s && ./s 60000000`. Stock Arch 7.2.4 aborted after
~16.5 M and ~8 M copies (203 s, 101 s); with the patch it ran 40 M copies clean. RX 7900 XTX (gfx1100), kernel 7.2.6.
