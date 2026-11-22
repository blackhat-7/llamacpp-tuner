"""sconvtest.py OUTDIR: test.loom: the fused short-conv kernels against the originals, bitwise, on the whole arena,
the capture buffer and the residual output; spans spec and non-spec, one at a fork (reader != writer), one fresh.
Original order: stat K, commit K, stat V, commit V, kv_prep ring; stat c2304, commit c2304, add.
Fused order: stat2, kvring+commit2; stat_add c2304, commit c2304. bench.loom: the same at t = 2 and 4."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kx
import sconvgen

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
S = kx.src()
rng = np.random.default_rng(13)
NEG = os.environ.get("NEG")
names = ["g_ink_sconv_stat_c1024", "g_ink_sconv_commit_cap_c1024", "g_ink_kv_prep_store_swa_ring", "g_ink_sconv_stat_c2304",
         "g_ink_sconv_commit_cap_c2304", "g_ink_add"]
orig = [kx.grab(S, n, n.replace("g_", "o_")).replace('%chan_off = index.cast %chan_off_arg : index to offset', '%chan_off_nn = index.assume %chan_off_arg [range(%chan_off_arg, 0, 1048575)] : index\n  %chan_off = index.cast %chan_off_nn : index to offset').replace("index.div %chan_off_arg, %c12", "index.div %chan_off_nn, %c12") for n in names]
mine = sconvgen.module(S)
for n in ("g_ink_sconv_stat2_c1024", "g_ink_kvring_commit2", "g_ink_sconv_stat_add_c2304"):
    mine = mine.replace(f'kernel.def export("{n}") @{n}(', f"kernel.def target(@w32) @{n}(")
lib = [kx.HEAD, "".join(kx.funcs_used(S, "".join(orig) + mine).values()), *orig, mine]
ARENA = 6_000_000 // 4
CAP = 16 * 8 * 48 * 6656
SWA = 600000


def npy(name, a):
    np.save(os.path.join(out, name), a)
    return name


def case(tag, t, spans, rand=True):
    """spans: list of (length, start, row, spec)."""
    meta = np.zeros(4 + 9 * len(spans), np.int32)
    meta[0] = len(spans)
    info = np.zeros((t, 4), np.int32)
    first = 0
    for s, (length, start, row, spec) in enumerate(spans):
        meta[4 + 9 * s:4 + 9 * s + 9] = [length, start, 0, row, first, 0, 4 if spec else 0, 0, 0]
        for j in range(length):
            info[first + j] = [0, start + j, 1, s]
        first += length
    orig = np.zeros((2, 5), np.int64)
    orig[0] = [0, 0, 0, -1, 0]
    orig[1] = [400000, 200000, 0, spans[1][1] if len(spans) > 1 else -1, 0]  # row 1 at its fork: reader != writer
    def F(name, shape, gen, dtype="f32"):
        n = "x".join(map(str, shape))
        if rand:
            return f'  %{name} = check.file.read.npy path("{npy(f"{tag}_{name}.npy", gen())}") : tensor<{n}x{dtype}>\n'
        return f"  %{name} = check.generate.fill value({0.5 if dtype == 'f32' else 0}) : tensor<{n}x{dtype}>\n"
    s = f"  %t = check.literal value({t}) : index\n  %spans = check.literal value({len(spans)}) : index\n"
    s += "  %layer = check.literal value(1) : index\n  %off_k = check.literal value(0) : index\n  %off_v = check.literal value(12288) : index\n"
    s += "  %off_ap = check.literal value(24576) : index\n" + f"  %swa = check.literal value({SWA}) : index\n"
    s += F("meta", (meta.size,), lambda: meta, "i32") + F("info", (t, 4), lambda: info, "i32") + F("orig", (2, 5), lambda: orig, "i64")
    s += F("kraw", (t, 1024), lambda: rng.normal(0, 1, (t, 1024)).astype(np.float32))
    s += F("vraw", (t, 1024), lambda: rng.normal(0, 1, (t, 1024)).astype(np.float32))
    s += F("y", (t, 2304), lambda: rng.normal(0, 1, (t, 2304)).astype(np.float32))
    s += F("x", (t, 2304), lambda: rng.normal(0, 1, (t, 2304)).astype(np.float32))
    s += F("kw", (1024, 4), lambda: rng.normal(0, 0.5, (1024, 4)).astype(np.float32))
    s += F("vw", (1024, 4), lambda: rng.normal(0, 0.5, (1024, 4)).astype(np.float32))
    s += F("aw", (2304, 4), lambda: rng.normal(0, 0.5, (2304, 4)).astype(np.float32))
    a0 = rng.normal(0, 1, ARENA).astype(np.float32)
    for nm in ("arena", "arena_e"):
        s += F(nm, (ARENA,), lambda: a0)
    for nm in ("cap", "cap_e"):
        s += f"  %{nm} = check.generate.fill value(0.0) : tensor<{CAP}xf32>\n"
    for nm in ("kc", "vc", "kc_e", "vc_e"):
        s += f"  %{nm} = check.generate.fill value(0.0) : tensor<{t}x1024xf32>\n"
    for nm in ("y2_e", "o", "o_e"):
        s += f"  %{nm} = check.generate.fill value(0.0) : tensor<{t}x2304xf32>\n"
    return s


def T(t, n):
    return f"tensor<{t}x{n}xf32>"


def base(t, nm):
    A, C, I = f"tensor<{ARENA}xf32>", f"tensor<{CAP}xf32>", f"tensor<{t}x4xi32>"
    common = f"tensor<{nm}xi32>, {I}, tensor<2x5xi64>, {A}"
    s = ""
    for raw, w, kc, off in (("%kraw", "%kw", "%kc_e", "%off_k"), ("%vraw", "%vw", "%vc_e", "%off_v")):
        s += (f"  kernel.launch @o_ink_sconv_stat_c1024[%t](%t, %spans, %layer, {off}, %meta, %info, %orig, %arena_e, {raw}, {w}, {kc}) : "
              f"[index](index, index, index, index, {common}, {T(t, 1024)}, tensor<1024x4xf32>, {T(t, 1024)})\n"
              f"  kernel.launch @o_ink_sconv_commit_cap_c1024[%spans](%t, %spans, %layer, {off}, %meta, %orig, %arena_e, {raw}, %cap_e) : "
              f"[index](index, index, index, index, tensor<{nm}xi32>, tensor<2x5xi64>, {A}, {T(t, 1024)}, {C})\n")
    s += (f"  kernel.launch @o_ink_kv_prep_store_swa_ring[%t](%t, %spans, %swa, %meta, %info, %orig, %kc_e, %vc_e, %arena_e) : "
          f"[index](index, index, index, {common.replace(', ' + A, '')}, {T(t, 1024)}, {T(t, 1024)}, {A})\n")
    s += (f"  kernel.launch @o_ink_sconv_stat_c2304[%t](%t, %spans, %layer, %off_ap, %meta, %info, %orig, %arena_e, %y, %aw, %y2_e) : "
          f"[index](index, index, index, index, {common}, {T(t, 2304)}, tensor<2304x4xf32>, {T(t, 2304)})\n"
          f"  kernel.launch @o_ink_sconv_commit_cap_c2304[%spans](%t, %spans, %layer, %off_ap, %meta, %orig, %arena_e, %y, %cap_e) : "
          f"[index](index, index, index, index, tensor<{nm}xi32>, tensor<2x5xi64>, {A}, {T(t, 2304)}, {C})\n"
          f"  kernel.launch @o_ink_add[%t](%t, %x, %y2_e, %o_e) : [index](index, {T(t, 2304)}, {T(t, 2304)}, {T(t, 2304)})\n")
    return s


def fused(t, nm):
    A, C, I = f"tensor<{ARENA}xf32>", f"tensor<{CAP}xf32>", f"tensor<{t}x4xi32>"
    common = f"tensor<{nm}xi32>, {I}, tensor<2x5xi64>, {A}"
    one = f"index, index, index, {common}, {T(t, 1024)}, tensor<1024x4xf32>, {T(t, 1024)}"
    vw = "%kw" if NEG else "%vw"
    return (f"  kernel.launch @g_ink_sconv_stat2_c1024[%t](%t, %spans, %layer, %off_k, %meta, %info, %orig, %arena, %kraw, %kw, %kc, "
            f"%spans, %layer, %off_v, %meta, %info, %orig, %arena, %vraw, {vw}, %vc) : [index](index, {one}, {one})\n"
            f"  kernel.launch @g_ink_kvring_commit2[%t, %spans](%t, %spans, %swa, %meta, %info, %orig, %kc, %vc, %arena, %layer, %off_k, %off_v, "
            f"%kraw, %vraw, %cap) : [index, index](index, index, index, {common.replace(', ' + A, '')}, {T(t, 1024)}, {T(t, 1024)}, {A}, "
            f"index, index, index, {T(t, 1024)}, {T(t, 1024)}, {C})\n"
            f"  kernel.launch @g_ink_sconv_stat_add_c2304[%t](%t, %spans, %layer, %off_ap, %meta, %info, %orig, %arena, %y, %aw, %x, %o) : "
            f"[index](index, index, index, index, {common}, {T(t, 2304)}, tensor<2304x4xf32>, {T(t, 2304)}, {T(t, 2304)})\n"
            f"  kernel.launch @o_ink_sconv_commit_cap_c2304[%spans](%t, %spans, %layer, %off_ap, %meta, %orig, %arena, %y, %cap) : "
            f"[index](index, index, index, index, tensor<{nm}xi32>, tensor<2x5xi64>, {A}, {T(t, 2304)}, {C})\n")


CASES = [("s2spec", 4, [(2, 100, 0, True), (2, 50, 1, True)]), ("s2plain", 4, [(2, 100, 0, False), (2, 50, 1, False)]),
         ("s1fresh", 3, [(3, 0, 0, False)]), ("s1spec", 2, [(2, 7, 0, True)])]
test = list(lib)
for tag, t, spans in CASES:
    nm = 4 + 9 * len(spans)
    body = case(tag, t, spans) + base(t, nm) + fused(t, nm)
    body += f"  check.expect.bitwise actual(%arena) expected(%arena_e) : tensor<{ARENA}xf32>\n"
    body += f"  check.expect.bitwise actual(%cap) expected(%cap_e) : tensor<{CAP}xf32>\n"
    body += f"  check.expect.bitwise actual(%o) expected(%o_e) : {T(t, 2304)}\n"
    test.append(f"check.case public @{tag} {{\n{body}  check.return\n}}\n")
open(os.path.join(out, "test.loom"), "w").write("\n".join(test))

bench = list(lib)
for t in (2, 4):
    spans = [(t, 100, 0, True)]
    nm = 4 + 9
    s = case(f"b{t}", t, spans)
    bench.append(f"check.case public @base_t{t} {{\n{s}{base(t, nm) * 4}  check.return\n}}\ncheck.benchmark<@base_t{t}> @base_t{t}_bm\n")
    bench.append(f"check.case public @fused_t{t} {{\n{s}{fused(t, nm) * 4}  check.return\n}}\ncheck.benchmark<@fused_t{t}> @fused_t{t}_bm\n")
open(os.path.join(out, "bench.loom"), "w").write("\n".join(bench))
