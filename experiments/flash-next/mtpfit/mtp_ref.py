"""mtp_ref.py - one Strata MTP draft step in numpy, stage by stage as src/core/mtp.cpp (engine 210ebbdd) runs it.

Citations are r5-upstream paths. Float math throughout: the engine quantizes activations to q8_1 / q8_0 before
every Q8_0 / Q2_0 product and may round hyper-connection activations to BF16, so expect agreement to ~1e-2
relative, not bit for bit (README "Parity").

    python mtp_ref.py --rt ~/Documents/projects/Strata-data/mtp/rt-basalt-q2b --check-formats --run
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

N, HC, LR = 2560, 4, 320                 # n_embd, hyper-connection streams, bottleneck (kernels/gr.hpp:44)
NH, NKV, HD, NROT = 24, 2, 256, 64       # kernels/qsa.hpp:94-111
NE, TOPK, FF = 512, 10, 640              # experts, router top-k, expert width (tools/mtp_rt.py:28)
EPS, THETA = 1e-6, 1e7                   # mtp.cpp:57, kernels/qsa.hpp:125
BLOB = 3 * (N * FF * 18 // 64)           # one expert in experts.bin (tools/mtp_rt.py:29)
DATA = Path.home() / "Documents/projects/Strata-data"
GGUF = (Path.home() / "Documents/projects/llamacpp-tuner/tmp/models/ukisai%2FSwift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF"
        / "IQ3_XXS/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf")


# ---------------------------------------------------------------------------------------------- formats
def bf16(u16: np.ndarray) -> np.ndarray:
    return (u16.astype(np.uint32) << 16).view(np.float32)


def deq_q8_0(b: np.ndarray) -> np.ndarray:
    """ggml Q8_0: 34-byte blocks, fp16 d then 32 int8 (tools/mtp_rt.py:33-44 writes it)."""
    b = b.reshape(-1, 34)
    d = b[:, :2].copy().view(np.float16).astype(np.float32)
    return (b[:, 2:].view(np.int8).astype(np.float32) * d).reshape(-1)


def deq_q2_0(codes: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Q2_0 planes: codes [..., 16] bytes (4 weights a byte, low bits first), d [...] fp16; w = (code - 1) * d
    (tools/mtp_pack.py:61-63 packs, :109-114 decodes; include/strata/kernels/cpu/expert.hpp:39)."""
    c = np.stack([(codes >> s) & 3 for s in (0, 2, 4, 6)], axis=-1).reshape(*codes.shape[:-1], 64)
    return (c.astype(np.float32) - 1.0) * d.astype(np.float32)[..., None]


def load_dense(rt: Path) -> dict[str, np.ndarray]:
    """dense.txt lines: name kind rows cols offset bytes (tools/mtp_rt.py:11). Norm weights are stored as 1 + w."""
    raw = np.fromfile(rt / "dense.bin", dtype=np.uint8)
    W = {}
    for line in (rt / "dense.txt").read_text().splitlines():
        name, kind, rows, cols, off, nb = line.split()
        rows, cols, off, nb = int(rows), int(cols), int(off), int(nb)
        b = raw[off:off + nb]
        if kind == "f32":
            W[name] = b.view("<f4").copy()
        elif kind == "bf16":
            W[name] = bf16(b.view("<u2")).reshape(rows, cols)
        else:
            W[name] = deq_q8_0(b).reshape(rows, cols)
    return W


class Experts:
    """experts.bin: per expert [gate/up codes, interleaved rows 2r = gate r, 2r+1 = up r][down codes]
    [gate/up fp16 scales][down fp16 scales] (tools/mtp_rt.py:44-56, include/strata/kernels/cpu/expert.hpp)."""

    def __init__(self, rt: Path):
        self.m = np.memmap(rt / "experts.bin", dtype=np.uint8, mode="r")
        assert self.m.size == NE * BLOB, self.m.size

    def __getitem__(self, e: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        b = np.asarray(self.m[e * BLOB:(e + 1) * BLOB])
        ncg, ncd = 2 * FF * (N // 64) * 16, N * (FF // 64) * 16
        nsg = 2 * FF * (N // 64) * 2
        gu = deq_q2_0(b[:ncg].reshape(2 * FF, N // 64, 16),
                      b[ncg + ncd:ncg + ncd + nsg].view("<f2").reshape(2 * FF, N // 64)).reshape(2 * FF, N)
        dn = deq_q2_0(b[ncg:ncg + ncd].reshape(N, FF // 64, 16),
                      b[ncg + ncd + nsg:].view("<f2").reshape(N, FF // 64)).reshape(N, FF)
        return gu[0::2], gu[1::2], dn


class MainModel:
    """The main model's pieces the drafter and the teacher use: token_embd rows (IQ3_S), output.weight rows (Q5_K)
    restricted to the draft vocabulary (the draft head is the main head's subset rows, mtp.cpp:1074-1076), and the
    output hyper-connection mixer (output_hc_*: the main head's only 'norm', core/layer.cpp:1249-1285)."""

    def __init__(self, gguf_path: Path, vocab: np.ndarray, pack: Path = DATA / "packs/swift-iq3_xxs"):
        sys.path.insert(0, os.environ.get("STRATA_GGUF_PY", str(Path.home() / "Documents/projects/Strata/third_party/llama.cpp/gguf-py")))
        import gguf
        self.gguf = gguf
        r = gguf.GGUFReader(str(gguf_path))
        t = {x.name: x for x in r.tensors}
        self.emb_t, self.head_t = t["token_embd.weight"], t["output.weight"]
        self.vocab = vocab
        self._head = None
        # output_hc_* in the engine's form (norm stored as 1 + w), from the pack's dense pool (index.txt cols 3-4)
        idx = {l.split()[0]: l.split() for l in (pack / "index.txt").read_text().splitlines() if not l.startswith("#")}
        pool = np.memmap(pack / "dense.bin", dtype=np.uint8, mode="r")
        def get(name):
            off, nb = int(idx[name][3]), int(idx[name][4])
            return np.asarray(pool[off:off + nb])
        self.hc_norm = get("output_hc_norm.weight").view("<f4").copy()
        self.hc_down = bf16(get("output_hc_down.weight").view("<u2")).reshape(LR, HC * N)
        self.hc_up = bf16(get("output_hc_up.weight").view("<u2")).reshape(HC * N, LR)

    def rows(self, t, ids) -> np.ndarray:
        data = np.asarray(t.data)   # (n_rows, row_bytes) uint8
        return self.gguf.quants.dequantize(data[np.asarray(ids)], t.tensor_type).reshape(len(ids), N).astype(np.float32)

    def embed(self, tok: int) -> np.ndarray:
        return self.rows(self.emb_t, [tok])[0]

    def head(self) -> np.ndarray:
        if self._head is None:
            self._head = self.rows(self.head_t, self.vocab)
        return self._head

    def teacher_logits(self, R: np.ndarray) -> np.ndarray:
        """Main-model logits over the draft vocabulary from a final residual stack R [HC*N]."""
        mixed, _ = gr_read(R.reshape(HC, N), self.hc_norm, self.hc_down, self.hc_up, None)
        return self.head() @ mixed


# ---------------------------------------------------------------------------------------------- operations
def silu(x): return x / (1.0 + np.exp(-x))
def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))


def rms(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + EPS) * w


def gr_read(R, w_norm, w_down, w_up, w_inject):
    """Hyper-connection read (kernels/gr.hpp:94-118, cuda/gr.cu:126-157 per-stream RMS, :251-255 mean)."""
    xn = rms(R, w_norm.reshape(HC, N))
    x = xn.reshape(-1)
    lo = silu((w_down @ x) / HC)                       # the /hc is inside the silu (gr.hpp:108-109)
    mixed = (xn * sigmoid((w_up @ lo).reshape(HC, N))).mean(0)
    return mixed, (w_inject @ x if w_inject is not None else None)


def gr_write(R, out, inject):
    """R[c] += out * 2 sigmoid(inject[c] / hc) (kernels/gr.hpp:121-127, cuda/gr.cu:283-293)."""
    return R + out[None, :] * (2.0 * sigmoid(inject / HC))[:, None]


def rope(x: np.ndarray, pos: int) -> np.ndarray:
    """NEOX pairs (i, i + 32) on the first 64 of 256 dims, base 1e7 (kernels/qsa.hpp:94-125; GGUF rope keys)."""
    half = NROT // 2
    f = pos * THETA ** (-np.arange(half, dtype=np.float64) * 2.0 / NROT)
    c, s = np.cos(f).astype(np.float32), np.sin(f).astype(np.float32)
    y = x.copy()
    a, b = x[..., :half], x[..., half:NROT]
    y[..., :half], y[..., half:NROT] = a * c - b * s, a * s + b * c
    return y


def front(W, R_in, emb, pos):
    """record_front (mtp.cpp:745-862): the two input branches, the attention read, this cell's K and V."""
    e2 = W["fc_embedding.weight"] @ rms(emb, W["pre_fc_norm_embedding.weight"])                    # :771-773
    hn = rms(R_in.reshape(-1), W["pre_fc_norm_hidden.weight"]).reshape(HC, N)   # pooled over 10240 (:776-779)
    R = (W["fc_hidden.weight"] @ hn.T).T + e2[None, :]                          # per stream, then broadcast (:780-785)
    mixed, inj = gr_read(R, W["attn_hyper_connection.hc_norm.weight"],                               # :787-806
                         W["attn_hyper_connection.input_mix_weight_down.weight"],
                         W["attn_hyper_connection.input_mix_weight_up.weight"],
                         W["attn_hyper_connection.block_inject_weight.weight"])
    k = (W["self_attn.k_proj.weight"] @ mixed).reshape(NKV, HD)                                    # :809-810
    v = (W["self_attn.v_proj.weight"] @ mixed).reshape(NKV, HD)
    k = rope(rms(k, W["self_attn.k_norm.weight"]), pos)                                              # :820-831
    return R, mixed, inj, k, v


def rest(W, ex: Experts, R, mixed, inj, q_pos, K, V, head):
    """record_rest (mtp.cpp:872-1131) for one row: dense attention over the cells K, V [cells, NKV, HD]."""
    qf = (W["self_attn.q_proj.weight"] @ mixed).reshape(NH, 2 * HD)                                 # :891
    q = rope(rms(qf[:, :HD], W["self_attn.q_norm.weight"]), q_pos)                                    # :894-903
    kh = np.repeat(K, NH // NKV, axis=1)                                     # head h reads kv head h // 12
    vh = np.repeat(V, NH // NKV, axis=1)
    s = np.einsum("hd,chd->hc", q, kh) / np.sqrt(HD)                         # cuda/qsa_decode_attn.cu:699
    p = np.exp(s - s.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    att = np.einsum("hc,chd->hd", p, vh) * sigmoid(qf[:, HD:])               # gate = 2nd half (:912, native_qsa.hpp:30)
    bo = W["self_attn.o_proj.weight"] @ att.reshape(-1)                                              # :914
    R = gr_write(R, bo, inj)                                                 # the MLP read applies it first (:917-933)
    x, inj2 = gr_read(R, W["mlp_hyper_connection.hc_norm.weight"],
                      W["mlp_hyper_connection.input_mix_weight_down.weight"],
                      W["mlp_hyper_connection.input_mix_weight_up.weight"],
                      W["mlp_hyper_connection.block_inject_weight.weight"])
    lg = W["mlp.gate.weight"] @ x                                            # router (:1003-1009)
    pr = np.exp(lg - lg.max())
    pr /= pr.sum()
    ids = np.argsort(-pr, kind="stable")[:TOPK]                              # kernels/router_top10.hpp:3-5
    w = pr[ids] / max(pr[ids].sum(), 2.0 ** -14)
    y = np.zeros(N, np.float32)
    for e, we in zip(ids, w):                                                # routed experts (:1016-1020)
        g, u, d = ex[int(e)]
        y += we * (d @ (silu(g @ x) * (u @ x)))
    sh = W["mlp.shared_expert.down_proj.weight"] @ (silu(W["mlp.shared_expert.gate_proj.weight"] @ x)
                                                    * (W["mlp.shared_expert.up_proj.weight"] @ x))
    y += sh * sigmoid(W["mlp.shared_expert_gate.weight"] @ x)                # added plain (shared_expert.hpp:1-6, 85-99)
    R = gr_write(R, y, inj2)                                                                         # :1040-1053
    out, _ = gr_read(R, W["hyper_connection_mixer.hc_norm.weight"],          # final mixer, no inject (:1101-1106)
                     W["hyper_connection_mixer.input_mix_weight_down.weight"],
                     W["hyper_connection_mixer.input_mix_weight_up.weight"], None)
    logits = head @ out if head is not None else None                        # draft-vocab rows of output.weight (:1110-1112)
    return R, logits, ids, w


class Cache:
    """The drafter's own K/V, one entry per cell (absolute position = cell, mtp.cpp:1495-1501)."""
    def __init__(self):
        self.K, self.V = {}, {}

    def kv(self, upto: int, window: int = 8192):
        cells = [c for c in sorted(self.K) if upto - window < c <= upto]   # --mtp-window (generate.cpp:573)
        return np.stack([self.K[c] for c in cells]), np.stack([self.V[c] for c in cells])


def step(W, ex, cache: Cache, R_in, emb, cell, head):
    """One draft step at `cell`: R_in is the main model's final residual stack of the previous position (step 1)
    or the previous step's output R (chain, verify_kernels.hpp:112-115); emb is the next token's embedding."""
    R, mixed, inj, k, v = front(W, R_in, emb, cell)
    cache.K[cell], cache.V[cell] = k, v
    K, V = cache.kv(cell)
    return rest(W, ex, R, mixed, inj, cell, K, V, head)


# ---------------------------------------------------------------------------------------------- checks
def rel(a, b): return float(np.sqrt(((a - b) ** 2).mean()) / (np.sqrt((b ** 2).mean()) + 1e-30))


def check_formats(rt: Path, tensors: Path = DATA / "mtp/tensors") -> None:
    """Decode the runtime files against the BF16 source tensors (base Qwen MTP, tools/mtp_fetch.py)."""
    def src(name, shape):
        return bf16(np.fromfile(tensors / f"{name}.bin", dtype="<u2")).reshape(shape)
    W = load_dense(rt)
    print(f"{rt.name}: {len(W)} dense tensors")
    for short, full in [("fc_hidden.weight", "mtp.fc_hidden.weight"), ("self_attn.q_proj.weight", "mtp.layers.0.self_attn.q_proj.weight"),
                        ("hyper_connection_mixer.input_mix_weight_down.weight", "mtp.hyper_connection_mixer.input_mix_weight_down.weight")]:
        print(f"  {short}: rel err vs BF16 source {rel(W[short], src(full, W[short].shape)):.4f}")
    nw = W["pre_fc_norm_hidden.weight"]
    print(f"  pre_fc_norm_hidden: stored - (1 + source) max abs {np.abs(nw - 1.0 - src('mtp.pre_fc_norm_hidden.weight', nw.shape)).max():.2e}")
    ex = Experts(rt)
    gu = np.memmap(tensors / "mtp.layers.0.mlp.experts.gate_up_proj.bin", dtype="<u2", mode="r", shape=(NE, 2 * FF, N))
    dn = np.memmap(tensors / "mtp.layers.0.mlp.experts.down_proj.bin", dtype="<u2", mode="r", shape=(NE, N, FF))
    for e in (0, 255, 511):
        g, u, d = ex[e]
        print(f"  expert {e}: rel err gate {rel(g, bf16(gu[e, :FF])):.3f} up {rel(u, bf16(gu[e, FF:])):.3f} "
              f"down {rel(d, bf16(dn[e])):.3f}")


def run(rt: Path, vocab_path: Path, gguf_path: Path, steps: int = 3, seed: int = 0) -> None:
    W, ex = load_dense(rt), Experts(rt)
    vocab = np.fromfile(vocab_path, dtype="<i4")
    main = MainModel(gguf_path, vocab)
    head = main.head()
    rng = np.random.default_rng(seed)
    cache = Cache()
    ctx = 8                                   # synthetic prompt cells: K/V only, as the prompt prefill writes them
    for c in range(ctx):
        _, _, _, k, v = front(W, rng.standard_normal(HC * N).astype(np.float32), main.embed(int(rng.integers(1000, 20000))), c)
        cache.K[c], cache.V[c] = k, v
    R_in = rng.standard_normal(HC * N).astype(np.float32)
    tok = int(vocab[100])
    t_logits = main.teacher_logits(R_in)
    print(f"teacher logits over the draft vocab: shape {t_logits.shape}, finite {np.isfinite(t_logits).all()}")
    for j in range(steps):
        R_in, logits, ids, w = step(W, ex, cache, R_in.reshape(-1), main.embed(tok), ctx + j, head)
        top = int(np.argmax(logits))
        p = np.exp(logits - logits.max()); p /= p.sum()
        print(f"step {j}: R {R_in.shape} finite {np.isfinite(R_in).all()} |R| {np.abs(R_in).mean():.3f}; logits {logits.shape} "
              f"finite {np.isfinite(logits).all()}; experts {ids.tolist()} w-sum {w.sum():.4f}; draft {vocab[top]} p {p[top]:.3f}")
        tok = int(vocab[top])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rt", type=Path, default=DATA / "mtp/rt-basalt-q2b")
    ap.add_argument("--vocab", type=Path, default=DATA / "draft_vocab_agent.bin")
    ap.add_argument("--gguf", type=Path, default=GGUF)
    ap.add_argument("--check-formats", action="store_true")
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()
    if a.check_formats:
        check_formats(DATA / "mtp/rt")       # base dense: decodes to the BF16 source within q8_0 error
        check_formats(a.rt)                   # Basalt dense: differs from base by its fit
    if a.run:
        run(a.rt, a.vocab, a.gguf)
    return 0


if __name__ == "__main__":
    sys.exit(main())
