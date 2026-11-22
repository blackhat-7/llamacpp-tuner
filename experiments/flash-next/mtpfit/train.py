"""train.py - fit the MTP draft layer's dense weights to Swift by KL to the main model (experts frozen at q2b).

The forward is mtp_ref.py's (which cites src/core/mtp.cpp) in torch, batched over a transcript's positions:

  step 1, base i:  (R_i, x_{i+1}) at cell i, attends cells <= i          -> target: main logits at i + 1
  step k, base i:  (step k-1's output R at i, x_{i+k}) at cell i + k - 1,
                   attends real cells <= i plus its own chain cells      -> target: main logits at i + k
  loss = sum_k decay^(k-1) KL(main || draft) over the draft vocabulary.

    python train.py --dry                                    # CPU: random weights, real shapes, one step + export
    python train.py --data DUMPDIR --init RT --out RT_SD     # GPU session (README step 5)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mtp_ref as ref  # noqa: E402
from mtp_ref import EPS, FF, HC, HD, LR, N, NH, NKV, NROT, THETA, TOPK  # noqa: E402
from torch.utils.checkpoint import checkpoint  # noqa: E402

ENGINE_TOOLS = Path.home() / "Documents/projects/Strata-wt/r5-upstream/tools"
FROZEN = ("self_attn.indexer.",)   # the engine's draft attention is dense: the indexer is never read (mtp.cpp:366)


def rms(x, w):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * w


def rope(x, pos):
    half = NROT // 2
    inv = THETA ** (-torch.arange(half, dtype=torch.float64, device=x.device) * 2.0 / NROT)
    f = pos.double()[:, None] * inv[None, :]
    c, s = f.cos().float()[:, None, :], f.sin().float()[:, None, :]
    a, b = x[..., :half], x[..., half:NROT]
    return torch.cat([a * c - b * s, a * s + b * c, x[..., NROT:]], -1)


def ffn(xs, offs, g, u, d):
    """The routed experts on rows sorted by expert (offs: each expert's end row) as three grouped matmuls; gate/up
    [E, FF, N] and down [E, N, FF] as stored."""
    h = F.silu(F.grouped_mm(xs, g.transpose(1, 2), offs=offs)) * F.grouped_mm(xs, u.transpose(1, 2), offs=offs)
    return F.grouped_mm(h, d.transpose(1, 2), offs=offs)


def q2_ffn(xs, offs, cg, cu, cd, sg, su, sd):
    """ffn() from the experts' Q2_0 planes (code - 1, int8) and scales: w = (code - 1) * d per 64 weights."""
    def deq(c, s):
        return (c.view(*c.shape[:-1], -1, 64).float() * s[..., None].float()).flatten(-2).to(xs.dtype)
    return ffn(xs, offs, deq(cg, sg), deq(cu, su), deq(cd, sd))


class MTP(torch.nn.Module):
    def __init__(self, W: dict[str, np.ndarray], gate, up, down, head, scales=None):
        """W: dense tensors (mtp_ref.load_dense); head [V, N] frozen. Experts frozen: gate/up [E, FF, N], down [E, N, FF]
        weights; or with scales (gate, up, down scales [E, rows, cols / 64], fp32) gate/up/down are the Q2_0 planes
        (code - 1, int8) and only the scales train, so experts.bin keeps the format the engine reads."""
        super().__init__()
        self.p = torch.nn.ParameterDict({k.replace(".", "__"): torch.nn.Parameter(torch.from_numpy(np.ascontiguousarray(v)),
                                                                                    requires_grad=not k.startswith(FROZEN))
                                         for k, v in W.items()})
        self.scaled = scales is not None
        for name, sc in zip(("sg", "su", "sd"), scales or ()):
            self.p[f"experts__{name}"] = torch.nn.Parameter(torch.from_numpy(sc))
        for name, t in (("gate", gate), ("up", up), ("down", down), ("head", head)):
            self.register_buffer(name, t, persistent=False)

    def w(self, name):
        return self.p[name.replace(".", "__")]

    def gr_read(self, R, name, inject=True):
        B = R.shape[0]
        xn = rms(R, self.w(f"{name}.hc_norm.weight").view(HC, N))
        x = xn.reshape(B, HC * N)
        lo = F.silu(x @ self.w(f"{name}.input_mix_weight_down.weight").T / HC)
        mixed = (xn * torch.sigmoid((lo @ self.w(f"{name}.input_mix_weight_up.weight").T).view(B, HC, N))).mean(1)
        return mixed, (x @ self.w(f"{name}.block_inject_weight.weight").T if inject else None)

    @staticmethod
    def gr_write(R, out, inj):
        return R + out[:, None, :] * (2.0 * torch.sigmoid(inj / HC))[:, :, None]

    def front(self, R_in, emb, pos):
        B = R_in.shape[0]
        e2 = rms(emb, self.w("pre_fc_norm_embedding.weight")) @ self.w("fc_embedding.weight").T
        hn = rms(R_in, self.w("pre_fc_norm_hidden.weight")).view(B, HC, N)
        R = hn @ self.w("fc_hidden.weight").T + e2[:, None, :]
        mixed, inj = self.gr_read(R, "attn_hyper_connection")
        k = rms((mixed @ self.w("self_attn.k_proj.weight").T).view(B, NKV, HD), self.w("self_attn.k_norm.weight"))
        v = (mixed @ self.w("self_attn.v_proj.weight").T).view(B, NKV, HD)
        return R, mixed, inj, rope(k, pos), v

    def attend(self, mixed, pos, K, V, mask, xK=None, xV=None):
        """K, V [C, NKV, HD] real cells, mask [B, C]; xK, xV [B, m, NKV, HD] this row's chain cells (own included)."""
        B, C, G = mixed.shape[0], K.shape[0], NH // NKV
        qf = (mixed @ self.w("self_attn.q_proj.weight").T).view(B, NH, 2 * HD)
        q = rope(rms(qf[..., :HD], self.w("self_attn.q_norm.weight")), pos).view(B, NKV, G, HD)
        s = torch.einsum("bgjd,cgd->bgjc", q, K).masked_fill(~mask[:, None, None, :], float("-inf"))
        if xK is not None:
            s = torch.cat([s, torch.einsum("bgjd,bmgd->bgjm", q, xK)], -1)
        p = torch.softmax(s / math.sqrt(HD), -1)
        o = torch.einsum("bgjc,cgd->bgjd", p[..., :C], V)
        if xV is not None:
            o = o + torch.einsum("bgjm,bmgd->bgjd", p[..., C:], xV)
        att = o.reshape(B, NH, HD) * torch.sigmoid(qf[..., HD:])
        return att.reshape(B, NH * HD) @ self.w("self_attn.o_proj.weight").T

    def moe(self, x):
        pr = torch.softmax(x @ self.w("mlp.gate.weight").T, -1)
        top, ids = pr.topk(TOPK, -1)
        wts = top / top.sum(-1, keepdim=True).clamp_min(2.0 ** -14)
        order = ids.flatten().argsort(stable=True)   # the assignments grouped by expert
        offs = torch.bincount(ids.flatten(), minlength=len(self.gate)).cumsum(0).to(torch.int32)
        rows = order // TOPK
        xs = x[rows].to(torch.bfloat16)
        if self.scaled:   # 128 experts at a time, recomputed in backward: all 512 dequantized at once would not fit
            ends, outs = [0] + offs.tolist(), []
            for i in range(0, len(self.gate), 128):
                j = min(i + 128, len(self.gate))
                outs.append(checkpoint(q2_ffn, xs[ends[i]:ends[j]], offs[i:j] - ends[i], self.gate[i:j], self.up[i:j],
                                       self.down[i:j], *(self.p[f"experts__{n}"][i:j] for n in ("sg", "su", "sd")),
                                       use_reentrant=False))
            out = torch.cat(outs)
        else:
            out = ffn(xs, offs, self.gate, self.up, self.down)
        y = torch.zeros_like(x).index_add(0, rows, out.float() * wts.flatten()[order, None])
        sh =(F.silu(x @ self.w("mlp.shared_expert.gate_proj.weight").T) * (x @ self.w("mlp.shared_expert.up_proj.weight").T)) \
            @ self.w("mlp.shared_expert.down_proj.weight").T
        return y + sh * torch.sigmoid(x @ self.w("mlp.shared_expert_gate.weight").T)

    def rest(self, R, mixed, inj, pos, K, V, mask, xK=None, xV=None):
        R = self.gr_write(R, self.attend(mixed, pos, K, V, mask, xK, xV), inj)
        x, inj2 = self.gr_read(R, "mlp_hyper_connection")
        R = self.gr_write(R, self.moe(x), inj2)
        out, _ = self.gr_read(R, "hyper_connection_mixer", inject=False)
        return R, out @ self.head.T.to(out.dtype)

    def chain(self, R, emb, steps, q0=0, window=8192):
        """R [n, HC*N] main residuals, emb [n, N] token embeddings of the same positions -> draft logits per step.
        Every cell's K/V is real; only bases q0.. are drafted from (the completion: drafts never start in a prompt).
        Step k's row j is base q0 + j (bases q0 .. n-1-k); its target is the main model's logits at q0 + j + k."""
        n = R.shape[0]
        cells = torch.arange(n - 1, device=R.device)
        Rk, mixed, inj, K, V = self.front(R[:-1], emb[1:], cells)
        base = cells[q0:]
        d = base[:, None] - cells[None, :]
        Rk, logits = self.rest(Rk[q0:], mixed[q0:], inj[q0:], base, K, V, (d >= 0) & (d < window))
        out, xK, xV = [logits], [], []
        for k in range(2, steps + 1):
            m = n - k - q0                                     # bases q0 .. n-1-k
            if m <= 0:
                break
            b = base[:m]
            Rf, mixed, inj, kk, vv = self.front(Rk[:m].reshape(m, HC * N), emb[b + k], b + k - 1)
            xK, xV = [t[:m] for t in xK] + [kk], [t[:m] for t in xV] + [vv]
            Rk, logits = self.rest(Rf, mixed, inj, b + k - 1, K, V, (d[:m] >= 0) & (d[:m] < window - k + 1),
                                   torch.stack(xK, 1), torch.stack(xV, 1))
            out.append(logits)
        return out


class Teacher(torch.nn.Module):
    """Main-model log-probs over the draft vocabulary: output_hc mixer + output.weight rows (core/layer.cpp:1249-1285)."""
    def __init__(self, hc_norm, hc_down, hc_up, head):
        super().__init__()
        for name, t in (("n", hc_norm), ("dn", hc_down), ("up", hc_up), ("head", head)):
            self.register_buffer(name, t, persistent=False)

    @torch.no_grad()
    def forward(self, R):
        B = R.shape[0]
        xn = rms(R.view(B, HC, N), self.n.view(HC, N))
        lo = F.silu(xn.reshape(B, -1) @ self.dn.T / HC)
        mixed = (xn * torch.sigmoid((lo @ self.up.T).view(B, HC, N))).mean(1)
        return torch.log_softmax(mixed @ self.head.T.to(mixed.dtype), -1)


def kl(t_logp, logits):
    return F.kl_div(torch.log_softmax(logits.float(), -1), t_logp.float(), log_target=True, reduction="batchmean")


def tv(t_logp, logits):
    """Total variation: 1 - sum_v min(p, q), one minus the plain speculative-sampling acceptance."""
    return 0.5 * (t_logp.float().exp() - torch.softmax(logits.float(), -1)).abs().sum(-1).mean()


def cut(logp, top_k=20, top_p=0.95):
    """The sampler's candidates (the user's top_k 20, top_p 0.95): the top_k log-probs, the rest of the shortest head
    reaching top_p set to -inf."""
    v, i = logp.topk(top_k, -1)
    p = torch.softmax(v, -1)
    return v.masked_fill(p.cumsum(-1) - p >= top_p, float("-inf")), i


@torch.no_grad()
def accept(t_logp, logits, gen=None):
    """greedy: the draft's argmax is the main model's; sampled: sum_v min(p, q), plain speculative sampling's temp-1
    acceptance. With gen also coupled: the engine's own temp-1 rule (top_k -> top_p -> a Gumbel-max pick with the same
    noise per token id on both sides, core/coupled_draft.hpp) over 4 noise draws, and KL(main || draft)."""
    lq = torch.log_softmax(logits.float(), -1)
    out = [(logits.argmax(-1) == t_logp.argmax(-1)).float().mean().item(),
           torch.minimum(t_logp.exp(), lq.exp()).sum(-1).mean().item()]
    if gen is not None:
        (tv_, ti), (qv, qi) = cut(t_logp.float()), cut(lq)
        hits = 0.0
        for _ in range(4):
            g = -torch.log(-torch.log(torch.rand(lq.shape, device=lq.device, generator=gen).clamp_min(1e-30)))
            pt = ti.gather(-1, (tv_ + g.gather(-1, ti)).argmax(-1, keepdim=True))
            pq = qi.gather(-1, (qv + g.gather(-1, qi)).argmax(-1, keepdim=True))
            hits += (pt == pq).float().mean().item() / 4
        out += [hits, F.kl_div(lq, t_logp.float(), log_target=True, reduction="batchmean").item()]
    return out


def loss_on(model, teacher, R, emb, steps, decay, q0=0, tv_w=0.0, gen=None):
    """One transcript chunk: R [n, HC*N] float, emb [n, N]. Returns (loss, [accept() per step]); the loss per step is
    KL(main || draft) + tv_w * TV, weighted decay^(k-1)."""
    t = teacher(R)                                       # t[i] = main log-probs for the token after position i
    loss, stats = 0.0, []
    for k, logits in enumerate(model.chain(R, emb, steps, q0), start=1):
        tk = t[q0 + k: q0 + k + logits.shape[0]]
        lk = kl(tk, logits) + (tv_w * tv(tk, logits) if tv_w else 0.0)
        loss = loss + decay ** (k - 1) * lk
        stats.append(accept(tk, logits, gen))
    return loss, stats


# ---------------------------------------------------------------------------------------------- data, export
def transcripts(dirs: list[Path], split: str, min_new: int = 16):
    """Index entries of every dump directory with at least min_new completion rows."""
    for d in dirs:
        for line in (d / "index.jsonl").read_text().splitlines():
            r = json.loads(line)
            if r["split"] == split and r["n"] - (r["n_prompt"] - 1) >= min_new:
                yield r


def chunk(r, max_len, device):
    """The transcript's residual rows (BF16 [n][HC*N]) and ids, cut to its last max_len positions; q0 = the first
    completion base (position n_prompt - 1 predicts the first completion token) within the cut."""
    m = np.memmap(r["r_file"], dtype="<u2", mode="r")
    n = r["n"]
    s = max(0, n - max_len)
    raw = np.asarray(m[(r["r_row0"] + s) * HC * N:(r["r_row0"] + n) * HC * N]).reshape(n - s, HC * N)
    R = torch.from_numpy(ref.bf16(raw)).to(device)
    return R, r["ids"][s:n], max(0, r["n_prompt"] - 1 - s)


def export(model, init_rt: Path, out: Path):
    """dense.bin at dense.txt's offsets, quantized by the engine's own tools/mtp_rt.py q8_0; experts, layout and
    vocabulary from init_rt. Norm weights are trained in their stored (1 + w) form, so they are written as they are."""
    os.environ.setdefault("STRATA_GGUF_PY", str(Path.home() / "Documents/projects/Strata/third_party/llama.cpp/gguf-py"))
    sys.path.insert(0, str(ENGINE_TOOLS))
    from mtp_rt import q8_0
    raw = bytearray((init_rt / "dense.bin").read_bytes())
    for line in (init_rt / "dense.txt").read_text().splitlines():
        name, kind, rows, cols, off, nb = line.split()
        off, nb = int(off), int(nb)
        w = model.w(name).detach().float().cpu().numpy()
        if kind == "q8_0":
            b = q8_0(w)
        elif kind == "bf16":   # round to nearest even
            u = w.astype("<f4").view("<u4").astype(np.uint64)
            b = (((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype("<u2")).tobytes()
        else:
            b = w.astype("<f4").tobytes()
        assert len(b) == nb, (name, len(b), nb)
        raw[off:off + nb] = b
    out.mkdir(parents=True, exist_ok=True)
    (out / "dense.bin").write_bytes(raw)
    for f in ("dense.txt", "draft_vocab.bin"):
        shutil.copyfile(init_rt / f, out / f)
    if model.scaled:
        export_scales(model, init_rt, out)
    elif not (out / "experts.bin").exists():
        (out / "experts.bin").symlink_to((init_rt / "experts.bin").resolve())


QSPLIT = (2 * FF * (N // 64) * 16, N * (FF // 64) * 16, 2 * FF * (N // 64) * 2)   # gate/up codes, down codes, gate/up scales


def planes(codes):
    """Q2_0 code bytes [rows, cols / 64, 16] (4 weights a byte, low bits first) -> int8 code - 1 [rows, cols]."""
    c = np.stack([(codes >> s) & 3 for s in (0, 2, 4, 6)], axis=-1)
    return c.reshape(codes.shape[0], -1).astype(np.int8) - 1


def expert_parts(rt: Path):
    """experts.bin (mtp_ref.Experts' layout) as int8 planes gate/up [E, FF, N], down [E, N, FF] and fp32 scales."""
    m = np.memmap(rt / "experts.bin", dtype=np.uint8, mode="r").reshape(ref.NE, ref.BLOB)
    ncg, ncd, nsg = QSPLIT
    cg, cu = np.empty((ref.NE, FF, N), np.int8), np.empty((ref.NE, FF, N), np.int8)
    cd = np.empty((ref.NE, N, FF), np.int8)
    sg, su = np.empty((ref.NE, FF, N // 64), np.float32), np.empty((ref.NE, FF, N // 64), np.float32)
    sd = np.empty((ref.NE, N, FF // 64), np.float32)
    for e in range(ref.NE):
        b = np.asarray(m[e])
        gu = planes(b[:ncg].reshape(2 * FF, N // 64, 16))           # rows 2r = gate r, 2r + 1 = up r
        cg[e], cu[e] = gu[0::2], gu[1::2]
        cd[e] = planes(b[ncg:ncg + ncd].reshape(N, FF // 64, 16))
        s = b[ncg + ncd:ncg + ncd + nsg].view("<f2").reshape(2 * FF, N // 64)
        sg[e], su[e] = s[0::2], s[1::2]
        sd[e] = b[ncg + ncd + nsg:].view("<f2").reshape(N, FF // 64)
    ex = ref.Experts(rt)
    for e in (0, ref.NE - 1):   # the planes and scales give mtp_ref's own dequantization
        g, u, d = ex[e]
        assert np.array_equal((cg[e].reshape(FF, -1, 64) * sg[e][..., None]).reshape(FF, N), g)
        assert np.array_equal((cd[e].reshape(N, -1, 64) * sd[e][..., None]).reshape(N, FF), d)
    return cg, cu, cd, sg, su, sd


def export_scales(model, init_rt: Path, out: Path):
    """experts.bin with the trained scales as fp16 in place; codes as init_rt has them."""
    raw = np.fromfile(init_rt / "experts.bin", dtype=np.uint8).reshape(ref.NE, ref.BLOB)
    ncg, ncd, nsg = QSPLIT
    sg, su, sd = (model.p[f"experts__{n}"].detach().float().cpu().numpy() for n in ("sg", "su", "sd"))
    raw[:, ncg + ncd:ncg + ncd + nsg] = np.stack([sg, su], 2).astype("<f2").view(np.uint8).reshape(ref.NE, -1)
    raw[:, ncg + ncd + nsg:] = sd.astype("<f2").view(np.uint8).reshape(ref.NE, -1)
    path = out / "experts.bin"
    path.unlink(missing_ok=True)
    raw.tofile(path)


# ---------------------------------------------------------------------------------------------- main
def dry(steps: int) -> int:
    """CPU: real shapes for every dense tensor (rt-basalt-q2b's layout), random values, 16 experts, 512-token vocab."""
    torch.manual_seed(0)
    rt = ref.DATA / "mtp/rt-basalt-q2b"
    g = torch.Generator().manual_seed(0)
    W = {}
    for line in (rt / "dense.txt").read_text().splitlines():
        name, kind, rows, cols = line.split()[:4]
        shape = (int(cols),) if kind == "f32" else (int(rows), int(cols))
        W[name] = (torch.ones(shape) if kind == "f32" else torch.randn(shape, generator=g) * 0.02).numpy()
    E, V, n = 16, 512, 12
    gate, up = (torch.randn(E, FF, N) * 0.02).bfloat16(), (torch.randn(E, FF, N) * 0.02).bfloat16()
    down, head = (torch.randn(E, N, FF) * 0.02).bfloat16(), torch.randn(V, N) * 0.02
    W["mlp.gate.weight"] = W["mlp.gate.weight"][:E]
    model = MTP(W, gate, up, down, head)
    teacher = Teacher(torch.ones(HC * N), torch.randn(LR, HC * N) * 0.02, torch.randn(HC * N, LR) * 0.02, head)
    R, emb = torch.randn(n, HC * N), torch.randn(n, N)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    t0 = time.time()
    loss, stats = loss_on(model, teacher, R, emb, steps, 0.6)
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
    opt.step()
    print(f"dry step: loss {loss.item():.4f} grad-norm {gn:.3f} steps {[tuple(round(v, 3) for v in s) for s in stats]} "
          f"({time.time() - t0:.1f} s); trainable {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.1f} M")
    W["mlp.gate.weight"] = np.zeros((512, N), np.float32)   # export needs the real router shape
    model.p["mlp__gate__weight"] = torch.nn.Parameter(torch.from_numpy(W["mlp.gate.weight"]))
    out = Path(os.environ.get("MTPFIT_DRY_OUT", "/tmp/mtpfit-dry"))
    export(model, rt, out)
    back = ref.load_dense(out)
    errs = {k: ref.rel(back[k], model.w(k).detach().numpy()) for k in ("fc_hidden.weight", "self_attn.q_proj.weight",
                                                                      "mlp.gate.weight", "pre_fc_norm_hidden.weight")}
    print(f"export round trip to {out}: rel err {', '.join(f'{k} {v:.4f}' for k, v in errs.items())}")
    return 0


def cached(path: Path, make):
    if not path.exists():
        np.save(path, make().astype(np.float16))
    return np.load(path).astype(np.float32)


def setup(a, dev):
    W = ref.load_dense(a.init)
    vocab = np.fromfile(a.vocab, dtype="<i4")
    main = ref.MainModel(a.gguf, vocab)
    cache = HERE / "data"
    head = torch.from_numpy(cached(cache / "head_draftvocab.npy", main.head)).to(dev, torch.bfloat16)
    path = cache / "token_embd.npy"   # the whole table once (fp16, ~1.3 GB): kept fp16 on the device, rows cast on use
    if not path.exists():
        np.save(path, main.rows(main.emb_t, np.arange(len(main.emb_t.data))).astype(np.float16))
    emb = torch.from_numpy(np.load(path)).to(dev)
    if getattr(a, "train_scales", False):
        cg, cu, cd, *scales = expert_parts(a.init)
        model = MTP(W, *(torch.from_numpy(x).to(dev) for x in (cg, cu, cd)), head, scales).to(dev)
    else:
        ex = ref.Experts(a.init)
        g, u, d = zip(*(ex[e] for e in range(ref.NE)))
        model = MTP(W, *(torch.from_numpy(np.stack(x)).to(dev, torch.bfloat16) for x in (g, u, d)), head).to(dev)
    teacher = Teacher(*(torch.from_numpy(x).to(dev) for x in (main.hc_norm, main.hc_down, main.hc_up)), head)
    return model, teacher, emb


def evaluate(model, teacher, emb, held, a, dev):
    """Per step the mean of accept()'s greedy, sampled, coupled and KL over the held-out transcripts, and the chained
    greedy, sampled and coupled acceptance: the expected drafts accepted per round when every step is drafted."""
    acc = np.zeros((a.steps, 4))
    gen = torch.Generator(device=dev).manual_seed(1234)   # the same noise draws for every checkpoint
    with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
        for r in held:
            R, ids, q0 = chunk(r, a.max_len, dev)
            acc += np.array(loss_on(model, teacher, R, emb[torch.tensor(ids, device=dev)].float(), a.steps, a.decay, q0,
                                    gen=gen)[1])
    acc /= max(1, len(held))
    return acc, [float(np.cumprod(acc[:, c]).sum()) for c in range(3)]


def report(tag, acc, ch):
    return (f"{tag} per step (greedy, sampled, coupled, KL) {acc.round(4).tolist()} chained greedy {ch[0]:.4f} "
            f"sampled {ch[1]:.4f} coupled {ch[2]:.4f}")


def train(a) -> int:
    dev = torch.device(a.device)
    t_end = time.time() + 60 * a.minutes   # one bounded gpu.lock hold; rerun with the same --ckpt to resume
    model, teacher, emb = setup(a, dev)
    train_set = list(transcripts(a.data, "train"))
    held = list(transcripts(a.data, "heldout"))[: a.n_held]
    params = [p for p in model.parameters() if p.requires_grad]
    groups = [{"params": [p for n, p in model.p.items() if p.requires_grad and not n.startswith("experts__")], "base": a.lr}]
    if model.scaled:
        groups.append({"params": [p for n, p in model.p.items() if n.startswith("experts__")], "base": a.scale_lr})
    opt = torch.optim.AdamW(groups, lr=a.lr, weight_decay=0.0)
    st = {"it": 0, "best": -1.0, "bad": 0, "log": []}
    if a.ckpt.exists():
        ck = torch.load(a.ckpt, map_location=dev)
        model.p.load_state_dict(ck["p"])
        opt.load_state_dict(ck["opt"])
        st = ck["st"]
        print(f"resumed {a.ckpt} at it {st['it']} (best {st['best']:.4f})", flush=True)
    else:
        acc, ch = evaluate(model, teacher, emb, held, a, dev)
        st["log"].append((0, None, acc.tolist(), ch))
        print(report(f"it 0 held-out ({len(held)}):", acc, ch), flush=True)
    print(f"{len(train_set)} train transcripts, {sum(r['n'] - r['n_prompt'] + 1 for r in train_set)} completion rows", flush=True)
    order = np.concatenate([np.random.default_rng(e).permutation(len(train_set)) for e in range(a.epochs)])
    t0, lsum = time.time(), 0.0
    while st["it"] < len(order) and time.time() < t_end:
        if (HERE.parent / "STOP").exists():
            print("STOP file: ending early", flush=True)
            break
        # linear warm-up, then a cosine down to min_lr * lr at the last iteration
        cos = 0.5 * (1.0 + math.cos(math.pi * st["it"] / max(1, len(order) - 1)))
        f = min(1.0, (st["it"] + 1) / a.warmup) * (a.min_lr + (1.0 - a.min_lr) * cos)
        for gr in opt.param_groups:
            gr["lr"] = gr["base"] * f
        R, ids, q0 = chunk(train_set[order[st["it"]]], a.max_len, dev)
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            loss, _ = loss_on(model, teacher, R, emb[torch.tensor(ids, device=dev)].float(), a.steps, a.decay, q0,
                              a.tv)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        st["it"] += 1
        lsum += loss.item()
        if st["it"] % 20 == 0:
            print(f"it {st['it']} loss {lsum / 20:.4f} ({(time.time() - t0) / 20:.2f} s/it)", flush=True)
            t0, lsum = time.time(), 0.0
        if st["it"] % a.eval_every == 0 or st["it"] == len(order):
            acc, ch = evaluate(model, teacher, emb, held, a, dev)
            st["log"].append((st["it"], loss.item(), acc.tolist(), ch))
            print(report(f"it {st['it']} held-out", acc, ch), flush=True)
            score = ch[0] + ch[2]
            if score > st["best"]:
                st["best"], st["bad"] = score, 0
                export(model, a.init, a.out)
                print(f"  best so far: exported to {a.out}", flush=True)
            else:
                st["bad"] += 1
            torch.save({"p": model.p.state_dict(), "opt": opt.state_dict(), "st": st}, a.ckpt)
            if st["bad"] >= a.patience:
                print(f"early stop: best {st['best']:.4f}", flush=True)
                break
    torch.save({"p": model.p.state_dict(), "opt": opt.state_dict(), "st": st}, a.ckpt)
    print(f"saved {a.ckpt} at it {st['it']} of {len(order)}", flush=True)
    return 0


def eval_dir(a) -> int:
    """Held-out acceptance of a drafter directory as exported (q8_0 dense round trip included)."""
    dev = torch.device(a.device)
    a.init = a.eval
    model, teacher, emb = setup(a, dev)
    held = list(transcripts(a.data, "heldout"))[: a.n_held]
    acc, ch = evaluate(model, teacher, emb, held, a, dev)
    print(report(f"{a.eval} held-out ({len(held)}):", acc, ch))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--data", type=Path, nargs="+", help="gen_data.py dump directories (index.jsonl)")
    ap.add_argument("--init", type=Path, default=ref.DATA / "mtp/rt-basalt-q2b")
    ap.add_argument("--out", type=Path, default=ref.DATA / "mtp/rt-sd")
    ap.add_argument("--eval", type=Path, help="only measure this drafter directory on the held-out split")
    ap.add_argument("--ckpt", type=Path, default=HERE / "data/ckpt.pt")
    ap.add_argument("--vocab", type=Path, default=ref.DATA / "draft_vocab_agent.bin")
    ap.add_argument("--gguf", type=Path, default=ref.GGUF)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--decay", type=float, default=0.6)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=3072)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--n-held", type=int, default=40)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--tv", type=float, default=0.0, help="weight of a total-variation term next to the KL")
    ap.add_argument("--train-scales", action="store_true", help="also train the routed experts' Q2_0 scales")
    ap.add_argument("--scale-lr", type=float, default=1e-5)
    ap.add_argument("--min-lr", type=float, default=0.1, help="the cosine schedule's final lr, as a fraction of --lr")
    ap.add_argument("--minutes", type=float, default=15)
    a = ap.parse_args()
    if a.dry:
        return dry(a.steps)
    return eval_dir(a) if a.eval else train(a)


if __name__ == "__main__":
    sys.exit(main())
