"""float64 reference for the test in DIR; prints max error of the original and new kernels vs the reference."""
import sys, numpy as np
d = sys.argv[1]
CB = np.array([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12], np.float64)
def ue4m3(b):
    x0 = b.astype(np.uint8).astype(np.int64); x = x0 & 127
    norm = ((x + 952) << 20).astype(np.uint32).view(np.float32).astype(np.float64)
    v = np.where(x < 8, x * 0.0009765625, norm)
    return np.where(x0 == 127, 0.0, v)
def deq(w, rows, k):
    b = w.view(np.uint8).reshape(-1, rows, k // 64, 36)
    sc = ue4m3(b[..., :4])  # [E,rows,blk,4]
    q = b[..., 4:].reshape(*b.shape[:3], 4, 8)
    lo, hi = CB[q & 15], CB[q >> 4]
    vals = np.concatenate([lo, hi], -1) * sc[..., None]  # [E,rows,blk,4,16]
    return vals.reshape(b.shape[0], rows, k)
def xdeq(xb, rows, k):
    xb = xb.view(np.uint8)
    codes = xb[:rows * k].view(np.int8).reshape(rows, k).astype(np.float64)
    meta = xb[rows * k:].view(np.float32).reshape(rows, k // 32, 2)
    return codes * np.repeat(meta[:, :, 0].astype(np.float64), 32, 1)
wgu = deq(np.load(f"{d}/wgu.npy"), 2304, 2304); wdn = deq(np.load(f"{d}/wdn.npy"), 2304, 1152)
gus = np.load(f"{d}/gus.npy").astype(np.float64)
sig = lambda z: 1 / (1 + np.exp(-z))
for T in (1, 2, 4):
    ids = np.load(f"{d}/ids{T}.npy"); xg = xdeq(np.load(f"{d}/xg{T}.npy"), T, 2304); xd = xdeq(np.load(f"{d}/xd{T}.npy"), 8 * T, 1152)
    act = np.zeros((8 * T, 1152)); ys = np.zeros((8 * T, 2304)); gmag = np.zeros((8*T,1152)); dmag = np.zeros((8*T,2304))
    for a, e in enumerate(ids):
        s = wgu[e] @ xg[a // 8] * gus[e]
        gmag[a] = (np.abs(wgu[e]) @ np.abs(xg[a // 8]) * gus[e])[:1152]
        g, u = s[:1152], s[1152:]
        act[a] = 4 * (2 * sig(g / 2) - 1) * sig(g) * 25 * (2 * sig(u * 0.08) - 1)
        ys[a] = wdn[e] @ xd[a]; dmag[a] = np.abs(wdn[e]) @ np.abs(xd[a])
    for nm, ref, mag in (("act", act, None), ("ys", ys, dmag)):
        for k in "on":
            got = np.load(f"{d}/t{T}_{nm}_{k}.npy").reshape(ref.shape).astype(np.float64)
            err = np.abs(got - ref)
            rel = (err / mag).max() if mag is not None else (err / (np.abs(ref) + 1e-3)).max()
            print(f"T={T} {nm} {'orig' if k=='o' else 'new '}: max abs err {err.max():.3g}  max err/(sum|terms|) or rel {rel:.3g}  ref max {np.abs(ref).max():.3g}")
    on = np.abs(np.load(f"{d}/t{T}_ys_o.npy") - np.load(f"{d}/t{T}_ys_n.npy")).max()
    print(f"T={T} ys orig vs new max abs diff {on:.3g}")
