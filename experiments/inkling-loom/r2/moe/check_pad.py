#!/usr/bin/env python3
"""Checks gen_pad_test.py outputs. usage: check_pad.py OUTDIR [--neg]: exit 1 on any mismatch."""
import sys
import numpy as np
d, bad = sys.argv[1], 0
for T, nlive in ((4, 3), (8, 5), (8, 7), (2, 2)):
    g = f"{T}_{nlive}"
    io, wo = np.load(f"{d}/{g}_tk_ids_o.npy"), np.load(f"{d}/{g}_tk_w_o.npy")
    il, wl = np.load(f"{d}/{g}_tk_ids_n.npy"), np.load(f"{d}/{g}_tk_w_n.npy")
    ok = (il[:nlive] == io[:nlive]).all() and (wl[:nlive].view(np.int32) == wo[:nlive].view(np.int32)).all() \
        and (il[nlive:] == -1).all() and (wl[nlive:] == 0).all()
    print(g, "router", ok); bad += not ok
    ids = np.load(f"{d}/ids{g}.npy")
    for t, N in (("gn", 1152), ("dn", 2304), ("gq", 1152), ("dq", 2304)):
        o = np.load(f"{d}/{g}_{t}_o.npy").reshape(-1, N); n = np.load(f"{d}/{g}_{t}_n.npy").reshape(-1, N)
        live = ids >= 0
        if "--neg" in sys.argv:  # negative control: pretend every slot is live, so -1 slots must match the original
            live = np.ones_like(live)
        ok = (o[live].view(np.int32) == n[live].view(np.int32)).all() and (n[~live] == 0).all() and live.sum() > 0 and (~live).sum() + ("--neg" in sys.argv) > 0
        print(g, t, ok, f"live slots {live.sum()} dead {(~live).sum()}"); bad += not ok
sys.exit(1 if bad else 0)
