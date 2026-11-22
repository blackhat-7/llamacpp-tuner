#!/usr/bin/env python3
"""Compare the q8 attention rows written by test_e.loom: old kernels vs new (and a shifted negative control).
usage: check_e.py DIR"""
import glob, os, sys
import numpy as np

def rows(path):
    b = np.load(path).view(np.uint8)
    n = b.size // 3840
    codes = b[: n * 3072].view(np.int8).reshape(n, 96, 32).astype(np.float32)
    meta = b[n * 3072 :].view(np.float32).reshape(n, 96, 2)
    return codes * meta[:, :, :1], meta

bad = 0
for old in sorted(glob.glob(os.path.join(sys.argv[1], "e*_old.npy"))):
    a, ma = rows(old)
    b, mb = rows(old.replace("_old", "_new"))
    err = np.abs(a - b).max(axis=2) / np.maximum(ma[:, :, 0], 1e-30)  # in units of the group's q8 step
    ok = err.max() <= 1.01 and np.abs(ma[:, :, 0] - mb[:, :, 0]).max() <= 1e-3 * np.abs(ma[:, :, 0]).max()
    # negative control: the same rows against another head's rows must fail
    neg = np.abs(a - np.roll(a, 1, axis=1)).max() / np.abs(ma[:, :, 0]).max()
    print(f"{os.path.basename(old)[:-8]:12s} max err {err.max():.3f} steps, mean {err.mean():.4f}; neg control {neg:.1f} steps {'OK' if ok and neg > 2 else 'FAIL'}")
    bad += not (ok and neg > 2)
sys.exit(bad)
