"""Compare per-verify-step kernel time between two loom-serve profile logs (dispatch lines)."""
import re, sys
def load(p):
    d = {}
    for l in open(p):
        m = re.search(r"dispatch (\S+)\s+count=(\d+) total=([\d.]+) (us|ms)", l)
        if m:
            t = float(m[3]) * (1000 if m[4] == "ms" else 1)
            d[m[1]] = (int(m[2]), t)
    return d
a, b = load(sys.argv[1]), load(sys.argv[2])
sa, sb = a["g_spec_accept"][0], b["g_spec_accept"][0]
print(f"verify steps: A {sa}  B {sb}")
rows = []
for k in set(a) | set(b):
    ca, ta = a.get(k, (0, 0)); cb, tb = b.get(k, (0, 0))
    rows.append((tb / sb - ta / sa, k, ta / sa, tb / sb, ca / sa, cb / sb))
rows.sort(reverse=True)
print(f"{'kernel':45s} {'A us/step':>10s} {'B us/step':>10s} {'delta':>8s} {'callsA':>7s} {'callsB':>7s}")
for d, k, x, y, ca, cb in rows[:int(sys.argv[3]) if len(sys.argv) > 3 else 30]:
    print(f"{k:45s} {x:10.1f} {y:10.1f} {d:8.1f} {ca:7.1f} {cb:7.1f}")
print("total", sum(r[2] for r in rows), sum(r[3] for r in rows))
