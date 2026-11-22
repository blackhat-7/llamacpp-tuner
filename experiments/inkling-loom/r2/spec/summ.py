"""Table of decode t/s per label and job (repeats joined by /) from session jsonl files."""
import json, sys, collections
rows = collections.defaultdict(dict)
jobs = []
for fn in sys.argv[1:]:
    for l in open(fn):
        d = json.loads(l)
        if d['name'] == 'warm': continue
        rows[d['label']].setdefault(d['name'], []).append(d['decode_tps'])
        if d['name'] not in jobs: jobs.append(d['name'])
print(f"{'label':34s}" + "".join(f"{j[-12:]:>13s}" for j in jobs))
for lab, r in rows.items():
    print(f"{lab[-34:]:34s}" + "".join(f"{'/'.join(str(x) for x in r[j]) if j in r else '-':>13s}" for j in jobs))
