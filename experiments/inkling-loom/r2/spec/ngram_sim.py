"""Offline prompt-lookup estimate: how much of a greedy output an n-gram drafter would accept."""
import json, sys
from tokenizers import Tokenizer

tok = Tokenizer.from_file('/home/illusion/Documents/projects/llamacpp-tuner/tmp/loom/kv8w4096/tokenizer.json')
rec = [json.loads(l) for l in open(sys.argv[1])]
jobs = {j['name']: j for j in json.load(open(sys.argv[2]))}
label, name = sys.argv[3], sys.argv[4]
r = next(x for x in rec if x['label'] == label and x['name'] == name)
q = jobs[name]['question']
P = tok.encode(q).ids
O = tok.encode(r['reasoning']).ids + tok.encode(r['content']).ids
seq = P + O
C = {1: 9.5, 2: 13.2, 4: 17.7, 8: 26.2}

def ngram(i, nmax=4):
    # longest suffix (ending at i) match among earlier positions; return (len, follow start)
    best = (0, -1)
    for q in range(i - 1, -1, -1):
        m = 0
        while m < nmax and q - m >= 0 and seq[q - m] == seq[i - m]:
            m += 1
        if m > best[0]:
            best = (m, q + 1)
            if m == nmax:
                break
    return best

def run(minm, dng, p3):
    i, t, toks, ngsteps, ngtoks = len(P) - 1, 0.0, 0, 0, 0
    while i < len(seq) - 1:
        m, f = ngram(i)
        if m >= minm:
            acc = 0
            while acc < dng and f + acc <= i and i + 1 + acc < len(seq) and seq[f + acc] == seq[i + 1 + acc]:
                acc += 1
            span = 1 << (dng).bit_length()  # padded shape of dng+1
            t += C[min(span, 8)]; gained = acc + 1; ngsteps += 1; ngtoks += gained
        else:
            e = 1 + p3 + p3**2 + p3**3
            t += C[4]; gained = e
        toks += gained; i += max(1, round(gained)) if m < minm else gained
    return toks / t * 1000, ngsteps, ngtoks

print(f'{name}: prompt {len(P)} out {len(O)}')
p3 = float(sys.argv[5])
base = (1 + p3 + p3**2 + p3**3) / C[4] * 1000
print(f'dflash d3 only: {base:.1f} t/s (model)')
for minm in (2, 3, 4):
    for dng in (3, 7):
        tps, s, g = run(minm, dng, p3)
        print(f'ngram if match>={minm}, depth {dng}: {tps:.1f} t/s  ngram steps {s}, tokens from them {g} ({g/s if s else 0:.2f}/step)')
