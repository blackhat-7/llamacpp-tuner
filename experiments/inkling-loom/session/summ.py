import sys, json
for l in sys.stdin:
    try: r = json.loads(l)
    except Exception: print(l[:300]); continue
    md = r.get('metrics_delta', {}); p = sum(v for k, v in md.items() if 'draft_proposed' in k); a = sum(v for k, v in md.items() if 'draft_accepted' in k)
    print(r.get('name', r.get('event')), r.get('prompt_tokens'), r.get('gen_tokens'), r.get('ttft_s'), r.get('decode_tps'), 'acc=%.2f' % (a / p) if p else '', r.get('peak_vram_gb', r.get('vram_gb')), flush=True)
