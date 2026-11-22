#!/usr/bin/env python3
"""One-wave MoE router top-8 for Inkling-2-Mini (128 experts): a drop-in for g_ink_moe_topk.

usage: gen_topk.py NAME [--live] > kernel.loom
Same launch ABI and the same results bit for bit: score = logistic(logit) + bias, 8 rounds of "highest score,
lowest expert id on ties", weights = prob / (sum of the 8 probs in round order + 1e-20) * 8 [* down scale].
The upstream kernel uses a 128-lane workgroup (4 waves) and 24 workgroup reductions through LDS and barriers;
here one wave holds 4 experts per lane (e = lane + 32 j) and every reduction is a subgroup reduction.
--live adds an %info argument ([tokens]x4 i32 from g_token_info; column 2 = live): padding tokens of a
power-of-two verify shape get ids -1 and weights 0, so the expert kernels can skip them. Live tokens are unchanged.
"""
import sys

NAME = sys.argv[1]
LIVE = "--live" in sys.argv[2:]
o = [f'''kernel.def export("{NAME}") @{NAME}(%tokens: index) {{
  %c1 = index.constant 1 : index
  %c32 = index.constant 32 : index
  kernel.launch.config workgroups(%tokens, %c1, %c1) workgroup_size(%c32, %c1, %c1) : index
}} launch(%tokens_arg: index, %use_dn_scale: i32, %logits: buffer, %bias: buffer, %dn_scale: buffer, %ids: buffer, %wts: buffer{", %info: buffer" if LIVE else ""}) {{
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 64)] : index
  %tok_id = kernel.workgroup.id<x> : index
  %tok = index.assume %tok_id [lt(%tok_id, %tokens)] : index
  %lane_raw = kernel.workitem.id<x> : index
  %lane = index.assume %lane_raw [range(%lane_raw, 0, 31)] : index
  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c0_i32 = scalar.constant 0 : i32
  %zero_f = scalar.constant 0.0 : f32
  %removed = scalar.constant -3.40282347e+38 : f32
  %big_id = scalar.constant 2147483647 : i32
  %minus1 = scalar.constant -1 : i32
  %lv = buffer.view %logits[%zero] : buffer -> view<[%tokens]x128xf32>
  %bv = buffer.view %bias[%zero] : buffer -> view<128xf32>
  %sv = buffer.view %dn_scale[%zero] : buffer -> view<128xf32>
  %iv = buffer.view %ids[%zero] : buffer -> view<[%tokens]x8xi32>
  %wv = buffer.view %wts[%zero] : buffer -> view<[%tokens]x8xf32>
  %is_l0 = index.cmp eq, %lane, %c0 : index
''']
if LIVE:
    o.append('''  %infv = buffer.view %info[%zero] : buffer -> view<[%tokens]x4xi32>
  %c2i = index.constant 2 : index
  %live32 = view.load %infv[%tok, %c2i] : view<[%tokens]x4xi32> -> i32
  %live = scalar.cmpi ne, %live32, %c0_i32 : i32
''')
w = o.append
for j in range(4):
    w(f'''  %cj{j} = index.constant {32 * j} : index
  %e{j}_v = index.add %lane, %cj{j} : index
  %e{j} = index.assume %e{j}_v [range(%e{j}_v, {32 * j}, {32 * j + 31})] : index
  %e32_{j} = index.cast %e{j} : index to i32
  %lgt{j} = view.load %lv[%tok, %e{j}] : view<[%tokens]x128xf32> -> f32
  %bs{j} = view.load %bv[%e{j}] : view<128xf32> -> f32
  %prob{j} = scalar.logisticf<afn> %lgt{j} : f32
  %sc{j}_0 = scalar.addf %prob{j}, %bs{j} : f32
  %mk{j}_0 = scalar.addi %minus1, %c0_i32 : i32
''')
w("  %sum_0 = scalar.addf %zero_f, %zero_f : f32\n")
for r in range(8):
    n = r + 1
    w(f'''  %lm01_{r} = scalar.maxnumf %sc0_{r}, %sc1_{r} : f32
  %lm23_{r} = scalar.maxnumf %sc2_{r}, %sc3_{r} : f32
  %lm_{r} = scalar.maxnumf %lm01_{r}, %lm23_{r} : f32
  %best_{r} = kernel.subgroup.reduce<maxnumf> %lm_{r} : f32
''')
    for j in range(4):
        w(f'''  %isb{j}_{r} = scalar.cmpf oeq, %sc{j}_{r}, %best_{r} : f32
  %cand{j}_{r} = scf.select %isb{j}_{r}, %e32_{j}, %big_id : i32
''')
    w(f'''  %cl01_{r} = scalar.minsi %cand0_{r}, %cand1_{r} : i32
  %cl23_{r} = scalar.minsi %cand2_{r}, %cand3_{r} : i32
  %cl_{r} = scalar.minsi %cl01_{r}, %cl23_{r} : i32
  %win_{r} = kernel.subgroup.reduce<minsi> %cl_{r} : i32
{"  %wid_" + str(r) + " = scf.select %live, %win_" + str(r) + ", %minus1 : i32" + chr(10) if LIVE else ""}  scf.if %is_l0 {{
    %cr_{r} = index.constant {r} : index
    view.store {"%wid_" + str(r) if LIVE else "%win_" + str(r)}, %iv[%tok, %cr_{r}] : i32, view<[%tokens]x8xi32>
  }}
  %r32_{r} = scalar.constant {r} : i32
''')
    mine = "%zero_f"
    for j in range(4):
        w(f'''  %won{j}_{r} = scalar.cmpi eq, %win_{r}, %e32_{j} : i32
  %mine{j}_{r} = scf.select %won{j}_{r}, %prob{j}, {mine} : f32
  %mk{j}_{n} = scf.select %won{j}_{r}, %r32_{r}, %mk{j}_{r} : i32
  %sc{j}_{n} = scf.select %won{j}_{r}, %removed, %sc{j}_{r} : f32
''')
        mine = f"%mine{j}_{r}"
    w(f'''  %wp_{r} = kernel.subgroup.reduce<addf> {mine} : f32
  %sum_{n} = scalar.addf %sum_{r}, %wp_{r} : f32
''')
w('''  %eps_n = scalar.constant 1e-20 : f32
  %wsum = scalar.addf %sum_8, %eps_n : f32
  %sc_k = scalar.constant 8.0 : f32
  %has_ds = scalar.cmpi ne, %use_dn_scale, %c0_i32 : i32
''')
for j in range(4):
    w(f'''  %writer{j} = scalar.cmpi sge, %mk{j}_8, %c0_i32 : i32
  scf.if %writer{j} {{
    %kb{j} = scalar.assume %mk{j}_8 [range(%mk{j}_8, 0, 7)] : i32
    %bk{j} = index.cast %kb{j} : i32 to index
    %wn{j} = scalar.divf %prob{j}, %wsum : f32
    %wns{j} = scalar.mulf %wn{j}, %sc_k : f32
    %wf{j} = scf.if %has_ds -> (f32) {{
      %ds{j} = view.load %sv[%e{j}] : view<128xf32> -> f32
      %wds{j} = scalar.mulf %wns{j}, %ds{j} : f32
      scf.yield %wds{j} : f32
    }} else {{
      scf.yield %wns{j} : f32
    }}
{"    %wl" + str(j) + " = scf.select %live, %wf" + str(j) + ", %zero_f : f32" + chr(10) if LIVE else ""}    view.store {"%wl" if LIVE else "%wf"}{j}, %wv[%tok, %bk{j}] : f32, view<[%tokens]x8xf32>
  }}
''')
w("  kernel.return\n}\n")
print("".join(o))
