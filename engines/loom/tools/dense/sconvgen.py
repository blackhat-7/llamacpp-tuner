"""Short-conv kernels of an Inkling visit (2..63 tokens), bit for bit:
- g_ink_sconv_stat2_c1024: g_ink_sconv_stat_c1024 for K and V in one launch (grid z picks).
- g_ink_kvring_commit2: g_ink_kv_prep_store_swa_ring (workgroups [0, tokens)) and g_ink_sconv_commit_cap_c1024
  for K and V (8 workgroups per span after those) in one launch. Commits run after both stats (previous launch),
  as before, so the conv history they overwrite is never read in the same launch.
- g_ink_sconv_stat_add_c2304: g_ink_sconv_stat_c2304 followed by g_ink_add(a, stat) per element."""

import re

from ctrlgen import kernel_func, multi


def with_ids(text, ids):
    """Replace `%name = kernel.workgroup.id<d> : index` by a function parameter %wg_d."""
    for d in ids:
        text = re.sub(
            rf"(%\w+) = kernel\.workgroup\.id<{d}> : index",
            rf"\1 = index.assume %wg_{d} [range(%wg_{d}, 0, 1048575)] : index",
            text,
        )
    return text


def module(src):
    out = []
    t, ps = kernel_func(src, "g_ink_sconv_stat_c1024", "g_ink_sconv_stat2_body")
    out += [
        t,
        multi(
            "g_ink_sconv_stat2_c1024",
            "g_ink_sconv_stat2_body",
            ps,
            2,
            "z",
            "%c256 = index.constant 256 : index\n  %cb = index.constant 4 : index\n"
            "  kernel.launch.config workgroups(%tokens, %cb, %cP) workgroup_size(%c256, %c1, %c1) : index",
        ),
    ]
    kv, kps = kernel_func(src, "g_ink_kv_prep_store_swa_ring", "g_ink_kvring_body")
    cm, cps = kernel_func(src, "g_ink_sconv_commit_cap_c1024", "g_ink_commit_body")
    kv = with_ids(kv, "x").replace(
        "(%tokens_arg: index,", "(%wg_x: index, %tokens_arg: index,", 1
    )
    cm = with_ids(cm, "xy").replace(
        "(%tokens_arg: index,", "(%wg_x: index, %wg_y: index, %tokens_arg: index,", 1
    )
    out += [
        kv,
        cm,
        """kernel.def export("g_ink_kvring_commit2") @g_ink_kvring_commit2(%tokens: index, %spans: index) {
  %c1 = index.constant 1 : index
  %c8 = index.constant 8 : index
  %c256 = index.constant 256 : index
  %cs = index.mul %spans, %c8 : index
  %n = index.add %tokens, %cs : index
  kernel.launch.config workgroups(%n, %c1, %c1) workgroup_size(%c256, %c1, %c1) : index
} launch(%tokens_arg: index, %spans_arg: index, %swa_off_arg: index, %metadata: buffer, %info: buffer, %origins: buffer, %k_in: buffer, %v_in: buffer, %arena: buffer, %layer_arg: index, %off_k_arg: index, %off_v_arg: index, %k_raw: buffer, %v_raw: buffer, %cap: buffer) {
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %c4 = index.constant 4 : index
  %c8 = index.constant 8 : index
  %wg = kernel.workgroup.id<x> : index
  %is_kv = index.cmp ult, %wg, %tokens : index
  scf.if %is_kv {
    func.call @g_ink_kvring_body(%wg, %tokens_arg, %spans_arg, %swa_off_arg, %metadata, %info, %origins, %k_in, %v_in, %arena) : (index, index, index, index, buffer, buffer, buffer, buffer, buffer, buffer)
  } else {
    %c = index.sub %wg, %tokens : index
    %span = index.div %c, %c8 : index
    %r = index.rem %c, %c8 : index
    %is_v = index.cmp uge, %r, %c4 : index
    %cb = index.rem %r, %c4 : index
    scf.if %is_v {
      func.call @g_ink_commit_body(%span, %cb, %tokens_arg, %spans_arg, %layer_arg, %off_v_arg, %metadata, %origins, %arena, %v_raw, %cap) : (index, index, index, index, index, index, buffer, buffer, buffer, buffer, buffer)
    } else {
      func.call @g_ink_commit_body(%span, %cb, %tokens_arg, %spans_arg, %layer_arg, %off_k_arg, %metadata, %origins, %arena, %k_raw, %cap) : (index, index, index, index, index, index, buffer, buffer, buffer, buffer, buffer)
    }
  }
  kernel.return
}
""",
    ]
    # stat + add: the stat body with its two stores replaced by stores of a + value.
    st, sps = kernel_func(src, "g_ink_sconv_stat_c2304", "g_ink_sconv_stat_add_body")
    st = st.replace(
        "%weight: buffer, %out: buffer)",
        "%weight: buffer, %a: buffer, %out: buffer)",
        1,
    )
    st = st.replace(
        "view.store %y, %out_view[%token, %ch] : f32, view<[%tokens]x2304xf32>",
        "%a_view = buffer.view %a[%zero] : buffer -> view<[%tokens]x2304xf32>\n"
        "    %ay = view.load %a_view[%token, %ch] : view<[%tokens]x2304xf32> -> f32\n"
        "    %sy = scalar.addf %ay, %y : f32\n"
        "    view.store %sy, %out_view[%token, %ch] : f32, view<[%tokens]x2304xf32>",
        1,
    )
    st = st.replace(
        "view.store %zero_f, %out_view[%token, %chz] : f32, view<[%tokens]x2304xf32>",
        "%a_view = buffer.view %a[%zero] : buffer -> view<[%tokens]x2304xf32>\n"
        "      %az = view.load %a_view[%token, %chz] : view<[%tokens]x2304xf32> -> f32\n"
        "      %sz = scalar.addf %az, %zero_f : f32\n"
        "      view.store %sz, %out_view[%token, %chz] : f32, view<[%tokens]x2304xf32>",
        1,
    )
    assert st.count("%a_view = buffer.view") == 2
    out += [
        st,
        """kernel.def export("g_ink_sconv_stat_add_c2304") @g_ink_sconv_stat_add_c2304(%tokens: index) {
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  %cblocks = index.constant 9 : index
  kernel.launch.config workgroups(%tokens, %cblocks, %c1) workgroup_size(%c256, %c1, %c1) : index
} launch(%tokens_arg: index, %spans_arg: index, %layer_arg: index, %chan_off_arg: index, %metadata: buffer, %info: buffer, %origins: buffer, %arena: buffer, %cur: buffer, %weight: buffer, %a: buffer, %out: buffer) {
  func.call @g_ink_sconv_stat_add_body(%tokens_arg, %spans_arg, %layer_arg, %chan_off_arg, %metadata, %info, %origins, %arena, %cur, %weight, %a, %out) : (index, index, index, index, buffer, buffer, buffer, buffer, buffer, buffer, buffer, buffer)
  kernel.return
}
""",
    ]
    return "\n".join(out)
