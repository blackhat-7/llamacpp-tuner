#!/usr/bin/env bash
# Rebuild our Inkling-2-Mini Loom package: the upstream package plus 8-bit KV, attention v2, a drafter window,
# round 2 (attention, prefill, dense and MoE kernels, dynamic draft spans) and our presets, then the same
# package with image input (<output>v). See README.md.
# Usage: engines/loom/build.sh [--force] [--output DIR]
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LCT_HOME=${LCT_HOME:-$(cd "$here/../.." && pwd)/tmp}
loom_version=0.2.0
package=loom-fun/unsloth-Inkling-2-Mini-loom
revision=277f099b5003b725e099d48caa1a2c0db1f9b650
window=4096
mmproj_revision=ea0c79b2ecd73669e774c8519b79bfabfddbc027
mmproj_f16_sha256=3bff75ffeb2d6e0371ba7bd42e30d26a7d98aea681fa44b03b5a20891cfa2c5f
mmproj_q8_sha256=313c8ae7ebd3dc1740c6b3eaa5b21857a2182c2e2c3569a41b4f4972169d7525

die() { echo "build.sh: $*" >&2; exit 1; }

force=0
out=$LCT_HOME/loom/kv8-r2
while (($#)); do
  case $1 in
    --force) force=1 ;;
    --output) out=${2:?--output needs a directory}; shift ;;
    -h | --help) sed -n '2,4p' "$0"; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
  shift
done

venv=$LCT_HOME/loom/venv
[[ -x $venv/bin/python ]] || uv venv --python 3.12 "$venv"
[[ -x $venv/bin/loom-serve ]] || uv pip install --python "$venv/bin/python" "hrx-loom[hf]==$loom_version"
have=$("$venv/bin/python" -c 'import importlib.metadata as m; print(m.version("hrx-loom"))')
[[ $have == "$loom_version" ]] || die "$venv has hrx-loom $have, this build needs $loom_version"
py=("$venv/bin/python" -B)

src=$LCT_HOME/loom/Inkling-2-Mini
[[ -f $src/model.json ]] ||
  "$venv/bin/loom-fetch" --no-input --output "$src" "huggingface:$package@$revision"
# The card's document origin names the package revision it came from.
grep -q "$package/resolve/$revision/" "$src/model.json" || die "$src is not $package at $revision"

mkdir -p "$(dirname "$out")"
tmp=$out.partial
rm -rf "$tmp"
trap 'rm -rf "$tmp"' EXIT
mkdir -p "$tmp/program" "$tmp/parameter"
ln "$src"/parameter/*.gguf "$tmp/parameter/"
cp "$src"/{head_weights.loom,llm_config.json,tokenizer.json,weights.loom} "$tmp/"
cp "$src"/program/{control,prepare,service,storage}.loom "$tmp/program/"

patch --quiet --fuzz=0 -p1 -d "$tmp" <"$here/patches/storage-kv8.diff"
"${py[@]}" "$here/tools/transform.py" "$src/program/llm.loom" "$tmp/program/llm.loom"
"${py[@]}" "$here/tools/dsp_window.py" "$tmp/program/llm.loom" "$window"
# Round 2 (2026-11-22). Order matters: each tool anchors on the text the ones before it leave.
"${py[@]}" "$here/tools/attn/attn.py" "$tmp"
"${py[@]}" "$here/tools/prefill/prefill.py" "$tmp" >/dev/null
"${py[@]}" "$here/tools/dense/dense_edit.py" "$tmp" >/dev/null
"${py[@]}" "$here/tools/moe/moe_opt.py" "$tmp" >/dev/null
"${py[@]}" "$here/tools/spec/dynspan.py" "$tmp" --lengths 2 --thresholds 90 --shift 3
"${py[@]}" - "$src/model.json" "$here/presets.json" "$tmp/model.json" <<'PY'
import json
import sys

card = json.load(open(sys.argv[1]))
card["presets"].update(json.load(open(sys.argv[2])))
with open(sys.argv[3], "w") as f:
    json.dump(card, f, indent=2)
PY

# Replace $2 (an existing output) with $1 (a fresh build) unless they match.
install() {
  local new=$1 dst=$2
  if [[ -e $dst ]]; then
    # Files only in the old output (such as a saved copy of the card) do not make it out of date.
    if ! diff -rq "$dst" "$new" | grep -qv "^Only in $dst"; then
      echo "$dst is up to date"
      rm -rf "$new"
      return
    fi
    ((force)) || die "$dst differs from this build; rerun with --force to replace it"
    rm -rf "$dst"
  fi
  mv "$new" "$dst"
  echo "built $dst"
}
install "$tmp" "$out"

# Image input (2026-11-22): ${out}v = $out + the inkling2v image component (tools/vision/vision.py). The Q8_0
# projector is quantized from the pinned F16 one; the F16 encoder (0.64 GB) does not fit beside 196K context.
mmproj_dir=$LCT_HOME/loom/vision
mmproj=$mmproj_dir/mmproj-Q8_0.gguf
if [[ ! -f $mmproj ]]; then
  f16=$("$venv/bin/hf" download unsloth/Inkling-2-Mini-GGUF mmproj-F16.gguf --revision "$mmproj_revision" | tail -n 1)
  f16=${f16#path=}
  echo "$mmproj_f16_sha256  $f16" | sha256sum --quiet -c - || die "$f16: unexpected sha256"
  mkdir -p "$mmproj_dir"
  uv run --python 3.12 --no-project --with gguf==0.19.0 --with numpy==2.5.5 \
    python -I "$here/tools/vision/quantize_mmproj.py" "$f16" "$mmproj.partial"
  mv "$mmproj.partial" "$mmproj"
fi
echo "$mmproj_q8_sha256  $mmproj" | sha256sum --quiet -c - || die "$mmproj: unexpected sha256"
rm -rf "${out}v.partial"
trap 'rm -rf "${out}v.partial"' EXIT
"${py[@]}" "$here/tools/vision/vision.py" "$out" "${out}v.partial" "$mmproj"
install "${out}v.partial" "${out}v"
trap - EXIT
