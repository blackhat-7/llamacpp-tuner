#!/usr/bin/env bash
# Rebuild our Inkling-2-Mini Loom package: the upstream package plus 8-bit KV, attention v2, a drafter window and
# our presets. See README.md.
# Usage: engines/loom/build.sh [--force] [--output DIR]
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LCT_HOME=${LCT_HOME:-$(cd "$here/../.." && pwd)/tmp}
loom_version=0.2.0
package=loom-fun/unsloth-Inkling-2-Mini-loom
revision=277f099b5003b725e099d48caa1a2c0db1f9b650
window=4096

die() { echo "build.sh: $*" >&2; exit 1; }

force=0
out=$LCT_HOME/loom/kv8w4096
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
"${py[@]}" - "$src/model.json" "$here/presets.json" "$tmp/model.json" <<'PY'
import json
import sys

card = json.load(open(sys.argv[1]))
card["presets"].update(json.load(open(sys.argv[2])))
with open(sys.argv[3], "w") as f:
    json.dump(card, f, indent=2)
PY

if [[ -e $out ]]; then
  # Files only in the old output (such as a saved copy of the card) do not make it out of date.
  if ! diff -rq "$out" "$tmp" | grep -qv "^Only in $out"; then
    echo "$out is up to date"
    exit 0
  fi
  ((force)) || die "$out differs from this build; rerun with --force to replace it"
  rm -rf "$out"
fi
mv "$tmp" "$out"
trap - EXIT
echo "built $out"
