#!/usr/bin/env bash
# Recreate one of the named llama.cpp builds the profiles use, at the path they use. See README.md.
# Usage: engines/llama.cpp/build.sh NAME [--force]     NAME: b12415-vk | h2o | hip
# Env: LCT_HOME (default <repo>/tmp), HIP_ARCH (default gfx1100), ROCM_PATH (default /opt/rocm), JOBS (default 8).
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LCT_HOME=${LCT_HOME:-$(cd "$here/../.." && pwd)/tmp}
HIP_ARCH=${HIP_ARCH:-gfx1100}
ROCM_PATH=${ROCM_PATH:-/opt/rocm}
JOBS=${JOBS:-8}

die() { echo "build.sh: $*" >&2; exit 1; }

name=${1:-}
force=0
case ${2:-} in
  "") ;;
  --force) force=1 ;;
  *) die "unknown argument: $2" ;;
esac

# Prebuilt Vulkan release: unpack, check the checksum GitHub publishes for the asset.
vulkan_release() {
  local build=$1 commit=$2 sha256=$3 out=$4
  local asset=llama-b$build-bin-ubuntu-vulkan-x64.tar.gz
  if [[ -e $out ]]; then
    if "$out/llama-server" --version 2>&1 | grep -q "build $build, commit $commit"; then
      echo "$out is up to date"
      return
    fi
    ((force)) || die "$out is not release b$build; rerun with --force to replace it"
  fi
  local tmp=$out.partial
  rm -rf "$tmp"
  trap 'rm -rf "$tmp"' EXIT
  mkdir -p "$tmp"
  curl -fsSL --retry 3 -o "$tmp/$asset" "https://github.com/ggml-org/llama.cpp/releases/download/b$build/$asset"
  echo "$sha256  $tmp/$asset" | sha256sum -c --quiet
  tar -xzf "$tmp/$asset" -C "$tmp" --strip-components=1
  rm "$tmp/$asset"
  "$tmp/llama-server" --version 2>&1 | grep -q "build $build, commit $commit" || die "$asset is not commit $commit"
  rm -rf "$out"
  mv "$tmp" "$out"
  trap - EXIT
  echo "built $out"
}

# HIP build of SRC at COMMIT into SRC/build-hip. SRC may be lct's managed checkout (lct setup builds into
# SRC/build), so only build-hip is ours; the checkout moves to COMMIT only with --force. UI pins the prebuilt web UI
# version: a shallow fetch makes llama.cpp ask for UI "b1", which does not exist, and "latest" no longer fits old code.
hip_build() {
  local url=$1 commit=$2 src=$3 ui=${4:-}
  local out=$src/build-hip
  if [[ -x $out/bin/llama-server ]]; then
    local have
    have=$(sed -n 's/.*LLAMA_COMMIT = "\([0-9a-f]*\)".*/\1/p' "$out/common/build-info.cpp" 2>/dev/null || true)
    if [[ -n $have && $commit == "$have"* ]]; then
      echo "$out/bin/llama-server is up to date"
      return
    fi
    ((force)) || die "$out was built from ${have:-an unknown commit}, not $commit; rerun with --force to rebuild"
  fi
  if [[ ! -d $src/.git ]]; then
    [[ -e $src ]] && die "$src exists but is not a git checkout"
    git init -q "$src"
    git -C "$src" remote add origin "$url"
  fi
  if [[ $(git -C "$src" rev-parse HEAD 2>/dev/null) != "$commit" ]]; then
    # An existing checkout at another commit is someone else's tree; never move it silently.
    if git -C "$src" rev-parse -q --verify HEAD >/dev/null; then
      ((force)) || die "$src is at $(git -C "$src" rev-parse --short HEAD), not $commit; rerun with --force to check it out"
    fi
    git -C "$src" cat-file -e "$commit^{commit}" 2>/dev/null || git -C "$src" fetch -q --depth 1 "$url" "$commit"
    git -C "$src" -c advice.detachedHead=false checkout -q "$commit"
  fi
  rm -rf "$out"
  local ui_env=()
  [[ -n $ui ]] && ui_env=(HF_UI_VERSION="$ui")
  # GGML_HIP and GPU_TARGETS are the profiles' builds' flags (their CMakeCache.txt). --rocm-path: CMake 4.5 cannot
  # identify ROCm's clang without it.
  env "${ui_env[@]}" cmake -S "$src" -B "$out" -DCMAKE_BUILD_TYPE=Release -DGGML_HIP=ON -DGPU_TARGETS="$HIP_ARCH" \
    -DCMAKE_HIP_COMPILER="$ROCM_PATH/llvm/bin/clang++" \
    -DCMAKE_HIP_COMPILER_ROCM_ROOT="$ROCM_PATH" -DCMAKE_PREFIX_PATH="$ROCM_PATH" \
    -DCMAKE_HIP_FLAGS="--rocm-path=$ROCM_PATH"
  [[ -n ${CONFIGURE_ONLY:-} ]] && { echo "configured $out"; return; }
  cmake --build "$out" --target llama-server -j "$JOBS"
  echo "built $out/bin/llama-server"
}

case $name in
  b12415-vk)
    vulkan_release 12415 4e5116f47 be4413d83d54e062009f162c2a40056175ab291f67e1963402277215b86a0749 \
      "$LCT_HOME/llama.cpp-b12415-vk" ;;
  h2o)
    hip_build https://github.com/pseudotensor/llama.cpp 6245df86208e383bd12ffeee374c52b0c0f94a3a \
      "$LCT_HOME/llama.cpp-h2o" b11713 ;;
  hip)
    hip_build https://github.com/ggml-org/llama.cpp.git 17197474510622a3b4ea7d0909d70b606f542b96 \
      "$LCT_HOME/llama.cpp" ;;
  -h | --help) sed -n '2,4p' "$0" ;;
  *) die "usage: build.sh b12415-vk|h2o|hip [--force]" ;;
esac
