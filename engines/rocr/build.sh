#!/usr/bin/env bash
# Rebuild ROCm 7.2.4's runtime (libhsa-runtime64, Arch's hsa-rocr) with our BlitSdma fix, in user space.
# Usage: engines/rocr/build.sh [--force]
# Env: LCT_HOME (default <repo>/tmp), ROCM_PATH (default /opt/rocm), JOBS (default 8).
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LCT_HOME=${LCT_HOME:-$(cd "$here/../.." && pwd)/tmp}
ROCM_PATH=${ROCM_PATH:-/opt/rocm}
JOBS=${JOBS:-8}
url=https://github.com/ROCm/rocm-systems.git
commit=97f5574fe2fdc7bef44fb01545347912ee9f1779   # tag rocm-7.2.4
patch=$here/patches/rocr-7.2.4-blit-sdma-oob.patch

die() { echo "build.sh: $*" >&2; exit 1; }

force=0
case ${1:-} in
  "") ;;
  --force) force=1 ;;
  *) die "unknown argument: $1" ;;
esac

out=$LCT_HOME/rocr-7.2.4-fix
if [[ -e $out/lib/libhsa-runtime64.so.1 ]] && ((!force)); then
  echo "$out exists; rerun with --force to rebuild it"
  exit 0
fi

src=$LCT_HOME/rocr-7.2.4-fix.src
if [[ ! -d $src/.git ]]; then
  git init -q "$src"
  git -C "$src" remote add origin "$url"
  git -C "$src" sparse-checkout set projects/rocr-runtime
fi
git -C "$src" cat-file -e "$commit^{commit}" 2>/dev/null ||
  git -C "$src" fetch -q --depth 1 --filter=blob:none origin "$commit"
git -C "$src" -c advice.detachedHead=false checkout -q -f "$commit"
git -C "$src" apply "$patch"

# Arch's package flags: its libstdc++ assertions turn the over-read into an abort, so build the same way.
export CFLAGS="-march=x86-64 -mtune=generic -O2 -pipe -fno-plt -fexceptions"
export CXXFLAGS="$CFLAGS -Wp,-D_GLIBCXX_ASSERTIONS"
rm -rf "$src/build" "$out.previous"
# Set the working install aside until the new one is done: profiles and Strata configs name this path, and a failed
# rebuild must not leave them on the stock runtime without a word.
[[ -e $out ]] && mv "$out" "$out.previous"
if cmake -S "$src/projects/rocr-runtime" -B "$src/build" -G Ninja -DCMAKE_BUILD_TYPE=None \
  -DCMAKE_INSTALL_PREFIX="$out" -DCMAKE_INSTALL_LIBDIR=lib -DCMAKE_PREFIX_PATH="$ROCM_PATH" -DBUILD_SHARED_LIBS=ON &&
  ninja -C "$src/build" -j "$JOBS" && ninja -C "$src/build" install; then
  rm -rf "$out.previous"
  echo "built $out"
else
  rm -rf "$out"
  [[ -e $out.previous ]] && mv "$out.previous" "$out"
  die "the build failed; $out holds the previous build again"
fi
