#!/usr/bin/env bash
# Rebuild our Strata engine: upstream v0.2.9 + patches/, as a HIP engine, a CPU image reader and (with nvcc) the
# CUDA expert-helper plugin, installed under $LCT_HOME/strata/data/engines/ with the source tree hash in each name.
# Then render configs/ into $LCT_HOME/strata/configs/.  See README.md.
#
# usage: engines/strata/build.sh [--force]      (--force replaces outputs that differ)
# env:   LCT_HOME STRATA_SRC STRATA_DATA HIP_ARCH CUDA_ARCH CUDAHOSTCXX CUDAHOSTCC NVCC ROCM_PATH JOBS PYTHON ALLOWED_HOST
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)

UPSTREAM=https://github.com/Niko1221/Strata.git
TAG=v0.2.9
TAG_COMMIT=c62d5e3a15b40ea51c96aae9e8d36928965f53ab
LLAMA_URL=https://github.com/ggml-org/llama.cpp.git
LLAMA_COMMIT=3cf03257f219afbe7334045ff7c6a06ac68c627d   # upstream's own pin (setup.py LLAMA_CPP_COMMIT)

FORCE=0
for arg in "$@"; do
  case $arg in
    --force) FORCE=1 ;;
    -h | --help) sed -n '2,8p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

export LCT_HOME=${LCT_HOME:-$REPO/tmp}
export STRATA_SRC=${STRATA_SRC:-$LCT_HOME/strata/src}
export STRATA_DATA=${STRATA_DATA:-$LCT_HOME/strata/data}
export ROCM_PATH=${ROCM_PATH:-/opt/rocm}
HIP_ARCH=${HIP_ARCH:-gfx1100}
CUDA_ARCH=${CUDA_ARCH:-89-real}
# the CUDA build needs the system compilers: a Nix gcc's loader cannot find libcuda.so.1
CUDAHOSTCXX=${CUDAHOSTCXX:-/usr/bin/g++}
CUDAHOSTCC=${CUDAHOSTCC:-/usr/bin/gcc}
JOBS=${JOBS:-$(nproc)}
PYTHON=${PYTHON:-python3}
LLAMA=$LCT_HOME/strata/llama.cpp-$LLAMA_COMMIT
MADE=()

die() { echo "build.sh: $*" >&2; exit 1; }
git_src() { git -C "$STRATA_SRC" "$@"; }

# Copy SRC to DST.  A DST that differs needs --force.  The copy goes to a temp file renamed over DST, so a DST that
# is a symlink is replaced as a link (with --force) and its target is never written.
install_file() {
  local src=$1 dst=$2 mode=$3 tmp
  if [[ -e $dst || -L $dst ]]; then
    if cmp -s "$src" "$dst"; then
      echo "same: $dst"
      return
    fi
    ((FORCE)) || die "$dst exists and differs; --force replaces it"
  fi
  mkdir -p "$(dirname "$dst")"
  tmp=$(mktemp "$dst.XXXXXX")
  install -m "$mode" "$src" "$tmp"
  mv -fT "$tmp" "$dst"
  MADE+=("$dst")
}

# 1. The source checkout at v0.2.9 + patches/, on branch lct.
if [[ ! -d $STRATA_SRC/.git ]]; then
  [[ -e $STRATA_SRC ]] && die "$STRATA_SRC exists and is not a git checkout"
  git clone "$UPSTREAM" "$STRATA_SRC"
fi
[[ -z $(git_src status --porcelain --untracked-files=no) ]] || die "$STRATA_SRC has uncommitted changes"
git_src rev-parse -q --verify "$TAG^{commit}" >/dev/null || git_src fetch origin tag "$TAG"
[[ $(git_src rev-parse "$TAG^{commit}") == "$TAG_COMMIT" ]] || die "$TAG is not $TAG_COMMIT in $STRATA_SRC"

# The tree the patches give, worked out in a scratch index, so an up-to-date checkout is left untouched and
# ninja rebuilds nothing.
patches=("$HERE"/patches/*.patch)
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT
GIT_INDEX_FILE=$scratch/index git_src read-tree "$TAG"
for p in "${patches[@]}"; do GIT_INDEX_FILE=$scratch/index git_src apply --cached "$p"; done
TREE=$(GIT_INDEX_FILE=$scratch/index git_src write-tree)

if [[ $(git_src rev-parse -q --verify "lct^{tree}" || true) != "$TREE" ]]; then
  old=$(git_src rev-parse -q --verify lct || true)
  [[ -n $old ]] && echo "branch lct was $old; recreating it"
  git_src checkout -q -B lct "$TAG"
  git_src -c user.name=lct -c user.email=lct@localhost am -q "${patches[@]}"
elif [[ $(git_src symbolic-ref -q --short HEAD || true) != lct ]]; then
  git_src checkout -q lct
fi
[[ $(git_src rev-parse 'HEAD^{tree}') == "$TREE" ]] || die "HEAD's tree is not the patches' tree $TREE"
ID=${TREE:0:12}
echo "source: $STRATA_SRC (branch lct, tree $TREE)"

# 2. llama.cpp at upstream's pin: ggml for the engine, mtmd for the image reader.
if [[ ! -d $LLAMA/.git ]]; then
  rm -rf "$LLAMA.part"
  git init -q "$LLAMA.part"
  git -C "$LLAMA.part" fetch -q --depth 1 "$LLAMA_URL" "$LLAMA_COMMIT"
  git -C "$LLAMA.part" checkout -q FETCH_HEAD
  mv -T "$LLAMA.part" "$LLAMA"
fi
[[ $(git -C "$LLAMA" rev-parse HEAD) == "$LLAMA_COMMIT" ]] || die "$LLAMA is not at $LLAMA_COMMIT"

# 3. The .venv that serve/server.py runs in, as upstream's setup.py makes it, plus ninja and cmake for the build.
VENV=$STRATA_SRC/.venv
REQ_ID=$(cat "$STRATA_SRC/requirements.txt" "$STRATA_SRC/requirements-build.txt" | sha256sum | cut -c1-16)
if [[ $(cat "$VENV/.lct-requirements" 2>/dev/null || true) != "$REQ_ID" ]]; then
  [[ -x $VENV/bin/python ]] || "$PYTHON" -m venv "$VENV"
  "$VENV/bin/python" -m pip install -q -r "$STRATA_SRC/requirements.txt" -r "$STRATA_SRC/requirements-build.txt"
  echo "$REQ_ID" >"$VENV/.lct-requirements"
fi
# appended, so a system cmake or ninja wins; the venv's fill in what is missing
export PATH="$PATH:$VENV/bin"

configure() { [[ -f $2/build.ninja ]] || cmake -G Ninja -S "$1" -B "$2" -DCMAKE_BUILD_TYPE=Release "${@:3}"; }

# 4. The HIP engine (flags of upstream's AMD source build: setup.py's engine step).
export HIP_PLATFORM=amd HIP_COMPILER=clang HIP_RUNTIME=rocclr HIP_PATH=$ROCM_PATH
configure "$STRATA_SRC" "$STRATA_SRC/build-lct-hip" \
  -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_BUILD_TESTS=OFF \
  -DSTRATA_PREFILL_MMQ=ON -DSTRATA_Q6K_EXPERTS=ON -DSTRATA_MMQ_KQUANTS=ON \
  -DCMAKE_HIP_ARCHITECTURES="$HIP_ARCH" -DCMAKE_HIP_COMPILER="$ROCM_PATH/llvm/bin/clang++" \
  -DCMAKE_HIP_COMPILER_ROCM_ROOT="$ROCM_PATH" -DCMAKE_PREFIX_PATH="$ROCM_PATH" \
  "-DCMAKE_HIP_FLAGS=--rocm-path=$ROCM_PATH --rocm-device-lib-path=$ROCM_PATH/amdgcn/bitcode" \
  -DSTRATA_GGML_DIR="$LLAMA"
cmake --build "$STRATA_SRC/build-lct-hip" --target strata -j "$JOBS"

# 5. The image reader on the CPU (the configs set "gpu": false).
configure "$STRATA_SRC/tools/vision" "$STRATA_SRC/build-lct-vision" \
  -DLLAMA_DIR="$LLAMA" -DSTRATA_VISION_CUDA=OFF -DSTRATA_PORTABLE=OFF
cmake --build "$STRATA_SRC/build-lct-vision" --target strata-vision -j "$JOBS"

# 6. The NVIDIA helper plugin.  Its window size must match the HIP engine's (16), or the engine refuses it.
# A shared library: the static kernel libraries it links must be PIC too.
NVCC=${NVCC:-$(command -v nvcc || true)}
if [[ -n $NVCC ]]; then
  configure "$STRATA_SRC" "$STRATA_SRC/build-lct-cuda" \
    -DSTRATA_ENABLE_CUDA=ON -DSTRATA_ENABLE_HIP=OFF -DSTRATA_BUILD_TESTS=OFF \
    -DSTRATA_PREFILL_MMQ=OFF -DSTRATA_Q6K_EXPERTS=ON -DSTRATA_MMQ_KQUANTS=ON -DSTRATA_WINDOW_MAX_T=16 \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" -DCMAKE_CUDA_COMPILER="$NVCC" \
    -DCUDAToolkit_ROOT="$(dirname "$(dirname "$NVCC")")" -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DCMAKE_CUDA_HOST_COMPILER="$CUDAHOSTCXX" -DCMAKE_CXX_COMPILER="$CUDAHOSTCXX" -DCMAKE_C_COMPILER="$CUDAHOSTCC" \
    -DSTRATA_GGML_DIR="$LLAMA"
  cmake --build "$STRATA_SRC/build-lct-cuda" --target strata_nvhelper -j "$JOBS"
else
  echo "nvcc not found (set NVCC): skipping the NVIDIA helper plugin"
fi

# 7. Install, named by tree hash so older builds stay beside new ones.
ENGINES=$STRATA_DATA/engines
export ENGINE=$ENGINES/strata-0.2.9-$HIP_ARCH-lct-$ID
export VISION=$ENGINES/strata-vision-lct-$ID
export NVHELPER=$ENGINES/libstrata_nvhelper-lct-$ID.so
install_file "$STRATA_SRC/build-lct-hip/strata" "$ENGINE" 755
install_file "$STRATA_SRC/build-lct-vision/bin/strata-vision" "$VISION" 755
if [[ -n $NVCC ]]; then
  install_file "$STRATA_SRC/build-lct-cuda/libstrata_nvhelper.so" "$NVHELPER" 755
fi
# small inputs kept in git: the Swift expert profiles and the agent draft vocabulary
install_file "$REPO/bench/strata-compare/share/swift-1.5-iq3_xxs.bin" "$STRATA_DATA/profiles/swift-1.5-iq3_xxs.bin" 644
install_file "$REPO/bench/strata-compare/share/draft_vocab_agent.bin" "$STRATA_DATA/draft_vocab_agent.bin" 644
install_file "$REPO/bench/strata-compare/share/swift-1.5-mixb-learned-frozen13.bin" \
  "$STRATA_DATA/profiles/swift-1.5-mixb-learned-frozen13.bin" 644

# 8. The run configs.
render_args=()
((FORCE)) && render_args+=(--force)
"$PYTHON" -I "$HERE/render.py" "${render_args[@]}"

echo "made:"
printf '  %s\n' "${MADE[@]:-nothing new}"
