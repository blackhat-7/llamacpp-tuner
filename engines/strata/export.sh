#!/usr/bin/env bash
# Write the checkout's commits on top of v0.2.9 back into patches/, replacing the old series.
#
# usage: engines/strata/export.sh [BRANCH]      (default lct; the checkout is $STRATA_SRC, as in build.sh)
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LCT_HOME=${LCT_HOME:-$(cd "$HERE/../.." && pwd)/tmp}
STRATA_SRC=${STRATA_SRC:-$LCT_HOME/strata/src}
BRANCH=${1:-lct}

git -C "$STRATA_SRC" rev-parse -q --verify "$BRANCH^{commit}" >/dev/null || {
  echo "export.sh: no branch $BRANCH in $STRATA_SRC" >&2
  exit 1
}
rm -f "$HERE"/patches/*.patch
# --zero-commit: the same commits export to the same files, whichever checkout they were applied in
git -C "$STRATA_SRC" format-patch -q --no-signature --zero-commit -o "$HERE/patches" "v0.2.9..$BRANCH"
patches=("$HERE"/patches/*.patch)
# keep a personal email out of git: our author line gets the GitHub noreply address
sed -i -E 's/^(From: blackhat-7) <[^>]*>$/\1 <44942105+blackhat-7@users.noreply.github.com>/' "${patches[@]}"
echo "${#patches[@]} patches in $HERE/patches (tree $(git -C "$STRATA_SRC" rev-parse "$BRANCH^{tree}"))"
