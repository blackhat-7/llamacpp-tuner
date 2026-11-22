#!/usr/bin/env bash
# Publish the fitted MTP drafter to Hugging Face with the license files its sources require, check the upload,
# and record its exact revision in README.md. Needs a write token (it runs `hf auth login` if you have none).
# Usage: engines/strata/publish-drafter.sh [--dry-run] [DRAFTER_DIR]
#   DRAFTER_DIR defaults to $STRATA_DATA/mtp/rt-sd2-swift. --dry-run stages the upload in
#   $LCT_HOME/scratch/publish-drafter for a look and stops.
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$here/../.." && pwd)
LCT_HOME=${LCT_HOME:-$repo_root/tmp}
STRATA_DATA=${STRATA_DATA:-$LCT_HOME/strata/data}
dry=0
if [[ ${1:-} == --dry-run ]]; then dry=1; shift; fi
src=${1:-$STRATA_DATA/mtp/rt-sd2-swift}
for f in dense.bin dense.txt draft_vocab.bin experts.bin; do
  [[ -f $src/$f ]] || { echo "publish-drafter.sh: $src/$f is missing" >&2; exit 1; }
done
stage=$LCT_HOME/scratch/publish-drafter
rm -rf "$stage"
mkdir -p "$stage"
# Hard links when possible: the files are 787 MB.
for f in dense.bin dense.txt draft_vocab.bin experts.bin; do
  ln "$src/$f" "$stage/$f" 2>/dev/null || cp "$src/$f" "$stage/$f"
done

if ((!dry)); then
  uv run --project "$repo_root" hf auth whoami >/dev/null 2>&1 || uv run --project "$repo_root" hf auth login
fi

DRY=$dry STAGE=$stage README="$here/README.md" uv run --project "$repo_root" python -I - <<'PY'
import hashlib
import os
import re
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

NAME = "strata-flash-next-mtp-sd2"  # Swift's license (section 8) keeps its name out of repo names
SWIFT, QWEN = "ukisai/Swift-Qwen3.8-Flash-Next", "Qwen/Qwen3.8-Flash-Next"
stage, readme, dry = Path(os.environ["STAGE"]), Path(os.environ["README"]), os.environ["DRY"] == "1"

# Swift's license asks for its text, Qwen's, and Swift's NOTICE to travel with any derivative.
for name, (repo, file) in {
    "LICENSE": (SWIFT, "LICENSE"),
    "LICENSE-QWEN": (SWIFT, "LICENSE-QWEN"),
}.items():
    stage.joinpath(name).write_bytes(Path(hf_hub_download(repo, file)).read_bytes())
swift_notice = Path(hf_hub_download(SWIFT, "NOTICE")).read_text().strip()
stage.joinpath("NOTICE").write_text(
    "These are modified weights: an MTP draft layer whose dense part was refit to Swift by self-distillation.\n\n"
    f"{swift_notice}\n\n"
    "Copyright 2026 UkisAI. Swift Contribution licensed under the Swift Open License v1.0 (LICENSE).\n"
    "Derivative of Qwen3.8-Flash-Next, Copyright (c) 2026 Qwen, Qwen Community License 1.0 (LICENSE-QWEN).\n"
    "Starting drafter: Basalt (https://github.com/jesdga95/basalt); its code is MIT, its drafter weights are\n"
    "under the Qwen Community License 1.0.\n"
)
stage.joinpath("README.md").write_text(f"""---
license: other
license_name: swift-open-license-1.0
license_link: LICENSE
base_model: {SWIFT}
tags: [strata, mtp, speculative-decoding]
---
# Strata MTP drafter for Flash-Next, refit by self-distillation (rt-sd2)

A multi-token-prediction draft layer in the Strata engine's own format (`dense.bin`, `dense.txt`, `experts.bin`,
`draft_vocab.bin`); Strata loads it with `--mtp DIR`. Drafts only propose tokens and the main model checks every
one, so it speeds up decoding without changing the output.

- Made from Basalt's self-distilled drafter for {QWEN} (https://github.com/jesdga95/basalt), whose experts come
  from the model's own MTP block. Its dense part was refit to Swift ({SWIFT}) by self-distillation on Swift's own
  outputs. These are modified weights.
- Effect on a 7900 XTX: drafts held per round +3% offline, temperature-1 decode about +3%.
- Engine build, run configs and the fit scripts: https://github.com/blackhat-7/llamacpp-tuner
  (`engines/strata/`, `experiments/flash-next/mtpfit/`).
- License: the Swift Open License v1.0 (`LICENSE`) and the Qwen Community License 1.0 (`LICENSE-QWEN`); see
  `NOTICE`. Commercial use by an organization with gross revenue over US$1M needs a Swift Enterprise License
  from UkisAI (`LICENSE`, section 5).
""")
print(f"staged in {stage}:", ", ".join(sorted(p.name for p in stage.iterdir())))
if dry:
    raise SystemExit(0)

api = HfApi()
repo_id = f"{api.whoami()['name']}/{NAME}"
api.create_repo(repo_id, exist_ok=True)
commit = api.upload_folder(repo_id=repo_id, folder_path=stage, commit_message="Strata MTP drafter rt-sd2")
revision = commit.oid

# The Hub stores each large file's sha256; compare it with ours before recording the revision.
weights = ["dense.bin", "experts.bin", "draft_vocab.bin"]
remote = {i.path: i.lfs.sha256 for i in api.get_paths_info(repo_id, weights, revision=revision) if i.lfs}
for name in weights:
    local = hashlib.sha256(stage.joinpath(name).read_bytes()).hexdigest()
    if name in remote and remote[name] != local:
        raise SystemExit(f"{name}: the Hub has {remote[name]}, the local file is {local}")

command = (
    f"`uv run hf download {repo_id} --revision {revision} --include '*.bin' dense.txt "
    '--local-dir "$LCT_HOME/strata/data/mtp/rt-sd2-swift"`'
)
text = readme.read_text()
new = re.sub(
    r"<!-- drafter-source -->.*?<!-- /drafter-source -->",
    f"<!-- drafter-source -->Download: {command}.<!-- /drafter-source -->",
    text,
    flags=re.S,
)
readme.write_text(new)
print(f"uploaded https://huggingface.co/{repo_id} at {revision}; README.md now says how to download it")
PY
((dry)) || rm -rf "$stage"
