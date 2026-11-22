"""Render the Strata run configs: configs/*.json with ${NAME} filled in, into $LCT_HOME/strata/configs/.

The names come from the environment (build.sh sets them): LCT_HOME, STRATA_SRC, STRATA_DATA, ENGINE, VISION,
NVHELPER, ROCM_PATH (default /opt/rocm), ALLOWED_HOST (default localhost: an extra Host name the server answers to).
A file that exists and differs is kept unless --force. A symlink is always kept and never written through.

usage: python3 -I render.py [--force]
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from string import Template

HERE = Path(__file__).resolve().parent
REQUIRED = ("LCT_HOME", "STRATA_SRC", "STRATA_DATA", "ENGINE", "VISION", "NVHELPER")


def fill(value: object, names: dict[str, str]) -> object:
    if isinstance(value, dict):
        return {k: fill(v, names) for k, v in value.items()}
    if isinstance(value, list):
        return [fill(v, names) for v in value]
    if isinstance(value, str):
        return Template(value).substitute(names)
    return value


def write(path: Path, text: str, force: bool) -> str:
    # a link points at a config someone keeps elsewhere (this machine's live ones): never replace or write through it
    if path.is_symlink():
        return f"kept (a link to {os.readlink(path)}; remove it to render here)"
    if path.exists():
        if path.read_text() == text:
            return "same"
        if not force:
            return "kept (differs; --force replaces it)"
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as tmp:
        tmp.write(text)
    os.chmod(tmp.name, 0o644)
    os.replace(tmp.name, path)
    return "made"


def main() -> None:
    force = sys.argv[1:] == ["--force"]
    if sys.argv[1:] not in ([], ["--force"]):
        sys.exit(__doc__)
    missing = [n for n in REQUIRED if not os.environ.get(n)]
    if missing:
        sys.exit(f"render.py: set {', '.join(missing)} (build.sh does)")
    names = {n: os.environ[n] for n in REQUIRED}
    names["ROCM_PATH"] = os.environ.get("ROCM_PATH", "/opt/rocm")
    names["ALLOWED_HOST"] = os.environ.get("ALLOWED_HOST", "localhost")

    out = Path(names["LCT_HOME"]) / "strata" / "configs"
    out.mkdir(parents=True, exist_ok=True)
    for template in sorted((HERE / "configs").glob("*.json")):
        text = (
            json.dumps(fill(json.loads(template.read_text()), names), indent=2) + "\n"
        )
        path = out / template.name
        print(f"{write(path, text, force)}: {path}")


if __name__ == "__main__":
    main()
