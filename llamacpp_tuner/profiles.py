"""Profiles: names for 'lct serve' arguments, and stacks of profiles started together.

Paths under the cache dir are stored as ${LCT_HOME}/..., so the file works on any
machine and in any checkout.
"""

import json
import re
import tomllib

from llamacpp_tuner.cache import get_cache_dir, get_profiles_path

VARIABLE = "${LCT_HOME}"


class ProfileError(Exception):
    """The profiles file is invalid, or a name is not in it."""


def _expand(text: str) -> str:
    return text.replace(VARIABLE, str(get_cache_dir()))


def _collapse(text: str) -> str:
    # Only whole path components, so a sibling such as <cache>2/ stays as it is.
    pattern = re.escape(str(get_cache_dir())) + r"(?=[/'\"\s]|$)"
    return re.sub(pattern, VARIABLE, text)


def _read() -> dict:
    path = get_profiles_path()
    if not path.is_file():
        return {}
    try:
        return tomllib.loads(path.read_text())
    except tomllib.TOMLDecodeError as error:
        raise ProfileError(f"Invalid {path}: {error}") from error


def load_profiles() -> dict[str, str]:
    """Read the profiles, each mapping a name to 'serve' arguments."""
    profiles = _read()
    profiles.pop("stacks", None)
    for name, value in profiles.items():
        if not isinstance(value, str):
            raise ProfileError(
                f"Alias '{name}' in {get_profiles_path()} must be a string of serve arguments."
            )
    return {name: _expand(value) for name, value in profiles.items()}


def load_stacks() -> dict[str, list[str]]:
    """Read the [stacks] table: a name for a set of profiles started together."""
    stacks = _read().get("stacks", {})
    for name, members in stacks.items():
        if not (isinstance(members, list) and all(isinstance(m, str) for m in members)):
            raise ProfileError(
                f"Stack '{name}' in {get_profiles_path()} must be a list of profile names."
            )
    return stacks


def write_profiles(profiles: dict[str, str]) -> None:
    """Replace every profile, keeping stacks. Rewriting the file drops its comments."""
    stacks = load_stacks()
    lines = [
        f"{json.dumps(key)} = {json.dumps(_collapse(value))}\n"
        for key, value in profiles.items()
    ]
    if stacks:
        lines.append("\n[stacks]\n")
        lines += [
            f"{json.dumps(key)} = {json.dumps(members)}\n"
            for key, members in stacks.items()
        ]
    get_profiles_path().write_text("".join(lines))


def expand_names(names: tuple[str, ...]) -> list[str]:
    """Turn stack and profile names into profile names, in order, without repeats."""
    profiles, stacks = load_profiles(), load_stacks()
    found: list[str] = []
    for name in names:
        for profile in stacks.get(name, [name]):
            if profile not in profiles:
                raise ProfileError(f"Unknown profile or stack: {profile}")
            if profile not in found:
                found.append(profile)
    return found
