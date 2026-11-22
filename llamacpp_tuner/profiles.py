"""Profiles: names for 'lct serve' arguments, and stacks of profiles started together.

Paths under the cache dir are stored as ${LCT_HOME}/..., so the file works on any
machine and in any checkout.
"""

import json
import re
import shlex
import tomllib

from llamacpp_tuner.cache import get_cache_dir, get_profiles_path

VARIABLE = "${LCT_HOME}"


class ProfileError(Exception):
    """The profiles file is invalid, or a name is not in it."""


def _plain_cache_dir() -> str | None:
    """The cache dir if it can stand unquoted in a profile's arguments."""
    cache = str(get_cache_dir())
    return cache if shlex.quote(cache) == cache else None


def _expand(text: str) -> str:
    if VARIABLE not in text:
        return text
    if not (cache := _plain_cache_dir()):
        raise ProfileError(
            f"{VARIABLE} in {get_profiles_path()} needs a cache dir without spaces "
            f"or quotes, not {get_cache_dir()}. Set LCT_HOME to such a path."
        )
    return text.replace(VARIABLE, cache)


def _collapse(text: str) -> str:
    if not (cache := _plain_cache_dir()):
        return text
    # Whole paths only: starting a value or after a space, quote, '=' or ':'
    # (--opt=PATH, VAR=PATH, PATH:PATH), and ending at a path boundary.
    pattern = r"(?<![^\s'\"=:])" + re.escape(cache) + r"(?=[/'\"\s:]|$)"
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
