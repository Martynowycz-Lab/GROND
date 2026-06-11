from __future__ import annotations

import json
import os
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_NAMES = (
    "grond.config.json",
    "grond.config.yaml",
    "grond.config.yml",
    ".grond.json",
    ".grond.yaml",
    ".grond.yml",
)

TOOL_ENV_VARS = {
    "shelxl": "SHELXL",
    "shelx2map": "SHELX2MAP",
    "shelxt": "SHELXT",
    "xds": "XDS",
    "xdsconv": "XDSCONV",
    "custom_refiner": "GROND_CUSTOM_REFINER",
}

TOOL_PATH_ENV_VARS: dict[str, str] = {}

TOOL_ALIASES = {
    "shelxl": ("shelxl", "shelxl2019"),
    "shelx2map": ("shelx2map",),
    "shelxt": ("shelxt",),
    "xds": ("xds",),
    "xdsconv": ("xdsconv",),
    "custom_refiner": (),
}


def _normalized_key(key: str) -> str:
    """Return the canonical key used for executable names in GROND configs."""

    return str(key).strip().lower().replace("-", "_")


def _expand_path_text(value: object) -> str:
    """Expand environment variables and user-home markers in a config value."""

    text = str(value or "").strip()
    if not text:
        return ""
    return os.path.expandvars(os.path.expanduser(text))


def find_tool_config(explicit: str = "", *, start: str | Path | None = None) -> Path | None:
    """Find the active GROND external-tool config file, if one exists.

    Search order is explicit `--config`, `GROND_CONFIG`, the current directory
    and its parents, then user-level config locations.  An explicit path must
    exist so typoed tool configs fail early.
    """

    if explicit:
        path = Path(explicit).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"GROND config file does not exist: {path}")
        return path.resolve()

    env_path = os.environ.get("GROND_CONFIG", "").strip()
    if env_path:
        path = Path(os.path.expandvars(os.path.expanduser(env_path)))
        if not path.exists():
            raise FileNotFoundError(f"GROND_CONFIG points to a missing file: {path}")
        return path.resolve()

    root = Path(start).expanduser() if start is not None else Path.cwd()
    if root.is_file():
        root = root.parent
    for parent in (root.resolve(), *root.resolve().parents):
        for name in DEFAULT_CONFIG_NAMES:
            candidate = parent / name
            if candidate.exists():
                return candidate.resolve()

    home = Path.home()
    for candidate in (
        home / ".config" / "grond" / "config.json",
        home / ".config" / "grond" / "config.yaml",
        home / ".grond.json",
        home / ".grond.yaml",
    ):
        if candidate.exists():
            return candidate.resolve()
    return None


def load_tool_config(path: str | Path | None) -> dict[str, Any]:
    """Load a JSON or YAML external-tool config file."""

    if path is None:
        return {}
    config_path = Path(path)
    suffix = config_path.suffix.lower()
    if suffix == ".json":
        data = json.loads(config_path.read_text())
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore[import-not-found]
        except ModuleNotFoundError as exc:
            raise RuntimeError("YAML GROND config files require PyYAML; use JSON or install grond[config].") from exc
        data = yaml.safe_load(config_path.read_text()) or {}
    else:
        raise ValueError(f"Unsupported GROND config suffix {suffix!r}; use .json, .yaml, or .yml")
    if not isinstance(data, Mapping):
        raise ValueError(f"GROND config must be a mapping/object: {config_path}")
    return dict(data)


def configured_tools(config: Mapping[str, Any] | None) -> dict[str, str]:
    """Return executable paths declared in a GROND config.

    The preferred layout is `{"tools": {"shelxl": "/path/to/shelxl"}}`.
    `executables`, `programs`, and top-level executable aliases are accepted to
    keep shared-lab config files forgiving.
    """

    config = config or {}
    tools: dict[str, str] = {}
    for section in ("tools", "executables", "programs"):
        section_data = config.get(section)
        if isinstance(section_data, Mapping):
            for key, value in section_data.items():
                normalized = _normalized_key(str(key))
                expanded = _expand_path_text(value)
                if expanded:
                    tools[normalized] = expanded
    for key, value in config.items():
        normalized = _normalized_key(str(key))
        if normalized in {"tools", "executables", "programs"}:
            continue
        if normalized in {"python", *TOOL_ENV_VARS.keys(), *TOOL_PATH_ENV_VARS.keys()}:
            expanded = _expand_path_text(value)
            if expanded:
                tools[normalized] = expanded
    return tools


def tool_env_updates(config: Mapping[str, Any] | None) -> dict[str, str]:
    """Return environment-variable updates implied by configured tool paths."""

    tools = configured_tools(config)
    return {env_name: tools[key] for key, env_name in TOOL_ENV_VARS.items() if tools.get(key)}


def resolve_path(
    name: str,
    *,
    configured: str = "",
    envvar: str = "",
    default_candidates: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Resolve one configured directory or file path without requiring executability."""

    checked: list[str] = []

    def resolve_candidate(candidate: str, source: str) -> dict[str, Any]:
        text = _expand_path_text(candidate)
        checked.append(text)
        path = Path(text)
        if path.exists():
            return {"available": True, "resolved": str(path.resolve()), "source": source}
        return {"available": False, "resolved": "", "source": source, "error": f"path does not exist: {text}"}

    env_value = os.environ.get(envvar, "") if envvar else ""
    payload: dict[str, Any] = {
        "name": name,
        "configured": _expand_path_text(configured),
        "envvar": envvar,
        "env_value": _expand_path_text(env_value),
        "checked": checked,
        "available": False,
        "resolved": "",
        "source": "",
    }

    result: dict[str, Any] | None = None
    if configured:
        result = resolve_candidate(configured, "config")
    elif env_value:
        result = resolve_candidate(env_value, f"env:{envvar}")
    else:
        for candidate in default_candidates:
            result = resolve_candidate(candidate, "default")
            if result.get("available"):
                break
    if result:
        payload.update(result)
    return payload


def resolve_executable(
    name: str,
    *,
    configured: str = "",
    envvar: str = "",
    aliases: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Resolve one external executable from config, environment, or PATH."""

    checked: list[str] = []

    def resolve_candidate(candidate: str, source: str) -> dict[str, Any]:
        text = _expand_path_text(candidate)
        checked.append(text)
        if "/" in text or text.startswith("."):
            path = Path(text)
            if path.exists():
                return {"available": True, "resolved": str(path.resolve()), "source": source}
            return {"available": False, "resolved": "", "source": source, "error": f"path does not exist: {text}"}
        found = shutil.which(text)
        if found:
            return {"available": True, "resolved": found, "source": source}
        return {"available": False, "resolved": "", "source": source, "error": f"not found on PATH: {text}"}

    env_value = os.environ.get(envvar, "") if envvar else ""
    payload: dict[str, Any] = {
        "name": name,
        "configured": _expand_path_text(configured),
        "envvar": envvar,
        "env_value": _expand_path_text(env_value),
        "aliases": list(aliases),
        "checked": checked,
        "available": False,
        "resolved": "",
        "source": "",
    }

    result: dict[str, Any] | None = None
    if configured:
        result = resolve_candidate(configured, "config")
    elif env_value:
        result = resolve_candidate(env_value, f"env:{envvar}")
    else:
        for alias in aliases:
            result = resolve_candidate(alias, "PATH")
            if result.get("available"):
                break
    if result:
        payload.update(result)
    return payload


def tool_statuses(config: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    """Report executable resolution for the external tools GROND can use."""

    tools = configured_tools(config)
    statuses: dict[str, dict[str, Any]] = {
        "python": resolve_executable("python", configured=tools.get("python", ""), aliases=("python3", "python")),
    }
    for name, envvar in TOOL_ENV_VARS.items():
        statuses[name] = resolve_executable(
            name,
            configured=tools.get(name, ""),
            envvar=envvar,
            aliases=TOOL_ALIASES.get(name, (name,)),
        )
    for name, envvar in TOOL_PATH_ENV_VARS.items():
        statuses[name] = resolve_path(name, configured=tools.get(name, ""), envvar=envvar)
    return statuses


def config_template() -> dict[str, Any]:
    """Return a commented-by-key JSON-compatible template for GROND tools."""

    return {
        "tools": {
            "python": "",
            "shelxl": "",
            "shelx2map": "",
            "shelxt": "",
            "xds": "",
            "xdsconv": "",
            "custom_refiner": "",
        },
        "endpoints": {
            "custom": {
                "command": "",
                "output_model": "",
                "note": "Optional command template using {model}, {reflections}, {out_dir}, and {cycles}.",
            },
        },
        "notes": "Use absolute paths on shared servers. Empty values mean GROND will try environment variables and PATH.",
    }


def write_config_template(path: str | Path, *, fmt: str = "json", overwrite: bool = False) -> Path:
    """Write a GROND external-tool config template."""

    out = Path(path).expanduser()
    if out.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing config: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    data = config_template()
    if fmt == "json":
        out.write_text(json.dumps(data, indent=2) + "\n")
    elif fmt == "yaml":
        lines = [
            "tools:",
            '  python: ""',
            '  shelxl: ""',
            '  shelx2map: ""',
            '  shelxt: ""',
            '  xds: ""',
            '  xdsconv: ""',
            '  custom_refiner: ""',
            "endpoints:",
            "  custom:",
            '    command: ""',
            '    output_model: ""',
            '    note: "Optional command template using {model}, {reflections}, {out_dir}, and {cycles}."',
            'notes: "Use absolute paths on shared servers. Empty values mean GROND will try environment variables and PATH."',
            "",
        ]
        out.write_text("\n".join(lines))
    else:
        raise ValueError("fmt must be 'json' or 'yaml'")
    return out.resolve()
