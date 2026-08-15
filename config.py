"""
config.py

YAML settings for the three scripts.

A config is built from two files: ``configs/common.yaml`` for the general
settings every model family shares, and ``configs/<family>.yaml`` for paths,
model lists, window sizes and tiling. The family file is merged on top. The
point of the split is that something like the left-censoring window then only
exists in one place -- so it will not accidentaly differ in CPM and RCM runs.

Used as::

    from config import load_config

    cfg = load_config("configs/cpm.yaml")
    cfg.echo()                      

    models = cfg["models"]
    nc = cfg.path("input_dir", model="ETH", period="Historical")
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict, Mapping

try:
    import yaml
except ImportError as exc: 
    raise ImportError(
        "PyYAML is required to read the configuration files.\n"
        "Install it with:  conda install pyyaml   (or: pip install pyyaml)"
    ) from exc


# Directory holding this file, used to resolve configs/common.yaml regardless
# of the working directory the script was launched from.
_REPO_ROOT = Path(__file__).resolve().parent
_COMMON_CONFIG = _REPO_ROOT / "configs" / "common.yaml"



# VALIDATION SCHEMA
# Keys that every merged configuration must define, with the type each must have. 
_REQUIRED = {
    "family": str,
    "models": list,
    "periods": list,
    "ws_list": list,
    "durations_h": list,
    "return_periods": list,
    "paths": dict,
    "ordinary_events": dict,
    "smev": dict,
    "uncertainty": dict,
    "output": dict,
}

_REQUIRED_PATHS = [
    "input_dir",
    "input_file",
    "output_dir",
    "staging_dir",
]

_REQUIRED_OE = [
    "var_name",
    "time_name",
    "min_rain",
    "separation_h",
    "time_res_min",
    "min_event_duration_min",
    "tolerance",
]


class ConfigError(ValueError):
    """Raised when a configuration file is missing or inconsistent."""



# MERGING
def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """Merge ``override`` onto ``base``, recursing into nested dicts.
    """
    merged = copy.deepcopy(dict(base))
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], Mapping)
            and isinstance(value, Mapping)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        raise ConfigError(f"Configuration file is empty: {path}")
    if not isinstance(data, dict):
        raise ConfigError(
            f"Configuration file must contain a mapping at the top level: {path}"
        )
    return data



# CONFIG OBJECT
class Config:
    """A merged, validated config: read-only dict access plus path()."""

    def __init__(self, data: Dict[str, Any], source: Path):
        self._data = data
        self.source = source

    # dict-like access 
    def __getitem__(self, key: str) -> Any:
        try:
            return self._data[key]
        except KeyError:
            raise KeyError(
                f"'{key}' is not defined in {self.source} or configs/common.yaml"
            ) from None

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def as_dict(self) -> Dict[str, Any]:
        """Return a deep copy of the merged settings."""
        return copy.deepcopy(self._data)

    def override(self, dotted_key: str, value: Any) -> None:
        """Set one setting by dotted key, e.g. ``"uncertainty.niter"``.

        For CLI overrides. Revalidates afterwards, so an override cannot leave
        the config in a state load_config() would have refused.
        """
        parts = dotted_key.split(".")
        node = self._data
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], dict):
                raise ConfigError(
                    f"Cannot override '{dotted_key}': '{part}' is not a "
                    f"section in the configuration."
                )
            node = node[part]
        node[parts[-1]] = value
        _validate(self._data, self.source)

    def dump(self, path: str | os.PathLike) -> None:
        """Dump the effective settings to YAML.
        """
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(self._data, fh, sort_keys=False,
                           default_flow_style=False)

    # convenience accessors 
    @property
    def family(self) -> str:
        return self._data["family"]

    @property
    def ws_suffix_for(self):
        """The _WS<n> suffix helper; ws=1 gives ""."""
        return lambda ws: "" if int(ws) == 1 else f"_WS{int(ws)}"

    def path(self, key: str, *, model: str = "", period: str = "") -> str:
        """Resolve a template under ``paths:``, e.g. "input_dir", "tas_file".

        Substitutes {MODEL}, {PERIOD} and {period_lower}.
        """
        paths = self._data["paths"]
        if key not in paths:
            raise KeyError(
                f"paths.{key} is not defined in {self.source}. "
                f"Available: {sorted(paths)}"
            )
        template = paths[key]
        try:
            return template.format(
                MODEL=model, PERIOD=period, period_lower=period.lower()
            )
        except KeyError as exc:
            raise ConfigError(
                f"paths.{key} in {self.source} uses an unknown placeholder {exc}. "
                f"Only {{MODEL}}, {{PERIOD}} and {{period_lower}} are supported."
            ) from None

    def input_file_path(self, model: str, period: str) -> str:
        """Full path to the precipitation NetCDF for one model and period."""
        return os.path.join(
            self.path("input_dir", model=model, period=period),
            self.path("input_file", model=model, period=period),
        )

    def output_dirs(self, model: str, period: str) -> Dict[str, str]:
        """Output subdirs for one model/period.
        """
        base = self.path("output_dir", model=model, period=period)
        return {
            "oe": os.path.join(base, "OE_details"),
            "quantiles": os.path.join(base, "quantiles"),
            "parameters": os.path.join(base, "parameters"),
            "ci": os.path.join(base, self._data["uncertainty"]["ci_subdir"]),
            "spatial": os.path.join(base, "Spatial_details"),
            "temp_scaling": os.path.join(base, "Temp_Scaling"),
        }

    def band_names(self) -> list:
        """Quantile-raster band names in written order (duration-major)."""
        return [
            f"{d}h-{rp}y"
            for d in self._data["durations_h"]
            for rp in self._data["return_periods"]
        ]

    # reporting 
    def echo(self, stream=None) -> None:
        """Print the effective config, common.yaml inheritance included.
        """
        import sys

        out = stream if stream is not None else sys.stdout
        print("=" * 74, file=out)
        print(f"EFFECTIVE CONFIGURATION  ({self.source})", file=out)
        print("=" * 74, file=out)
        print(yaml.safe_dump(self._data, sort_keys=False, default_flow_style=False),
              file=out, end="")
        print("=" * 74, file=out, flush=True)



# ENTRY POINT
def load_config(config_path: str | os.PathLike) -> Config:
    """Read common.yaml, merge the given family config on top, validate.
    """
    family_path = Path(config_path).expanduser()
    if not family_path.is_absolute():
        # Allow both "configs/cpm.yaml" from the repo root and a path relative
        # to the current working directory.
        candidate = _REPO_ROOT / family_path
        family_path = candidate if candidate.is_file() else family_path.resolve()

    common = _read_yaml(_COMMON_CONFIG)
    family = _read_yaml(family_path)
    merged = _deep_merge(common, family)

    _validate(merged, family_path)
    return Config(merged, family_path)


def _validate(cfg: Dict[str, Any], source: Path) -> None:
    """Sanity-check the merged config, with an error that says what to fix."""
    # required top-level keys and their types 
    for key, expected_type in _REQUIRED.items():
        if key not in cfg:
            raise ConfigError(
                f"Missing required setting '{key}'.\n"
                f"  Checked: {source} and {_COMMON_CONFIG}"
            )
        if not isinstance(cfg[key], expected_type):
            raise ConfigError(
                f"Setting '{key}' must be a {expected_type.__name__}, "
                f"got {type(cfg[key]).__name__}."
            )

    # non-empty lists
    for key in ("models", "periods", "ws_list", "durations_h", "return_periods"):
        if len(cfg[key]) == 0:
            raise ConfigError(f"Setting '{key}' must not be empty.")

    # required nested keys 
    for key in _REQUIRED_PATHS:
        if key not in cfg["paths"]:
            raise ConfigError(
                f"Missing required setting 'paths.{key}' in {source}."
            )
    for key in _REQUIRED_OE:
        if key not in cfg["ordinary_events"]:
            raise ConfigError(
                f"Missing required setting 'ordinary_events.{key}'."
            )

    # value sanity checks 
    ws_list = cfg["ws_list"]
    if any((not isinstance(w, int)) or w < 1 for w in ws_list):
        raise ConfigError(
            f"'ws_list' must contain positive integers, got {ws_list}."
        )

    durations = cfg["durations_h"]
    if any((not isinstance(d, int)) or d < 1 for d in durations):
        raise ConfigError(
            f"'durations_h' must contain positive integers, got {durations}."
        )

    time_res_min = cfg["ordinary_events"]["time_res_min"]
    for d in durations:
        if (d * 60) % time_res_min != 0:
            raise ConfigError(
                f"Duration {d} h is not a whole multiple of the data time "
                f"resolution ({time_res_min} min). The aggregation window "
                f"would not be an integer number of timesteps."
            )

    lc = cfg["smev"]["left_censoring"]
    if not (isinstance(lc, list) and len(lc) == 2):
        raise ConfigError(
            f"'smev.left_censoring' must be a two-element list, got {lc}."
        )
    if not (0.0 <= lc[0] < lc[1] <= 1.0):
        raise ConfigError(
            f"'smev.left_censoring' must satisfy 0 <= lower < upper <= 1, got {lc}."
        )

    rp = cfg["return_periods"]
    if any(r <= 1 for r in rp):
        raise ConfigError(
            f"'return_periods' must all be greater than 1 year, got {rp}. "
            f"A return period of 1 gives a zero exceedance probability."
        )

    unc = cfg["uncertainty"]
    if not isinstance(unc.get("enabled"), bool):
        raise ConfigError("'uncertainty.enabled' must be true or false.")
    if unc.get("niter", 0) < 2:
        raise ConfigError(
            f"'uncertainty.niter' must be at least 2, got {unc.get('niter')}."
        )
    ci = unc.get("ci_percentiles")
    if not (isinstance(ci, list) and len(ci) == 2 and 0 <= ci[0] < ci[1] <= 100):
        raise ConfigError(
            f"'uncertainty.ci_percentiles' must be [low, high] within "
            f"0-100 with low < high, got {ci}."
        )
