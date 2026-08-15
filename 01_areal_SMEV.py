"""
01_areal_SMEV.py

Apply SMEV (Simplified Metastatistical Extreme Value) to gridded NetCDF
precipitation, using mean-areal series built from a ``ws x ws`` sliding
moving-window average.

For each combination of (MODEL, PERIOD, window_size ws) the script:

  1. Reads the NetCDF and applies the ``ws x ws`` moving window.
     ws = 1 leaves the data on its native grid. A window containing any NaN
     gives NaN, so cells near the domain edge or the coast are never averaged
     from partial data.
  2. Extracts Ordinary Events (OE) and Annual Maxima (AMS) per cell and per
     duration. OEs are independent storms, separated by more than
     ``separation_h`` dry hours.
  3. Fits the SMEV Weibull on the left-censored OE of each cell and duration,
     and turns each fit into return levels.
  4. Optionally bootstraps confidence intervals on those return levels.

Run it as::

    python 01_areal_SMEV.py --config configs/cpm.yaml
    python 01_areal_SMEV.py --config configs/rcm.yaml

Everything is read from the YAML; nothing here needs editing. CPM and RCM
differ only in paths, model lists, window sizes and tiling -- the shared
method settings are in ``configs/common.yaml`` so the two remain consistent.

Outputs, written under ``paths.output_dir`` ({ws} is ``_WS<n>``, dropped when
ws = 1)::

    OE_details/OE_{MODEL}_{PERIOD}{ws}.parquet      ordinary events
    OE_details/AMS_{MODEL}_{PERIOD}{ws}.parquet     annual maxima

    parameters/{MODEL}_{PERIOD}_Shape{ws}.tif       Weibull shape
    parameters/{MODEL}_{PERIOD}_Scale{ws}.tif       Weibull scale
    parameters/{MODEL}_{PERIOD}_N{ws}.tif           mean number of OE per year
    parameters/{MODEL}_{PERIOD}_mAM{ws}.tif         mean annual maximum
        5 bands, one per duration, in ``durations_h`` order.

    quantiles/{MODEL}_{PERIOD}{ws}.tif              return levels
        30 bands, duration-major:
        1h-2y, 1h-5y, ... 1h-100y, 3h-2y, ... 24h-100y.

    quantiles_CIs/{MODEL}_{PERIOD}_CI_low{ws}.tif   lower bound
    quantiles_CIs/{MODEL}_{PERIOD}_CI_high{ws}.tif  upper bound
        Same geometry and band order as quantiles/; only if the bootstrap ran.

Uncertainty via bootstrap:
It is optional here and can be run two ways. 
In-pass, by setting ``uncertainty.enabled: true``, it runs inside
the main pass while each cell's OEs are still in memory. Alone, via
``--uncertainty-only``, it recomputes intervals from the OE parquet of an
earlier run without re-reading the NetCDF.

Both call the same function in ``bootstrap.py`` with the same per-cell seed,
so they agree to float32 rounding.

Notes:
* ``smev.py`` here is derived from pyTENAX v0.1.2 (MIT); see THIRD_PARTY_NOTICES.md.
* And SMEV is applied directly to the extracted OE values: we bypass the event
  extraction inside ``smev.py`` and use only
  ``SMEV.estimate_smev_parameters`` and ``SMEV.smev_return_values``. The
  extraction lives here because it has to run cell by cell over a grid, not
  on a single station series.
"""

import os
# Prevent oversubscription with multi-thread BLAS
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys
import time
import collections
import math
import shutil
import atexit
import argparse
import tempfile
import warnings
import traceback
import multiprocessing as mp
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import xarray as xr
import cftime

from scipy.signal import convolve2d

import pyarrow as pa
import pyarrow.parquet as pq

import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS

# SMEV class, derived from pyTENAX; see THIRD_PARTY_NOTICES.md
from smev import SMEV

# Repository modules
from config import load_config
from bootstrap import bootstrap_cell, cell_rng


# GDAL/PROJ (optional; helps some Windows envs)
_gdal_data = os.path.join(sys.prefix, "Library", "share", "gdal")
_proj_data = os.path.join(sys.prefix, "Library", "share", "proj")
if os.path.isdir(_gdal_data):
    os.environ.setdefault("GDAL_DATA", _gdal_data)
if os.path.isdir(_proj_data):
    os.environ.setdefault("PROJ_LIB", _proj_data)


# CONFIGURATION

# No paths or model lists live in this file; they all come from the YAML given on the
# command line. The path is also pushed into the environment because we
# use "spawn": workers re-import this module from scratch instead of
# inheriting the parent's memory, so each one has to find the config itself.

CONFIG_ENV_VAR = "AREAL_SMEV_CONFIG"

# Filled in by _apply_config(). Left as None so the module still imports with
# no config present (the tests rely on that); main() refuses to run until one has been applied.
CFG = None


def _apply_config(cfg) -> None:
    """Bind one config to this module's settings.
    """
    g = globals()
    g["CFG"] = cfg

    oe   = cfg["ordinary_events"]
    out  = cfg["output"]
    unc  = cfg["uncertainty"]
    tile = cfg["tiling"]
    par  = cfg["parallelism"]

    # run plan 
    g["PERIODS"]        = list(cfg["periods"])
    g["MODELS"]         = list(cfg["models"])
    g["DURATIONS_H"]    = list(cfg["durations_h"])
    g["WS_LIST"]        = list(cfg["ws_list"])
    g["RETURN_PERIODS"] = list(cfg["return_periods"])

    # ordinary-event extraction and SMEV fit 
    g["VAR_NAME"]               = oe["var_name"]
    g["TIME_NAME"]              = oe["time_name"]
    g["MIN_RAIN"]               = float(oe["min_rain"])
    g["SEPARATION_H"]           = int(oe["separation_h"])
    g["TIME_RES_MIN"]           = int(oe["time_res_min"])
    g["MIN_EVENT_DURATION_MIN"] = int(oe["min_event_duration_min"])
    g["TOLERANCE"]              = float(oe["tolerance"])
    g["LEFT_CENSORING"]         = list(cfg["smev"]["left_censoring"])

    # bootstrap 
    g["UNC_ENABLED"]        = bool(unc["enabled"])
    g["UNC_NITER"]          = int(unc["niter"])
    g["UNC_CI_PERCENTILES"] = list(unc["ci_percentiles"])
    g["UNC_RANDOM_SEED"]    = int(unc["random_seed"])

    # output containers 
    g["PARQUET_COMPRESSION"] = out["parquet"]["compression"]
    g["PARQUET_ZSTD_LEVEL"]  = int(out["parquet"]["zstd_level"])
    g["TIF_DTYPE"]           = out["geotiff"]["dtype"]
    g["TIF_COMPRESS"]        = out["geotiff"]["compress"]
    g["TIF_NODATA"]          = float(out["geotiff"]["nodata"])

    # tiling and parallelism 
    g["NY_TILE"]         = int(tile["ny_tile"])
    g["NX_TILE"]         = int(tile["nx_tile"])
    g["N_INNER_WORKERS"] = int(par["n_inner_workers"])
    g["N_OUTER_JOBS"]    = int(par["n_outer_jobs"])
    g["STAGING_DIR"]     = cfg["paths"]["staging_dir"]
    g["SPATIAL_SUBSET"]  = cfg.get("spatial_subset")


def _require_config():
    """Active config, or a message saying how to supply one."""
    if CFG is None:
        raise SystemExit(
            "No configuration loaded.\n"
            "Run this script with --config, for example:\n"
            "    python 01_areal_SMEV.py --config configs/cpm.yaml"
        )
    return CFG


# Picked up automatically by spawned workers on re-import.
if os.environ.get(CONFIG_ENV_VAR):
    _apply_config(load_config(os.environ[CONFIG_ENV_VAR]))



# Fixed settings:
# "spawn" : the inner pool is created inside an
# outer subprocess, and NetCDF/HDF5/GDAL handles must not survive a fork.
MP_START_METHOD = "spawn"

# Parquet column order.
OE_COLS  = ["lat", "lon", "duration_h", "year", "OE", "OE_i", "From_i", "To_i"]
AMS_COLS = ["lat", "lon", "duration_h", "year", "OE", "OE_i", "From_i", "To_i"]


# Path resolvers
def input_path_for(model: str, period: str) -> str:
    """NetCDF input path for (MODEL, PERIOD), from the paths templates."""
    return _require_config().input_file_path(model, period)


def output_dirs_for(period: str, model: str) -> Tuple[str, str, str]:
    """(OE_details, quantiles, parameters) dirs.
    """
    dirs = _require_config().output_dirs(model, period)
    return dirs["oe"], dirs["quantiles"], dirs["parameters"]


def ci_dir_for(period: str, model: str) -> str:
    """Where the bootstrap CI rasters go."""
    return _require_config().output_dirs(model, period)["ci"]


def ws_suffix(ws: int) -> str:
    return "" if int(ws) == 1 else f"_WS{int(ws)}"


# Simple timing helper
def _fmt_s(x: float) -> str:
    return f"{x:8.2f} s  ({x/60:6.2f} min)"


# MOVING WINDOW AVERAGE 
def moving_window_average_strict_2d(data_2d: np.ndarray, window_size: int) -> np.ndarray:
    """Centered moving-window average; output is NaN if ANY input in the window is NaN."""
    valid_mask = ~np.isnan(data_2d)
    data_filled = np.where(valid_mask, data_2d, 0.0)
    kernel = np.ones((window_size, window_size), dtype=float)
    filtered_sum = convolve2d(data_filled, kernel, mode="same", boundary="fill", fillvalue=0)
    valid_count  = convolve2d(valid_mask.astype(float), kernel, mode="same", boundary="fill", fillvalue=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(valid_count == kernel.sum(), filtered_sum / valid_count, np.nan)
    return result


def moving_window_average_strict_3d(series: np.ndarray, window_size: int) -> np.ndarray:
    T, Ny, Nx = series.shape
    out = np.empty((T, Ny, Nx), dtype=series.dtype)
    for t in range(T):
        out[t, :, :] = moving_window_average_strict_2d(
            series[t, :, :], window_size
        ).astype(series.dtype, copy=False)
    return out



# TIME + STORM LOGIC HELPERS 
NS_PER_H = np.timedelta64(1, "h") / np.timedelta64(1, "ns")

def hours_to_ns(h: float) -> int:
    return int(h * NS_PER_H)

def ensure_datetime64_ns(arr) -> np.ndarray:
    try:
        if hasattr(arr, "to_datetimeindex"):
            return arr.to_datetimeindex().values.astype("datetime64[ns]")
    except Exception:
        pass
    return pd.to_datetime(arr).values.astype("datetime64[ns]")

def parse_units_origin(units: str) -> np.datetime64:
    try:
        base = units.split("since", 1)[1].strip()
        return np.datetime64(pd.to_datetime(base), "ns")
    except Exception:
        return np.datetime64("1970-01-01T00:00:00", "ns")

def unit_to_seconds_factor(unit: str) -> float:
    u = unit.strip().lower()
    return {
        "seconds": 1.0, "second": 1.0, "secs": 1.0, "sec": 1.0, "s": 1.0,
        "minutes": 60.0, "minute": 60.0, "mins": 60.0, "min": 60.0,
        "hours": 3600.0, "hour": 3600.0, "hrs": 3600.0, "hr": 3600.0, "h": 3600.0,
        "days": 86400.0, "day": 86400.0, "d": 86400.0
    }.get(u, 1.0)

# Gregorian-family CF calendar names that pandas handles natively.
GREGORIAN_CALENDARS = {
    "", "gregorian", "standard", "proleptic_gregorian", "julian",
}


def _build_axis_from_cftime(time_da: xr.DataArray
                            ) -> Tuple[np.ndarray, np.ndarray]:
    """Synthetic monotonic datetime64[ns] axis + true civil years from cftime.

    Used for any non-gregorian CF calendar.
    The returned datetime64 axis is *fictitious* -- its absolute values
    are not meaningful, but the spacing in seconds matches the actual
    model timestep, which is all that the storm-separation /
    near-edge / gap-removal logic in SMEV needs.

    The true civil year of each timestep is taken directly from the
    cftime object and returned as ``years_override`` so that AMS
    year-binning and ``N = events / unique_years`` remain based on the calendar.
    """
    # Note: xarray, when it decodes a time variable on open, MOVES the CF
    # attributes (``calendar``, ``units``) from ``.attrs`` into ``.encoding``.
    # So we must read from both: ``attrs`` for files opened with
    # ``decode_times=False`` (or never-decoded), ``encoding`` for the normal
    # decoded path. Otherwise a 360_day file silently looks "gregorian" to
    # the dispatch, and ``pd.to_datetime`` later trips over the cftime
    # objects in ``time_da.values``.
    attrs    = getattr(time_da, "attrs", {}) or {}
    encoding = getattr(time_da, "encoding", {}) or {}
    cal   = attrs.get("calendar") or encoding.get("calendar") or "360_day"
    units = attrs.get("units")    or encoding.get("units")    or "days since 1949-12-01"
    vals  = time_da.values
    n     = vals.shape[0]

    try:
        if np.issubdtype(getattr(vals, "dtype", object), np.number):
            dts_cf = cftime.num2date(
                vals, units=units, calendar=cal,
                only_use_cftime_datetimes=True,
            )
        else:
            dts_cf = list(vals)
    except Exception:
        dts_cf = []
    years = np.array(
        [dt.year for dt in dts_cf] if dts_cf else np.zeros(n, dtype=int),
        dtype=np.int64,
    )

    # Estimate the timestep in seconds from the time variable itself.
    dt_seconds = None
    if n > 1 and np.issubdtype(getattr(vals, "dtype", object), np.number):
        unit  = units.split("since")[0].strip().lower()
        scale = unit_to_seconds_factor(unit)
        diffs = np.diff(vals).astype(np.float64) * float(scale)
        dt_seconds = float(np.median(diffs))
    elif n > 1 and dts_cf:
        base_units = "seconds since 1900-01-01 00:00:00"
        nums = cftime.date2num(dts_cf, base_units, calendar=cal).astype(np.float64)
        dt_seconds = float(np.median(np.diff(nums)))
    if dt_seconds is None or not np.isfinite(dt_seconds) or dt_seconds <= 0:
        dt_seconds = float(TIME_RES_MIN) * 60.0

    base_dt64 = parse_units_origin(units)
    offsets_s = (np.arange(n, dtype=np.int64) * int(round(dt_seconds)))
    dates_ns  = base_dt64 + offsets_s.astype("timedelta64[s]").astype("timedelta64[ns]")
    return dates_ns.astype("datetime64[ns]"), years


def time_for_calendar(time_da: xr.DataArray
                      ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Calendar based time axis builder.

    Dispatches purely on the ``calendar`` attribute of the time
    variable, so any combination of model x calendar in the CPM 
    routes to the correct branch automatically:

      * Gregorian family (``gregorian``, ``standard``,
        ``proleptic_gregorian``, ``julian``, or missing) -> pandas
        path. Year extraction downstream uses
        ``pd.Timestamp(date).year``, which is already correct, so
        ``years_override`` is ``None``.

      * Anything else (``360_day`` e.g., for MOHC / DWD / ICTP) ->
        cftime path. A synthetic monotonic datetime64 axis is built
        and the true civil years are returned in ``years_override``.

    Returns
    -------
    dates_ns : np.ndarray[datetime64[ns]]
        Monotonic time axis.
    years_override : Optional[np.ndarray[int64]]
        True civil years per timestep when the calendar is
        non-gregorian; ``None`` otherwise.
    """
    #   1. CF calendar string. After ``xr.open_dataset(decode_times=True)``
    #      (our default), xarray moves ``calendar`` and ``units`` from
    #      ``time_da.attrs`` into ``time_da.encoding``. Read from BOTH so
    #      this dispatch works on decoded and non-decoded files alike.
    #
    #   2. Value dtype. Regardless of what the calendar string says, if
    #      ``time_da.values`` is already a real ``datetime64[ns]`` array,
    #      the gregorian (pandas) branch is safe. If it is an object array,
    #      the cftime branch is mandatory: ``pd.to_datetime`` cannot
    #      convert cftime objects and will raise ``TypeError``.
    attrs    = getattr(time_da, "attrs", {}) or {}
    encoding = getattr(time_da, "encoding", {}) or {}
    cal = str(attrs.get("calendar") or encoding.get("calendar") or "").lower()

    # Value dtype takes precedence: a datetime64 axis is always safe for the
    # pandas path, an object axis is always required for the cftime path.
    dtype = getattr(time_da, "dtype", None)
    if dtype is not None and np.issubdtype(dtype, np.datetime64):
        return ensure_datetime64_ns(time_da.values), None
    if dtype is not None and dtype == object:
        return _build_axis_from_cftime(time_da)

    # Fall back to calendar string for unusual dtypes (e.g. numeric times
    # from a not-yet-decoded file).
    if cal in GREGORIAN_CALENDARS:
        return ensure_datetime64_ns(time_da.values), None
    return _build_axis_from_cftime(time_da)


def find_consecutive_above_threshold_1d(
    data_1d: np.ndarray,
    dates_ns: np.ndarray,
    min_rain: float,
    separation_h: int,
) -> List[np.ndarray]:
    """Group consecutive timesteps >= min_rain into storms separated by > separation_h hours.

    Vectorised version of the storm-detection loop: identifies all wet
    timesteps, computes inter-wet-timestep gaps in one numpy call, and uses
    ``np.split`` to break the index array at gaps that exceed the storm
    separation time. Functionally identical to the original elementwise
    loop but ~30-100x faster on long hourly time series.
    """
    wet_idx = np.where(data_1d >= min_rain)[0]
    if wet_idx.size == 0:
        return []
    if wet_idx.size == 1:
        return [wet_idx.astype(np.int64, copy=False)]

    sep_ns    = hours_to_ns(separation_h)
    wet_dates = dates_ns[wet_idx]
    # np.diff on datetime64[ns] returns timedelta64[ns]; cast to int64 ns
    diffs_ns  = np.diff(wet_dates).astype("timedelta64[ns]").astype(np.int64)
    # Break after every wet timestep whose successor is > sep_ns away
    break_after = np.where(diffs_ns > sep_ns)[0]
    if break_after.size == 0:
        return [wet_idx.astype(np.int64, copy=False)]
    return [s.astype(np.int64, copy=False)
            for s in np.split(wet_idx, break_after + 1)]


def remove_short_storms_and_near_gaps_1d(
    storms: List[np.ndarray],
    dates_ns: np.ndarray,
    time_res_min: int,
    min_event_duration_min: int,
    separation_h: int,
    check_gaps: bool = True,
) -> List[np.ndarray]:
    """Drop storms shorter than min_event_duration_min and storms near edges/gaps."""
    if not storms:
        return storms
    kept = []
    tr      = np.timedelta64(time_res_min, "m")
    min_dur = np.timedelta64(int(min_event_duration_min), "m")
    for s in storms:
        dur = (dates_ns[s[-1]] - dates_ns[s[0]]) + tr
        if dur >= min_dur:
            kept.append(s)
    storms = kept
    if not storms:
        return storms

    sep_ns = hours_to_ns(separation_h)

    # Near edges
    if (dates_ns[storms[0][0]] - dates_ns[0]).astype("timedelta64[ns]").astype(int) < sep_ns:
        storms.pop(0)
        if not storms:
            return storms
    if (dates_ns[-1] - dates_ns[storms[-1][-1]]).astype("timedelta64[ns]").astype(int) < sep_ns:
        storms.pop()
        if not storms:
            return storms

    if not check_gaps:
        return storms

    # Large internal gaps
    diffs = np.diff(dates_ns).astype("timedelta64[ns]").astype(int)
    gap_end_idx = np.where(diffs > sep_ns)[0]
    if gap_end_idx.size == 0:
        return storms
    gap_start_idx = gap_end_idx + 1
    time_res_ns = diffs[0] if diffs.size else hours_to_ns(TIME_RES_MIN / 60)

    to_del = set()
    for ge in gap_end_idx:
        end_date   = dates_ns[ge]
        start_date = end_date - np.timedelta64(sep_ns, "ns")
        window     = np.arange(start_date, end_date, np.timedelta64(int(time_res_ns), "ns"))
        for i, s in enumerate(storms):
            if np.intersect1d(dates_ns[s], window).size > 0:
                to_del.add(i)
    for gs in gap_start_idx:
        start_date = dates_ns[gs]
        end_date   = start_date + np.timedelta64(sep_ns, "ns")
        window     = np.arange(start_date, end_date, np.timedelta64(int(time_res_ns), "ns"))
        for i, s in enumerate(storms):
            if np.intersect1d(dates_ns[s], window).size > 0:
                to_del.add(i)
    if to_del:
        storms = [s for i, s in enumerate(storms) if i not in sorted(to_del)]
    return storms


def conv_event_mean_mm_per_h(segment: np.ndarray, duration_h: int) -> Tuple[float, int]:
    """Centered moving sum on the event segment."""
    k = max(1, int(duration_h * 60 // TIME_RES_MIN))
    event = np.convolve(segment, np.ones(k, dtype=np.float32), mode="same").astype(np.float32)
    if np.all(np.isnan(event)):
        return (np.nan, 0)
    loc = int(np.nanargmax(event))
    oe_mmh = float(event[loc]) / float(k)
    return (oe_mmh, loc)


# Per-cell SMEV helper (used inside workers, on one cell at a time)
def _fit_smev_cell(
    P: np.ndarray,
    years: np.ndarray,
    smev_engine: SMEV,
    return_periods_arr: np.ndarray,
    left_censoring: List[float],
) -> Optional[Tuple[float, float, float, np.ndarray]]:
    """Fit SMEV on the OE series of one (cell, duration). Returns
    (shape, scale, N, rl_array) or None when the fit is not possible.
    """
    mask = np.isfinite(P) & (P > 0)
    P = P[mask]
    years = years[mask]
    if P.size < 2:
        return None
    n_uy = int(np.unique(years).size)
    if n_uy == 0:
        return None
    n_val = float(P.size) / float(n_uy)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            shape, scale = smev_engine.estimate_smev_parameters(P, left_censoring)
            rl = smev_engine.smev_return_values(return_periods_arr, shape, scale, n_val)
        except Exception:
            return None
    if not np.isfinite(shape) or not np.isfinite(scale):
        return None
    return float(shape), float(scale), n_val, np.atleast_1d(np.asarray(rl, dtype=np.float64))


# OUTPUT SCHEMA
def coerce_oe_df(df: pd.DataFrame, lat_dtype: np.dtype, lon_dtype: np.dtype) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=OE_COLS)
    df = df.assign(
        lat        = df["lat"].astype(lat_dtype, copy=False),
        lon        = df["lon"].astype(lon_dtype, copy=False),
        duration_h = df["duration_h"].astype("int8", copy=False),
        year       = df["year"].astype("int16", copy=False),
        OE         = df["OE"].astype("float32", copy=False),
        OE_i       = df["OE_i"].astype("int32", copy=False),
        From_i     = df["From_i"].astype("int32", copy=False),
        To_i       = df["To_i"].astype("int32", copy=False),
    )
    return df[OE_COLS]


def coerce_ams_df(df: pd.DataFrame, lat_dtype: np.dtype, lon_dtype: np.dtype) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=AMS_COLS)
    df = df.assign(
        lat        = df["lat"].astype(lat_dtype, copy=False),
        lon        = df["lon"].astype(lon_dtype, copy=False),
        duration_h = df["duration_h"].astype("int8", copy=False),
        year       = df["year"].astype("int16", copy=False),
        OE         = df["OE"].astype("float32", copy=False),
        OE_i       = df["OE_i"].astype("int32", copy=False),
        From_i     = df["From_i"].astype("int32", copy=False),
        To_i       = df["To_i"].astype("int32", copy=False),
    )
    return df[AMS_COLS]


# EVEN-WS crop
def compute_even_ws_crop_indices(in_path: str, tdim: str, ydim: str, xdim: str, ws: int
                                 ) -> Tuple[int, int, int, int]:
    ds = xr.open_dataset(in_path)
    if SPATIAL_SUBSET is not None:
        ds = ds.sel(**SPATIAL_SUBSET)
    var = ds[VAR_NAME].transpose(tdim, ydim, xdim)
    slab = var.isel({tdim: 0}).values
    ds.close()

    valid_mask = ~np.isnan(slab)
    kernel = np.ones((ws, ws), dtype=float)
    valid_count = convolve2d(valid_mask.astype(float), kernel, mode="same",
                             boundary="fill", fillvalue=0)
    out_valid = (valid_count == kernel.sum())

    lat_idx = np.nonzero(out_valid.any(axis=1))[0]
    lon_idx = np.nonzero(out_valid.any(axis=0))[0]

    if lat_idx.size == 0 or lon_idx.size == 0:
        return 0, out_valid.shape[0] - 1, 0, out_valid.shape[1] - 1
    return int(lat_idx[0]), int(lat_idx[-1]), int(lon_idx[0]), int(lon_idx[-1])


# Parquet schema helpers
def _pa_float_type_from_np(np_dtype: np.dtype) -> pa.DataType:
    dt = np.dtype(np_dtype)
    if dt == np.dtype("float32"):
        return pa.float32()
    if dt == np.dtype("float64"):
        return pa.float64()
    return pa.from_numpy_dtype(dt)

def _make_oe_schema(lat_dtype: np.dtype, lon_dtype: np.dtype) -> pa.Schema:
    return pa.schema([
        ("lat",        _pa_float_type_from_np(lat_dtype)),
        ("lon",        _pa_float_type_from_np(lon_dtype)),
        ("duration_h", pa.int8()),
        ("year",       pa.int16()),
        ("OE",         pa.float32()),
        ("OE_i",       pa.int32()),
        ("From_i",     pa.int32()),
        ("To_i",       pa.int32()),
    ])

def _make_ams_schema(lat_dtype: np.dtype, lon_dtype: np.dtype) -> pa.Schema:
    return pa.schema([
        ("lat",        _pa_float_type_from_np(lat_dtype)),
        ("lon",        _pa_float_type_from_np(lon_dtype)),
        ("duration_h", pa.int8()),
        ("year",       pa.int16()),
        ("OE",         pa.float32()),
        ("OE_i",       pa.int32()),
        ("From_i",     pa.int32()),
        ("To_i",       pa.int32()),
    ])


def write_parquet(df: pd.DataFrame, path: str, schema: pa.Schema) -> None:
    if os.path.exists(path):
        os.remove(path)
    writer = pq.ParquetWriter(
        path, schema,
        compression=PARQUET_COMPRESSION,
        compression_level=PARQUET_ZSTD_LEVEL,
        use_dictionary=True,
    )
    try:
        if df is not None and not df.empty:
            writer.write_table(pa.Table.from_pandas(df, schema=schema, preserve_index=False))
    finally:
        writer.close()


# CRS / GEOTRANSFORM helpers
def detect_crs(ds: xr.Dataset) -> CRS:
    """Detect a rasterio CRS from CF-style metadata; fall back to EPSG:4326."""
    # 1) grid_mapping attribute on the data variable
    try:
        gm_name = ds[VAR_NAME].attrs.get("grid_mapping", None)
    except Exception:
        gm_name = None

    candidate_vars = []
    if gm_name and gm_name in ds.variables:
        candidate_vars.append(ds[gm_name])
    for cand in ("crs", "spatial_ref", "rotated_pole", "rotated_latitude_longitude"):
        if cand in ds.variables:
            candidate_vars.append(ds[cand])

    for cv in candidate_vars:
        attrs = getattr(cv, "attrs", {}) or {}
        # WKT
        for key in ("crs_wkt", "spatial_ref", "esri_pe_string"):
            wkt = attrs.get(key)
            if wkt:
                try:
                    return CRS.from_wkt(str(wkt))
                except Exception:
                    pass
        # EPSG
        for key in ("epsg_code", "EPSG"):
            if key in attrs:
                try:
                    return CRS.from_epsg(int(attrs[key]))
                except Exception:
                    pass
        # PROJ4
        for key in ("proj4", "proj4_string", "proj4_params"):
            if key in attrs:
                try:
                    return CRS.from_proj4(str(attrs[key]))
                except Exception:
                    pass
        # Grid mapping name heuristic
        gmn = str(attrs.get("grid_mapping_name", "")).lower()
        if gmn in ("latitude_longitude", "longitude_latitude"):
            return CRS.from_epsg(4326)

    return CRS.from_epsg(4326)


def build_transform_from_1d_coords(lat_vals: np.ndarray, lon_vals: np.ndarray
                                   ) -> Tuple["rasterio.transform.Affine", bool]:
    """Build a north-up GeoTIFF transform from 1D lat/lon arrays.

    Returns (transform, flip_lat) where flip_lat is True when the input lat
    array is increasing (south->north) and therefore the data must be flipped
    along the first (lat) axis before writing the TIF (which expects row 0 at
    the northern edge).
    """
    if lat_vals.ndim != 1 or lon_vals.ndim != 1:
        raise ValueError("areal_SMEV.py expects 1D lat/lon coordinate arrays.")

    lat_vals = np.asarray(lat_vals).astype("float64")
    lon_vals = np.asarray(lon_vals).astype("float64")
    nx, ny = lon_vals.size, lat_vals.size
    if nx < 2 or ny < 2:
        raise ValueError("Need at least 2x2 cells to build a transform.")

    dx = float(np.abs(lon_vals[1] - lon_vals[0]))
    dy = float(np.abs(lat_vals[1] - lat_vals[0]))
    west  = float(lon_vals.min() - dx / 2.0)
    east  = float(lon_vals.max() + dx / 2.0)
    south = float(lat_vals.min() - dy / 2.0)
    north = float(lat_vals.max() + dy / 2.0)
    transform = from_bounds(west, south, east, north, nx, ny)
    flip_lat = bool(lat_vals[0] < lat_vals[-1])
    return transform, flip_lat


def write_geotiff(
    path: str,
    data: np.ndarray,             # shape (bands, Ny, Nx) in source (NetCDF) orientation
    band_descriptions: List[str],
    crs: CRS,
    transform,
    flip_lat: bool,
):
    """Write a multi-band float32 GeoTIFF, flipping along lat if needed."""
    if data.ndim != 3:
        raise ValueError(f"Expected (bands, Ny, Nx); got shape {data.shape}.")
    nbands, ny, nx = data.shape
    if len(band_descriptions) != nbands:
        raise ValueError("band_descriptions length must equal number of bands.")

    if os.path.exists(path):
        os.remove(path)

    profile = {
        "driver":    "GTiff",
        "height":    ny,
        "width":     nx,
        "count":     nbands,
        "dtype":     TIF_DTYPE,
        "crs":       crs,
        "transform": transform,
        "compress":  TIF_COMPRESS,
        "nodata":    TIF_NODATA,
        "tiled":     True,
    }
    with rasterio.open(path, "w", **profile) as dst:
        for b in range(nbands):
            arr = data[b]
            if flip_lat:
                arr = arr[::-1, :]
            dst.write(arr.astype(TIF_DTYPE, copy=False), b + 1)
            dst.set_band_description(b + 1, band_descriptions[b])


# Parquet merge helper (combine worker "part" files into one final parquet)
def _merge_parquet_parts(final_path: str, part_paths: List[str], schema: pa.Schema) -> None:
    """Merge worker-written parquet parts into a single final parquet.

    Empty list -> writes an empty file with the schema only.
    Multiple parts are streamed row-group by row-group, so 
    memory stays bounded regardless of output size.
    """
    part_paths = [p for p in part_paths if p and os.path.exists(p)]

    if os.path.exists(final_path):
        os.remove(final_path)

    if not part_paths:
        writer = pq.ParquetWriter(
            final_path, schema,
            compression=PARQUET_COMPRESSION,
            compression_level=PARQUET_ZSTD_LEVEL,
            use_dictionary=True,
        )
        writer.close()
        return

    if len(part_paths) == 1:
        os.replace(part_paths[0], final_path)
        return

    writer = pq.ParquetWriter(
        final_path, schema,
        compression=PARQUET_COMPRESSION,
        compression_level=PARQUET_ZSTD_LEVEL,
        use_dictionary=True,
    )
    try:
        for p in part_paths:
            pf = pq.ParquetFile(p)
            for rg in range(pf.num_row_groups):
                writer.write_table(pf.read_row_group(rg))
    finally:
        writer.close()


# WORKER: process a batch of tiles in one subprocess
def _worker_process_tile_batch(
    in_path: str,
    model: str,
    ws: int,
    tdim: str, ydim: str, xdim: str,
    crop_lat0: int, crop_lon0: int,
    pad: int,
    tiles: List[Tuple[int, int, int, int]],
    lat_dtype_str: str,
    lon_dtype_str: str,
    tmp_dir: str,
    durations_h: List[int],
    return_periods: List[int],
    left_censoring: List[float],
    min_rain: float,
    separation_h: int,
    time_res_min: int,
    min_event_duration_min: int,
    grid_ny: int = 0,
    grid_nx: int = 0,
    flip_lat: bool = False,
    unc_enabled: bool = False,
    unc_niter: int = 1000,
    unc_ci_percentiles: Optional[List[float]] = None,
    unc_random_seed: int = 12345,
) -> Dict[str, Any]:
    """One worker = one OS subprocess processing several tiles serially.

    For each tile the worker:
      1. xarray-reads only its padded (lat, lon) slice from disk
      2. applies the moving-window average if ws>1, then crops
      3. drizzle-to-zero
      4. loops cells -> storm detection -> per-duration OE -> SMEV fit,
         and, when unc_enabled is set, the bootstrap for that cell.
         OE rows are flushed to parquet per j-row to bound memory
      5. accumulates tiny local (n_durations, Ny_tile, Nx_tile) result
         arrays that get returned to the master via IPC

    The uncertainty arguments are passed explicitly rather than read from
    module globals so that this function stays callable and testable on its
    own. grid_ny and grid_nx give the size of the full target grid, which the
    bootstrap needs in order to seed each cell by absolute position.

    Returns a summary dict with per-tile result arrays + paths to two
    parquet "part" files this worker wrote (OE and AMS).
    """
    if unc_ci_percentiles is None:
        unc_ci_percentiles = [5.0, 95.0]
    lat_dtype = np.dtype(lat_dtype_str)
    lon_dtype = np.dtype(lon_dtype_str)

    pid = os.getpid()
    oe_part_path  = os.path.join(tmp_dir, f"OE_part_pid{pid}.parquet")
    ams_part_path = os.path.join(tmp_dir, f"AMS_part_pid{pid}.parquet")

    oe_schema  = _make_oe_schema(lat_dtype, lon_dtype)
    ams_schema = _make_ams_schema(lat_dtype, lon_dtype)
    oe_writer = pq.ParquetWriter(
        oe_part_path, oe_schema,
        compression=PARQUET_COMPRESSION,
        compression_level=PARQUET_ZSTD_LEVEL,
        use_dictionary=True,
    )
    ams_writer = pq.ParquetWriter(
        ams_part_path, ams_schema,
        compression=PARQUET_COMPRESSION,
        compression_level=PARQUET_ZSTD_LEVEL,
        use_dictionary=True,
    )

    # SMEV engine constructed once per worker
    smev_engine = SMEV(
        return_period         = list(return_periods),
        durations             = [d * 60 for d in durations_h],
        time_resolution       = time_res_min,
        tolerance             = TOLERANCE,
        min_event_duration    = min_event_duration_min,
        storm_separation_time = separation_h,
        left_censoring        = list(left_censoring),
        min_rain              = min_rain,
    )
    rp_arr = np.asarray(return_periods, dtype=np.float64)
    ndur = len(durations_h)
    nrp  = len(return_periods)

    rows_oe_total = 0
    rows_ams_total = 0
    tile_results: List[Dict[str, Any]] = []
    timing = defaultdict(float)

    try:
        ds = xr.open_dataset(in_path)
        if SPATIAL_SUBSET is not None:
            ds = ds.sel(**SPATIAL_SUBSET)
        var = ds[VAR_NAME].transpose(tdim, ydim, xdim)
        time_da = ds[TIME_NAME] if TIME_NAME in ds.variables else ds["time"]
        dates_ns, years_override = time_for_calendar(time_da)
        Ny_base = var.sizes[ydim]
        Nx_base = var.sizes[xdim]

        for (y0, y1, x0, x1) in tiles:
            # TARGET -> BASE coords
            ty0 = crop_lat0 + y0
            ty1 = crop_lat0 + y1
            tx0 = crop_lon0 + x0
            tx1 = crop_lon0 + x1
            ry0 = max(0, ty0 - pad)
            ry1 = min(Ny_base, ty1 + pad)
            rx0 = max(0, tx0 - pad)
            rx1 = min(Nx_base, tx1 + pad)

            t0 = time.perf_counter()
            block = var.isel({ydim: slice(ry0, ry1),
                              xdim: slice(rx0, rx1)}).astype("float32").values
            timing["tile_read"] += time.perf_counter() - t0

            if ws > 1:
                t0 = time.perf_counter()
                block = moving_window_average_strict_3d(block, ws)
                timing["tile_moving_avg"] += time.perf_counter() - t0

            oy0 = ty0 - ry0
            oy1 = ty1 - ry0
            ox0 = tx0 - rx0
            ox1 = tx1 - rx0
            tile_np = block[:, oy0:oy1, ox0:ox1]
            del block

            lat_vals = ds[ydim].isel({ydim: slice(ty0, ty1)}).values
            lon_vals = ds[xdim].isel({xdim: slice(tx0, tx1)}).values

            # drizzle-to-zero 
            mask_nan = np.isnan(tile_np)
            tile_np = np.where(
                mask_nan, np.nan,
                np.where(tile_np >= min_rain, tile_np, 0.0)
            ).astype("float32", copy=False)

            Ny_t = y1 - y0
            Nx_t = x1 - x0
            shape_local = np.full((ndur, Ny_t, Nx_t), np.nan, dtype=np.float32)
            scale_local = np.full((ndur, Ny_t, Nx_t), np.nan, dtype=np.float32)
            n_local     = np.full((ndur, Ny_t, Nx_t), np.nan, dtype=np.float32)
            rl_local    = np.full((ndur * nrp, Ny_t, Nx_t), np.nan, dtype=np.float32)
            # Mean of annual maxima per (cell, duration). NaN where the cell
            # produced no AMS for that duration.
            mam_local   = np.full((ndur, Ny_t, Nx_t), np.nan, dtype=np.float32)
            # Allocated only when the bootstrap is running
            if unc_enabled:
                ci_low_local  = np.full((ndur * nrp, Ny_t, Nx_t), np.nan, dtype=np.float32)
                ci_high_local = np.full((ndur * nrp, Ny_t, Nx_t), np.nan, dtype=np.float32)
            else:
                ci_low_local = ci_high_local = None

            row_oe_buf:  List[tuple] = []
            row_ams_buf: List[tuple] = []

            t0 = time.perf_counter()
            for j in range(Ny_t):
                for i in range(Nx_t):
                    s = tile_np[:, j, i]
                    if np.all(np.isnan(s)):
                        continue

                    s_for_detect = np.where(np.isnan(s), -np.inf, s)
                    storms = find_consecutive_above_threshold_1d(
                        s_for_detect, dates_ns, min_rain, separation_h
                    )
                    if not storms:
                        continue
                    storms = remove_short_storms_and_near_gaps_1d(
                        storms, dates_ns,
                        time_res_min, min_event_duration_min, separation_h,
                        check_gaps=True,
                    )
                    if not storms:
                        continue

                    lat = float(lat_vals[j])
                    lon = float(lon_vals[i])

                    # Per-duration OE accumulator for *this cell only*
                    per_d_oe:   Dict[int, List[Tuple[float, int, int, int, int]]] = {d: [] for d in durations_h}
                    ams_by_yd:  Dict[Tuple[int, int], Tuple[float, int, int, int]] = {}

                    for s_idx in storms:
                        s0 = int(s_idx[0])
                        s1 = int(s_idx[-1])
                        seg = s[s0:s1 + 1]
                        for d in durations_h:
                            oe_mmh, loc_in_seg = conv_event_mean_mm_per_h(seg, d)
                            if np.isnan(oe_mmh):
                                continue
                            max_i = s0 + loc_in_seg
                            if years_override is not None:
                                year_val = int(years_override[max_i])
                            else:
                                year_val = int(pd.Timestamp(dates_ns[max_i]).year)
                            per_d_oe[d].append((float(oe_mmh), year_val, max_i, s0, s1))
                            key = (year_val, d)
                            cur = ams_by_yd.get(key)
                            if cur is None or oe_mmh > cur[0]:
                                ams_by_yd[key] = (float(oe_mmh), max_i, s0, s1)

                    # OE rows + SMEV fit per (cell, duration)
                    for d_idx, d in enumerate(durations_h):
                        lst = per_d_oe[d]
                        if not lst:
                            continue
                        for (oe_mmh, yr, max_i, s0, s1) in lst:
                            row_oe_buf.append(
                                (lat, lon, int(d), int(yr), float(oe_mmh),
                                 int(max_i), int(s0), int(s1))
                            )
                        P = np.fromiter((t[0] for t in lst), dtype=np.float64, count=len(lst))
                        years = np.fromiter((t[1] for t in lst), dtype=np.int64, count=len(lst))
                        fit = _fit_smev_cell(P, years, smev_engine, rp_arr, left_censoring)
                        if fit is None:
                            continue
                        shp, scl, nv, rl_arr_a = fit
                        shape_local[d_idx, j, i] = np.float32(shp)
                        scale_local[d_idx, j, i] = np.float32(scl)
                        n_local[d_idx, j, i]     = np.float32(nv)
                        for r_idx in range(nrp):
                            v = rl_arr_a[r_idx] if r_idx < rl_arr_a.size else np.nan
                            rl_local[d_idx * nrp + r_idx, j, i] = (
                                np.float32(v) if np.isfinite(v) else np.nan
                            )

                        # ---- optional bootstrap CIs ----
                        # Done here while this cell's OEs and years are still
                        # in memory. Note the seed uses the cell's position in the OUTPUT RASTER,
                        # not in the internal grid: the two differ by a row flip
                        # when lat runs south->north.
                        # Seeding off the raster keeps the result independent
                        # of the tiling, and lets --uncertainty-only (which
                        # only ever sees the raster) reproduce it exactly.
                        if unc_enabled:
                            raster_row = (
                                grid_ny - 1 - (y0 + j) if flip_lat else y0 + j
                            )
                            mask = np.isfinite(P) & (P > 0)
                            ci = bootstrap_cell(
                                P=P[mask],
                                years=years[mask],
                                niter=unc_niter,
                                n=nv,
                                rp=rp_arr,
                                left_censoring=left_censoring,
                                ci_percentiles=unc_ci_percentiles,
                                rng=cell_rng(
                                    unc_random_seed, d_idx,
                                    raster_row, x0 + i,
                                    grid_ny, grid_nx,
                                ),
                            )
                            if ci is not None:
                                base = d_idx * nrp
                                ci_low_local[base:base + nrp, j, i]  = ci[0]
                                ci_high_local[base:base + nrp, j, i] = ci[1]

                    # AMS rows for this cell, and accumulators for mean AMS
                    sum_by_d:   Dict[int, float] = {d: 0.0 for d in durations_h}
                    count_by_d: Dict[int, int]   = {d: 0   for d in durations_h}
                    for (year_val, d), (oe_mmh, max_i, s0, s1) in ams_by_yd.items():
                        row_ams_buf.append(
                            (lat, lon, int(d), int(year_val),
                             float(oe_mmh), int(max_i), int(s0), int(s1))
                        )
                        sum_by_d[d]   += float(oe_mmh)
                        count_by_d[d] += 1
                    # Mean AMS across years, one value per duration for this cell
                    for d_idx, d in enumerate(durations_h):
                        c = count_by_d[d]
                        if c > 0:
                            mam_local[d_idx, j, i] = np.float32(sum_by_d[d] / c)

                # flush after each j-row to bound memory 
                if row_oe_buf:
                    df = pd.DataFrame(row_oe_buf, columns=OE_COLS)
                    df = coerce_oe_df(df, lat_dtype, lon_dtype)
                    oe_writer.write_table(
                        pa.Table.from_pandas(df, schema=oe_schema, preserve_index=False)
                    )
                    rows_oe_total += len(df)
                    row_oe_buf.clear()
                if row_ams_buf:
                    df = pd.DataFrame(row_ams_buf, columns=AMS_COLS)
                    df = coerce_ams_df(df, lat_dtype, lon_dtype)
                    ams_writer.write_table(
                        pa.Table.from_pandas(df, schema=ams_schema, preserve_index=False)
                    )
                    rows_ams_total += len(df)
                    row_ams_buf.clear()
            timing["tile_compute_oe_smev"] += time.perf_counter() - t0

            tile_results.append({
                "y0": y0, "y1": y1, "x0": x0, "x1": x1,
                "shape_local": shape_local,
                "scale_local": scale_local,
                "n_local":     n_local,
                "rl_local":    rl_local,
                "mam_local":   mam_local,
                "ci_low_local":  ci_low_local,
                "ci_high_local": ci_high_local,
            })

            del tile_np

        ds.close()
    finally:
        oe_writer.close()
        ams_writer.close()

    # drop empty part files
    if rows_oe_total == 0 and os.path.exists(oe_part_path):
        os.remove(oe_part_path)
        oe_part_path = None
    if rows_ams_total == 0 and os.path.exists(ams_part_path):
        os.remove(ams_part_path)
        ams_part_path = None

    return {
        "pid":      pid,
        "oe_part":  oe_part_path,
        "ams_part": ams_part_path,
        "rows_oe":  rows_oe_total,
        "rows_ams": rows_ams_total,
        "tiles":    tile_results,
        "timing":   dict(timing),
    }


# Per-(model, ws)
def run_areal_smev_for_model_period_ws(model: str, period: str, ws: int) -> None:
    """Master run for ONE (model, period, ws) combination.

    Resolves the per-PERIOD input file and output directories, then launches
    an inner pool of tile workers to extract OE/AMS and fit SMEV. The inner
    pool uses MP_START_METHOD ("spawn") so it is safe to call this function
    itself from inside an outer ProcessPoolExecutor.
    """
    t_total0 = time.perf_counter()

    in_path = input_path_for(model, period)
    if not os.path.exists(in_path):
        print(f"[{model} {period} ws={ws}] Missing input: {in_path}")
        return

    oe_details_dir, quantiles_dir, parameters_dir = output_dirs_for(period, model)
    os.makedirs(oe_details_dir, exist_ok=True)
    os.makedirs(quantiles_dir,  exist_ok=True)
    os.makedirs(parameters_dir, exist_ok=True)
    os.makedirs(STAGING_DIR,    exist_ok=True)

    # 1) Inspect NetCDF  
    t0 = time.perf_counter()
    ds = xr.open_dataset(in_path)
    if SPATIAL_SUBSET is not None:
        ds = ds.sel(**SPATIAL_SUBSET)
    var = ds[VAR_NAME]
    tdim, ydim, xdim = var.dims
    var = var.transpose(tdim, ydim, xdim)

    Ny_base = var.sizes[ydim]
    Nx_base = var.sizes[xdim]
    lat_dtype = np.dtype(ds[ydim].values.dtype)
    lon_dtype = np.dtype(ds[xdim].values.dtype)
    crs = detect_crs(ds)
    t_inspect = time.perf_counter() - t0

    # 2) Cropping rules
    if ws == 1:
        crop_lat0, crop_lat1 = 0, Ny_base
        crop_lon0, crop_lon1 = 0, Nx_base
        pad = 0
    elif ws % 2 == 1:
        border = (ws - 1) // 2
        crop_lat0, crop_lat1 = border, Ny_base - border
        crop_lon0, crop_lon1 = border, Nx_base - border
        pad = border
    else:
        ds.close()
        lat_min, lat_max, lon_min, lon_max = compute_even_ws_crop_indices(
            in_path, tdim, ydim, xdim, ws
        )
        ds = xr.open_dataset(in_path)
        if SPATIAL_SUBSET is not None:
            ds = ds.sel(**SPATIAL_SUBSET)
        crop_lat0, crop_lat1 = lat_min, lat_max + 1
        crop_lon0, crop_lon1 = lon_min, lon_max + 1
        pad = (ws // 2) + 1

    Ny_tgt = crop_lat1 - crop_lat0
    Nx_tgt = crop_lon1 - crop_lon0
    if Ny_tgt <= 0 or Nx_tgt <= 0:
        print(f"[{model} ws={ws}] Target domain empty after cropping. Skipping.")
        ds.close()
        return

    # Lat/lon for the target (for TIF transform)
    lat_vals_tgt = ds[ydim].isel({ydim: slice(crop_lat0, crop_lat1)}).values
    lon_vals_tgt = ds[xdim].isel({xdim: slice(crop_lon0, crop_lon1)}).values
    ds.close()

    # 3) Build tile list (target coordinates)
    y_ranges = [(y, min(y + NY_TILE, Ny_tgt)) for y in range(0, Ny_tgt, NY_TILE)]
    x_ranges = [(x, min(x + NX_TILE, Nx_tgt)) for x in range(0, Nx_tgt, NX_TILE)]
    tiles = [(y0, y1, x0, x1) for (y0, y1) in y_ranges for (x0, x1) in x_ranges]

    # 4) Output paths 
    suffix      = ws_suffix(ws)
    oe_path     = os.path.join(oe_details_dir, f"OE_{model}_{period}{suffix}.parquet")
    ams_path    = os.path.join(oe_details_dir, f"AMS_{model}_{period}{suffix}.parquet")
    rl_out_path = os.path.join(quantiles_dir,  f"{model}_{period}{suffix}.tif")
    param_paths = {
        "Scale": os.path.join(parameters_dir, f"{model}_{period}_Scale{suffix}.tif"),
        "Shape": os.path.join(parameters_dir, f"{model}_{period}_Shape{suffix}.tif"),
        "N":     os.path.join(parameters_dir, f"{model}_{period}_N{suffix}.tif"),
        "mAM":   os.path.join(parameters_dir, f"{model}_{period}_mAM{suffix}.tif"),
    }

    # 5) Staging dir for parquet parts  
    tmp_dir = os.path.join(STAGING_DIR, f"_tmp_{model}_{period}_ws{ws}")
    os.makedirs(tmp_dir, exist_ok=True)

    # 5b) Raster geometry 
    transform, flip_lat = build_transform_from_1d_coords(lat_vals_tgt, lon_vals_tgt)

    # 6) Initialize empty output grids on the master 
    ndur = len(DURATIONS_H)
    nrp  = len(RETURN_PERIODS)
    shape_grid = np.full((ndur, Ny_tgt, Nx_tgt), np.nan, dtype=np.float32)
    scale_grid = np.full((ndur, Ny_tgt, Nx_tgt), np.nan, dtype=np.float32)
    n_grid     = np.full((ndur, Ny_tgt, Nx_tgt), np.nan, dtype=np.float32)
    rl_grid    = np.full((ndur * nrp, Ny_tgt, Nx_tgt), np.nan, dtype=np.float32)
    mam_grid   = np.full((ndur, Ny_tgt, Nx_tgt), np.nan, dtype=np.float32)
    if UNC_ENABLED:
        ci_low_grid  = np.full((ndur * nrp, Ny_tgt, Nx_tgt), np.nan, dtype=np.float32)
        ci_high_grid = np.full((ndur * nrp, Ny_tgt, Nx_tgt), np.nan, dtype=np.float32)
    else:
        ci_low_grid = ci_high_grid = None

    # 7) Split tiles across workers 
    n_workers = max(1, min(N_INNER_WORKERS, len(tiles)))
    tile_batches = np.array_split(np.array(tiles, dtype=object), n_workers)
    tile_batches = [list(map(tuple, b.tolist())) for b in tile_batches if len(b) > 0]

    # 8) Submit to inner ProcessPoolExecutor 
    # The inner pool runs the actual storm/SMEV computation per tile. We
    # use "spawn" so it is safe even when this function is itself called
    # from an outer subprocess.
    t0 = time.perf_counter()
    total_oe = 0
    total_ams = 0
    oe_parts:  List[str] = []
    ams_parts: List[str] = []
    worker_timings: List[dict] = []

    inner_ctx = mp.get_context(MP_START_METHOD)
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=inner_ctx) as ex:
        futs = []
        for batch in tile_batches:
            futs.append(ex.submit(
                _worker_process_tile_batch,
                in_path, model, ws,
                tdim, ydim, xdim,
                crop_lat0, crop_lon0, pad,
                batch,
                str(lat_dtype), str(lon_dtype),
                tmp_dir,
                DURATIONS_H, RETURN_PERIODS,
                LEFT_CENSORING,
                MIN_RAIN, SEPARATION_H, TIME_RES_MIN, MIN_EVENT_DURATION_MIN,
                Ny_tgt, Nx_tgt, flip_lat,
                UNC_ENABLED, UNC_NITER, UNC_CI_PERCENTILES, UNC_RANDOM_SEED,
            ))
        for fut in as_completed(futs):
            res = fut.result()
            for tr in res["tiles"]:
                y0, y1 = tr["y0"], tr["y1"]
                x0, x1 = tr["x0"], tr["x1"]
                shape_grid[:, y0:y1, x0:x1] = tr["shape_local"]
                scale_grid[:, y0:y1, x0:x1] = tr["scale_local"]
                n_grid[:,     y0:y1, x0:x1] = tr["n_local"]
                rl_grid[:,    y0:y1, x0:x1] = tr["rl_local"]
                mam_grid[:,   y0:y1, x0:x1] = tr["mam_local"]
                if UNC_ENABLED and tr.get("ci_low_local") is not None:
                    ci_low_grid[:,  y0:y1, x0:x1] = tr["ci_low_local"]
                    ci_high_grid[:, y0:y1, x0:x1] = tr["ci_high_local"]
            if res["oe_part"]:
                oe_parts.append(res["oe_part"])
            if res["ams_part"]:
                ams_parts.append(res["ams_part"])
            total_oe  += int(res.get("rows_oe", 0))
            total_ams += int(res.get("rows_ams", 0))
            worker_timings.append(res.get("timing", {}))
    t_workers = time.perf_counter() - t0

    # 9) Merge parquet parts 
    t0 = time.perf_counter()
    _merge_parquet_parts(oe_path,  oe_parts,  _make_oe_schema(lat_dtype, lon_dtype))
    _merge_parquet_parts(ams_path, ams_parts, _make_ams_schema(lat_dtype, lon_dtype))
    t_merge = time.perf_counter() - t0

    # 10) Cleanup staging 
    for p in oe_parts + ams_parts:
        if p and os.path.exists(p):
            try:
                os.remove(p)
            except Exception:
                pass
    try:
        if os.path.isdir(tmp_dir) and not os.listdir(tmp_dir):
            os.rmdir(tmp_dir)
    except Exception:
        pass

    # 11) Band metadata (transform was built at step 5b) 
    dur_band_names = [f"{d}h" for d in DURATIONS_H]
    rp_labels      = [f"{rp}y" for rp in RETURN_PERIODS]
    rl_band_names  = [f"{d_lbl}-{rp_lbl}" for d_lbl in dur_band_names for rp_lbl in rp_labels]

    # 12) Write GeoTIFFs 
    t0 = time.perf_counter()
    write_geotiff(param_paths["Scale"], scale_grid, dur_band_names, crs, transform, flip_lat)
    write_geotiff(param_paths["Shape"], shape_grid, dur_band_names, crs, transform, flip_lat)
    write_geotiff(param_paths["N"],     n_grid,     dur_band_names, crs, transform, flip_lat)
    write_geotiff(param_paths["mAM"],   mam_grid,   dur_band_names, crs, transform, flip_lat)
    write_geotiff(rl_out_path,          rl_grid,    rl_band_names,  crs, transform, flip_lat)

    # Confidence intervals, written into their own directory with exactly the
    # same geometry and band order as the return-level raster.
    if UNC_ENABLED:
        ci_dir = ci_dir_for(period, model)
        os.makedirs(ci_dir, exist_ok=True)
        ci_low_path  = os.path.join(ci_dir, f"{model}_{period}_CI_low{suffix}.tif")
        ci_high_path = os.path.join(ci_dir, f"{model}_{period}_CI_high{suffix}.tif")
        write_geotiff(ci_low_path,  ci_low_grid,  rl_band_names, crs, transform, flip_lat)
        write_geotiff(ci_high_path, ci_high_grid, rl_band_names, crs, transform, flip_lat)
    t_tif = time.perf_counter() - t0

    # 13) Summary 
    def _avg(key: str) -> float:
        if not worker_timings:
            return 0.0
        return float(np.mean([w.get(key, 0.0) for w in worker_timings]))

    print("\n" + "=" * 60)
    print(f"[{model} {period} ws={ws}] Completed in {_fmt_s(time.perf_counter() - t_total0)}")
    print(f"  Input        : {in_path}")
    print(f"  Target grid  : Ny={Ny_tgt}, Nx={Nx_tgt}  (crop from {Ny_base}x{Nx_base}, pad={pad})")
    print(f"  CRS          : {crs.to_string() if crs else 'EPSG:4326'}")
    print(f"  Workers      : {n_workers}, tiles: {len(tiles)} "
          f"(NY_TILE={NY_TILE}, NX_TILE={NX_TILE})")
    print(f"  OE  rows     : {total_oe:,}  -> {oe_path}")
    print(f"  AMS rows     : {total_ams:,} -> {ams_path}")
    print(f"  Quantile TIF : {rl_out_path}  ({len(rl_band_names)} bands)")
    print(f"  Param TIFs   : Scale/Shape/N/mAM in {parameters_dir}  ({len(dur_band_names)} bands each)")
    print(f"  Timings (wall):")
    print(f"    inspect NetCDF             : {_fmt_s(t_inspect)}")
    print(f"    parallel OE+SMEV (workers) : {_fmt_s(t_workers)}")
    print(f"    parquet merge              : {_fmt_s(t_merge)}")
    print(f"    GeoTIFF write              : {_fmt_s(t_tif)}")
    print(f"  Per-worker avg (s):")
    print(f"    tile_read                  : {_avg('tile_read'):8.2f}")
    print(f"    tile_moving_avg            : {_avg('tile_moving_avg'):8.2f}")
    print(f"    tile_compute_oe_smev       : {_avg('tile_compute_oe_smev'):8.2f}")
    print("=" * 60 + "\n")


# Outer-job wrapper (kept at module level so it is picklable by spawn)
def _run_one_job_with_errlog(model: str, period: str, ws: int) -> str:
    """Run one (model, period, ws) and return a short status string.

    Exceptions are caught here so that one failing job does not abort
    the outer pool. Stack traces are printed in the subprocess and a
    short summary line is returned to the master.
    """
    try:
        run_areal_smev_for_model_period_ws(model, period, ws)
        return f"OK  [{model} {period} ws={ws}]"
    except Exception as e:
        traceback.print_exc()
        return f"ERR [{model} {period} ws={ws}] {type(e).__name__}: {e}"


# MAIN 
#
# Parallelism strategy
# Two-level pool:
#
#   outer pool : N_OUTER_JOBS concurrent (MODEL x PERIOD x ws) jobs
#   inner pool : N_INNER_WORKERS tile workers per outer job
#
# Total peak processes ~ N_OUTER_JOBS * N_INNER_WORKERS. To run a single
# (model, period, ws) at a time (e.g. when memory is the bottleneck), set
# N_OUTER_JOBS = 1; the outer pool is then skipped entirely.
def run_all_jobs():
    # Build flat job list once, in a deterministic order
    jobs: List[Tuple[str, str, int]] = [
        (m, p, w)
        for m in MODELS
        for p in PERIODS
        for w in WS_LIST
    ]

    print("=" * 60)
    print("areal_SMEV  --  run plan")
    print("=" * 60)
    print(f"  MODELS              : {MODELS}")
    print(f"  PERIODS             : {PERIODS}")
    print(f"  WS_LIST             : {WS_LIST}")
    print(f"  DURATIONS_H         : {DURATIONS_H}")
    print(f"  RETURN_PERIODS      : {RETURN_PERIODS}")
    print(f"  jobs                : {len(jobs)} (model x period x ws)")
    print(f"  N_OUTER_JOBS        : {N_OUTER_JOBS}")
    print(f"  N_INNER_WORKERS     : {N_INNER_WORKERS}")
    print(f"  peak total workers  : {N_OUTER_JOBS * N_INNER_WORKERS}")
    print(f"  mp start method     : {MP_START_METHOD}")
    print(f"  STAGING_DIR         : {STAGING_DIR}")
    if UNC_ENABLED:
        print(f"  bootstrap           : ON  ({UNC_NITER} iterations, "
              f"{UNC_CI_PERCENTILES[0]:g}-{UNC_CI_PERCENTILES[1]:g} percentiles, "
              f"seed {UNC_RANDOM_SEED})")
        print(f"                        this roughly doubles the runtime")
    else:
        print(f"  bootstrap           : off "
              f"(set uncertainty.enabled: true to turn it on)")
    print("=" * 60)

    if N_OUTER_JOBS <= 1:
        # Sequential outer: one (model, period, ws) at a time, inner pool
        # uses N_INNER_WORKERS workers. Choose this when memory is tight
        # or when debugging.
        for (m, p, w) in jobs:
            status = _run_one_job_with_errlog(m, p, w)
            print(status)
        return

    # Concurrent outer pool
    outer_ctx = mp.get_context(MP_START_METHOD)

    # Per-model timing
    t_start_model = {m: time.perf_counter() for m in MODELS}
    remaining_per_model = {m: 0 for m in MODELS}
    for (m, _p, _w) in jobs:
        remaining_per_model[m] += 1

    with ProcessPoolExecutor(max_workers=N_OUTER_JOBS,
                             mp_context=outer_ctx) as outer_pool:
        futs = {
            outer_pool.submit(_run_one_job_with_errlog, m, p, w): (m, p, w)
            for (m, p, w) in jobs
        }
        for fut in as_completed(futs):
            (m, p, w) = futs[fut]
            try:
                status = fut.result()
            except Exception as e:
                status = f"ERR [{m} {p} ws={w}] {type(e).__name__}: {e}"
            print(status)

            # Per-model completion timer
            remaining_per_model[m] -= 1
            if remaining_per_model[m] == 0:
                dt = time.perf_counter() - t_start_model[m]
                print(f"Model {m} was completed in {dt:.1f} s " f"({dt/60:.2f} min).")


# BOOTSTRAP  (--uncertainty-only)
# Redoes the CIs from the parquet an earlier run wrote. 
def _years_for_oe(oe: pd.DataFrame, model: str, period: str) -> np.ndarray:
    """Calendar year of every OE in a parquet file.
    """
    if "year" in oe.columns:
        return oe["year"].to_numpy()

    in_path = input_path_for(model, period)
    if not os.path.exists(in_path):
        raise SystemExit(
            f"The OE parquet for {model} {period} has no 'year' column, so the\n"
            f"calendar has to be recovered from the source NetCDF, but that\n"
            f"file is missing:\n"
            f"    {in_path}\n"
            f"Either restore it, or re-run the extraction so the parquet is\n"
            f"written with a 'year' column."
        )

    ds = xr.open_dataset(in_path)
    try:
        time_da = ds[TIME_NAME] if TIME_NAME in ds.variables else ds["time"]
        dates_ns, years_override = time_for_calendar(time_da)
    finally:
        ds.close()

    oe_i = oe["OE_i"].to_numpy(dtype=np.int64)
    if years_override is not None:
        return np.asarray(years_override)[oe_i]
    return pd.DatetimeIndex(np.asarray(dates_ns)[oe_i]).year.to_numpy()


def run_uncertainty_only(model: str, period: str, ws: int) -> str:
    """Bootstrap one (model, period, ws) from its existing parquet output."""
    t_start = time.perf_counter()
    suffix = ws_suffix(ws)
    oe_dir, quantiles_dir, _ = output_dirs_for(period, model)
    ci_dir = ci_dir_for(period, model)

    oe_path = os.path.join(oe_dir, f"OE_{model}_{period}{suffix}.parquet")
    q_tif   = os.path.join(quantiles_dir, f"{model}_{period}{suffix}.tif")

    if not os.path.exists(oe_path):
        return f"[{model} {period} ws={ws}] SKIP: OE parquet not found"
    if not os.path.exists(q_tif):
        return (f"[{model} {period} ws={ws}] SKIP: quantiles GeoTIFF not found "
                f"(needed for the grid geometry)")

    os.makedirs(ci_dir, exist_ok=True)

    # Take the geometry straight from the return-level raster.
    with rasterio.open(q_tif) as src:
        crs = src.crs
        transform = src.transform
        ny, nx = src.height, src.width
        band_names = list(src.descriptions)

    ndur, nrp = len(DURATIONS_H), len(RETURN_PERIODS)
    if len(band_names) != ndur * nrp:
        return (f"[{model} {period} ws={ws}] SKIP: quantiles raster has "
                f"{len(band_names)} bands, expected {ndur * nrp}")

    oe = pd.read_parquet(oe_path)
    if oe.empty:
        return f"[{model} {period} ws={ws}] SKIP: OE parquet is empty"

    oe = oe.assign(year=_years_for_oe(oe, model, period))

    # Map each event's lat/lon onto the raster grid. Cell centres are derived
    # from the same affine transform the raster was written with.
    row_f, col_f = ~transform * (oe["lon"].to_numpy(np.float64),
                                 oe["lat"].to_numpy(np.float64))
    cols = np.floor(row_f).astype(np.int64)
    rows = np.floor(col_f).astype(np.int64)
    inside = (rows >= 0) & (rows < ny) & (cols >= 0) & (cols < nx)
    oe = oe.loc[inside].copy()
    oe["row"] = rows[inside]
    oe["col"] = cols[inside]

    ci_low  = np.full((ndur * nrp, ny, nx), np.nan, dtype=np.float32)
    ci_high = np.full((ndur * nrp, ny, nx), np.nan, dtype=np.float32)
    rp_arr = np.asarray(RETURN_PERIODS, dtype=np.float64)

    n_fitted = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for (dur_h, r, c), sub in oe.groupby(["duration_h", "row", "col"],
                                             sort=False):
            try:
                d_idx = DURATIONS_H.index(int(dur_h))
            except ValueError:
                continue

            P = sub["OE"].to_numpy(np.float64)
            years = sub["year"].to_numpy()
            keep = np.isfinite(P) & (P > 0)
            P, years = P[keep], years[keep]
            if P.size < 2:
                continue
            n_years = int(np.unique(years).size)
            if n_years == 0:
                continue

            ci = bootstrap_cell(
                P=P,
                years=years,
                niter=UNC_NITER,
                n=float(P.size) / float(n_years),
                rp=rp_arr,
                left_censoring=LEFT_CENSORING,
                ci_percentiles=UNC_CI_PERCENTILES,
                rng=cell_rng(UNC_RANDOM_SEED, d_idx, int(r), int(c), ny, nx),
            )
            if ci is None:
                continue
            base = d_idx * nrp
            ci_low[base:base + nrp, int(r), int(c)]  = ci[0]
            ci_high[base:base + nrp, int(r), int(c)] = ci[1]
            n_fitted += 1

    lo_path = os.path.join(ci_dir, f"{model}_{period}_CI_low{suffix}.tif")
    hi_path = os.path.join(ci_dir, f"{model}_{period}_CI_high{suffix}.tif")
    for out_path, data in ((lo_path, ci_low), (hi_path, ci_high)):
        with rasterio.open(
            out_path, "w", driver="GTiff",
            height=ny, width=nx, count=data.shape[0],
            dtype=TIF_DTYPE, crs=crs, transform=transform,
            nodata=TIF_NODATA, compress=TIF_COMPRESS,
        ) as dst:
            dst.write(data)
            for b, name in enumerate(band_names, start=1):
                dst.set_band_description(b, name)

    dt = time.perf_counter() - t_start
    return (f"[{model} {period} ws={ws}] bootstrap done in {dt/60:.2f} min "
            f"({n_fitted} cell-durations)")


def run_uncertainty_only_all():
    """Run the bootstrap over every job in the configuration."""
    jobs = [(m, p, w) for m in MODELS for p in PERIODS for w in WS_LIST]

    print("=" * 60)
    print("areal_SMEV  --  standalone bootstrap (--uncertainty-only)")
    print("=" * 60)
    print(f"  MODELS         : {MODELS}")
    print(f"  PERIODS        : {PERIODS}")
    print(f"  WS_LIST        : {WS_LIST}")
    print(f"  jobs           : {len(jobs)}")
    print(f"  iterations     : {UNC_NITER}")
    print(f"  percentiles    : {UNC_CI_PERCENTILES}")
    print(f"  random seed    : {UNC_RANDOM_SEED}")
    print("  reading existing OE parquet files; the NetCDF is not re-read")
    print("=" * 60)

    for (m, p, w) in jobs:
        try:
            print(run_uncertainty_only(m, p, w), flush=True)
        except Exception as exc:
            print(f"ERR [{m} {p} ws={w}] {type(exc).__name__}: {exc}", flush=True)


# COMMAND LINE
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Fit SMEV to areal precipitation extremes from gridded climate "
            "model output."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python 01_areal_SMEV.py --config configs/cpm.yaml\n"
            "  python 01_areal_SMEV.py --config configs/rcm.yaml\n"
            "  python 01_areal_SMEV.py --config configs/cpm.yaml "
            "--uncertainty-only\n"
        ),
    )
    parser.add_argument(
        "--config", required=True, metavar="FILE",
        help="YAML configuration file, e.g. configs/cpm.yaml",
    )
    parser.add_argument(
        "--uncertainty-only", action="store_true",
        help=(
            "Skip the extraction and fitting, and only recompute bootstrap "
            "confidence intervals from the parquet files an earlier run "
            "produced. Implies the bootstrap, regardless of "
            "uncertainty.enabled in the configuration."
        ),
    )
    parser.add_argument(
        "--niter", type=int, default=None, metavar="N",
        help=(
            "Override uncertainty.niter for this run. Useful for a quick, "
            "coarse pass before committing to the full iteration count."
        ),
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    cfg = load_config(args.config)

    if args.niter is not None:
        cfg.override("uncertainty.niter", int(args.niter))
    if args.uncertainty_only:
        cfg.override("uncertainty.enabled", True)

    effective_fd, effective_path = tempfile.mkstemp(
        prefix="areal_smev_effective_", suffix=".yaml"
    )
    os.close(effective_fd)
    cfg.dump(effective_path)
    atexit.register(lambda: os.path.exists(effective_path)
                    and os.remove(effective_path))
    os.environ[CONFIG_ENV_VAR] = effective_path

    _apply_config(cfg)
    cfg.echo()

    if args.uncertainty_only:
        run_uncertainty_only_all()
    else:
        run_all_jobs()


if __name__ == "__main__":
    main()