"""
02_spatial_metrics.py

Spatial metrics describing the precipitation field at the time of ordinary events.

Stage 01 records when each ordinary event happened and how intense it was.
Here we go back to the precipitation field and describes what the event
looked like in space: how big its footprint was, and how concentrated the rain
was inside it.

For each (MODEL, PERIOD, ws, duration):

  1. Read the OEs stage 01 wrote to parquet.
  2. Load the matching precipitation NetCDF once, into shared memory.
  3. Aggregate the field over each event's duration window and compute the
     metrics.
  4. Write one parquet per model and period.

Run it as::

    python 02_spatial_metrics.py --config configs/cpm.yaml
    python 02_spatial_metrics.py --config configs/rcm.yaml

Needs 01_areal_SMEV.py to have run first.

Metrics:
``A_50``
    Area in grid cells of the smallest region holding that 50% of the
    event total. Small = compact and convective-looking, large = widespread.
``PCA_50``
    Area in grid cells reaching at least 50% of the peak intensity within that window.

The fractions (e.g., 50%, 10% ...) are set under ``spatial_metrics`` in configs/common.yaml.

Notes:
Two things dominate the runtime and both are parallel. The NetCDF read is
split into time slabs across several processes into one shared-memory buffer,
because a single-stream HDF5 read is capped by the per-chunk cache. The metrics 
themselves run in processes rather than threads, so the per-event Python loop is 
not fighting over one GIL.

For a full CPM domain that array can be tens of GB. It is held once, for the
lifetime of a model/period job.
"""

import os
# Single-thread the BLAS libs inside each worker process.
os.environ.setdefault("OMP_NUM_THREADS",       "1")
os.environ.setdefault("MKL_NUM_THREADS",       "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS",  "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS",   "1")

import sys
import time
import atexit
import traceback
import multiprocessing as mp
from multiprocessing import shared_memory

import numpy as np
import pandas as pd
import xarray as xr

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Tuple, List, Optional

import argparse
import tempfile
from pathlib import Path

# Repository modules
from config import load_config


# CONFIGURATION
# Settings come from the YAML given with --config; see configs/ and config.py.
CONFIG_ENV_VAR = "AREAL_SMEV_CONFIG"

CFG = None


def _apply_config(cfg) -> None:
    """Bind one config to this module's settings."""
    g = globals()
    g["CFG"] = cfg

    oe  = cfg["ordinary_events"]
    out = cfg["output"]
    sm  = cfg["spatial_metrics"]
    par = cfg["parallelism"]

    g["MODELS"]      = list(cfg["models"])
    g["PERIODS"]     = list(cfg["periods"])
    g["WS_LIST"]     = list(cfg["ws_list"])
    g["DURATIONS_H"] = list(cfg["durations_h"])

    g["VAR_NAME"]     = oe["var_name"]
    g["TIME_NAME"]    = oe["time_name"]
    g["MIN_RAIN"]     = float(oe["min_rain"])
    g["TIME_RES_MIN"] = int(oe["time_res_min"])

    g["STAGING_DIR"] = os.path.join(cfg["paths"]["staging_dir"], "spatial")

    g["N_LOAD_WORKERS"]   = int(par["n_load_workers"])
    g["MAX_TASK_WORKERS"] = int(par["max_task_workers"])

    g["PARQUET_COMPRESSION"] = out["parquet"]["compression"]
    g["PARQUET_ZSTD_LEVEL"]  = int(out["parquet"]["zstd_level"])
    g["ARROW_BATCH_SIZE"]    = int(sm["arrow_batch_size"])
    g["WRITE_BUFFER_ROWS"]   = int(sm["write_buffer_rows"])
    g["COORD_DECIMALS"]      = int(sm["coord_decimals"])

    g["A_FRACTIONS"] = list(sm["a_fractions"])
    g["PCA_ALPHAS"]  = list(sm["pca_alphas"])
    g["A_COLS"]      = [f"A_{int(f * 100)}"   for f in sm["a_fractions"]]
    g["PCA_COLS"]    = [f"PCA_{int(a * 100)}" for a in sm["pca_alphas"]]


def _require_config():
    """Active config, or a message saying how to provide one."""
    if CFG is None:
        raise SystemExit(
            "No configuration loaded.\n"
            "Run this script with --config, for example:\n"
            "    python 02_spatial_metrics.py --config configs/cpm.yaml"
        )
    return CFG


# Picked up automatically by spawned workers on re-import.
if os.environ.get(CONFIG_ENV_VAR):
    _apply_config(load_config(os.environ[CONFIG_ENV_VAR]))


# "spawn" is the only safe start method once nested concurrency meets the
# NetCDF/HDF5 libraries.
MP_START_METHOD = "spawn"

QUIET = False


# Module-level worker globals
_WORKER_SHM     = None
_WORKER_PRECIP  = None

def _worker_init(shm_name: str,
                 shape: Tuple[int, int, int],
                 dtype_str: str):
    """ProcessPoolExecutor initializer.
    """
    global _WORKER_SHM, _WORKER_PRECIP
    _WORKER_SHM = shared_memory.SharedMemory(name=shm_name)
    _WORKER_PRECIP = np.ndarray(shape,
                                dtype=np.dtype(dtype_str),
                                buffer=_WORKER_SHM.buf)


# Helpers
def _fmt_s(x: float) -> str:
    return f"{x:8.2f} s  ({x/60:6.2f} min)"

def ws_suffix(ws: int) -> str:
    return "" if ws == 1 else f"_WS{ws}"

def parquet_in_dir(model: str, period: str) -> str:
    """Where stage 01 put the OE/AMS parquet."""
    return _require_config().output_dirs(model, period)["oe"]

def oe_parquet_path(model: str, period: str, ws: int) -> str:
    return os.path.join(parquet_in_dir(model, period),
                        f"OE_{model}_{period}{ws_suffix(ws)}.parquet")

def ams_parquet_path(model: str, period: str, ws: int) -> str:
    return os.path.join(parquet_in_dir(model, period),
                        f"AMS_{model}_{period}{ws_suffix(ws)}.parquet")

def nc_local_path(model: str, period: str) -> str:
    """Precipitation NetCDF for (MODEL, PERIOD).
    """
    return _require_config().input_file_path(model, period)

def out_dir(model: str, period: str) -> str:
    """Where the spatial-metric parquet goes."""
    return _require_config().output_dirs(model, period)["spatial"]

def duration_k(duration_h: int) -> int:
    return max(1, int(duration_h * 60 // TIME_RES_MIN))

def time_window_bounds_same(t: int, k: int) -> Tuple[int, int]:
    p = k // 2
    if k % 2 == 1:
        return (t - p, t + p)
    return (t - p, t + p - 1)

def spatial_window_bounds_same(j: np.ndarray, i: np.ndarray, ws: int):
    p = ws // 2
    if ws % 2 == 1:
        return j - p, j + p, i - p, i + p
    return j - p, j + p - 1, i - p, i + p - 1

def prefix2d(a: np.ndarray) -> np.ndarray:
    return np.pad(a, ((1, 0), (1, 0)), mode="constant").cumsum(0).cumsum(1)

def rect_sum(P, y0, y1, x0, x1):
    return P[y1 + 1, x1 + 1] - P[y0, x1 + 1] - P[y1 + 1, x0] + P[y0, x0]

def drizzle_to_zero(arr: np.ndarray) -> np.ndarray:
    m = np.isnan(arr)
    return np.where(m, np.nan,
                    np.where(arr >= MIN_RAIN, arr, 0.0)).astype("float32",
                                                                copy=False)

def rg_uniform(ws: int) -> float:
    if ws <= 1:
        return 0.0
    return float(np.sqrt((ws * ws - 1) / 6.0))


# Coordinate mapping (lat/lon ↔ j/i)
def build_coord_maps(ds_nc: xr.Dataset, var: xr.DataArray):
    tdim, ydim, xdim = var.dims
    lat_vals = ds_nc[ydim].values
    lon_vals = ds_nc[xdim].values
    lat_r = np.round(lat_vals.astype("float64"), COORD_DECIMALS)
    lon_r = np.round(lon_vals.astype("float64"), COORD_DECIMALS)
    lat_map = {float(v): int(idx) for idx, v in enumerate(lat_r)}
    lon_map = {float(v): int(idx) for idx, v in enumerate(lon_r)}
    return tdim, ydim, xdim, lat_vals, lon_vals, lat_map, lon_map

def map_latlon_to_ji(lat, lon, lat_map, lon_map):
    lat_r = np.round(lat.astype("float64"), COORD_DECIMALS)
    lon_r = np.round(lon.astype("float64"), COORD_DECIMALS)
    j = pd.Series(lat_r).map(lat_map).to_numpy(dtype="int32")
    i = pd.Series(lon_r).map(lon_map).to_numpy(dtype="int32")
    if np.any(pd.isna(j)) or np.any(pd.isna(i)):
        bad = np.where(pd.isna(j) | pd.isna(i))[0][:10]
        raise ValueError(
            f"Failed to map some lat/lon to grid indices (example rows {bad})"
        )
    return j, i


# Read duration-mean intensity field
def read_duration_mean_field_bbox_np(precip: np.ndarray,
                                     t: int, k: int,
                                     y0: int, y1: int,
                                     x0: int, x1: int) -> np.ndarray:
    T = precip.shape[0]
    L, R = time_window_bounds_same(t, k)
    Lc = max(0, L)
    Rc = min(T - 1, R)

    slab = precip[Lc:Rc + 1, y0:y1 + 1, x0:x1 + 1]
    if slab.dtype != np.float32:
        slab = slab.astype("float32", copy=False)
    slab = drizzle_to_zero(slab)
    s = np.sum(slab, axis=0, dtype=np.float32)
    return (s / float(k)).astype("float32", copy=False)


# metric computation 
def compute_all_metrics(F: np.ndarray,
                        ws: int,
                        j_loc: np.ndarray,
                        i_loc: np.ndarray) -> Dict[str, np.ndarray]:
    Ny, Nx = F.shape
    n = j_loc.shape[0]

    out: Dict[str, np.ndarray] = {
        "cv":      np.full(n, np.nan, dtype="float32"),
        "Rg_norm": np.full(n, np.nan, dtype="float32"),
    }
    for col in A_COLS + PCA_COLS:
        out[col] = np.full(n, np.nan, dtype="float32")

    if n == 0:
        return out

    valid = ~np.isnan(F)
    F0    = np.where(valid, F, 0.0).astype("float32", copy=False)

    P_val = prefix2d(valid.astype("int32"))
    P_sum = prefix2d(F0.astype("float64"))
    P_sq  = prefix2d((F0.astype("float64") ** 2))

    F64 = F0.astype("float64")
    yy, xx = np.meshgrid(np.arange(Ny, dtype="float64"),
                         np.arange(Nx, dtype="float64"),
                         indexing="ij")
    P_Fy  = prefix2d(F64 * yy)
    P_Fx  = prefix2d(F64 * xx)
    P_Fy2 = prefix2d(F64 * yy * yy)
    P_Fx2 = prefix2d(F64 * xx * xx)

    y0, y1, x0, x1 = spatial_window_bounds_same(j_loc, i_loc, ws)
    y0c = np.clip(y0, 0, Ny - 1)
    y1c = np.clip(y1, 0, Ny - 1)
    x0c = np.clip(x0, 0, Nx - 1)
    x1c = np.clip(x1, 0, Nx - 1)

    cntV = rect_sum(P_val, y0c, y1c, x0c, x1c).astype("float64")
    ws2  = float(ws * ws)
    ok_full = cntV == ws2

    if np.any(ok_full):
        sumF = rect_sum(P_sum, y0c, y1c, x0c, x1c)[ok_full]
        sumQ = rect_sum(P_sq,  y0c, y1c, x0c, x1c)[ok_full]
        mean = sumF / ws2
        var  = np.maximum(sumQ / ws2 - mean * mean, 0.0)
        std  = np.sqrt(var)
        cv_v = np.full(mean.shape, np.nan, dtype="float32")
        gm   = mean > 0
        cv_v[gm] = (std[gm] / mean[gm]).astype("float32")
        out["cv"][ok_full] = cv_v

    sumF_all = rect_sum(P_sum, y0c, y1c, x0c, x1c)
    ok_rg = ok_full & (sumF_all > 0.0)
    if np.any(ok_rg):
        sF = sumF_all[ok_rg]
        meanY = rect_sum(P_Fy,  y0c, y1c, x0c, x1c)[ok_rg] / sF
        meanX = rect_sum(P_Fx,  y0c, y1c, x0c, x1c)[ok_rg] / sF
        varY  = np.maximum(rect_sum(P_Fy2, y0c, y1c, x0c, x1c)[ok_rg] / sF
                           - meanY * meanY, 0.0)
        varX  = np.maximum(rect_sum(P_Fx2, y0c, y1c, x0c, x1c)[ok_rg] / sF
                           - meanX * meanX, 0.0)
        rg    = np.sqrt(varY + varX)
        rg_u  = rg_uniform(ws)
        if rg_u > 0.0:
            out["Rg_norm"][ok_rg] = (rg / rg_u).astype("float32")
        else:
            out["Rg_norm"][ok_rg] = 0.0
    ok_zero = ok_full & (sumF_all == 0.0)
    if np.any(ok_zero):
        out["Rg_norm"][ok_zero] = 0.0

    y0p, y1p, x0p, x1p = y0, y1, x0, x1
    in_bounds = (y0p >= 0) & (y1p < Ny) & (x0p >= 0) & (x1p < Nx)
    if not np.any(in_bounds):
        return out

    valid_idx = np.where(in_bounds)[0]
    n_valid = valid_idx.size

    dy = np.arange(ws, dtype="int32")
    dx = np.arange(ws, dtype="int32")
    DY, DX = np.meshgrid(dy, dx, indexing="ij")
    DY = DY.ravel()
    DX = DX.ravel()

    y0_v = y0p[valid_idx]
    x0_v = x0p[valid_idx]
    rows = (y0_v[:, None] + DY[None, :])
    cols = (x0_v[:, None] + DX[None, :])

    W = F[rows, cols]
    has_nan = np.isnan(W).any(axis=1)
    W_safe  = np.where(np.isnan(W), 0.0, W).astype("float32", copy=False)
    totals  = W_safe.sum(axis=1, dtype="float64")
    ok = (~has_nan) & (totals > 0.0)
    if not np.any(ok):
        return out

    W_ok   = W_safe[ok]
    tot_ok = totals[ok]
    ok_global = valid_idx[ok]
    N = ws * ws

    W_sorted_desc = np.sort(W_ok, axis=1)[:, ::-1]
    W_cumsum = np.cumsum(W_sorted_desc, axis=1, dtype="float64")
    for f, col in zip(A_FRACTIONS, A_COLS):
        target = f * tot_ok
        ge = W_cumsum >= target[:, None]
        k_first = np.argmax(ge, axis=1)
        frac = ((k_first + 1).astype("float64") / float(N)).astype("float32")
        out[col][ok_global] = frac

    p_max = W_ok.max(axis=1)
    pos = p_max > 0
    if np.any(pos):
        W_pm = W_ok[pos]
        pm   = p_max[pos]
        og   = ok_global[pos]
        for a, col in zip(PCA_ALPHAS, PCA_COLS):
            thresh = (a * pm)[:, None]
            cnt = (W_pm >= thresh).sum(axis=1, dtype="int32")
            frac = (cnt.astype("float64") / float(N)).astype("float32")
            out[col][og] = frac

    return out


# Thresholds, event loaders 
def compute_pointwise_thresholds_for_duration(oe_path, duration_h,
                                              lat_map, lon_map, Ny, Nx):
    dset = ds.dataset(oe_path, format="parquet")
    filt = (ds.field("duration_h") == duration_h)

    cid_chunks, oe_chunks = [], []
    scanner = dset.scanner(columns=["lat", "lon", "OE"],
                           filter=filt, batch_size=ARROW_BATCH_SIZE)
    n_rows = 0
    for batch in scanner.to_batches():
        lat = batch.column(0).to_numpy(zero_copy_only=False)
        lon = batch.column(1).to_numpy(zero_copy_only=False)
        oe  = batch.column(2).to_numpy(zero_copy_only=False).astype("float32",
                                                                     copy=False)
        j, i = map_latlon_to_ji(lat, lon, lat_map, lon_map)
        cid  = (j.astype("int64") * Nx + i.astype("int64")).astype("int32",
                                                                    copy=False)
        cid_chunks.append(cid)
        oe_chunks.append(oe)
        n_rows += cid.size

    n_cells = Ny * Nx
    if n_rows == 0:
        nan = np.full(n_cells, np.nan, dtype="float32")
        return nan, nan.copy(), nan.copy(), nan.copy()

    cell_id = np.concatenate(cid_chunks)
    oe_all  = np.concatenate(oe_chunks)
    order   = np.argsort(cell_id, kind="mergesort")
    cell_id = cell_id[order]
    oe_all  = oe_all[order]

    thr85 = np.full(n_cells, np.nan, dtype="float32")
    thr90 = np.full(n_cells, np.nan, dtype="float32")
    thr95 = np.full(n_cells, np.nan, dtype="float32")
    thr99 = np.full(n_cells, np.nan, dtype="float32")

    cuts   = np.flatnonzero(np.diff(cell_id)) + 1
    starts = np.r_[0, cuts]
    ends   = np.r_[cuts, n_rows]
    for s, e in zip(starts, ends):
        cid = int(cell_id[s])
        v   = oe_all[s:e]
        thr85[cid] = np.quantile(v, 0.85, method="higher").astype("float32")
        thr90[cid] = np.quantile(v, 0.90, method="higher").astype("float32")
        thr95[cid] = np.quantile(v, 0.95, method="higher").astype("float32")
        thr99[cid] = np.quantile(v, 0.99, method="higher").astype("float32")

    return thr85, thr90, thr95, thr99

def load_ams_events_for_duration(ams_path, duration_h, lat_map, lon_map):
    dset = ds.dataset(ams_path, format="parquet")
    filt = (ds.field("duration_h") == duration_h)
    tab = dset.to_table(
        columns=["lat", "lon", "year", "OE", "OE_i", "From_i", "To_i"],
        filter=filt,
    )
    if tab.num_rows == 0:
        return {"n": 0}

    pdf  = tab.to_pandas()
    lat  = pdf["lat"].to_numpy()
    lon  = pdf["lon"].to_numpy()
    oe   = pdf["OE"].to_numpy(dtype="float32", copy=False)
    oe_i = pdf["OE_i"].to_numpy(dtype="int32", copy=False)
    fr   = pdf["From_i"].to_numpy(dtype="int32", copy=False)
    to   = pdf["To_i"].to_numpy(dtype="int32", copy=False)
    year = pdf["year"].to_numpy(dtype="int16", copy=False)

    storm_len_h = ((to - fr + 1).astype("float32")
                   * (TIME_RES_MIN / 60.0)).astype("float32", copy=False)

    j, i = map_latlon_to_ji(lat, lon, lat_map, lon_map)
    order = np.argsort(oe_i, kind="mergesort")
    return {
        "n":           tab.num_rows,
        "lat":         lat[order],
        "lon":         lon[order],
        "oe_i":        oe_i[order],
        "oe":          oe[order],
        "storm_len_h": storm_len_h[order],
        "year":        year[order],
        "j":           j[order].astype("int32", copy=False),
        "i":           i[order].astype("int32", copy=False),
    }

def load_tail_events_for_duration(oe_path, duration_h,
                                  thr85, thr90, thr95, thr99,
                                  lat_map, lon_map, Ny, Nx):
    dset = ds.dataset(oe_path, format="parquet")
    filt = (ds.field("duration_h") == duration_h)

    lat_list, lon_list = [], []
    oe_i_list, fr_list, to_list = [], [], []
    pclass_list, oe_list = [], []

    scanner = dset.scanner(
        columns=["lat", "lon", "OE", "OE_i", "From_i", "To_i"],
        filter=filt, batch_size=ARROW_BATCH_SIZE,
    )

    kept, total = 0, 0
    for batch in scanner.to_batches():
        lat = batch.column(0).to_numpy(zero_copy_only=False)
        lon = batch.column(1).to_numpy(zero_copy_only=False)
        oe  = batch.column(2).to_numpy(zero_copy_only=False).astype("float32",
                                                                     copy=False)
        oe_i = batch.column(3).to_numpy(zero_copy_only=False).astype("int32",
                                                                      copy=False)
        fr   = batch.column(4).to_numpy(zero_copy_only=False).astype("int32",
                                                                      copy=False)
        to   = batch.column(5).to_numpy(zero_copy_only=False).astype("int32",
                                                                      copy=False)

        j, i = map_latlon_to_ji(lat, lon, lat_map, lon_map)
        cid  = (j.astype("int64") * Nx + i.astype("int64")).astype("int32",
                                                                    copy=False)

        t85 = thr85[cid]; t90 = thr90[cid]
        t95 = thr95[cid]; t99 = thr99[cid]

        pclass = np.zeros_like(oe, dtype="int8")
        pclass[oe >= t85] = 85
        pclass[oe >= t90] = 90
        pclass[oe >= t95] = 95
        pclass[oe >= t99] = 99

        keep = pclass >= 85
        if np.any(keep):
            lat_list.append(lat[keep]); lon_list.append(lon[keep])
            oe_i_list.append(oe_i[keep])
            fr_list.append(fr[keep]);   to_list.append(to[keep])
            pclass_list.append(pclass[keep])
            oe_list.append(oe[keep])
            kept += int(np.sum(keep))
        total += oe.size

    if kept == 0:
        return {"n": 0}

    lat_all = np.concatenate(lat_list)
    lon_all = np.concatenate(lon_list)
    oe_i_all = np.concatenate(oe_i_list).astype("int32", copy=False)
    fr_all   = np.concatenate(fr_list).astype("int32", copy=False)
    to_all   = np.concatenate(to_list).astype("int32", copy=False)
    pclass_all = np.concatenate(pclass_list).astype("int8", copy=False)
    oe_all   = np.concatenate(oe_list).astype("float32", copy=False)

    storm_len_h = ((to_all - fr_all + 1).astype("float32")
                   * (TIME_RES_MIN / 60.0)).astype("float32", copy=False)
    j_all, i_all = map_latlon_to_ji(lat_all, lon_all, lat_map, lon_map)
    order = np.argsort(oe_i_all, kind="mergesort")

    return {
        "n":           kept,
        "lat":         lat_all[order],
        "lon":         lon_all[order],
        "oe_i":        oe_i_all[order],
        "oe":          oe_all[order],
        "storm_len_h": storm_len_h[order],
        "pctl_class":  pclass_all[order],
        "j":           j_all[order].astype("int32", copy=False),
        "i":           i_all[order].astype("int32", copy=False),
    }


# Schema + writer
def build_schema(lat_dtype, lon_dtype, kind: str) -> pa.Schema:
    fields = [
        ("lat",        pa.from_numpy_dtype(np.dtype(lat_dtype))),
        ("lon",        pa.from_numpy_dtype(np.dtype(lon_dtype))),
        ("duration_h", pa.int8()),
    ]
    if kind == "AMS":
        fields.append(("year", pa.int16()))
    else:
        fields.append(("pctl_class", pa.int8()))
    fields += [
        ("OE_i",        pa.int32()),
        ("OE",          pa.float32()),
        ("storm_len_h", pa.float32()),
        ("cv",          pa.float32()),
        ("Rg_norm",     pa.float32()),
    ]
    for col in A_COLS + PCA_COLS:
        fields.append((col, pa.float32()))
    return pa.schema(fields)

def write_metrics_for_events(precip: Optional[np.ndarray],
                             ws: int, duration_h: int,
                             lat_dtype, lon_dtype,
                             events: Dict, out_path: str, kind: str,
                             Ny: int, Nx: int):
    n = int(events.get("n", 0))
    if n == 0:
        return

    schema = build_schema(lat_dtype, lon_dtype, kind)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.exists(out_path):
        os.remove(out_path)

    writer = pq.ParquetWriter(out_path, schema,
                              compression=PARQUET_COMPRESSION,
                              compression_level=PARQUET_ZSTD_LEVEL,
                              use_dictionary=True)

    buf = {name: [] for name in schema.names}
    buf_n = 0

    def flush():
        nonlocal buf_n
        if buf_n == 0:
            return
        out = {k: np.concatenate(v) for k, v in buf.items()}
        table = pa.Table.from_pydict(out, schema=schema)
        writer.write_table(table)
        for k in buf:
            buf[k] = []
        buf_n = 0

    k_steps = duration_k(duration_h)

    oe_i = events["oe_i"]
    cuts = np.flatnonzero(np.diff(oe_i)) + 1
    starts = np.r_[0, cuts]
    ends   = np.r_[cuts, n]

    t0 = time.perf_counter()
    for s, e in zip(starts, ends):
        cnt = e - s
        t_idx = int(oe_i[s])

        if ws == 1:
            oe_vals = events["oe"][s:e].astype("float32", copy=False)
            nan_m   = np.isnan(oe_vals)
            wet     = (~nan_m) & (oe_vals >= MIN_RAIN)

            cv_arr  = np.where(nan_m, np.nan,
                                np.where(wet, 0.0, np.nan)).astype("float32")
            rg_arr  = np.where(nan_m, np.nan,
                                np.where(wet, 0.0, np.nan)).astype("float32")
            metric_extra = {col: np.full(cnt, np.nan, dtype="float32")
                             for col in A_COLS + PCA_COLS}
        else:
            j = events["j"][s:e]
            i = events["i"][s:e]
            p = ws // 2
            y0 = max(0, int(j.min()) - p)
            y1 = min(Ny - 1, int(j.max()) + p)
            x0 = max(0, int(i.min()) - p)
            x1 = min(Nx - 1, int(i.max()) + p)

            F = read_duration_mean_field_bbox_np(precip,
                                                 t_idx, k_steps,
                                                 y0, y1, x0, x1)

            j_loc = (j - y0).astype("int32", copy=False)
            i_loc = (i - x0).astype("int32", copy=False)

            mres = compute_all_metrics(F, ws, j_loc, i_loc)
            cv_arr  = mres["cv"]
            rg_arr  = mres["Rg_norm"]
            metric_extra = {col: mres[col] for col in A_COLS + PCA_COLS}

        buf["lat"].append(events["lat"][s:e].astype(lat_dtype, copy=False))
        buf["lon"].append(events["lon"][s:e].astype(lon_dtype, copy=False))
        buf["duration_h"].append(np.full(cnt, duration_h, dtype="int8"))

        if kind == "AMS":
            buf["year"].append(events["year"][s:e].astype("int16", copy=False))
        else:
            buf["pctl_class"].append(events["pctl_class"][s:e].astype("int8",
                                                                      copy=False))

        buf["OE_i"].append(np.full(cnt, t_idx, dtype="int32"))
        buf["OE"].append(events["oe"][s:e].astype("float32", copy=False))
        buf["storm_len_h"].append(events["storm_len_h"][s:e].astype("float32",
                                                                     copy=False))
        buf["cv"].append(cv_arr.astype("float32", copy=False))
        buf["Rg_norm"].append(rg_arr.astype("float32", copy=False))
        for col in A_COLS + PCA_COLS:
            buf[col].append(metric_extra[col].astype("float32", copy=False))

        buf_n += cnt
        if buf_n >= WRITE_BUFFER_ROWS:
            flush()

    flush()
    writer.close()

    if not QUIET:
        print(f"    [{kind} ws={ws:>2} d={duration_h:>2}h] wrote {n:,} rows "
              f"in {_fmt_s(time.perf_counter()-t0)} -> {os.path.basename(out_path)}",
              flush=True)


# Merge per-duration parquets 
def merge_parquet_files(final_path: str, part_paths: List[str]):
    part_paths = [p for p in part_paths if p and os.path.exists(p)]
    if not part_paths:
        return

    pf0 = pq.ParquetFile(part_paths[0])
    schema = pf0.schema_arrow

    if os.path.exists(final_path):
        os.remove(final_path)

    writer = pq.ParquetWriter(final_path, schema,
                              compression=PARQUET_COMPRESSION,
                              compression_level=PARQUET_ZSTD_LEVEL,
                              use_dictionary=True)
    try:
        for pth in part_paths:
            pf = pq.ParquetFile(pth)
            for rg in range(pf.num_row_groups):
                writer.write_table(pf.read_row_group(rg))
    finally:
        writer.close()


# PARALLEL NC LOAD 
def _load_slab_into_shm(nc_path: str, var_name: str,
                        tdim: str, ydim: str, xdim: str,
                        t0: int, t1: int) -> Tuple[int, int]:
    """Worker process: read [t0:t1] of var_name from nc_path and write it
    into the shared-memory precip array attached by the initializer.

    Each worker has its OWN HDF5 file handle and chunk cache, so reads
    run in parallel and aggregate close to disk speed.
    """
    global _WORKER_PRECIP
    ds_nc = xr.open_dataset(nc_path, decode_times=False)
    try:
        var = ds_nc[var_name].transpose(tdim, ydim, xdim)
        slab = var.isel({tdim: slice(t0, t1)}).values
        if slab.dtype != np.float32:
            slab = slab.astype("float32", copy=False)
        _WORKER_PRECIP[t0:t1] = slab
    finally:
        ds_nc.close()
    return t0, t1


def _task_one_proc(model: str, period: str,
                   ws: int, duration_h: int,
                   tmp_dir: str,
                   lat_map: Dict, lon_map: Dict,
                   Ny: int, Nx: int,
                   lat_dtype_str: str, lon_dtype_str: str
                   ) -> Dict[str, Optional[str]]:
    """Worker-process.
    """
    global _WORKER_PRECIP
    res = {"ws": ws, "d": duration_h, "ams_part": None, "oe_part": None}

    ams_in = ams_parquet_path(model, period, ws)
    oe_in  = oe_parquet_path(model, period, ws)
    have_ams = os.path.exists(ams_in)
    have_oe  = os.path.exists(oe_in)
    if not have_ams and not have_oe:
        return res

    lat_dtype = np.dtype(lat_dtype_str)
    lon_dtype = np.dtype(lon_dtype_str)

    try:
        if have_ams:
            ev_ams = load_ams_events_for_duration(
                ams_in, duration_h, lat_map, lon_map
            )
            if ev_ams.get("n", 0) > 0:
                ams_part = os.path.join(
                    tmp_dir,
                    f"AMS_part_{model}_{period}{ws_suffix(ws)}_D{duration_h}h.parquet"
                )
                write_metrics_for_events(
                    _WORKER_PRECIP if ws > 1 else None,
                    ws, duration_h,
                    lat_dtype, lon_dtype,
                    ev_ams, ams_part, kind="AMS",
                    Ny=Ny, Nx=Nx,
                )
                res["ams_part"] = ams_part

        if have_oe:
            thr85, thr90, thr95, thr99 = (
                compute_pointwise_thresholds_for_duration(
                    oe_in, duration_h, lat_map, lon_map, Ny, Nx
                )
            )
            ev_oe = load_tail_events_for_duration(
                oe_in, duration_h, thr85, thr90, thr95, thr99,
                lat_map, lon_map, Ny, Nx
            )
            if ev_oe.get("n", 0) > 0:
                oe_part = os.path.join(
                    tmp_dir,
                    f"OE_part_{model}_{period}{ws_suffix(ws)}_D{duration_h}h.parquet"
                )
                write_metrics_for_events(
                    _WORKER_PRECIP if ws > 1 else None,
                    ws, duration_h,
                    lat_dtype, lon_dtype,
                    ev_oe, oe_part, kind="OE_TAIL",
                    Ny=Ny, Nx=Nx,
                )
                res["oe_part"] = oe_part
    except Exception as exc:
        tb = traceback.format_exc()
        print(f"[ws={ws} d={duration_h}h] TASK ERROR: {exc}\n{tb}", flush=True)

    return res


# Per (model, period) run 
def process_model_period(model: str, period: str):
    print(f"\n{'='*64}")
    print(f"  {model} | {period}")
    print(f"{'='*64}", flush=True)

    nc_path = nc_local_path(model, period)
    if not os.path.exists(nc_path):
        print(f"  NC not found -> skip: {nc_path}", flush=True)
        return

    ws_has_any = []
    for ws in WS_LIST:
        if (os.path.exists(ams_parquet_path(model, period, ws)) or
            os.path.exists(oe_parquet_path (model, period, ws))):
            ws_has_any.append(ws)
    if not ws_has_any:
        print("  No source parquets for any ws -> skip", flush=True)
        return

    # 1) Peek at NC: shape, coords, dtypes
    nc_size_gb = os.path.getsize(nc_path) / 1e9
    print(f"  Opening NC ({nc_size_gb:.1f} GB on disk): {nc_path}", flush=True)
    t_meta = time.perf_counter()
    ds_peek = xr.open_dataset(nc_path, decode_times=False)
    if VAR_NAME not in ds_peek:
        ds_peek.close()
        print(f"  NC missing var '{VAR_NAME}' -> skip", flush=True)
        return
    var = ds_peek[VAR_NAME]
    tdim, ydim, xdim = var.dims
    var = var.transpose(tdim, ydim, xdim)

    T  = int(var.sizes[tdim])
    Ny = int(var.sizes[ydim])
    Nx = int(var.sizes[xdim])

    _, _, _, lat_vals, lon_vals, lat_map, lon_map = build_coord_maps(ds_peek, var)
    lat_dtype = np.dtype(lat_vals.dtype)
    lon_dtype = np.dtype(lon_vals.dtype)
    ds_peek.close()

    nbytes = T * Ny * Nx * 4
    print(f"  Grid: T={T:,} Ny={Ny} Nx={Nx}  ({nbytes/1e9:.2f} GB float32)",
          flush=True)
    print(f"  Metadata read in {_fmt_s(time.perf_counter()-t_meta)}",
          flush=True)

    # 2) Allocate shared memory for precip 
    shm = shared_memory.SharedMemory(create=True, size=nbytes)
    shape = (T, Ny, Nx)
    dtype_str = "float32"

    # Best-effort cleanup if interpreter exits before our finally runs.
    def _cleanup_shm(name=shm.name):
        try:
            s = shared_memory.SharedMemory(name=name)
            s.close()
            s.unlink()
        except Exception:
            pass
    atexit.register(_cleanup_shm)

    ctx = mp.get_context(MP_START_METHOD)

    try:
        # 3) Parallel NC load 
        n_load = min(N_LOAD_WORKERS, max(1, T // 100))
        slab_size = (T + n_load - 1) // n_load
        slabs = [(t0, min(t0 + slab_size, T)) for t0 in range(0, T, slab_size)]

        print(f"  Parallel NC load: {len(slabs)} time-slabs across "
              f"{n_load} processes", flush=True)

        t_load = time.perf_counter()
        with ProcessPoolExecutor(
                max_workers=n_load,
                mp_context=ctx,
                initializer=_worker_init,
                initargs=(shm.name, shape, dtype_str)) as load_ex:
            futs = [
                load_ex.submit(_load_slab_into_shm,
                               nc_path, VAR_NAME, tdim, ydim, xdim, t0, t1)
                for (t0, t1) in slabs
            ]
            done = 0
            for fut in as_completed(futs):
                fut.result()
                done += 1
                if done == len(futs) or done % max(1, len(futs)//6) == 0:
                    print(f"    load progress {done}/{len(futs)} slabs  "
                          f"({_fmt_s(time.perf_counter()-t_load)})",
                          flush=True)

        load_time = time.perf_counter() - t_load
        print(f"  Loaded {nbytes/1e9:.2f} GB in {_fmt_s(load_time)} "
              f"({nbytes/1e9/load_time:.0f} MB/s aggregate)", flush=True)

        # 4) Stage temp dir 
        tmp_dir = os.path.join(STAGING_DIR, f"_{model}_{period}")
        os.makedirs(tmp_dir, exist_ok=True)

        # 5) Run all (ws, duration) tasks on a process pool 
        tasks: List[Tuple[int, int]] = [
            (ws, d) for ws in ws_has_any for d in DURATIONS_H
        ]
        n_task = min(MAX_TASK_WORKERS, len(tasks))
        print(f"  Dispatching {len(tasks)} tasks to {n_task} processes "
              f"(OMP={os.environ.get('OMP_NUM_THREADS','1')} per worker)",
              flush=True)

        ws_to_parts: Dict[int, Dict[str, List[str]]] = {
            ws: {"ams_parts": [], "oe_parts": []} for ws in ws_has_any
        }

        t_tasks = time.perf_counter()
        with ProcessPoolExecutor(
                max_workers=n_task,
                mp_context=ctx,
                initializer=_worker_init,
                initargs=(shm.name, shape, dtype_str)) as task_ex:
            futs = {
                task_ex.submit(
                    _task_one_proc,
                    model, period, ws, d, tmp_dir,
                    lat_map, lon_map, Ny, Nx,
                    str(lat_dtype), str(lon_dtype),
                ): (ws, d)
                for ws, d in tasks
            }
            done = 0
            n_total = len(futs)
            for fut in as_completed(futs):
                ws, d = futs[fut]
                done += 1
                try:
                    r = fut.result()
                    if r.get("ams_part"):
                        ws_to_parts[ws]["ams_parts"].append(r["ams_part"])
                    if r.get("oe_part"):
                        ws_to_parts[ws]["oe_parts"].append(r["oe_part"])
                except Exception as exc:
                    print(f"  [ws={ws} d={d}h] EXCEPTION: {exc}", flush=True)
                    traceback.print_exc()
                if done % 5 == 0 or done == n_total:
                    print(f"  task progress: {done}/{n_total} "
                          f"({100*done/n_total:.1f}%)  "
                          f"{_fmt_s(time.perf_counter()-t_tasks)}",
                          flush=True)

        print(f"  All tasks done in {_fmt_s(time.perf_counter()-t_tasks)}",
              flush=True)

        # 6) Merge per ws 
        outbase = out_dir(model, period)
        os.makedirs(outbase, exist_ok=True)

        for ws in ws_has_any:
            ams_parts = sorted(ws_to_parts[ws]["ams_parts"])
            oe_parts  = sorted(ws_to_parts[ws]["oe_parts"])

            ams_final = os.path.join(
                outbase,
                f"AMS_METRICS_{model}_{period}{ws_suffix(ws)}.parquet"
            )
            oe_final = os.path.join(
                outbase,
                f"OE_TAIL_METRICS_{model}_{period}{ws_suffix(ws)}.parquet"
            )

            if ams_parts:
                t1 = time.perf_counter()
                merge_parquet_files(ams_final, ams_parts)
                print(f"  [ws={ws:>2}] merged AMS -> {os.path.basename(ams_final)}"
                      f"  in {_fmt_s(time.perf_counter()-t1)}", flush=True)
            if oe_parts:
                t1 = time.perf_counter()
                merge_parquet_files(oe_final, oe_parts)
                print(f"  [ws={ws:>2}] merged OE_TAIL -> {os.path.basename(oe_final)}"
                      f"  in {_fmt_s(time.perf_counter()-t1)}", flush=True)

            for p in ams_parts + oe_parts:
                try:
                    if p and os.path.exists(p):
                        os.remove(p)
                except OSError:
                    pass

        try:
            if os.path.isdir(tmp_dir) and not os.listdir(tmp_dir):
                os.rmdir(tmp_dir)
        except OSError:
            pass

    finally:
        # 7) Always free the shared memory 
        try:
            shm.close()
        except Exception:
            pass
        try:
            shm.unlink()
        except Exception:
            pass


# MAIN
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Extract spatial metrics for the precipitation events found by "
            "01_areal_SMEV.py."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python 02_spatial_metrics.py --config configs/cpm.yaml\n"
            "  python 02_spatial_metrics.py --config configs/rcm.yaml\n"
        ),
    )
    parser.add_argument(
        "--config", required=True, metavar="FILE",
        help="YAML configuration file, e.g. configs/cpm.yaml",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cfg = load_config(args.config)

    fd, effective_path = tempfile.mkstemp(
        prefix="areal_smev_effective_", suffix=".yaml"
    )
    os.close(fd)
    cfg.dump(effective_path)
    atexit.register(lambda: os.path.exists(effective_path)
                    and os.remove(effective_path))
    os.environ[CONFIG_ENV_VAR] = effective_path

    _apply_config(cfg)
    cfg.echo()

    run_all()


def run_all():
    print("=" * 64)
    print("  SPATIAL-METRICS EXTRACTION ")
    print(f"  Metrics: storm_len_h, cv, Rg_norm, "
          f"{', '.join(A_COLS)}, {', '.join(PCA_COLS)}")
    print(f"  Models:        {MODELS}")
    print(f"  Periods:       {PERIODS}")
    print(f"  WS:            {WS_LIST}")
    print(f"  Durations:     {DURATIONS_H} h")
    print(f"  Load workers:  {N_LOAD_WORKERS} processes (parallel slab reads)")
    print(f"  Task workers:  {MAX_TASK_WORKERS} processes")
    print(f"  Staging:       {STAGING_DIR}")
    print("=" * 64, flush=True)

    os.makedirs(STAGING_DIR, exist_ok=True)
    t_total = time.perf_counter()

    for period in PERIODS:
        for model in MODELS:
            try:
                process_model_period(model, period)
            except Exception as exc:
                print(f"\n[{model} {period}] UNHANDLED ERROR: {exc}", flush=True)
                traceback.print_exc()

    print(f"\nALL DONE in {_fmt_s(time.perf_counter()-t_total)}", flush=True)


if __name__ == "__main__":
    mp.set_start_method(MP_START_METHOD, force=True)
    main()