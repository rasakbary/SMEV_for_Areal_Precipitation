"""
03_temp_scaling.py

How strongly extreme precipitation intensifies with temperature, from quantile
regressions of event intensity against the temperature preceding each event.

For each (MODEL, PERIOD, ws):

  1. Load the OEs from 01_areal_SMEV.py and the matching tas NetCDF.
  2. Pair each event with T_pre24, the mean temperature over the 24 h before
     the event peak.
  3. Keep the tail events only -- top 15 / 10 / 5 / 1 % per grid cell and
     duration.
  4. Fit log(intensity) ~ T_pre24 at each level in
     ``temperature_scaling.quantiles``, per cell and duration.
  5. Write the tail events and the fitted rates to parquet.

Run it as::

    python 03_temp_scaling.py --config configs/cpm.yaml
    python 03_temp_scaling.py --config configs/rcm.yaml

Needs 01_areal_SMEV.py to have run first.

Outputs, in Temp_Scaling/ under ``paths.output_dir``::

    OE_TAIL_Temp_{MODEL}_{PERIOD}{ws}.parquet      tail events with T_pre24
    OE_TAIL_TScaling_{MODEL}_{PERIOD}{ws}.parquet  scaling rates and intervals

Temperature convention:
* T_pre24 is the mean of ``tas`` over the 24 h before the peak, peak hour excluded.
* ws = 1 takes the single grid cell; ws > 1 averages over the same ws x ws
  window as the areal precipitation, so the two stay consistent.
* Events in the first 24 h of the record are dropped -- incomplete window.
* ``tas`` is converted from K to degC.

tas does not have to be hourly. Its step is inferred from its own time axis and
the pre-event window converted to that many steps: e.g., 24 for hourly, 8 for
3-hourly. OE_i is always an hourly index, so it maps onto the tas axis as
``OE_i // tas_dt``. The step has to divide 24 evenly; anything else raises
so it doesn't quietly misalign the window.

Notes:
* tas is read into one float32 array per model/period.
* For ws > 1 the spatial averaging is precomputed with a uniform filter -- a
  separable running sum, so the cost does not depend on window size. After
  that the T_pre24 lookup is the same chunked fancy-indexing as ws = 1.
* The quantile regressions run across a process pool.
"""
 
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from scipy.ndimage import uniform_filter
from statsmodels.regression.quantile_regression import QuantReg
from multiprocessing import Pool
import os
import time
import gc
import atexit
import argparse
import tempfile

# Repository modules
from config import load_config


# CONFIGURATION

# Settings come from the YAML given with --config; see configs/ and config.py.
CONFIG_ENV_VAR = "AREAL_SMEV_CONFIG"

VALID_MODES = ('tpre24', 'tpre24_SAVE_MEANS', 'scaling', 'both')

CFG = None


def _apply_config(cfg) -> None:
    """Bind one config to this module's settings."""
    g = globals()
    g["CFG"] = cfg

    ts  = cfg["temperature_scaling"]
    oe  = cfg["ordinary_events"]
    par = cfg["parallelism"]

    g["MODELS"]    = list(cfg["models"])
    g["PERIODS"]   = list(cfg["periods"])
    g["WS_LIST"]   = list(cfg["ws_list"])
    g["DURATIONS"] = list(cfg["durations_h"])

    g["RUN_MODE"]  = ts["run_mode"]
    if g["RUN_MODE"] not in VALID_MODES:
        raise SystemExit(
            f"Invalid run mode {g['RUN_MODE']!r}. Choose one of: "
            f"{', '.join(VALID_MODES)}"
        )

    g["QUANTILES"]             = list(ts["quantiles"])
    g["MIN_EVENTS_REGRESSION"] = int(ts["min_events_regression"])
    g["TEMP_WINDOW_H"]         = int(ts["temp_window_h"])

    g["TIME_RES_MIN"] = int(oe["time_res_min"])

    g["N_PROC"]     = int(par["n_proc_regression"])
    g["CHUNK_SIZE"] = int(par["chunk_size"])


def _require_config():
    """Active config, or a message saying how to supply one."""
    if CFG is None:
        raise SystemExit(
            "No configuration loaded.\n"
            "Run this script with --config, for example:\n"
            "    python 03_temp_scaling.py --config configs/cpm.yaml"
        )
    return CFG


# Picked up automatically by spawned workers on re-import.
if os.environ.get(CONFIG_ENV_VAR):
    _apply_config(load_config(os.environ[CONFIG_ENV_VAR]))


# DATA LOADING
def get_suffix(ws: int) -> str:
    return '' if ws == 1 else f'_WS{ws}'


def load_oe(model: str, period: str, ws: int) -> pd.DataFrame:
    """The OEs stage 01 wrote."""
    suffix = get_suffix(ws)
    oe_dir = _require_config().output_dirs(model, period)["oe"]
    path = os.path.join(oe_dir, f'OE_{model}_{period}{suffix}.parquet')
    return pd.read_parquet(path)


def infer_tas_dt_hours(ds: xr.Dataset) -> int:
    freq = xr.infer_freq(ds['time'])           # e.g. 'h', '3h', 'D'; None if irregular
    if freq is None:
        raise ValueError("tas time axis is irregular; cannot infer dt.")
    dt = pd.to_timedelta(pd.tseries.frequencies.to_offset(freq))
    dt_hours = dt.total_seconds() / 3600.0
    if dt_hours <= 0 or abs(dt_hours - round(dt_hours)) > 1e-6:
        raise ValueError(f"Inferred non-integer tas dt = {dt_hours} h.")
    dt_hours = int(round(dt_hours))
    if 24 % dt_hours != 0:
        raise ValueError(f"tas dt = {dt_hours} h does not divide 24 h evenly.")
    return dt_hours


def load_tas_numpy(model: str, period: str):
    """Load tas into a contiguous float32 numpy array in °C.

    Also infers the temporal resolution (dt, in hours) from the file's
    own time axis and returns it.
    """
    path = _require_config().path('tas_file', model=model, period=period)
    ds = xr.open_dataset(path)
    lat_vals  = ds.coords['lat'].values
    lon_vals  = ds.coords['lon'].values

    tas_dt = infer_tas_dt_hours(ds)

    print(f"  Loading tas (time={ds.sizes['time']}, "
          f"lat={len(lat_vals)}, lon={len(lon_vals)}, dt={tas_dt} h) ...")

    da = ds['tas']
    da = da.squeeze(drop = True) # drop any degenerate dimensions
    da = da.transpose('time', 'lat', 'lon')  # ensure time is first axis
    print(f"  tas shape = {da.shape}")

    t0 = time.time()
    tas_np = (da.values - 273.15).astype(np.float32)
    print(f"  Loaded in {(time.time()-t0)/60:.2f} min  ({tas_np.nbytes/1e9:.1f} GB)")
    ds.close()
    return tas_np, lat_vals, lon_vals, tas_dt
 
 
def precompute_spatial_avg(tas_np: np.ndarray, ws: int) -> np.ndarray:
    """
    Pre-compute the w×w spatial moving average of tas_np using
    scipy.ndimage.uniform_filter.
 
    Returns a float32 array of the same shape as tas_np.
    """
    print(f"    Pre-averaging tas over {ws}×{ws} window ...")
    t0 = time.time()
    tas_avg = uniform_filter(tas_np, size=[1, ws, ws],
                             mode='nearest').astype(np.float32)
    print(f"    Done in {(time.time()-t0)/60:.2f} min")
    return tas_avg
 
 

# T_pre24
def compute_T_pre24(oe_df: pd.DataFrame,
                    tas_arr: np.ndarray,
                    lat_vals: np.ndarray,
                    lon_vals: np.ndarray,
                    tas_dt: int = 1) -> np.ndarray:
    """
    Compute T_pre24 for every event via chunked fancy-indexing.

    tas_dt : int
        Temporal resolution of tas in hours (e.g., 1 for hourly, 3 for 3-hourly).
        OE_i is always an hourly index; it is converted to the tas time
        axis as OE_i // tas_dt. The 24 h pre-event window then corresponds
        to (24 // tas_dt) consecutive tas steps.
    """
    n = len(oe_df)
    T_pre24 = np.full(n, np.nan, dtype=np.float32)
 
    # Step 1: map (lat, lon) → grid indices
    uniq = oe_df[['lat', 'lon']].drop_duplicates()
    u_lats = uniq['lat'].values.astype(np.float64)
    u_lons = uniq['lon'].values.astype(np.float64)
 
    u_lat_idx = np.argmin(
        np.abs(lat_vals.astype(np.float64)[None, :] - u_lats[:, None]), axis=1
    ).astype(np.int32)
    u_lon_idx = np.argmin(
        np.abs(lon_vals.astype(np.float64)[None, :] - u_lons[:, None]), axis=1
    ).astype(np.int32)
 
    uniq = uniq.copy()
    uniq['_lat_idx'] = u_lat_idx
    uniq['_lon_idx'] = u_lon_idx
    idx_df = oe_df[['lat', 'lon']].merge(uniq, on=['lat', 'lon'], how='left')
 
    ev_lat_idx = idx_df['_lat_idx'].values
    ev_lon_idx = idx_df['_lon_idx'].values
    ev_oe_i    = oe_df['OE_i'].values.astype(np.int64)
    del idx_df, uniq
    gc.collect()
 
    # Step 2: chunked fancy-indexing
    n_steps = 24 // tas_dt                  # 24 for hourly, 8 for 3-hourly
    offsets = np.arange(-n_steps, 0, dtype=np.int64)
    min_oe_i = 24     
 
    for start in range(0, n, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, n)

        oe_chunk  = ev_oe_i[start:end]
        lat_chunk = ev_lat_idx[start:end]
        lon_chunk = ev_lon_idx[start:end]

        valid = oe_chunk >= min_oe_i
        if not np.any(valid):
            continue

        oe_v  = oe_chunk[valid]
        lat_v = lat_chunk[valid]
        lon_v = lon_chunk[valid]

        # Convert hourly OE_i → tas time index
        oe_v_tas = oe_v // tas_dt

        t_idx = oe_v_tas[:, None] + offsets[None, :]
        vals  = tas_arr[t_idx, lat_v[:, None], lon_v[:, None]]
        means = vals.mean(axis=1)

        positions = np.where(valid)[0] + start
        T_pre24[positions] = means

    return T_pre24
 
 

# TAIL IDENTIFICATION
def identify_tail_events(oe_df: pd.DataFrame) -> pd.DataFrame:
    """
    Top 15/10/5/1 % of OE per grid cell and duration.
    """
    valid = oe_df.loc[oe_df['T_pre24'].notna()]
 
    thresholds = (
        valid.groupby(['lat', 'lon', 'duration_h'])['OE']
        .quantile([0.85, 0.90, 0.95, 0.99])
        .unstack(level=-1)
        .rename(columns={0.85: 'q85', 0.90: 'q90', 0.95: 'q95', 0.99: 'q99'})
    )
 
    merged = valid.merge(thresholds, on=['lat', 'lon', 'duration_h'], how='left')
    tail = merged.loc[merged['OE'] >= merged['q85']].copy()
    del merged
    gc.collect()
 
    tail['tail'] = 'p85'
    tail.loc[tail['OE'] >= tail['q90'], 'tail'] = 'p90'
    tail.loc[tail['OE'] >= tail['q95'], 'tail'] = 'p95'
    tail.loc[tail['OE'] >= tail['q99'], 'tail'] = 'p99'
 
    tail = tail.drop(columns=['q85', 'q90', 'q95', 'q99'], errors='ignore')
    tail['tail'] = pd.Categorical(
        tail['tail'], categories=['p85', 'p90', 'p95', 'p99'], ordered=True
    )
    return tail



# TAIL AGGREGATION — mean OE & T_pre24 per (lat, lon, duration_h, tail)
def aggregate_tail_means(oe_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each grid cell & duration, compute per-tail means of OE and T_pre24.

    Percentile tails:
        p85 -> top 15% (OE >= q85)
        p90 -> top 10% (OE >= q90)
        p95 -> top  5% (OE >= q95)
        p99 -> top  1% (OE >= q99)

    Returns one row per (lat, lon, duration_h, tail) with mean_OE,
    mean_T_pre24 and n_events.
    """
    valid = oe_df.loc[oe_df['T_pre24'].notna()]

    thresholds = (
        valid.groupby(['lat', 'lon', 'duration_h'])['OE']
        .quantile([0.85, 0.90, 0.95, 0.99])
        .unstack(level=-1)
        .rename(columns={0.85: 'q85', 0.90: 'q90', 0.95: 'q95', 0.99: 'q99'})
        .reset_index()
    )

    merged = valid.merge(thresholds, on=['lat', 'lon', 'duration_h'], how='left')

    tail_map = {'p85': 'q85', 'p90': 'q90', 'p95': 'q95', 'p99': 'q99'}

    out = []
    for tail_name, qcol in tail_map.items():
        sel = merged.loc[merged['OE'] >= merged[qcol]]
        agg = (
            sel.groupby(['lat', 'lon', 'duration_h'])
            .agg(mean_OE=('OE', 'mean'),
                 mean_T_pre24=('T_pre24', 'mean'),
                 n_events=('OE', 'size'))
            .reset_index()
        )
        agg['tail'] = tail_name
        out.append(agg)

    result = pd.concat(out, ignore_index=True)
    result['tail'] = pd.Categorical(
        result['tail'], categories=['p85', 'p90', 'p95', 'p99'], ordered=True
    )
    return result[['lat', 'lon', 'duration_h', 'tail', 'mean_OE', 'mean_T_pre24', 'n_events']]
 

# QUANTILE REGRESSION — per (lat, lon, duration_h)
def _fit_qr_one_cell(args):
    """Worker for multiprocessing Pool."""
    import warnings
    lat_val, lon_val, dur, oe_vals, t_vals = args
 
    mask = (oe_vals > 0) & np.isfinite(t_vals)
    oe_v = oe_vals[mask]
    t_v  = t_vals[mask]
 
    if len(oe_v) < MIN_EVENTS_REGRESSION:
        return []
 
    y = np.log(oe_v).astype(np.float64)
    X = np.column_stack([np.ones(len(t_v), dtype=np.float64),
                         t_v.astype(np.float64)])
 
    results = []
    for q in QUANTILES:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                qr_result = QuantReg(y, X).fit(q=q, max_iter=10000)
 
            converged = qr_result.mle_retvals.get('converged', True) \
                        if hasattr(qr_result, 'mle_retvals') else True
 
            beta   = qr_result.params[1]
            ci     = qr_result.conf_int(alpha=0.05)
            ci_lo  = ci[1, 0]
            ci_hi  = ci[1, 1]
 
            results.append({
                'lat':         lat_val,
                'lon':         lon_val,
                'duration_h':  dur,
                'quantile':    q,
                'slope_pct_K': round((np.exp(beta)  - 1) * 100, 3),
                'slope_CI_lo': round((np.exp(ci_lo) - 1) * 100, 3),
                'slope_CI_hi': round((np.exp(ci_hi) - 1) * 100, 3),
                'intercept':   round(qr_result.params[0], 5),
                'n_events':    len(oe_v),
                'converged':   converged,
            })
        except Exception:
            results.append({
                'lat':         lat_val,
                'lon':         lon_val,
                'duration_h':  dur,
                'quantile':    q,
                'slope_pct_K': np.nan,
                'slope_CI_lo': np.nan,
                'slope_CI_hi': np.nan,
                'intercept':   np.nan,
                'n_events':    len(oe_v),
                'converged':   False,
            })
 
    return results
 
 
def compute_scaling(oe_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fit quantile regression per (lat, lon, duration_h).
    """
    valid = oe_df.loc[oe_df['T_pre24'].notna() & (oe_df['OE'] > 0)]
    grouped = valid.groupby(['lat', 'lon', 'duration_h'])
    n_groups = len(grouped)
 
    print(f"    Building work items for {n_groups:,} groups ...")
    t0 = time.time()
    work_items = []
    for (lat_val, lon_val, dur), grp in grouped:
        work_items.append((
            lat_val, lon_val, dur,
            grp['OE'].values.astype(np.float32),
            grp['T_pre24'].values.astype(np.float32),
        ))
    print(f"    Work items built in {(time.time()-t0)/60:.2f} min")
 
    del valid, grouped
    gc.collect()
 
    print(f"    Fitting: {n_groups:,} groups × {len(QUANTILES)} quantiles "
          f"= {n_groups * len(QUANTILES):,} fits ({N_PROC} processes) ...")
 
    all_results = []
    with Pool(processes=N_PROC) as pool:
        for i, batch in enumerate(
            pool.imap_unordered(_fit_qr_one_cell, work_items, chunksize=500)
        ):
            all_results.extend(batch)
            if (i + 1) % 50000 == 0:
                print(f"      ... {i+1:,}/{n_groups:,} groups done")
 
    del work_items
    gc.collect()
 
    n_converged = sum(1 for r in all_results if r.get('converged', True))
    n_failed    = len(all_results) - n_converged
    print(f"    Scaling complete: {len(all_results):,} results "
          f"({n_converged:,} converged, {n_failed:,} hit iteration limit)")
 
    return pd.DataFrame(all_results)
 
 

# MAIN
def process_model_period(model: str, period: str):
    print(f"\n{'='*60}")
    print(f"  MODEL: {model}  |  PERIOD: {period}")
    print(f"{'='*60}")
 
    # 1) Load raw temperature array once
    t0 = time.time()
    tas_np, lat_vals, lon_vals, tas_dt = load_tas_numpy(model, period)
    print(f"  Temperature ready in {(time.time()-t0)/60:.2f} min (tas dt = {tas_dt} h)")
 
    for ws in WS_LIST:
        print(f"\n  --- WS = {ws} (suffix='{get_suffix(ws)}') ---")
 
        # 2) Prepare the temperature array for this ws
        if ws == 1:
            tas_arr = tas_np
        else:
            tas_arr = precompute_spatial_avg(tas_np, ws)
 
        # 3) Load OE data
        t1 = time.time()
        try:
            oe_df = load_oe(model, period, ws)
        except FileNotFoundError:
            print(f"    [SKIP] OE file not found.")
            if ws > 1:
                del tas_arr
                gc.collect()
            continue
        print(f"    Loaded {len(oe_df):,} OE records in {(time.time()-t1)/60:.2f} min")
 
        # 4) Compute T_pre24
        t2 = time.time()
        oe_df['T_pre24'] = compute_T_pre24(
            oe_df, tas_arr, lat_vals, lon_vals, tas_dt=tas_dt
        )
        n_valid = oe_df['T_pre24'].notna().sum()
        print(f"    T_pre24 done in {(time.time()-t2)/60:.1f} min  "
              f"({n_valid:,} valid, {len(oe_df)-n_valid:,} dropped)")
 
        # Free pre-averaged array (ws>1) before tail/scaling
        if ws > 1:
            del tas_arr
            gc.collect()
 
        suffix = get_suffix(ws)
        out_dir = Path(_require_config().output_dirs(model, period)["temp_scaling"])

        # 5) Tail file  (per-event OR per-tail means, depending on selected run mode)
        if RUN_MODE in ('tpre24', 'tpre24_SAVE_MEANS', 'both'):
            t3 = time.time()
            out_dir.mkdir(parents=True, exist_ok=True)

            if RUN_MODE == 'tpre24_SAVE_MEANS':
                tail_df = aggregate_tail_means(oe_df)
                print(f"    Tail means: {len(tail_df):,} rows  "
                      f"(p85={(tail_df['tail']=='p85').sum():,}, "
                      f"p90={(tail_df['tail']=='p90').sum():,}, "
                      f"p95={(tail_df['tail']=='p95').sum():,}, "
                      f"p99={(tail_df['tail']=='p99').sum():,})  "
                      f"[{(time.time()-t3)/60:.1f} min]")
                path_tail = out_dir / f'OE_TAIL_Temp_{model}_{period}{suffix}.parquet'
                tail_df.to_parquet(path_tail, index=False, engine='pyarrow')
            else:
                tail_df = identify_tail_events(oe_df)
                print(f"    Tail: {len(tail_df):,}  "
                      f"(p85={(tail_df['tail']=='p85').sum():,}, "
                      f"p90={(tail_df['tail']=='p90').sum():,}, "
                      f"p95={(tail_df['tail']=='p95').sum():,}, "
                      f"p99={(tail_df['tail']=='p99').sum():,})  "
                      f"[{(time.time()-t3)/60:.1f} min]")
                cols = ['lat', 'lon', 'duration_h', 'OE', 'OE_i', 'T_pre24', 'tail']
                path_tail = out_dir / f'OE_TAIL_Temp_{model}_{period}{suffix}.parquet'
                tail_df[cols].to_parquet(path_tail, index=False, engine='pyarrow')

            print(f"    Saved {path_tail.name}  ({len(tail_df):,} rows)")
            del tail_df
            gc.collect()

        # 6) Quantile regression  (only in 'scaling' / 'both')
        if RUN_MODE in ('scaling', 'both'):
            t4 = time.time()
            scaling_df = compute_scaling(oe_df)
            del oe_df
            gc.collect()
            print(f"    Scaling done in {(time.time()-t4)/60:.1f} min")

            out_dir.mkdir(parents=True, exist_ok=True)
            path_scal = out_dir / f'OE_TAIL_TScaling_{model}_{period}{suffix}.parquet'
            scaling_df.to_parquet(path_scal, index=False, engine='pyarrow')
            print(f"    Saved {path_scal.name}  ({len(scaling_df):,} rows)")
            del scaling_df
            gc.collect()
        else:
            del oe_df
            gc.collect()
 
    del tas_np
    gc.collect()
    print(f"\n  Model-period complete. Temperature array freed.")
 
 
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Fit temperature scaling of extreme precipitation by quantile "
            "regression."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python 03_temp_scaling.py --config configs/cpm.yaml\n"
            "  python 03_temp_scaling.py --config configs/rcm.yaml "
            "--run-mode both\n"
        ),
    )
    parser.add_argument(
        "--config", required=True, metavar="FILE",
        help="YAML configuration file, e.g. configs/cpm.yaml",
    )
    parser.add_argument(
        "--run-mode", choices=VALID_MODES, default=None,
        help=(
            "Which part of the analysis to run. Overrides "
            "temperature_scaling.run_mode in the configuration."
        ),
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cfg = load_config(args.config)

    if args.run_mode is not None:
        cfg.override("temperature_scaling.run_mode", args.run_mode)

    # Hand workers the effective config, overrides included (see stage 01).
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

    print(f"  run mode: {RUN_MODE}")

    total_t0 = time.time()
    for model in MODELS:
        for period in PERIODS:
            process_model_period(model, period)
    print(f"\n{'='*60}")
    print(f"  ALL DONE in {(time.time()-total_t0)/60:.1f} min")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()