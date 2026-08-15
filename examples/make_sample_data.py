"""
Generate a small synthetic dataset for trying out the codes.

The real analysis runs on multi-terabyte CORDEX-FPSCONV archives, which cannot
be redistributed. This script builds a sample with the same
structure -- variable names, dimension order, units and calendar -- so that
anyone can run and confirm the code works before using it on real data.

Usage
-----
    python examples/make_sample_data.py --out-dir examples/sample_data

Then run the analysis against the matching configuration::

    python 01_areal_SMEV.py --config configs/example.yaml
    python 02_spatial_metrics.py --config configs/example.yaml
    python 03_temp_scaling.py --config configs/example.yaml

Both a precipitation file (`pr`, mm/h) and a temperature file (`tas`, K) are
written for each period, since stage 03 needs temperature.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import xarray as xr


# A deliberately small grid: large enough to exercise the tiling and the
# moving-window averaging, small enough to run in well under a minute.
N_LAT, N_LON = 12, 14
LAT0, LON0 = 46.0, 10.0
GRID_STEP = 0.0275  # degrees, roughly 3 km, comparable to a CPM grid

PERIODS = {
    "Historical": "1996-01-01",
    "Rcp85": "2091-01-01",
}
N_YEARS = 10


def _precipitation(n_time: int, n_lat: int, n_lon: int, rng) -> np.ndarray:
    """Intermittent hourly precipitation with a heavy upper tail.
    """
    # Wet-hour probability varies smoothly in space so the maps are not noise.
    lat_f = np.linspace(0.6, 1.4, n_lat)[:, None]
    lon_f = np.linspace(1.3, 0.7, n_lon)[None, :]
    field = lat_f * lon_f

    # Seasonal cycle
    hours = np.arange(n_time)
    season = 1.0 + 0.6 * np.sin(2 * np.pi * (hours / (365.25 * 24)) - np.pi / 2)

    base_p = 0.04
    p_wet = base_p * season[:, None, None] * field[None, :, :]

    wet = rng.random((n_time, n_lat, n_lon)) < p_wet
    # Persist storms: a wet hour makes the next hour more likely to be wet.
    for lag in (1, 2, 3):
        wet[lag:] |= wet[:-lag] & (rng.random((n_time - lag, n_lat, n_lon)) < 0.45)

    intensity = 3.0 * rng.weibull(0.75, size=(n_time, n_lat, n_lon))
    pr = np.where(wet, intensity, 0.0).astype("float32")

    n_extreme = max(1, n_time // 4000)
    for _ in range(n_extreme):
        t = int(rng.integers(0, n_time))
        y = int(rng.integers(0, n_lat))
        x = int(rng.integers(0, n_lon))
        pr[t, max(0, y - 1):y + 2, max(0, x - 1):x + 2] += rng.uniform(20, 60)

    return pr


def _temperature(n_time: int, n_lat: int, n_lon: int, rng, warming: float) -> np.ndarray:
    """Hourly 2 m temperature in kelvin, with seasonal and diurnal cycles."""
    hours = np.arange(n_time)
    seasonal = 10.0 * np.sin(2 * np.pi * (hours / (365.25 * 24)) - np.pi / 2)
    diurnal = 4.0 * np.sin(2 * np.pi * (hours % 24) / 24.0 - np.pi / 2)
    lapse = np.linspace(2.0, -2.0, n_lat)[:, None] * np.ones((1, n_lon))

    tas = (
        285.0
        + warming
        + seasonal[:, None, None]
        + diurnal[:, None, None]
        + lapse[None, :, :]
        + rng.normal(0.0, 1.5, size=(n_time, n_lat, n_lon))
    )
    return tas.astype("float32")


def _write(path: str, data: np.ndarray, name: str, units: str,
           long_name: str, times, lats, lons) -> None:
    ds = xr.Dataset(
        {name: (("time", "lat", "lon"), data, {
            "units": units,
            "long_name": long_name,
        })},
        coords={
            "time": ("time", times),
            "lat": ("lat", lats, {"units": "degrees_north",
                                  "standard_name": "latitude"}),
            "lon": ("lon", lons, {"units": "degrees_east",
                                  "standard_name": "longitude"}),
        },
        attrs={
            "title": "Synthetic sample data for the areal-SMEV analysis",
            "comment": (
                "Randomly generated for testing only. Not a climate simulation."
            ),
            "Conventions": "CF-1.8",
        },
    )
    encoding = {name: {"zlib": True, "complevel": 4, "dtype": "float32"}}
    ds.to_netcdf(path, encoding=encoding)
    ds.close()
    size_mb = os.path.getsize(path) / 1e6
    print(f"  wrote {path}  ({data.shape[0]} steps, {size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument(
        "--out-dir", default="examples/sample_data",
        help="Directory for the generated NetCDF files.",
    )
    parser.add_argument(
        "--years", type=int, default=N_YEARS,
        help=f"Years per period (default {N_YEARS}).",
    )
    parser.add_argument(
        "--seed", type=int, default=20260101,
        help="Random seed, so the sample data is reproducible.",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    lats = LAT0 + GRID_STEP * np.arange(N_LAT)
    lons = LON0 + GRID_STEP * np.arange(N_LON)

    for period, start in PERIODS.items():
        rng = np.random.default_rng(args.seed + hash(period) % 1000)
        times = pd.date_range(start=start, periods=args.years * 365 * 24,
                              freq="h")
        n_time = times.size
        print(f"{period}: {args.years} years, grid {N_LAT}x{N_LON}")

        pr = _precipitation(n_time, N_LAT, N_LON, rng)
        _write(
            os.path.join(args.out_dir, f"SAMPLE_pr_{period}_merged.nc"),
            pr, "pr", "mm h-1", "precipitation rate",
            times, lats, lons,
        )

        warming = 0.0 if period == "Historical" else 3.5
        tas = _temperature(n_time, N_LAT, N_LON, rng, warming)
        _write(
            os.path.join(args.out_dir, f"SAMPLE_tas_{period}_merged.nc"),
            tas, "tas", "K", "near-surface air temperature",
            times, lats, lons,
        )

    print("\nSample data ready. Try:")
    print("  python 01_areal_SMEV.py --config configs/example.yaml")


if __name__ == "__main__":
    main()
