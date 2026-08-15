# Areal SMEV: SMEV applied to areal precipitation over space and its temperature scaling

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21953199.svg)](https://doi.org/10.5281/zenodo.21953199

This repository fits the **Simplified Metastatistical Extreme Value (SMEV)**
model to areal precipitation in convection-permitting (CPM) and
regional (RCM) climate model output, computes precipitation return levels, describes the spatial structure 
of the underlying ordinary events, and measures how their intensity scales with temperature.

> **Status:** the analysis code is complete and reproducible.

---

## What the code does

It contains three stages. Each one reads what the previous stage
wrote, so they must be run in order.

| Stage | Script | What it produces |
|-------|--------|------------------|
| 1 | `01_areal_SMEV.py` | Ordinary events, annual maxima, Weibull parameters, return levels, and optional bootstrap confidence intervals |
| 2 | `02_spatial_metrics.py` | Spatial footprint and concentration metrics for each event |
| 3 | `03_temp_scaling.py` | Temperature scaling rates from quantile regression |

**Stage 1** builds areal precipitation series by averaging over a moving
`ws × ws` window, so that the same analysis can be repeated across spatial
scales from a single grid cell up to tens of kilometres. It extracts
independent storms, fits SMEV to the left-censored ordinary events of every
grid cell and duration, and converts each fit into return levels.

**Stage 2** returns to the precipitation field and describes what each event
looked like in space: how large its footprint was, and how concentrated the
rain was within it.

**Stage 3** pairs every event with the mean temperature over the 24 hours
before its peak, and fits quantile regressions of intensity against that
temperature to obtain scaling rates.

---

## Installation

```bash
conda env create -f environment.yml
conda activate areal-smev
```

Conda is strongly preferred: `rasterio` and `netCDF4` depend on GDAL and HDF5
system libraries that pip will not install. A `requirements.txt` is provided
as a fallback if you already have those libraries.

---

## Try it without any data

The real analysis runs on CORDEX-FPSCONV archives that cannot be
redistributed. To check the installation works, generate a small synthetic
dataset and run all three stages against it:

```bash
python examples/make_sample_data.py
python 01_areal_SMEV.py     --config configs/example.yaml
python 02_spatial_metrics.py --config configs/example.yaml
python 03_temp_scaling.py    --config configs/example.yaml
```

This takes a couple of minutes on a laptop. The **results are meaningless**; but every stage runs 
exactly the code path the real analysis uses.

---

## Running on real data

All settings live in YAML files.

```bash
python 01_areal_SMEV.py --config configs/cpm.yaml     # convection-permitting
python 01_areal_SMEV.py --config configs/rcm.yaml     # regional
```

A configuration is assembled from two files:

```
configs/common.yaml     method settings shared by every model family
configs/cpm.yaml        CPM paths, model list, window sizes, tiling
configs/rcm.yaml        the same for RCMs
```

The family file is merged on top of `common.yaml`. Method settings — event
separation, censoring window, durations, return periods — live only in
`common.yaml`, so a CPM run and an RCM run cannot mistakenly end up using
different definitions.

### What you need to change

Edit the `paths:` block of `configs/cpm.yaml` and `configs/rcm.yaml` to point
at your data, and set `models:` and `periods:` to the runs you want.

Input files are expected as NetCDF with a `pr` variable in mm/h and a `tas`
variable in kelvin, on a 1-D latitude/longitude grid.

---

## Uncertainty

Bootstrap confidence intervals are optional. There are two ways to run this step.

**During the main run** — set `uncertainty.enabled: true` in the
configuration. The bootstrap runs while each cell's events are still in
memory.

**Afterwards, on their own** — recompute intervals from the parquet files an
earlier run produced, without touching the NetCDF again:

```bash
python 01_areal_SMEV.py --config configs/cpm.yaml --uncertainty-only
python 01_areal_SMEV.py --config configs/cpm.yaml --uncertainty-only --niter 5000
```

Both modes call the same function and seed each grid cell by its position in
the output raster.

---

## Outputs

Written under `paths.output_dir`. `{ws}` is the suffix `_WS<n>`, omitted for
`ws = 1`.

```
OE_details/
    OE_{MODEL}_{PERIOD}{ws}.parquet          ordinary events
    AMS_{MODEL}_{PERIOD}{ws}.parquet         annual maxima
parameters/
    {MODEL}_{PERIOD}_Shape{ws}.tif           Weibull shape
    {MODEL}_{PERIOD}_Scale{ws}.tif           Weibull scale
    {MODEL}_{PERIOD}_N{ws}.tif               mean events per year
    {MODEL}_{PERIOD}_mAM{ws}.tif             mean annual maximum
quantiles/
    {MODEL}_{PERIOD}{ws}.tif                 return levels, 30 bands
quantiles_CIs/
    {MODEL}_{PERIOD}_CI_low{ws}.tif          lower interval bound
    {MODEL}_{PERIOD}_CI_high{ws}.tif         upper interval bound
Spatial_details/
    OE_TAIL_METRICS_{MODEL}_{PERIOD}{ws}.parquet
    AMS_METRICS_{MODEL}_{PERIOD}{ws}.parquet
Temp_Scaling/
    OE_TAIL_Temp_{MODEL}_{PERIOD}{ws}.parquet
    OE_TAIL_TScaling_{MODEL}_{PERIOD}{ws}.parquet
```

The parameter rasters carry one band per duration. The return-level and
interval rasters carry one band per (duration, return period), ordered
duration-major: `1h-2y, 1h-5y, … 1h-100y, 3h-2y, … 24h-100y`. Band
descriptions are written into the files.

---

## Hardware and runtime

This code was developed for a shared compute server and is parallel at two
levels: concurrent `(model, period, window size)` jobs, each with its own pool
of tile workers.

The defaults in `configs/cpm.yaml` assume a large machine — 24 inner workers,
and a precipitation array that can reach tens of gigabytes held in shared
memory. **On a smaller machine, reduce `n_inner_workers`, `n_load_workers`,
`max_task_workers`, and the tile sizes under `tiling:`,** or the run will
exhaust memory. Peak worker count is `n_outer_jobs × n_inner_workers`; keep it
below the physical core count.

The per-worker memory budget for stage 1 is roughly:

```
(ny_tile + 2·pad) × (nx_tile + 2·pad) × n_timesteps × 4 bytes
```

where `pad = ws // 2 + 1`.

---

## Tests

```bash
python -m pytest tests/ -v
```

This just test and checks the fast bootstrap version implemented here 
against the reference implementation in `smev.py`. `bootstrap.py` replaces 
the per-iteration `statsmodels` fit with a closed-form solution and batches iterations 
by resample length; the tests drive both implementations with identical year draws so that 
any disagreement is numerical rather than a difference in random sampling. On the test cells
they agree to within float32 resolution.

---

## Repository layout

```
01_areal_SMEV.py       stage 1: ordinary events, SMEV fit, return levels
02_spatial_metrics.py  stage 2: spatial metrics per event
03_temp_scaling.py     stage 3: temperature scaling
smev.py                SMEV estimator (derived from pyTENAX -- see below)
bootstrap.py           year-block bootstrap, shared by both uncertainty modes
config.py              loads, merges and validates the YAML configuration
configs/               common.yaml + one file per model family
examples/              synthetic data generator, for trying the pipeline
tests/                 bootstrap equivalence tests
```

---

## Data availability

The precipitation and temperature inputs are CORDEX-FPSCONV simulations and
are not redistributed here.

---

## Citation

> Akbary, R. (2026). *Areal SMEV: SMEV applied to areal precipitation 
> over space and its temperature scaling* (Version v1.0.0). Zenodo. 
> https://doi.org/10.5281/zenodo.21953199

---

## Licence and credit

This repository is released under the MIT Licence; see [LICENSE](LICENSE).

`smev.py` is a lightly modified copy of the SMEV implementation from
**[pyTENAX](https://github.com/PetrVey/pyTENAX) v0.1.2**, © 2024 Petr Vey,
also MIT licensed. The numerical core of the estimator is unchanged; the
modifications are documented in full in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

**If you use this code, please cite pyTENAX and the SMEV method papers as well
as this repository.** The relevant references are listed in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
