"""
bootstrap.py

Year-block bootstrap CIs for SMEV return levels. Single implementation, used
by both modes of ``01_areal_SMEV.py`` -- in-pass and ``--uncertainty-only`` --
so the two cannot be different.

For one cell and one duration: resample whole calendar years with replacement
``niter`` times (years as blocks, which keeps the within-year dependence
between events), refit the Weibull on each resample with the same censoring
window as the point estimate, turn each fit into return levels, then take two
percentiles of the resulting (niter, n_rp) matrix -- 5th and 95th by default,
so a 90 % interval.

This is a vectorised rewrite of the loop in
``smev.SMEV.smev_bootstrap_uncertainty`` (pyTENAX v0.1.2; see
THIRD_PARTY_NOTICES.md). 

``tests/test_bootstrap.py`` checks that against ``smev.py``. Driven with identical 
year draws the worst relative difference across the test cells was ~5e-8, below float32 
resolution (1.2e-7), so the two agree exactly at the precision actually written to disk. 
Speed-up on those cells was ~10x, though that depends on the events per cell and the BLAS build.

Randomness enters only via the ``rng`` argument. cell_rng() derives it from a
base seed plus the cell position.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

import numpy as np


__all__ = ["bootstrap_cell", "cell_rng"]


def cell_rng(
    base_seed: int,
    duration_index: int,
    row: int,
    col: int,
    n_rows: int,
    n_cols: int,
) -> np.random.Generator:
    """Generator for one (duration, row, col) cell of the output raster.
    """
    cell_id = (duration_index * n_rows + int(row)) * n_cols + int(col)
    return np.random.default_rng(np.random.SeedSequence([base_seed, cell_id]))


def bootstrap_cell(
    P: np.ndarray,
    years: np.ndarray,
    niter: int,
    n: float,
    rp: np.ndarray,
    left_censoring: Sequence[float],
    ci_percentiles: Sequence[float],
    rng: np.random.Generator,
) -> Optional[np.ndarray]:
    """Year-block bootstrap of SMEV return levels for one cell and duration.

    `P` must already be filtered to finite, positive values, and `years` gives
    the calendar year of each entry (these are the resampling blocks). `n` is
    the mean OE per year from the point-estimate fit, and `left_censoring`
    must match what the point estimate used.

    Returns (2, len(rp)) float32, low percentile on row 0, or None if no
    iteration gave a usable fit (too few events, or too few distinct years).

    Iterations that fail to produce a finite fit stay NaN and drop out of the
    final percentile, which mirrors the try/except in the reference version.
    """
    blocks = np.unique(years)
    n_blocks = blocks.size
    if n_blocks == 0:
        return None

    # Split the events once, by year, so each iteration only concatenates.
    block_values = [P[years == b] for b in blocks]
    block_lengths = np.array([v.size for v in block_values], dtype=np.int64)

    # randy[:, k] selects which years form the k-th resample.
    randy = rng.integers(0, n_blocks, size=(n_blocks, niter))
    total_lengths = block_lengths[randy].sum(axis=0)

    quantile = 1.0 - 1.0 / np.asarray(rp, dtype=np.float64)
    out = np.full((niter, quantile.size), np.nan, dtype=np.float64)

    # Group iterations by resample length: all iterations of a given length
    # share the same plotting positions, so they can be fitted in one go.
    for length in np.unique(total_lengths):
        cols = np.where(total_lengths == length)[0]
        if length < 2:
            continue

        sample = np.empty((cols.size, length), dtype=np.float64)
        for j, it in enumerate(cols):
            sample[j] = np.concatenate([block_values[k] for k in randy[:, it]])
        sample.sort(axis=1)

        # Plotting positions and the left-censoring window, matching
        # SMEV.estimate_smev_parameters.
        ecdf = np.arange(1, 1 + length) / (1 + length)
        fidx = max(1, math.floor(length * left_censoring[0]))
        tidx = math.ceil(length * left_censoring[1])
        to_use = np.arange(fidx - 1, tidx) if fidx == 1 else np.arange(fidx, tidx)
        if to_use.size < 2:
            continue

        # Closed-form OLS of log(x) on log(log(1/(1-F))).
        x = np.log(np.log(1.0 / (1.0 - ecdf[to_use])))
        Y = np.log(sample[:, to_use])
        x_mean = x.mean()
        x_centred = x - x_mean
        sxx = float((x_centred * x_centred).sum())
        if sxx == 0.0:
            continue
        slope = (Y * x_centred).sum(axis=1) / sxx
        intercept = Y.mean(axis=1) - slope * x_mean

        with np.errstate(divide="ignore", invalid="ignore"):
            shape = 1.0 / slope
            scale = np.exp(intercept)
            base = (-1.0) * np.log(1.0 - quantile[None, :] ** (1.0 / n))
            out[cols] = scale[:, None] * base ** (1.0 / shape[:, None])

    if not np.isfinite(out).any():
        return None

    low, high = np.nanpercentile(out, list(ci_percentiles), axis=0)
    return np.vstack([low, high]).astype(np.float32)
