"""
Equivalence tests for the fast bootstrap in ``bootstrap.py`` implemented here.

``bootstrap.bootstrap_cell`` is a vectorised rewrite of the loop in
``smev.SMEV.smev_bootstrap_uncertainty`` (derived from pyTENAX v0.1.2). It is
only legitimate to publish return-level confidence intervals from the fast
version if it agrees with the reference implementation.

Run with:
    python -m pytest tests/ -v
or without pytest:
    python tests/test_bootstrap.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bootstrap import bootstrap_cell, cell_rng  # noqa: E402
from smev import SMEV  # noqa: E402


# HELPERS

LEFT_CENSORING = [0.85, 1]
RETURN_PERIODS = np.array([2, 5, 10, 20, 50, 100], dtype=float)


def _synthetic_cell(seed=0, n_years=30, per_year=40, shape=0.7, scale=5.0):
    """Weibull-distributed ordinary events spread over whole years."""
    rng = np.random.default_rng(seed)
    years = np.repeat(np.arange(1990, 1990 + n_years), per_year)
    P = scale * rng.weibull(shape, size=years.size)
    P = P[np.isfinite(P) & (P > 0)]
    years = years[: P.size]
    return P, years


class _FixedRNG:
    """Stands in for a Generator, returning a predetermined index matrix.

    Lets the fast implementation be driven with exactly the same year draws as
    the reference implementation.
    """

    def __init__(self, randy):
        self._randy = randy

    def integers(self, low, high=None, size=None):
        assert self._randy.shape == size, (
            f"unexpected draw shape {size}, expected {self._randy.shape}"
        )
        return self._randy


def _reference_bootstrap(P, years, randy, n, rp, left_censoring):
    """Reference bootstrap: the pyTENAX loop, driven by a fixed ``randy``.

    This mirrors ``SMEV.smev_bootstrap_uncertainty``, but takes the
    year-index matrix as an argument instead of drawing it internally, so the
    comparison isolates the numerics.
    """
    engine = SMEV(
        return_period=list(rp),
        durations=[1],
        time_resolution=60,
        left_censoring=list(left_censoring),
    )
    blocks = np.unique(years)
    n_blocks = blocks.size
    niter = randy.shape[1]
    rl_unc = np.full((niter, len(rp)), np.nan)

    for ii in range(niter):
        pr = []
        for iy in range(n_blocks):
            selected = years == blocks[randy[iy, ii]]
            pr.append(P[selected])
        pr = np.concatenate(pr)
        try:
            smev_shape, smev_scale = engine.estimate_smev_parameters(
                pr, list(left_censoring)
            )
            rl_unc[ii, :] = engine.smev_return_values(
                rp, smev_shape, smev_scale, n
            )
        except Exception:
            pass
    return rl_unc


def _compare(P, years, niter=200, seed=1234, label=""):
    """Run both implementations on identical draws and return the max error."""
    blocks = np.unique(years)
    n_blocks = blocks.size
    n = float(P.size) / float(n_blocks)

    draw_rng = np.random.default_rng(seed)
    randy = draw_rng.integers(0, n_blocks, size=(n_blocks, niter))

    # Reference: per-iteration statsmodels OLS.
    ref = _reference_bootstrap(P, years, randy, n, RETURN_PERIODS, LEFT_CENSORING)
    ref_lo, ref_hi = np.nanpercentile(ref, [5.0, 95.0], axis=0)

    # Fast: closed-form OLS, batched by resample length.
    fast = bootstrap_cell(
        P=P,
        years=years,
        niter=niter,
        n=n,
        rp=RETURN_PERIODS,
        left_censoring=LEFT_CENSORING,
        ci_percentiles=[5.0, 95.0],
        rng=_FixedRNG(randy),
    )
    assert fast is not None, f"{label}: fast bootstrap returned None"

    # Compare as float32, since that is what the fast version writes to disk.
    ref_lo32 = ref_lo.astype(np.float32)
    ref_hi32 = ref_hi.astype(np.float32)
    rel_lo = np.abs(fast[0] - ref_lo32) / np.maximum(np.abs(ref_lo32), 1e-12)
    rel_hi = np.abs(fast[1] - ref_hi32) / np.maximum(np.abs(ref_hi32), 1e-12)
    return float(np.nanmax(np.concatenate([rel_lo, rel_hi]))), fast, (ref_lo32, ref_hi32)


# TESTS

def test_matches_reference_typical_cell():
    """Fast and reference bootstraps agree on a well-populated cell."""
    P, years = _synthetic_cell(seed=0)
    err, _, _ = _compare(P, years, label="typical")
    assert err < 1e-5, f"relative error {err:.3e} exceeds 1e-5"


def test_matches_reference_across_several_cells():
    """Agreement holds across cells with different sizes and shapes."""
    cases = [
        dict(seed=1, n_years=10, per_year=20, shape=0.5, scale=2.0),
        dict(seed=2, n_years=20, per_year=15, shape=0.9, scale=8.0),
        dict(seed=3, n_years=45, per_year=60, shape=0.7, scale=5.0),
        dict(seed=4, n_years=12, per_year=100, shape=1.2, scale=3.0),
    ]
    worst = 0.0
    for case in cases:
        P, years = _synthetic_cell(**case)
        err, _, _ = _compare(P, years, niter=120, label=str(case))
        worst = max(worst, err)
        assert err < 1e-5, f"case {case}: relative error {err:.3e}"
    print(f"    worst relative error across cells: {worst:.3e}")


def test_uneven_events_per_year():
    """Years contributing different numbers of events still agree.
    """
    rng = np.random.default_rng(7)
    years_list, p_list = [], []
    for y in range(1990, 2020):
        k = int(rng.integers(5, 80))
        years_list.append(np.full(k, y))
        p_list.append(5.0 * rng.weibull(0.7, size=k))
    years = np.concatenate(years_list)
    P = np.concatenate(p_list)
    err, _, _ = _compare(P, years, niter=150, label="uneven")
    assert err < 1e-5, f"relative error {err:.3e} exceeds 1e-5"


def test_returns_none_for_degenerate_input():
    """Cells that cannot be fitted return None rather than raising."""
    rng = np.random.default_rng(0)
    # A single event: every resample has length < 2.
    out = bootstrap_cell(
        P=np.array([3.0]),
        years=np.array([2000]),
        niter=50,
        n=1.0,
        rp=RETURN_PERIODS,
        left_censoring=LEFT_CENSORING,
        ci_percentiles=[5.0, 95.0],
        rng=rng,
    )
    assert out is None

    # No events at all.
    out = bootstrap_cell(
        P=np.array([]),
        years=np.array([]),
        niter=50,
        n=1.0,
        rp=RETURN_PERIODS,
        left_censoring=LEFT_CENSORING,
        ci_percentiles=[5.0, 95.0],
        rng=rng,
    )
    assert out is None


def test_output_shape_and_ordering():
    """Output is (2, n_return_periods) with the low percentile first."""
    P, years = _synthetic_cell(seed=11)
    out = bootstrap_cell(
        P=P,
        years=years,
        niter=100,
        n=float(P.size) / np.unique(years).size,
        rp=RETURN_PERIODS,
        left_censoring=LEFT_CENSORING,
        ci_percentiles=[5.0, 95.0],
        rng=np.random.default_rng(3),
    )
    assert out is not None
    assert out.shape == (2, RETURN_PERIODS.size)
    assert out.dtype == np.float32
    assert np.all(out[0] <= out[1]), "low percentile must not exceed high"
    # Return levels increase with return period.
    assert np.all(np.diff(out[0]) > 0)
    assert np.all(np.diff(out[1]) > 0)


def test_seeding_is_reproducible_and_position_dependent():
    """Same cell and seed give identical results; different cells differ."""
    P, years = _synthetic_cell(seed=5)
    n = float(P.size) / np.unique(years).size
    kwargs = dict(
        P=P, years=years, niter=100, n=n, rp=RETURN_PERIODS,
        left_censoring=LEFT_CENSORING, ci_percentiles=[5.0, 95.0],
    )
    a = bootstrap_cell(rng=cell_rng(12345, 0, 4, 7, 100, 100), **kwargs)
    b = bootstrap_cell(rng=cell_rng(12345, 0, 4, 7, 100, 100), **kwargs)
    c = bootstrap_cell(rng=cell_rng(12345, 0, 4, 8, 100, 100), **kwargs)
    assert np.array_equal(a, b), "same cell and seed must reproduce exactly"
    assert not np.array_equal(a, c), "different cells must draw differently"


def test_interval_brackets_point_estimate():
    """The bootstrap interval contains the point estimate.

    A sanity check on the pairing of the fit and the resampling: if the two
    used different censoring windows or a different n, this would fail.
    """
    P, years = _synthetic_cell(seed=21, n_years=40, per_year=50)
    n = float(P.size) / np.unique(years).size
    engine = SMEV(
        return_period=list(RETURN_PERIODS),
        durations=[1],
        time_resolution=60,
        left_censoring=LEFT_CENSORING,
    )
    shape, scale = engine.estimate_smev_parameters(P, LEFT_CENSORING)
    point = np.atleast_1d(
        engine.smev_return_values(RETURN_PERIODS, shape, scale, n)
    )
    ci = bootstrap_cell(
        P=P, years=years, niter=400, n=n, rp=RETURN_PERIODS,
        left_censoring=LEFT_CENSORING, ci_percentiles=[5.0, 95.0],
        rng=cell_rng(12345, 0, 0, 0, 10, 10),
    )
    assert ci is not None
    inside = (point >= ci[0]) & (point <= ci[1])
    assert inside.all(), (
        f"point estimate outside the 90 % interval at return periods "
        f"{RETURN_PERIODS[~inside]}"
    )


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for fn in tests:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"  FAIL  {fn.__name__}: {exc}")
        except Exception as exc:  # pragma: no cover
            failures += 1
            print(f"  ERROR {fn.__name__}: {type(exc).__name__}: {exc}")
    print()
    print(f"{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
