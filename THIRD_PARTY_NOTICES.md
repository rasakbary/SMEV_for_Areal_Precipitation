# Third-party notices

This repository includes code derived from third-party open-source projects.
The notices below are reproduced as required by the respective licences.

---

## pyTENAX

**File in this repository:** `smev.py`

**Upstream project:** pyTENAX
**Upstream version:** v0.1.2
**Upstream file:** `src/pyTENAX/smev.py`
**Repository:** https://github.com/PetrVey/pyTENAX
**Release used:** https://github.com/PetrVey/pyTENAX/releases/tag/v0.1.2
**Licence:** MIT

### What was taken

`smev.py` in this repository is a lightly modified copy of the `SMEV` class
from pyTENAX v0.1.2, including the optional numba kernels
(`_smev_inner_loop_numba_seq`, `_smev_inner_loop_numba`).

### What was changed

The modifications are presentational and do not alter the statistical method.
Specifically:

- Docstrings converted from Google style to NumPy style, and expanded
  (notably for the two numba kernels).
- internal local variables were lowercased accordingly 
  (`RP` -> `rp`, `M` -> `n_blocks`, `RL_unc` -> `rl_unc`, `Pr` -> `pr`, `Bid` -> `bid`).
- Type hints on `smev_return_values` were widened to accept and return
  array-like input as well as scalars.

The numerical core of the estimator — `estimate_smev_parameters`
(Gringorten-style plotting position, left-censoring window, OLS fit in
log-log Weibull space) and `smev_return_values` — is algorithmically
unchanged from upstream.

### Upstream licence text

```
MIT License

Copyright (c) 2024 Petr Vey

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### How to cite pyTENAX and the underlying method

If you use this code, please also credit the upstream work:

- **Software:** Vey, P. *pyTENAX* (v0.1.2). https://github.com/PetrVey/pyTENAX
- **MEV method:** Marani, M., & Ignaccolo, M. (2015). A metastatistical
  approach to rainfall extremes. *Advances in Water Resources*, 79, 121-126.
  https://doi.org/10.1016/j.advwatres.2015.03.001
- **Simplified MEV (SMEV):** Marra, F., Nikolopoulos, E. I., Anagnostou, E. N.,
  & Morin, E. (2018). Metastatistical Extreme Value analysis of hourly rainfall
  from short records: Estimation of high quantiles and impact of measurement
  errors. *Advances in Water Resources*, 117, 27-39.
  https://doi.org/10.1016/j.advwatres.2018.05.001
