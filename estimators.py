"""Four estimators of the largest Lyapunov exponent; neighbour search, fitting and bootstrap live in core."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

import core
from core import Z, block_bootstrap, fit_region, neighbours, saturation_onset

NAN = float("nan")


@dataclass
class Result:
    method: str
    lam: float = NAN                       # lambda1, 1/time
    ci: dict | None = None                 # block-bootstrap interval, None = refused
    se_ols: float = NAN                    # OLS slope SE: a LOWER bound (correlated residuals)
    window: tuple[int, int] | None = None  # fit window in samples
    fit: dict | None = None
    reliable: bool = False
    notes: list[str] = field(default_factory=list)
    plot: dict = field(default_factory=dict)
    p_surr: float = NAN
    surr: np.ndarray = field(default_factory=lambda: np.zeros(0))
    check: str = ""


def _log_divergence(Y: np.ndarray, rows: np.ndarray, cols: np.ndarray, masks: list[np.ndarray],
                    ks: range, window: tuple[int, int] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """For each reference i (grouped rows) and step k: ln < |Y_{i+k} - Y_{j+k}| >_j over its neighbours j,
    one neighbourhood per mask. Returns the reference-averaged curves (masks x ks) and, over window=(a,b),
    each reference's OLS slope. Accumulated step by step; no pairs x horizon array is ever formed."""
    if len(rows) * (Y.shape[1] + 4) * 8 * 4 > (core.MEM_BYTES or 2**30):   # per-step temporaries
        raise core.PipelineError(f"{len(rows)} pairs exceed the memory budget; raise --max-mem-gb")
    _, g = np.unique(rows, return_inverse=True)
    G = int(g.max()) + 1 if g.size else 0
    cnt = [np.bincount(g[mk], minlength=G).astype(float) for mk in masks]
    curves = np.full((len(masks), len(ks)), NAN)
    acc = np.zeros((len(masks), 5, G))
    for c, k in enumerate(ks):
        d = np.linalg.norm(Y[rows + k] - Y[cols + k], axis=1)
        for l, mk in enumerate(masks):
            with np.errstate(divide="ignore", invalid="ignore"):
                ly = np.log(np.bincount(g[mk], d[mk], G) / cnt[l])   # zero distances -> -inf -> dropped
            ok = np.isfinite(ly)
            if ok.any():
                curves[l, c] = ly[ok].mean()
            if window and window[0] <= k < window[1]:
                t, a = k - window[0], acc[l]
                a[0, ok] += 1
                a[1, ok] += t
                a[2, ok] += t * t
                a[3, ok] += ly[ok]
                a[4, ok] += t * ly[ok]
    n, s1, s2, sy, sty = acc.transpose(1, 0, 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        den = n * s2 - s1**2
        slopes = np.where(den > 0, (n * sty - s1 * sy) / den, NAN)
    return curves, slopes


def _finish(res: Result, units: np.ndarray, block: int, n_boot: int, rng: np.random.Generator) -> Result:
    """lambda1 = mean of per-unit values; block-bootstrap interval or a refusal."""
    res.lam = float(np.nanmean(units))
    if n_boot == 0:                         # surrogate runs need lambda only
        return res
    res.ci = block_bootstrap(units, block, n_boot, rng)
    if res.ci is None:
        res.notes.append(f"too few blocks of {block} units for an interval (< {core.MIN_BLOCKS}); refused")
    return res


def _adaptive(Y: np.ndarray, pairs_for: callable, horizon0: int, period: float, mid: int):
    """Extend the horizon (doubling, capped at N/2) until the divergence curve approaches its plateau."""
    H = max(horizon0, 5)
    while True:
        rows, cols, masks, extra = pairs_for(len(Y) - H)
        curves, _ = _log_divergence(Y, rows, cols, masks, range(H))
        if saturation_onset(curves[mid], period) is not None or H >= len(Y) // 2:
            return curves, rows, cols, masks, extra, H
        H = min(2 * H, len(Y) // 2)


def rosenstein(Y: np.ndarray, dt: float, theiler: int, rmin: float, period: float, r2_min: float,
               horizon0: int, n_boot: int, rng: np.random.Generator,
               window: tuple[int, int] | None = None) -> Result:
    """Rosenstein, Collins & De Luca (1993), Physica D 65:117.

    y(k) = < ln d_j(k) >_j,   d_j(k) = |Y_{j+k} - Y_{nn(j)+k}|,   lambda1 = dy/dk / dt in the scaling region,
    nn(j) the nearest neighbour with |j - nn(j)| > theiler and d > rmin. Logarithms are averaged, never
    distances (ln<d> adds a spurious term from the variance of finite-time exponents). The tree holds
    Y[:M], M = N - horizon, so both trajectories stay in range; zero distances are dropped."""
    res = Result("Rosenstein")

    def pairs(M: int):
        nb = neighbours(Y, theiler, rmin, k=1, n_tree=M, queries=np.arange(M))
        return nb.rows, nb.cols, [np.ones(len(nb.rows), bool)], nb

    if window is None:
        curves, rows, cols, masks, nb, H = _adaptive(Y, pairs, horizon0, period, 0)
        fit = fit_region(curves[0], dt, period, r2_min)
        res.plot = {"y": curves[0], "fit": fit}
        res.notes.append(f"{nb.excluded_theiler} Theiler-, {nb.excluded_rmin} r_min-excluded, "
                         f"{nb.unresolved} unresolved")
        if not fit["ok"]:
            res.notes.append("no scaling region (R^2, one e-fold, one period); no estimate")
            return res
        window, res.fit, res.se_ols = (fit["a"], fit["b"]), fit, fit["se_ols"]
    else:
        rows, cols, masks, _ = pairs(len(Y) - window[1])
    res.window = window
    _, slopes = _log_divergence(Y, rows, cols, masks, range(*window), window)
    _finish(res, slopes[0] / dt, theiler, n_boot, rng)
    res.reliable = res.ci is not None
    return res


def kantz(Y: np.ndarray, dt: float, theiler: int, rmin: float, ladder: np.ndarray, c_ladder: np.ndarray,
          period: float, r2_min: float, horizon0: int, n_boot: int, rng: np.random.Generator,
          max_pairs: int | None = None, window: tuple[int, int] | None = None) -> Result:
    """Kantz (1994), Phys. Lett. A 185:77.

    S(eps, k) = < ln( < |Y_{i+k} - Y_{j+k}| >_{j in U_eps(i)} ) >_i : distances are averaged WITHIN each
    neighbourhood first, then logged, then averaged over references. lambda1 = dS/dk / dt in the scaling
    region at the middle radius; the slope spread over the eps ladder (same window) is the collapse diagnostic.
    References are strided so the pair count stays <= max_pairs (default N: Rosenstein's per-step cost)."""
    res = Result("Kantz")
    n = len(Y)
    mid = len(ladder) // 2
    stride = min(theiler, max(1, math.ceil(c_ladder[-1] * n * n / (max_pairs or n))))  # >= 1 ref per window

    def pairs(M: int):
        nb = neighbours(Y, theiler, rmin, radius=float(ladder[-1]), queries=np.arange(0, M, stride), n_tree=M)
        return nb.rows, nb.cols, [nb.dist <= e for e in ladder], nb

    block = math.ceil(theiler / stride)
    if window is None:
        curves, rows, cols, masks, nb, H = _adaptive(Y, pairs, horizon0, period, mid)
        fit = fit_region(curves[mid], dt, period, r2_min)
        res.plot = {"curves": curves, "ladder": ladder, "fit": fit}
        res.notes.append(f"{len(np.unique(rows))} references (stride {stride}), {len(rows)} pairs")
        if not fit["ok"]:
            res.notes.append("no scaling region at the middle radius; no estimate")
            return res
        window, res.fit, res.se_ols = (fit["a"], fit["b"]), fit, fit["se_ols"]
        a, b = window
        spread = [core._ols(c[a:b])[0] / dt for c in curves if np.all(np.isfinite(c[a:b]))]
        res.plot["slopes"] = spread
    else:
        rows, cols, masks, _ = pairs(n - window[1])
        masks = [masks[mid]]
        mid = 0
    res.window = window
    _, slopes = _log_divergence(Y, rows, cols, [masks[mid]], range(*window), window)
    _finish(res, slopes[0] / dt, block, n_boot, rng)
    res.reliable = res.ci is not None
    if "slopes" in res.plot and res.ci is not None:
        width = np.ptp(res.plot["slopes"])
        collapse = width <= res.ci["hi"] - res.ci["lo"]
        res.notes.append(f"slope spread over {len(ladder)} radii {width:.4f} = {width / (res.ci['hi'] - res.ci['lo']):.2f}"
                         f" CI widths ({'collapse' if collapse else 'no full collapse: inspect the eps family'})")
    return res


def wolf(Y: np.ndarray, dt: float, theiler: int, rmin: float, d_min: float, d_max: float, evolve: int,
         n_boot: int, rng: np.random.Generator) -> Result:
    """Wolf, Swift, Swinney & Vastano (1985), Physica D 16:285.

    Follow the reference trajectory i and one neighbour j for `evolve` samples:
    lambda1 = sum ln(d'/d) / (n_evolutions * evolve * dt). When d' > d_max the neighbour is replaced by the
    point in [d_min, d_max] (Theiler-excluded) minimising the angle to the current separation vector.
    Stops when the REFERENCE reaches the end. `evolve` comes from a preliminary lambda, not the period."""
    res = Result("Wolf")
    n = len(Y)
    M = n - evolve
    tree = cKDTree(Y[:M])

    def replace(i: int, v: np.ndarray | None) -> tuple[int, bool]:
        nb = neighbours(Y, theiler, rmin, radius=d_max, queries=np.array([i]), n_tree=M, tree=tree)
        if v is not None and nb.cols.size:
            band = nb.dist >= d_min
            c, dd = (nb.cols[band], nb.dist[band]) if band.any() else (nb.cols, nb.dist)
            cos = np.abs((Y[c] - Y[i]) @ v) / (dd * np.linalg.norm(v))
            return int(c[np.argmax(cos)]), bool(band.any())
        nn = neighbours(Y, theiler, rmin, k=1, queries=np.array([i]), n_tree=M, tree=tree)
        if not nn.cols.size:
            raise core.PipelineError("Wolf: no neighbour for the reference trajectory")
        return int(nn.cols[0]), v is None

    i, (j, _) = 0, replace(0, None)
    lnr, n_ren, n_off = [], 0, 0
    while i + evolve < n:
        if j >= M:
            j, inband = replace(i, Y[min(j, n - 1)] - Y[i])
            n_ren, n_off = n_ren + 1, n_off + (not inband)
        d0 = np.linalg.norm(Y[j] - Y[i])
        d1 = np.linalg.norm(Y[j + evolve] - Y[i + evolve])
        if d1 > 0:
            lnr.append(math.log(d1 / d0))
        i, j = i + evolve, j + evolve
        if (d1 > d_max or d1 == 0) and i < M:
            j, inband = replace(i, Y[j] - Y[i])
            n_ren, n_off = n_ren + 1, n_off + (not inband)
    lnr = np.asarray(lnr)
    run = np.cumsum(lnr) / (np.arange(1, len(lnr) + 1) * evolve * dt)
    res.plot = {"running": run, "t": np.arange(1, len(lnr) + 1) * evolve * dt}
    _finish(res, lnr / (evolve * dt), math.ceil(theiler / evolve), n_boot, rng)
    half = run[len(run) // 2:]
    drift = core._ols(half)[0] * len(half) if len(half) > 2 else NAN
    plateau = res.ci is not None and abs(drift) <= Z * res.ci["se"]
    res.notes.append(f"evolve {evolve} samples, {n_ren} renormalisations ({n_off} outside the distance band), "
                     f"running-lambda drift over 2nd half {drift:+.4f} ({'plateaued' if plateau else 'NOT plateaued'})")
    res.reliable = bool(plateau)
    if not plateau:
        res.notes.append("running estimate has not plateaued: result refused as unreliable")
    return res


def jacobian(Y: np.ndarray, dt: float, theiler: int, rmin: float, k_nb: int, n_boot: int,
             rng: np.random.Generator) -> Result:
    """Sano & Sawada (1985), PRL 55:1082.

    Local linear map A_i from the k_nb neighbours of Y_i: (Y_{j+1} - Y_{i+1}) = A_i (Y_j - Y_i), solved by
    SVD (pseudo-inverse, never normal equations). Q_{i+1} R_i = A_i Q_i; lambda_l = < ln |R_i[l,l]| > / dt
    after discarding the first `theiler` steps of frame alignment. Reports lambda1 and, as a consistency
    check, the exponent nearest zero tested against its own bootstrap SE (a flow has one zero exponent)."""
    res = Result("Jacobian")
    n, m = Y.shape
    M = n - 1
    nb = neighbours(Y, theiler, rmin, k=k_nb, n_tree=M, queries=np.arange(M))
    sel = nb.found[nb.rows]
    R = nb.rows[sel].reshape(-1, k_nb)[:, 0]
    C = nb.cols[sel].reshape(-1, k_nb)
    if nb.unresolved:
        res.notes.append(f"{nb.unresolved} steps without {k_nb} neighbours skipped")
    dX, dX1 = Y[C] - Y[R][:, None], Y[C + 1] - Y[R + 1][:, None]
    sv = np.linalg.svd(dX, compute_uv=False)
    cond = sv[:, 0] / sv[:, -1]
    A = np.swapaxes(np.linalg.pinv(dX) @ dX1, 1, 2)
    Q = np.eye(m)
    logs = np.empty((len(A), m))
    for t in range(len(A)):
        Q, Rm = np.linalg.qr(A[t] @ Q)
        logs[t] = np.log(np.abs(np.diag(Rm)))
    logs = logs[theiler:] / dt
    if len(logs) == 0:
        res.notes.append("record shorter than the frame-alignment transient")
        return res
    lams = logs.mean(0)
    top = int(np.argmax(lams))
    _finish(res, logs[:, top], theiler, n_boot, rng)
    res.notes.append(f"k={k_nb} neighbours, condition number median {np.median(cond):.3g}, max {cond.max():.3g}")
    well = np.median(cond) < 1 / math.sqrt(np.finfo(float).eps)
    res.reliable = res.ci is not None and bool(well)
    if m >= 2:
        rest = [l for l in range(m) if l != top]
        z0 = min(rest, key=lambda l: abs(lams[l]))
        ci0 = block_bootstrap(logs[:, z0], theiler, n_boot, rng) if n_boot else None
        if ci0 is None:
            res.check = "zero-exponent check: refused (too few blocks)"
        else:
            nse = abs(lams[z0]) / ci0["se"]
            ok = nse <= Z
            res.check = f"zero-exponent check: {'PASS' if ok else 'FAIL'} ({nse:.2f} SE; lambda={lams[z0]:+.4f})"
            res.reliable &= bool(ok)
    res.plot = {"lams": lams}
    return res
