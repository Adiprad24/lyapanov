"""Shared machinery: loading, spectrum, timescales, embedding, neighbours, fitting, bootstrap, surrogates."""
from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import signal, stats
from scipy.ndimage import uniform_filter1d
from scipy.spatial import cKDTree

# --- cited constants (never tuned) ---
RTOL = 15.0                    # Kennel, Brown & Abarbanel (1992) PRA 45:3403, distance-ratio test
ATOL = 2.0                     # Kennel et al. (1992), attractor-size test
ALPHA = 0.05                   # significance level of every test and interval
INV_E = math.exp(-1.0)         # e-folding threshold
MAD_TO_SIGMA = 1.4826          # MAD -> Gaussian sigma (Rousseeuw & Croux 1993)
SECOND_DIFF = math.sqrt(6.0)   # std of x[i+1] - 2x[i] + x[i-1] for unit white noise
MIN_CYCLES = 5                 # a frequency must complete >= 5 cycles in the record (task specification)
K0, KGROW = 32, 4              # incremental neighbour schedule: start at k = 32, multiply by 4 (task specification)
# --- direct consequences of ALPHA ---
Z = float(stats.norm.isf(ALPHA / 2))                # two-sided normal quantile
N_SURR = math.ceil(1 / ALPHA) - 1                   # smallest rank test that can reach p = alpha
MIN_BLOCKS = math.ceil(2 / ALPHA)                   # each alpha/2 tail of a percentile interval holds >= 1 block
MIN_PAIRS = math.ceil(-math.log(ALPHA) / ALPHA)     # 0 false of n bounds a fraction by alpha ("rule of three")

NAN = float("nan")
WARNINGS: list[str] = []
MEM_BYTES: float = 0.0


class PipelineError(Exception):
    """A refusal: the data or arguments do not support an answer."""


def warn(msg: str) -> None:
    WARNINGS.append(msg)


def set_memory(gb: float | None) -> None:
    """Working-memory budget: --max-mem-gb, else a quarter of available RAM (room for the 3-4 temporaries
    each vectorised step allocates alongside its accumulator)."""
    global MEM_BYTES
    if gb is not None:
        MEM_BYTES = gb * 2**30
        return
    try:
        MEM_BYTES = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 4
    except (ValueError, OSError, AttributeError):
        MEM_BYTES = 2**30


# ----------------------------------------------------------------------------------------- loading
def load(path: str, column: str | None) -> tuple[np.ndarray, np.ndarray, str, list[str]]:
    """Read time + signal columns, skipping non-numeric lines; Tecplot VARIABLES names columns."""
    names: list[str] | None = None
    last_text: list[str] = []
    rows: list[list[float]] = []
    tstr: list[str] = []
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue
            if s.upper().startswith("VARIABLES"):
                rhs = s.split("=", 1)[-1]
                names = re.findall(r'"([^"]*)"', rhs) or rhs.replace(",", " ").split()
                continue
            parts = s.replace(",", " ").split()
            try:
                vals = [float(p) for p in parts]
            except ValueError:
                last_text = [p for p in parts if p != "#"]
                continue
            if not rows and names is None and len(last_text) == len(vals):
                names = last_text
            if rows and len(vals) != len(rows[0]):
                raise PipelineError(f"row {len(rows) + 1} has {len(vals)} columns, expected {len(rows[0])}")
            rows.append(vals)
            tstr.append(parts[0])
    if len(rows) < 2 or len(rows[0]) < 2:
        raise PipelineError("need a time column and at least one signal column")
    data = np.asarray(rows)
    ncol = data.shape[1]
    if not names or len(names) != ncol:
        names = ["t"] + [f"col{i}" for i in range(1, ncol)]
    if column is None:
        c = 1
    elif column.isdigit():
        c = int(column)
        if not 1 <= c < ncol:
            raise PipelineError(f"--column {c} out of range 1..{ncol - 1} (0 is time)")
    else:
        low = [n.lower() for n in names]
        if column.lower() not in low[1:]:
            raise PipelineError(f"--column {column!r} not among {names[1:]}")
        c = low.index(column.lower(), 1)
    return data[:, 0], data[:, c], names[c], tstr


def time_tolerance(tstr: list[str]) -> float:
    """Half a unit in the last printed place of the time column."""
    tol = 0.0
    for s in tstr:
        m = re.fullmatch(r"[-+]?\d*\.?(\d*)(?:[eEdD]([-+]?\d+))?", s)
        if m:
            tol = max(tol, 0.5 * 10.0 ** (int(m.group(2) or 0) - len(m.group(1))))
    return tol


def deduplicate(t: np.ndarray, x: np.ndarray, tol: float) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Merge restart overlaps: times equal within tol keep the LAST occurrence (stable sort)."""
    order = np.argsort(t, kind="stable")
    group = np.cumsum(np.r_[True, np.diff(t[order]) > tol]) - 1
    keep = np.full(group[-1] + 1, -1)
    np.maximum.at(keep, group, order)              # largest original index = last occurrence
    t, x = t[keep], x[keep]
    dts = np.diff(t)
    if np.any(dts <= tol):
        raise PipelineError("time not strictly increasing after de-duplication")
    dt = float(np.median(dts))
    if np.any(np.abs(dts - dt) > 2 * tol + 1e-9 * dt):
        raise PipelineError(f"non-uniform sampling: dt ranges {dts.min():.6g}..{dts.max():.6g}")
    return t, x, dt, len(order) - len(keep)


def noise_sigma(x: np.ndarray) -> float:
    """MAD-based white-noise std from second differences (robust to isolated spikes)."""
    d2 = np.diff(x, 2)
    return MAD_TO_SIGMA * float(np.median(np.abs(d2 - np.median(d2)))) / SECOND_DIFF


def validate(t: np.ndarray, x: np.ndarray) -> dict:
    """Refuse non-finite or constant data; report robust outliers and linear-trend significance."""
    bad = ~np.isfinite(x)
    if bad.any():
        raise PipelineError(f"{bad.sum()} non-finite values (first at t={t[bad][0]:.6g}); refusing to interpolate")
    if np.ptp(x) == 0:
        raise PipelineError("signal column is constant")
    n = len(x)
    med = np.median(x)
    sig = MAD_TO_SIGMA * np.median(np.abs(x - med))
    thr = stats.norm.isf(ALPHA / (2 * n))           # Bonferroni over n samples
    n_out = int(np.sum(np.abs(x - med) > thr * sig)) if sig > 0 else 0
    lr = stats.linregress(t, x)
    res = x - (lr.intercept + lr.slope * t)
    r1 = np.clip(np.corrcoef(res[:-1], res[1:])[0, 1], -0.999999, 0.999999)
    n_eff = max(3.0, n * (1 - r1) / (1 + r1))       # AR(1) effective sample size
    tstat = lr.slope / (lr.stderr * math.sqrt((n - 2) / max(n_eff - 2, 1)))
    p = float(2 * stats.t.sf(abs(tstat), max(n_eff - 2, 1)))
    if n_out:
        warn(f"{n_out} robust outliers beyond {thr:.1f} MAD-sigma (kept)")
    if p < ALPHA:
        warn(f"significant linear trend: {lr.slope * (t[-1] - t[0]) / np.std(x):+.2f} std over record (p={p:.3g})")
    return {"n_outliers": n_out, "trend_std": float(lr.slope * (t[-1] - t[0]) / np.std(x)), "trend_p": p}


def detect_transient(x: np.ndarray, period: float, sig_n: float) -> int:
    """First sample of the first block (MIN_CYCLES periods long) whose median Hilbert envelope has reached
    the stationary level of the second half (median - Z robust sigma)."""
    blk = max(1, int(round(MIN_CYCLES * period)))
    nb = len(x) // blk
    if nb < 4:
        return 0
    env = np.abs(signal.hilbert(x - np.median(x)))[: nb * blk].reshape(nb, blk)
    env = np.median(env, 1)            # median: Hilbert end effects (~1 period) touch < half a block
    ref = env[nb // 2:]
    s = max(MAD_TO_SIGMA * np.median(np.abs(ref - np.median(ref))), sig_n)
    first = int(np.argmax(env >= np.median(ref) - Z * s))
    return first * blk


def decimation_factor(x: np.ndarray, dt: float, sig_n: float) -> tuple[int, float]:
    """Largest factor keeping the highest frequency with significant power (above the white-noise floor at
    ALPHA) at <= 1 radian per sample -- the small-step regime the local-linear (Jacobian) map and the
    second-difference noise estimate assume -- with a further factor 2 clearing the FIR transition band."""
    nper = max(16, len(x) // MIN_CYCLES)
    f, p = signal.welch(x - x.mean(), fs=1 / dt, nperseg=nper)
    dof = 2 * (2 * len(x) // nper - 1)               # Welch, 50 % overlap
    floor = 2 * sig_n**2 * dt * stats.chi2.isf(ALPHA / len(f), dof) / dof      # Bonferroni over bins
    sig = np.nonzero(p[1:] > floor)[0]
    f_hi = f[1:][sig[-1]] if sig.size else f[-1]
    q = max(1, int(1 / (4 * math.pi * f_hi * dt)))
    while q > 1 and max(_factors(q)) > 13:
        q -= 1
    return q, float(f_hi)


def _factors(q: int) -> list[int]:
    out, d = [], 2
    while q > 1:
        while q % d == 0:
            out.append(d)
            q //= d
        d += 1
    return out or [1]


def decimate(x: np.ndarray, q: int) -> np.ndarray:
    """Zero-phase FIR decimation in stages <= 13, mean removed first (scipy zero-pads the edges)."""
    mu = x.mean()
    y = x - mu
    stages, f = [], 1
    for p in sorted(_factors(q), reverse=True):     # combine primes greedily into stages <= 13
        if f * p <= 13:
            f *= p
        else:
            stages.append(f)
            f = p
    stages.append(f)
    for s in stages:
        if s > 1:
            y = signal.decimate(y, s, ftype="fir", zero_phase=True)
    return y + mu


# ----------------------------------------------------------------------------------------- spectrum
def amplitude_spectrum(x: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    w = np.hanning(len(x))
    return np.fft.rfftfreq(len(x), dt), np.abs(np.fft.rfft((x - x.mean()) * w)) * 2 / w.sum()


def dominant_frequency(x: np.ndarray, dt: float, band: tuple[float, float]) -> float:
    """Hann-window peak inside `band` only, refined by parabolic interpolation of log amplitude
    (refinements beyond half a bin rejected)."""
    f, a = amplitude_spectrum(x, dt)
    sel = np.nonzero((f >= band[0]) & (f <= band[1]))[0]
    if sel.size == 0:
        raise PipelineError(f"analysis band {band} contains no frequency bins")
    k = sel[np.argmax(a[sel])]
    d = 0.0
    if 0 < k < len(a) - 1 and np.all(a[k - 1:k + 2] > 0):
        la, lb, lc = np.log(a[k - 1:k + 2])
        den = la - 2 * lb + lc
        if den < 0:
            d = 0.5 * (la - lc) / den
            d = d if abs(d) <= 0.5 else 0.0
    return float((k + d) * (f[1] - f[0]))


def analysis_band(n: int, dt: float, kind: str, f_shed: float | None,
                  band: tuple[float, float] | None) -> tuple[tuple[float, float], str]:
    """Explicit band, else +-f_s/2 around the expected harmonic (f_s lift, 2 f_s drag), else >= MIN_CYCLES."""
    nyq = 0.5 / dt
    if band is not None:
        return band, "--f-band"
    if f_shed is not None:
        c = f_shed * (2 if kind == "drag" else 1)
        return (c - f_shed / 2, min(c + f_shed / 2, nyq)), f"{kind}: centred on {c:.4g} (f_s={f_shed:.4g})"
    if kind == "drag":
        warn("drag without --shedding-freq: band cannot be centred on 2 f_s; using the default band")
    return (MIN_CYCLES / (n * dt), nyq), f"default: >= {MIN_CYCLES} cycles in record"


def f_dom_uncertainty(x: np.ndarray, dt: float, band: tuple[float, float], f0: float,
                      n_boot: int, rng: np.random.Generator) -> float:
    """Bootstrap SE of f_dom over record segments each holding MIN_CYCLES dominant cycles."""
    L = int(math.ceil(MIN_CYCLES / (f0 * dt)))
    k = len(x) // L
    if k < 2:
        return float("nan")
    est = np.array([dominant_frequency(x[i * L:(i + 1) * L], dt, band) for i in range(k)])
    return float(np.std(est[rng.integers(0, k, (n_boot, k))].mean(1), ddof=1))


# ----------------------------------------------------------------------------------------- timescales
def acf(x: np.ndarray) -> np.ndarray:
    """Biased autocorrelation via FFT."""
    n = len(x)
    y = x - x.mean()
    F = np.fft.rfft(y, 2 ** int(math.ceil(math.log2(2 * n))))
    r = np.fft.irfft(np.abs(F) ** 2)[:n]
    return r / r[0]


def correlation_time(x: np.ndarray, dt: float, f_dom: float, sig_n: float) -> dict:
    """dominant period, ACF-envelope 1/e decay, and envelope-modulation period, all in samples.
    The Theiler window is the largest finite one."""
    n = len(x)
    P = 1.0 / (f_dom * dt)
    r = acf(x)
    ra = np.abs(signal.hilbert(r[: n // MIN_CYCLES]))
    below = np.nonzero(ra < INV_E)[0]
    acf_decay = float(below[0]) if below.size else float("nan")
    if not below.size:
        warn(f"ACF envelope does not fall below 1/e within N/{MIN_CYCLES}: periodic or quasi-periodic")
    # spectrum of the Hilbert amplitude envelope; edges (one period) trimmed against Hilbert end effects
    e = np.abs(signal.hilbert(x - x.mean()))[int(P): n - int(P)]
    f, a = amplitude_spectrum(e, dt)
    sel = np.nonzero((f >= MIN_CYCLES / (len(e) * dt)) & (f < f_dom / 2))[0]   # modulation slower than carrier/2
    env_period = float("nan")
    if sel.size > 2:
        p = a**2
        k = sel[np.argmax(p[sel])]
        lo, hi = np.searchsorted(f, [f[k] / math.sqrt(2), f[k] * math.sqrt(2)])   # octave background
        bg = np.median(p[max(lo, sel[0]):hi + 1])
        # periodogram ordinate / median ~ Exp(1)/ln2; Bonferroni over the searched bins
        if p[k] > bg * math.log2(sel.size / ALPHA) and a[k] > sig_n:
            env_period = 1.0 / (f[k] * dt)
    vals = {"dominant_period": P, "acf_decay": acf_decay, "envelope_period": env_period}
    finite = {k: v for k, v in vals.items() if np.isfinite(v)}
    binding = max(finite, key=finite.get)
    return {**vals, "theiler": int(math.ceil(finite[binding])), "binding": binding, "acf": r}


# ----------------------------------------------------------------------------------------- embedding
def embed(x: np.ndarray, tau: int, m: int) -> np.ndarray:
    """Y_i = [x_i, x_{i+tau}, ..., x_{i+(m-1)tau}]."""
    w = np.lib.stride_tricks.sliding_window_view(x, (m - 1) * tau + 1)
    return np.ascontiguousarray(w[:, ::tau])


def ami(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Average mutual information I(lag), histogram estimator, Sturges (1926) bins."""
    n = len(x)
    bins = int(math.ceil(math.log2(n))) + 1
    b = np.minimum(((x - x.min()) / np.ptp(x) * bins).astype(int), bins - 1)
    out = np.zeros(max_lag + 1)
    for lag in range(max_lag + 1):
        u, v = b[: n - lag], b[lag:]
        pj = np.bincount(u * bins + v, minlength=bins * bins).reshape(bins, bins) / len(u)
        pu, pv = pj.sum(1), pj.sum(0)
        nz = pj > 0
        out[lag] = np.sum(pj[nz] * np.log(pj[nz] / np.outer(pu, pv)[nz]))
    return out


def delay_tau(x: np.ndarray, period: float, longest: float, r: np.ndarray,
              prominence: float) -> tuple[int, str, np.ndarray]:
    """First minimum of smoothed AMI whose rise to the next maximum exceeds `prominence` x the total decline;
    fall back to AMI < I(1)/e, then the ACF zero crossing, then a quarter period."""
    max_lag = int(min(len(x) // 2, math.ceil(2 * longest)))
    I = ami(x, max_lag)
    # smoothing width period/16: a quarter of the nominal quarter-period minimum, so it cannot erase it
    Is = uniform_filter1d(I, max(1, int(round(period / 16))), mode="nearest")
    decline = Is[1] - Is[1:].min()
    for k in range(2, max_lag):
        if Is[k] <= Is[k - 1] and Is[k] < Is[k + 1]:
            j = k
            while j < max_lag and Is[j + 1] >= Is[j]:
                j += 1
            if Is[j] - Is[k] >= prominence * decline:
                return k, "AMI first prominent minimum", I
    below = np.nonzero(I[1:] < I[1] * INV_E)[0]
    if below.size:
        warn("AMI has no prominent minimum; tau from AMI < I(1)/e")
        return int(below[0] + 1), "AMI < I(1)/e", I
    z = np.nonzero(r[1:] <= 0)[0]
    if z.size:
        warn("AMI fallback failed; tau from ACF first zero crossing")
        return int(z[0] + 1), "ACF zero crossing", I
    warn("all AMI/ACF criteria failed; tau = quarter period")
    return max(1, int(round(period / 4))), "quarter period", I


def r_min(sig_n: float, R_A: float, m: int) -> float:
    """Distances below this are noise or floating-point recurrence, not dynamics."""
    return max(sig_n, math.sqrt(np.finfo(float).eps) * R_A * math.sqrt(m))


def fnn(x: np.ndarray, tau: int, theiler: int, sig_n: float) -> tuple[int, list[dict]]:
    """Kennel, Brown & Abarbanel (1992) false nearest neighbours, both tests, Theiler window applied."""
    R_A = float(np.std(x))
    table: list[dict] = []
    for m in range(1, len(x)):
        n_vec = len(x) - m * tau
        if n_vec <= 2 * theiler + 1:
            break
        Y = embed(x, tau, m)[:n_vec]
        nb = neighbours(Y, theiler, r_min(sig_n, R_A, m), k=1)
        d = nb.dist
        dx = np.abs(x[m * tau + nb.rows] - x[m * tau + nb.cols])
        false = (dx / d > RTOL) | (np.hypot(d, dx) / R_A > ATOL)
        table.append({"m": m, "fnn": float(false.mean()) if d.size else float("nan"), "usable": int(d.size),
                      "excluded": nb.excluded_theiler + nb.excluded_rmin, "unresolved": nb.unresolved})
        if d.size < MIN_PAIRS:
            raise PipelineError(f"FNN at m={m}: only {d.size} usable pairs (< {MIN_PAIRS}); no verdict")
        if table[-1]["fnn"] <= ALPHA:
            return m, table
        if m > 1 and table[-1]["fnn"] >= table[-2]["fnn"]:
            best = min(table, key=lambda r: r["fnn"])
            warn(f"FNN never fell to {ALPHA}; minimum {best['fnn']:.3f} at m={best['m']} (noise?)")
            return best["m"], table
    raise PipelineError("FNN: record too short for any embedding dimension")


# ----------------------------------------------------------------------------------------- neighbours
@dataclass
class Neighbours:
    rows: np.ndarray         # query index into Y
    cols: np.ndarray         # neighbour index into Y
    dist: np.ndarray
    found: np.ndarray        # per query: k neighbours (radius mode: >= 1) found
    excluded_theiler: int
    excluded_rmin: int
    unresolved: int


def neighbours(Y: np.ndarray, theiler: int, rmin: float, k: int = 1, radius: float | None = None,
               queries: np.ndarray | None = None, n_tree: int | None = None,
               tree: cKDTree | None = None) -> Neighbours:
    """k nearest (or all within `radius`) neighbours with |i-j| > theiler and distance > rmin.
    Queries incrementally: k = K0, then only unresolved rows again with k *= KGROW."""
    n_tree = len(Y) if n_tree is None else n_tree
    tree = cKDTree(Y[:n_tree]) if tree is None else tree
    q_all = np.arange(n_tree) if queries is None else np.asarray(queries)
    ub = np.inf if radius is None else radius
    need = 1 if radius is not None else k
    out_r, out_c, out_d = [], [], []
    found = np.zeros(len(q_all), bool)
    ex_t = ex_r = unres = 0
    pos, kq = np.arange(len(q_all)), min(K0, n_tree)
    while pos.size:
        nxt = []
        chunk = max(1, int((MEM_BYTES or 2**30) // (kq * 64)))
        for s in range(0, pos.size, chunk):
            p = pos[s:s + chunk]
            qi = q_all[p]
            d, j = tree.query(Y[qi], k=kq, distance_upper_bound=ub)
            d, j = d.reshape(len(p), kq), j.reshape(len(p), kq)
            real = j < n_tree
            bt = real & (np.abs(j - qi[:, None]) <= theiler)
            br = real & ~bt & (d <= rmin)
            ok = real & ~bt & ~br
            full = kq >= n_tree
            if radius is None:
                ahead = np.cumsum(ok, 1) - ok
                take, looked = ok & (ahead < k), ahead < k
                done = (take.sum(1) >= k) | full
            else:
                take, looked = ok, real
                done = ~real[:, -1] | full
            ex_t += int((bt & looked)[done].sum())
            ex_r += int((br & looked)[done].sum())
            got = take[done].sum(1) >= need
            found[p[done]] = got
            unres += int((~got).sum())
            rr, cc = np.nonzero(take[done])
            out_r.append(qi[done][rr])
            out_c.append(j[done][rr, cc])
            out_d.append(d[done][rr, cc])
            nxt.append(p[~done])
        pos = np.concatenate(nxt)
        kq = min(kq * KGROW, n_tree)
    rows, cols, dist = (np.concatenate(a) if a else np.zeros(0) for a in (out_r, out_c, out_d))
    o = np.argsort(rows, kind="stable")
    return Neighbours(rows[o].astype(int), cols[o].astype(int), dist[o], found, ex_t, ex_r, unres)


# ----------------------------------------------------------------------------------------- correlation sum
def correlation_sum(Y: np.ndarray, theiler: int, rmin: float, R_A: float) -> dict:
    """C(eps) over pairs with |i-j| > theiler and distance > rmin, its local slope d log C / d log eps, and the
    plateau (dimension >= 1: point-like recurrence of a sampled periodic orbit is not a scaling region).
    Slope uncertainty: delete-one-block jackknife over MIN_BLOCKS blocks of reference index."""
    n, m = Y.shape
    n_r = int(math.ceil(math.log2(float(n) * n))) + 1          # Sturges on the number of pairs
    radii = np.geomspace(rmin, R_A * math.sqrt(m), n_r)
    tree = cKDTree(Y)
    blocks = np.array_split(np.arange(n), MIN_BLOCKS)
    blk = np.repeat(np.arange(MIN_BLOCKS), [len(b) for b in blocks])
    cnt = np.array([cKDTree(Y[b]).count_neighbors(tree, radii) for b in blocks], float)
    for lag in range(theiler + 1):                               # remove temporally close pairs (and self)
        ri = np.searchsorted(radii, np.linalg.norm(Y[lag:] - Y[: n - lag], axis=1))
        for owner in ([blk[: n - lag]] if lag == 0 else [blk[: n - lag], blk[lag:]]):
            h = np.bincount(owner * (n_r + 1) + ri, minlength=MIN_BLOCKS * (n_r + 1))
            cnt -= np.cumsum(h.reshape(MIN_BLOCKS, n_r + 1), 1)[:, :n_r]
    cnt -= cnt[:, :1]                                            # distances <= r_min are not dynamics
    i = np.arange(n)
    per = n - 1 - np.minimum(i, theiler) - np.minimum(n - 1 - i, theiler)
    valid = np.bincount(blk, per.astype(float))
    lr = np.log(radii)
    with np.errstate(divide="ignore", invalid="ignore"):
        C = cnt.sum(0) / valid.sum()
        s = np.gradient(np.log(C), lr)
        jk = np.array([np.gradient(np.log((cnt.sum(0) - cnt[b]) / (valid.sum() - valid[b])), lr)
                       for b in range(MIN_BLOCKS)])
    se = np.sqrt((MIN_BLOCKS - 1) / MIN_BLOCKS * np.sum((jk - jk.mean(0)) ** 2, 0))
    elig = (radii <= 0.5 * R_A) & np.isfinite(s) & np.all(np.isfinite(jk), 0) & (se > 0)
    best, best_key = None, None
    for a in np.nonzero(elig)[0]:
        for b in range(a + 1, n_r):
            if not elig[b]:
                break
            if lr[b] - lr[a] < 1.0:                              # plateau must span one e-fold of eps
                continue
            seg = s[a:b + 1]
            if seg.mean() + Z * se[a:b + 1].mean() < 1.0:          # a flow's trajectory has dimension >= 1
                continue
            dev = float(np.max(np.abs(seg - seg.mean()) / se[a:b + 1]))
            key = (dev <= Z, (lr[b] - lr[a]) if dev <= Z else -dev, -dev)
            if best_key is None or key > best_key:
                best, best_key = (a, b), key
    out = {"radii": radii, "C": C, "slope": s, "slope_se": se, "lo": None, "hi": None,
           "r_lo": NAN, "r_hi": NAN, "D2": NAN, "D2_se": NAN, "provisional": True}
    if best is None:
        warn("correlation sum: no radius range of one e-fold between r_min and 0.5 R_A; no scaling radii")
        return out
    a, b = best
    provisional = not best_key[0]
    if provisional:
        warn(f"no flat correlation-sum plateau (|slope - mean| <= {Z:.2f} SE); radii PROVISIONAL")
    return {**out, "lo": a, "hi": b, "r_lo": float(radii[a]), "r_hi": float(radii[b]),
            "D2": float(s[a:b + 1].mean()), "D2_se": float(se[a:b + 1].mean()), "provisional": provisional}


# ----------------------------------------------------------------------------------------- fitting
def _sg(y: np.ndarray, period: float) -> np.ndarray:
    """Savitzky-Golay first derivative over one dominant period (averages the oscillation out)."""
    w = min(max(5, int(period) | 1), len(y) - (1 - len(y) % 2))
    return signal.savgol_filter(y, w, 2, deriv=1) if w >= 3 else np.gradient(y)


def saturation_onset(y: np.ndarray, period: float) -> int | None:
    """First step where the smoothed curve comes within one e-fold (1 in ln units) of its maximum, provided a
    plateau is confirmed: the second half of the curve lies after it and grows at < 1/e the approach rate."""
    if len(y) < 5 or not np.all(np.isfinite(y)):
        return None
    w = min(max(5, int(period) | 1), len(y) - (1 - len(y) % 2))
    ys = signal.savgol_filter(y, w, 2) if w >= 3 else y
    top = float(ys.max())
    onset = int(np.argmax(ys >= top - 1.0))
    half = len(ys) // 2
    if onset == 0 or half < onset:
        return None
    return onset if _ols(ys[half:])[0] < INV_E * (ys[onset] - ys[0]) / onset else None


def _ols(y: np.ndarray) -> tuple[float, float, float]:
    k = np.arange(len(y), dtype=float)
    kc = k - k.mean()
    slope = float(np.sum(kc * (y - y.mean())) / np.sum(kc**2))
    res = y - y.mean() - slope * kc
    ss = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - float(np.sum(res**2)) / ss if ss > 0 else 0.0
    se = math.sqrt(np.sum(res**2) / max(len(y) - 2, 1) / np.sum(kc**2))
    return slope, se, r2


def fit_region(y: np.ndarray, dt: float, period: float, r2_min: float,
               window: tuple[int, int] | None = None) -> dict:
    """Scaling region of a divergence curve: R^2 >= r2_min, then the flatter half of local slopes, then the
    longest; must grow by >= one e-fold, span >= one dominant period (else the in-cycle oscillation of the curve passes for
    growth) and end before saturation. Slope per unit time; its OLS SE is a LOWER bound."""
    s = _sg(y, period) if np.all(np.isfinite(y)) and len(y) >= 5 else np.gradient(y)
    onset = saturation_onset(y, period)
    end = onset if onset is not None else len(y)
    if window is None:
        step = max(1, int(period // 2))          # boundaries finer than half the dominant period are not resolved
        cand = []
        for a in range(0, end, step):
            for b in range(a + max(3, 2 * step), end + 1, step):    # >= one dominant period
                seg = y[a:b]
                if not np.all(np.isfinite(seg)):
                    continue
                sl, se, r2 = _ols(seg)
                if r2 >= r2_min and sl * (b - a) >= 1.0:   # >= one e-fold: else exponential is untested
                    cand.append((float(np.std(s[a:b]) / sl), b - a, a, b))
        if not cand:
            return {"ok": False, "onset": onset, "local_slope": s / dt}
        cut = np.median([c[0] for c in cand])
        _, _, a, b = max((c for c in cand if c[0] <= cut), key=lambda c: (c[1], -c[0]))
    else:
        a, b = window
    sl, se, r2 = _ols(y[a:b])
    return {"ok": True, "a": a, "b": b, "slope": sl / dt, "se_ols": se / dt, "r2": r2, "onset": onset,
            "seconds": (b - a) * dt, "periods": (b - a) / period, "local_slope": s / dt}


# ----------------------------------------------------------------------------------------- uncertainty
def block_bootstrap(units: np.ndarray, block_length: int, n_boot: int, rng: np.random.Generator) -> dict | None:
    """Circular moving-block bootstrap of the mean of per-reference `units` (NaN ignored).
    Returns None (refusal) when fewer than MIN_BLOCKS blocks fit in the series."""
    if n_boot < 200:
        raise PipelineError("n_boot must be >= 200")
    u = np.asarray(units, float)
    n, L = len(u), max(1, int(block_length))
    if n // L < MIN_BLOCKS:
        return None
    nb = int(math.ceil(n / L))
    off = np.arange(L)
    boots = np.empty(n_boot)
    step = max(1, int((MEM_BYTES or 2**30) // (16 * nb * L)))
    for s in range(0, n_boot, step):
        st = rng.integers(0, n, (min(step, n_boot - s), nb))
        idx = ((st[:, :, None] + off) % n).reshape(len(st), -1)[:, :n]
        boots[s:s + len(st)] = np.nanmean(u[idx], 1)
    lo, hi = np.quantile(boots, [ALPHA / 2, 1 - ALPHA / 2])
    return {"lo": float(lo), "hi": float(hi), "se": float(np.std(boots, ddof=1)),
            "contains_zero": bool(lo <= 0 <= hi), "boots": boots}


# ----------------------------------------------------------------------------------------- surrogates
def iaaft(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Iterative amplitude-adjusted Fourier-transform surrogate (Schreiber & Schmitz 1996); iterates until
    the spectral error stops decreasing."""
    amp, xs = np.abs(np.fft.rfft(x)), np.sort(x)
    s, err = rng.permutation(x), np.inf
    while True:
        S = np.fft.rfft(s)
        s_new = xs[np.argsort(np.argsort(np.fft.irfft(amp * np.exp(1j * np.angle(S)), len(x))))]
        e = float(np.mean((np.abs(np.fft.rfft(s_new)) - amp) ** 2))
        if e >= err:
            return s
        s, err = s_new, e


def surrogate_test(x: np.ndarray, lam: float, estimate: Callable[[np.ndarray], float],
                   rng: np.random.Generator) -> tuple[float, np.ndarray]:
    """Rank p-value of lam against N_SURR IAAFT surrogates under the same estimator and settings.
    Statistic |lambda - median(surrogates)|, two-sided: Wolf and Jacobian give large positive values on
    stochastic surrogates, Rosenstein and Kantz give ~0, so only "different from the surrogates" is generic.
    The smallest attainable p is 1/(N_SURR + 1) = ALPHA."""
    lams = np.array([estimate(iaaft(x, rng)) for _ in range(N_SURR)])
    ok = lams[np.isfinite(lams)]
    if not np.isfinite(lam) or ok.size == 0:
        return float("nan"), lams
    c = np.median(ok)
    return float((1 + np.sum(np.abs(ok - c) >= abs(lam - c))) / (1 + ok.size)), lams
