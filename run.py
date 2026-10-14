"""Estimate the largest Lyapunov exponent of a CFD force-coefficient series by four methods.

    python run.py DATA.plt --signal-kind lift [--column NAME] [overrides]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

import core
import estimators as est
from core import PipelineError


def parse(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("data")
    p.add_argument("--column", help="signal column name or file index (0 is time); default 1")
    p.add_argument("--signal-kind", choices=["lift", "drag"], default="lift")
    p.add_argument("--shedding-freq", help="f_s in Hz, or the JSON written by the lift run")
    p.add_argument("--f-band", nargs=2, type=float, metavar=("LO", "HI"))
    p.add_argument("--t-start", type=float, help="override detected transient end")
    p.add_argument("--decimate", type=int, help="override decimation factor")
    p.add_argument("--theiler", type=int, help="override Theiler window (samples, after decimation)")
    p.add_argument("--tau", type=int)
    p.add_argument("--ami-prominence", type=float, default=0.01,
                   help="AMI minimum must rise by this fraction of the total AMI decline (default 0.01)")
    p.add_argument("--m", type=int)
    p.add_argument("--r-min", type=float, help="override distance floor (standardised units)")
    p.add_argument("--radii", nargs=2, type=float, metavar=("LO", "HI"), help="override scaling radii")
    p.add_argument("--fit-window", nargs=2, type=int, metavar=("A", "B"), help="Rosenstein/Kantz fit window")
    p.add_argument("--r2-min", type=float, default=0.98)
    p.add_argument("--wolf-evolve", type=int, help="override Wolf evolution time (samples)")
    p.add_argument("--jacobian-k", type=int, help="override Jacobian neighbourhood size")
    p.add_argument("--kantz-pairs", type=int, help="override Kantz pair budget (default N)")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--no-surrogates", action="store_true")
    p.add_argument("--max-mem-gb", type=float)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", help="output prefix (default: DATA stem + _lyap)")
    a = p.parse_args(argv)
    checks = [
        (os.path.isfile(a.data), f"no such file: {a.data}"),
        (a.n_boot >= 200, "--n-boot must be >= 200"),
        (0 < a.r2_min < 1, "--r2-min must be in (0, 1)"),
        (a.f_band is None or 0 <= a.f_band[0] < a.f_band[1], "--f-band needs 0 <= LO < HI"),
        (a.radii is None or 0 < a.radii[0] < a.radii[1], "--radii needs 0 < LO < HI"),
        (a.fit_window is None or 0 <= a.fit_window[0] < a.fit_window[1] - 2, "--fit-window needs 0 <= A < B-2"),
        (0 < a.ami_prominence < 1, "--ami-prominence must be in (0, 1)"),
        (a.r_min is None or a.r_min > 0, "--r-min must be > 0"),
        (a.max_mem_gb is None or a.max_mem_gb > 0, "--max-mem-gb must be > 0"),
    ] + [(getattr(a, k) is None or getattr(a, k) >= 1, f"--{k.replace('_', '-')} must be >= 1")
         for k in ("decimate", "theiler", "tau", "m", "wolf_evolve", "jacobian_k", "kantz_pairs")]
    for ok, msg in checks:
        if not ok:
            raise PipelineError(msg)
    if a.jacobian_k is not None and a.m is not None and a.jacobian_k <= a.m:
        raise PipelineError("--jacobian-k must exceed --m")
    if a.shedding_freq is not None:
        try:
            a.shedding_freq = float(a.shedding_freq)
        except ValueError:
            try:
                with open(a.shedding_freq) as fh:
                    a.shedding_freq = float(json.load(fh)["f_dom"])
            except (OSError, ValueError, KeyError) as e:
                raise PipelineError(f"--shedding-freq: not a number or a lift-run JSON ({e})")
        if a.shedding_freq <= 0:
            raise PipelineError("--shedding-freq must be > 0")
    a.out = a.out or os.path.splitext(a.data)[0] + "_lyap"
    d = os.path.dirname(os.path.abspath(a.out))
    if not (os.path.isdir(d) and os.access(d, os.W_OK)):
        raise PipelineError(f"output directory not writable: {d}")
    return a


def analyse(a: argparse.Namespace) -> tuple[dict, dict]:
    rng = np.random.default_rng(a.seed)
    core.set_memory(a.max_mem_gb)
    P: dict = {}
    # --- load and prepare
    t, x, name, tstr = core.load(a.data, a.column)
    t, x, dt, ndup = core.deduplicate(t, x, core.time_tolerance(tstr))
    P.update(column=name, n_raw=len(tstr), duplicates=ndup, dt_raw=dt)
    P["validation"] = core.validate(t, x)
    sig_raw = core.noise_sigma(x)
    band, why = core.analysis_band(len(x), dt, a.signal_kind, a.shedding_freq,
                                   tuple(a.f_band) if a.f_band else None)
    f0 = core.dominant_frequency(x, dt, band)
    i0 = core.detect_transient(x, 1 / (f0 * dt), sig_raw)
    P["t_start_detected"] = float(t[i0])
    if a.t_start is not None:
        i0 = int(np.searchsorted(t, a.t_start))
    if len(x) - i0 < 4:
        raise PipelineError("nothing left after the transient")
    t, x = t[i0:], x[i0:]
    sig_kept = core.noise_sigma(x)          # estimated on the most oversampled series, where it is valid
    q, f_sig = core.decimation_factor(x, dt, sig_kept)
    q = a.decimate or q
    x, dt = core.decimate(x, q), dt * q
    sig_n = sig_kept / math.sqrt(q) / x.std()   # FIR keeps 1/q of the white-noise band; then standardise
    x = (x - x.mean()) / x.std()                                    # standardise the retained window
    n = len(x)
    P.update(t_start=float(t[0]), decimate=q, f_significant=f_sig, dt=dt, n=n)
    # --- spectrum and timescales
    band, why = core.analysis_band(n, dt, a.signal_kind, a.shedding_freq, tuple(a.f_band) if a.f_band else None)
    f_dom = core.dominant_frequency(x, dt, band)
    f_glob = core.dominant_frequency(x, dt, (1 / (n * dt), 0.5 / dt))
    if a.f_band is None and abs(f_glob - f_dom) > 1 / (n * dt):
        core.warn(f"in-band peak {f_dom:.5g} differs from the global peak {f_glob:.5g} (drift?)")
    P.update(band=list(band), band_source=why, f_dom=f_dom,
             f_dom_se=core.f_dom_uncertainty(x, dt, band, f_dom, a.n_boot, rng))
    ct = core.correlation_time(x, dt, f_dom, sig_n)
    period = ct["dominant_period"]
    theiler = a.theiler or ct["theiler"]
    P.update({k: ct[k] for k in ("dominant_period", "acf_decay", "envelope_period", "binding")},
             theiler=theiler, noise_sigma=sig_n)
    if a.theiler:
        P["binding"] = "--theiler"
    # --- embedding
    longest = max(v for v in (period, ct["envelope_period"]) if np.isfinite(v))
    tau, tau_why, ami = (a.tau, "--tau", None) if a.tau else core.delay_tau(x, period, longest, ct["acf"],
                                                                                 a.ami_prominence)
    if a.m:
        m, fnn_tab = a.m, []
    else:
        m, fnn_tab = core.fnn(x, tau, theiler, sig_n)
    Y = core.embed(x, tau, m)
    R_A = float(np.std(x))
    rmin = a.r_min or core.r_min(sig_n, R_A, m)
    P.update(tau=tau, tau_source=tau_why, m=m, fnn=fnn_tab, r_min=rmin, n_vectors=len(Y))
    if len(Y) // theiler < core.MIN_BLOCKS:
        core.warn(f"only {len(Y) // theiler} Theiler windows in the record: intervals will be refused")
    cs = core.correlation_sum(Y, theiler, rmin, R_A)
    if a.radii:
        cs.update(r_lo=a.radii[0], r_hi=a.radii[1], provisional=False)
        sel = (cs["radii"] >= a.radii[0]) & (cs["radii"] <= a.radii[1])
        cs["lo"], cs["hi"] = int(np.argmax(sel)), int(len(sel) - 1 - np.argmax(sel[::-1]))
    P.update(radii=[cs["r_lo"], cs["r_hi"]], D2=cs["D2"], D2_se=cs["D2_se"], radii_provisional=cs["provisional"])
    # --- estimators
    win = tuple(a.fit_window) if a.fit_window else None
    common = dict(dt=dt, theiler=theiler, rmin=rmin, n_boot=a.n_boot, rng=rng)
    R = {}
    R["Rosenstein"] = est.rosenstein(Y, period=period, r2_min=a.r2_min, horizon0=theiler, window=win, **common)
    have_radii = cs["lo"] is not None
    if have_radii:
        ladder = cs["radii"][cs["lo"]:cs["hi"] + 1]
        c_ladder = cs["C"][cs["lo"]:cs["hi"] + 1]
        R["Kantz"] = est.kantz(Y, ladder=ladder, c_ladder=c_ladder, period=period, r2_min=a.r2_min,
                               horizon0=theiler, max_pairs=a.kantz_pairs, window=win, **common)
    else:
        R["Kantz"] = est.Result("Kantz", notes=["no scaling radii from the correlation sum: refused"])
    lam_pre = R["Rosenstein"].lam
    if a.wolf_evolve:
        evolve = a.wolf_evolve
    elif np.isfinite(lam_pre) and lam_pre > 0:
        evolve = max(1, int(round(1 / (lam_pre * dt))))           # one preliminary Lyapunov time
    else:
        evolve = None
    if evolve and have_radii:
        R["Wolf"] = est.wolf(Y, d_min=cs["r_lo"], d_max=cs["r_hi"], evolve=evolve, **common)
    else:
        R["Wolf"] = est.Result("Wolf", notes=["no positive preliminary lambda or no scaling radii: refused"])
    k_nb = a.jacobian_k or (max(m + 1, int(round(cs["C"][cs["hi"]] * len(Y)))) if have_radii else None)
    if k_nb:
        R["Jacobian"] = est.jacobian(Y, k_nb=k_nb, **common)
    else:
        R["Jacobian"] = est.Result("Jacobian", notes=["no scaling radii for the neighbourhood size: refused"])
    P.update(wolf_evolve=evolve, jacobian_k=k_nb)
    if cs["provisional"]:
        for k in ("Kantz", "Wolf", "Jacobian"):
            R[k].reliable = False
            R[k].notes.append("radii provisional (no correlation-sum plateau)")
    # --- surrogates: same tau, m, Theiler, r_min, radii, fit windows and evolution time
    if not a.no_surrogates:
        quiet = dict(common, n_boot=0)
        runs = {
            "Rosenstein": lambda Ys: est.rosenstein(Ys, period=period, r2_min=a.r2_min, horizon0=theiler,
                                                    window=R["Rosenstein"].window, **quiet).lam,
            "Kantz": lambda Ys: est.kantz(Ys, ladder=ladder, c_ladder=c_ladder, period=period, r2_min=a.r2_min,
                                          horizon0=theiler, max_pairs=a.kantz_pairs,
                                          window=R["Kantz"].window, **quiet).lam,
            "Wolf": lambda Ys: est.wolf(Ys, d_min=cs["r_lo"], d_max=cs["r_hi"], evolve=evolve, **quiet).lam,
            "Jacobian": lambda Ys: est.jacobian(Ys, k_nb=k_nb, **quiet).lam,
        }
        for k, fn in runs.items():
            r = R[k]
            if not np.isfinite(r.lam) or (k in ("Rosenstein", "Kantz") and r.window is None):
                continue
            r.p_surr, r.surr = core.surrogate_test(x, r.lam, lambda s: fn(core.embed(s, tau, m)), rng)
            if r.ci is not None and r.ci["lo"] > 0 and r.p_surr > core.ALPHA:
                r.reliable = False
                r.notes.append(f"positive exponent not distinguished from IAAFT surrogates (p={r.p_surr:.2f})")
    P["cs"] = cs
    P["ami"] = ami
    return P, R


def report(P: dict, R: dict) -> None:
    per = P["dominant_period"]
    print(f"column {P['column']}: {P['n_raw']} rows, {P['duplicates']} duplicates removed, dt={P['dt_raw']:.6g}")
    v = P["validation"]
    print(f"validation: {v['n_outliers']} robust outliers; trend {v['trend_std']:+.3f} std/record (p={v['trend_p']:.3g})")
    print(f"transient: detected t={P['t_start_detected']:.6g}, used t={P['t_start']:.6g}")
    print(f"decimation x{P['decimate']} (significant power to {P['f_significant']:.4g} Hz) -> dt={P['dt']:.6g}, N={P['n']}")
    print(f"band [{P['band'][0]:.4g}, {P['band'][1]:.4g}] Hz ({P['band_source']})")
    print(f"f_dom = {P['f_dom']:.5g} +- {P['f_dom_se']:.2g} Hz")
    for k in ("dominant_period", "acf_decay", "envelope_period"):
        mark = "  <- binds" if P["binding"] == k else ""
        print(f"  {k:16s} {P[k]:10.1f} samples{mark}")
    print(f"Theiler window {P['theiler']} samples ({P['binding']}); noise sigma {P['noise_sigma']:.3g}")
    print(f"tau = {P['tau']} ({P['tau_source']}; {P['tau'] / per:.3f} periods)")
    for r in P["fnn"]:
        print(f"  FNN m={r['m']}: {r['fnn']:.4f} over {r['usable']} usable pairs "
              f"({r['excluded']} excluded, {r['unresolved']} unresolved)")
    print(f"m = {P['m']}; r_min = {P['r_min']:.4g}")
    prov = " PROVISIONAL" if P["radii_provisional"] else ""
    print(f"scaling radii [{P['radii'][0]:.4g}, {P['radii'][1]:.4g}]{prov}; D2 = {P['D2']:.3f} +- {P['D2_se']:.3f}")
    print(f"Wolf evolve = {P['wolf_evolve']} samples; Jacobian k = {P['jacobian_k']}")
    for k in ("Rosenstein", "Kantz"):
        f = R[k].fit
        if f:
            print(f"{k} fit window [{f['a']}, {f['b']}) samples = {f['seconds']:.4g} s = {f['periods']:.2f} periods, "
                  f"R^2={f['r2']:.4f}, OLS SE {f['se_ols']:.2g} (lower bound)")
    for w in core.WARNINGS:
        print(f"warning: {w}")
    print()
    print(f"{'method':12s} {'lambda1':>9s}    {'95% CI':20s} {'per cycle':>9s}   {'p(surr)':>7s}   reliable")
    for k, r in R.items():
        ci = f"[{r.ci['lo']:+.4f}, {r.ci['hi']:+.4f}]" if r.ci else "refused"
        p = f"{r.p_surr:.2f}" if np.isfinite(r.p_surr) else "-"
        print(f"{k:12s} {r.lam:+9.4f}    {ci:20s} {r.lam * per * P['dt']:+9.3f}   {p:>7s}   "
              f"{'yes' if r.reliable else 'no'}   {r.check}")
    for k, r in R.items():
        for note in r.notes:
            print(f"  {k}: {note}")


def save(P: dict, R: dict, out: str) -> None:
    arrays = {"radii": P["cs"]["radii"], "C": P["cs"]["C"], "C_slope": P["cs"]["slope"]}
    if P["ami"] is not None:
        arrays["ami"] = P["ami"]
    for k, r in R.items():
        arrays[f"{k}_surr"] = r.surr
        for key, val in r.plot.items():
            if isinstance(val, (np.ndarray, list)) and key != "fit":
                arrays[f"{k}_{key}"] = np.asarray(val, float)
        if r.ci:
            arrays[f"{k}_boot"] = r.ci["boots"]
    np.savez(out + ".npz", **arrays)
    keep = {k: v for k, v in P.items() if k not in ("cs", "ami")}
    keep["results"] = {k: {"lambda1": r.lam, "ci": [r.ci["lo"], r.ci["hi"]] if r.ci else None,
                           "ci_contains_zero": r.ci["contains_zero"] if r.ci else None,
                           "se_ols_lower_bound": r.se_ols, "window": r.window, "p_surr": r.p_surr,
                           "reliable": r.reliable, "check": r.check, "notes": r.notes} for k, r in R.items()}
    keep["warnings"] = core.WARNINGS
    with open(out + ".json", "w") as fh:
        json.dump(_clean(keep), fh, indent=1, allow_nan=False)


def _clean(o):
    """JSON-safe copy: numpy scalars to Python, non-finite floats to null."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, np.ndarray)):
        return [_clean(v) for v in o]
    if hasattr(o, "item"):
        o = o.item()
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o if isinstance(o, (str, int, float, bool, type(None))) else str(o)


def plot(P: dict, R: dict, out: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dt = P["dt"]
    fig, ax = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    a = ax[0, 0]
    for k, c in (("Rosenstein", "C0"), ("Kantz", "C1")):
        r = R[k]
        y = r.plot.get("y") if k == "Rosenstein" else (r.plot["curves"][len(r.plot["ladder"]) // 2]
                                                       if "curves" in r.plot else None)
        if y is None:
            continue
        tt = np.arange(len(y)) * dt
        a.plot(tt, y, c, lw=1, label=k)
        if r.window:
            i, j = r.window
            a.axvspan(i * dt, j * dt, color=c, alpha=0.12)
            b = np.polyfit(tt[i:j], y[i:j], 1)
            a.plot(tt[i:j], np.polyval(b, tt[i:j]), "k--", lw=1)
    a.set(xlabel="time", ylabel="<ln divergence>", title="Divergence curves and fit windows")
    a.legend()
    a = ax[0, 1]
    r = R["Kantz"]
    if "curves" in r.plot:
        for e, y in zip(r.plot["ladder"], r.plot["curves"]):
            a.plot(np.arange(len(y)) * dt, y, lw=0.8, label=f"eps={e:.3g}")
        if r.window:
            a.axvspan(r.window[0] * dt, r.window[1] * dt, color="grey", alpha=0.15)
        a.legend(fontsize=6, ncol=2)
    a.set(xlabel="time", ylabel="S(eps, t)", title="Kantz eps family")
    a = ax[1, 0]
    r = R["Wolf"]
    if "running" in r.plot:
        a.plot(r.plot["t"], r.plot["running"], "C2")
        if r.ci:
            a.axhspan(r.ci["lo"], r.ci["hi"], color="C2", alpha=0.15)
    a.set(xlabel="time", ylabel="running lambda1", title="Wolf running estimate")
    a = ax[1, 1]
    for i, (k, r) in enumerate(R.items()):
        if r.surr.size:
            a.plot(np.full(r.surr.size, i), r.surr, "o", color="grey", ms=3, alpha=0.6)
        if np.isfinite(r.lam):
            a.plot(i, r.lam, "D", color=f"C{i}", ms=7)
            if r.ci:
                a.vlines(i, r.ci["lo"], r.ci["hi"], color=f"C{i}")
    a.axhline(0, color="k", lw=0.5)
    a.set_xticks(range(len(R)), list(R))
    a.set(ylabel="lambda1", title="Data (diamond, 95% CI) vs IAAFT surrogates (grey)")
    fig.savefig(out + ".png", dpi=300)
    fig.savefig(out + ".pdf")


def main(argv: list[str] | None = None) -> int:
    np.seterr(all="ignore")                 # non-finite intermediates are checked explicitly
    try:
        a = parse(sys.argv[1:] if argv is None else argv)
        P, R = analyse(a)
        report(P, R)
        save(P, R, a.out)
        plot(P, R, a.out)
        print(f"\nwrote {a.out}.npz, {a.out}.json, {a.out}.png, {a.out}.pdf")
        return 0
    except PipelineError as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as e:                  # never a bare traceback
        print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
