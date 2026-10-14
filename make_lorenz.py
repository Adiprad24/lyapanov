"""Generate the Lorenz (1963) validation series: the x-component, in the pipeline's input format."""
import sys

import numpy as np
from scipy.integrate import solve_ivp

SIGMA, RHO, BETA = 10.0, 28.0, 8.0 / 3.0          # classical parameters; accepted lambda1 = 0.9056
DT, N, T_TRANSIENT = 0.01, 50_000, 100.0           # sample interval, samples kept, transient discarded
RTOL, ATOL, X0 = 1e-12, 1e-12, (1.0, 1.0, 1.0)      # DOP853 tolerances and initial state

out = sys.argv[1] if len(sys.argv) > 1 else "lorenz.dat"
rhs = lambda _t, s: [SIGMA * (s[1] - s[0]), s[0] * (RHO - s[2]) - s[1], s[0] * s[1] - BETA * s[2]]
t = T_TRANSIENT + DT * np.arange(N)
sol = solve_ivp(rhs, (0.0, t[-1]), X0, method="DOP853", t_eval=t, rtol=RTOL, atol=ATOL)
with open(out, "w") as fh:
    fh.write(f"# Lorenz x(t): sigma={SIGMA} rho={RHO} beta={BETA:.10f} x0={X0}\n"
             f"# DOP853 rtol={RTOL} atol={ATOL}; t<{T_TRANSIENT} discarded; dt={DT}; N={N}\n"
             'TITLE = "Lorenz validation"\nVARIABLES = "t", "x"\nZONE T="lorenz"\n')
    np.savetxt(fh, np.column_stack([t - T_TRANSIENT, sol.y[0]]), fmt=["%.6f", "%.10e"])
