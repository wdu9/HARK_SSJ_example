import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize as opt


# simulate uncertainty response
def simulate_agg_unc(x, dY=0.01, rho_Y=0.84, rho_Z=1.0, hor=20, sig_Z=0.01):
    """
    dY:     scale of credit shock, in terms of its impact on Y
    rho_Y:  persistence of *realized* credit shock, in terms of Y
    rho_Z:  persistence of *expected* TFP process, chosen to maximize peak uncertainty
    hor:    horizon of simulation, chosen large enough for uncertainty to peak within horizon
    """

    # arbitrary normalization
    ss_X = 1.0

    # we search over x, which monotonically maps to sig_w / sig_Z
    sig_w = np.sqrt(1 / x - 1) * sig_Z * ss_X

    # solve Ricatti equation for steady state uncertainty
    p = [ss_X ** 2 / sig_w ** 2, 1 - rho_Z ** 2 - (ss_X * sig_Z / sig_w) ** 2, -(sig_Z ** 2)]
    ss_Sig = max(np.roots(p))

    # updating step
    def upd(Sig, X):
        return rho_Z ** 2 * Sig / (1 + (X / sig_w) ** 2 * Sig) + sig_Z ** 2

    # simulate response to shock
    X = np.exp(np.log(1 - dY) * rho_Y ** np.arange(hor)) * ss_X
    Sig = np.zeros(hor + 1)
    Sig[0] = ss_Sig
    for t in range(hor):
        Sig[t + 1] = upd(Sig[t], X[t])

    return Sig / ss_Sig - 1, sig_w / (sig_Z * ss_X)

def calibrate_sig_wZ():
    # find signal-to-noise ratio that maximizes peak uncertainty for 1%-shock
    x = opt.fminbound(lambda x: -max(simulate_agg_unc(x)[0]), 0, 1)
    peak, sig_wZ = simulate_agg_unc(x)

    print("Parameters maximizing peak uncertainty:")
    print(f"sig_w / sig_Z = {sig_wZ:.3f} * X_ss, yielding max peak of {100 * max(peak):.3f} percent")
    return sig_wZ, x

def make_figure_12():
    # make figure 12
    sig_wZ, x = calibrate_sig_wZ()

    ngrid = 100
    dy_grid = np.linspace(1e-12, 0.5, ngrid)
    peak = np.array([max(simulate_agg_unc(x, dY=dy_grid[k])[0]) for k in range(ngrid)])

    fig, ax = plt.subplots(constrained_layout=True, figsize=(4, 3))
    ax.plot(100 * dy_grid, 100 * peak, "-k", linewidth=2)
    ax.plot(100 * dy_grid, 100 * dy_grid, ":", color="0.7", linewidth=1.5)
    ax.set(xlabel=r"GDP loss on impact (in %)", ylabel=r"Uncertainty rise at peak (in %)")
    fig.savefig("output/fig12_agg_unc_by_impact.pdf")
    # fig.show()