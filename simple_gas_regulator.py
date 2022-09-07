import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import integrate


def inflow(t, tau):
    """ Exponentially declining gas inflow rate model. """

    m_inf = 10**11.5  # Total mass infall by t_g
    t_g = 14.  # Gyr

    a = m_inf/(tau*(1-np.exp(-t_g/tau)))

    return a*np.exp(-t/tau)


def dmgas_dt(t, m_gas, R, lam, eta, tau):

    return inflow(t, tau) - (1 - R)*eta*m_gas - lam*eta*m_gas


def dmstar_dt(t, m_star, mgas_vs_t, R, lam, eta, tau):

    mgas = np.interp(t, mgas_vs_t[:, 0], mgas_vs_t[:, 1])

    sfr = eta*mgas

    return (1 - R)*sfr


def dzmet_dt(t, zmet, mgas_vs_t, R, lam, eta, tau, y_zmet, z_infall):

    mgas = np.interp(t, mgas_vs_t[:, 0], mgas_vs_t[:, 1])

    return y_zmet*(1 - R)*eta + (z_infall - zmet)*inflow(t, tau)/mgas


def model_exp_inflow(axes, R, lam, eta, tau, y_zmet, z_infall, color="black"):

    t = np.arange(t_0, t_f, 0.001)

    int_mgas = integrate.solve_ivp(dmgas_dt, [t_0, t_f], [m_0],
                                   t_eval=t, args=[R, lam, eta, tau])

    mgas = np.squeeze(int_mgas["y"])

    sfr = eta*mgas

    mgas_vs_t = np.c_[t, mgas]

    int_mstar = integrate.solve_ivp(dmstar_dt, [t_0, t_f], [m_0], t_eval=t,
                                    args=[mgas_vs_t, R, lam, eta, tau])

    mstar = np.squeeze(int_mstar["y"])

    int_zmet = integrate.solve_ivp(dzmet_dt, [t_0, t_f], [z_0], t_eval=t,
                                   args=[mgas_vs_t, R, lam, eta, tau, y_zmet,
                                         z_infall])

    zgas = np.squeeze(int_zmet["y"])/0.0142
    zstar = np.cumsum(zgas*sfr)/np.cumsum(sfr)

    axes[0].plot(np.log10(t), np.log10(inflow(t, tau)) - 9., color=color,
                 label="$\\tau\ =\ $" + str(tau))

    axes[1].plot(np.log10(t), np.log10(mgas), color=color)
    axes[2].plot(np.log10(t), np.log10(mstar), color=color)
    axes[3].plot(np.log10(t), np.log10(sfr/mstar) - 9., color=color)
    axes[4].plot(np.log10(t), np.log10(zgas), color=color)
    axes[5].plot(np.log10(t), np.log10(zstar), color=color)

    axes[0].legend(frameon=False)


R = 0.441  # Mass return fraction (Chabrier)
lam = 1.  # Mass-loading factor
eta = 0.5  # Star-formation efficiency
y_zmet = 0.0631  # Metal yield (Chabrier)
z_infall = 0.  # Metal fraction for inflowing gas

m_0 = 1.  # Initial gas mass
z_0 = 0.  # Initial metal fraction
t_0 = 0.  # Start time
t_f = 14.  # End time

fig = plt.figure(figsize=(16, 9))
gs = mpl.gridspec.GridSpec(2, 3, wspace=0.3)

ax1 = plt.subplot(gs[0, 0])
ax2 = plt.subplot(gs[0, 1])
ax3 = plt.subplot(gs[0, 2])

ax4 = plt.subplot(gs[1, 0])
ax5 = plt.subplot(gs[1, 1])
ax6 = plt.subplot(gs[1, 2])

axes = [ax1, ax2, ax3, ax4, ax5, ax6]

colors = ["#FF595E", "#FFCA3A", "#8AC926", "#1982C4", "#6A4C93"][::-1]

tau_vals = [0.25, 0.5, 1., 2., 4.]
for i in range(len(tau_vals)):
    tau = tau_vals[i]
    model_exp_inflow(axes, R, lam, eta, tau, y_zmet, z_infall, color=colors[i])

ax1.set_ylabel("log$_{10}$(Inflow rate / $\mathrm{M_\odot}$ yr$^{-1}$)")
ax2.set_ylabel("log$_{10}(M_\mathrm{gas}\ /\ \mathrm{M_\odot})$")
ax3.set_ylabel("log$_{10}(M_\mathrm{*}\ /\ \mathrm{M_\odot})$")
ax4.set_ylabel("log$_{10}$(sSFR / yr$^{-1}$)")
ax5.set_ylabel("log$_{10}(Z_\mathrm{gas}\ /\ \mathrm{Z_\odot})$")
ax6.set_ylabel("log$_{10}(Z_*\ /\ \mathrm{Z_\odot})$")

for ax in axes:
    ax.set_xlim(np.log10(0.3), np.log10(14.))
    ax.set_xlabel("Age of Universe / Gyr")

    ax.set_xticks(np.log10(np.array([0.3, 1., 3., 10.])))
    ax.set_xticklabels(np.array([0.3, 1., 3., 10.]))

ax1.set_ylim(1., 3.)
ax2.set_ylim(8.25, 11.5)
ax3.set_ylim(8.25, 11.5)
ax4.set_ylim(-11.75, -8.)
ax5.set_ylim(-0.8, 1.2)
ax6.set_ylim(-1.1, 0.4)

plt.savefig("gas_regulator_result.pdf", bbox_inches="tight")
