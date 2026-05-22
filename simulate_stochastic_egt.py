#!/usr/bin/env python3

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from pathlib import Path
from scipy.integrate import solve_ivp

plt.rcParams.update({
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.size":         10,
    "axes.labelsize":    11,
    "axes.titlesize":    11,
})

K         = 1.0
BETA_NOTE = 0.25
Q_VAL     = 0.5
P_VALUES  = np.linspace(0.0, 1.0, 50)
FPS       = 15

SCENARIOS = {
    "Highest_Variance":         {"N": 0.4607, "G": 0.4107, "label": "Highest Variance (Max G)"},
    "Lowest_Variance":          {"N": 0.5898, "G": 0.1767, "label": "Lowest Variance (Min G)"},
    "Highest_Note_Utilization": {"N": 1.1655, "G": 0.1818, "label": "Highest Note Utilization (Max N)"},
    "Lowest_Note_Utilization":  {"N": 0.0328, "G": 0.2273, "label": "Lowest Note Utilization (Min N)"},
}

COL      = {"attend": "#C3E7D0", "mixed": "#F9E7C2", "skip": "#F3B7BD"}
COL_DARK = {"attend": "#B7DCC5", "mixed": "#E6AF56", "skip": "#E76D81"}
LABEL    = {"attend": "Attend ESS", "mixed": "Mixed ESS", "skip": "Skip ESS"}

OUT = Path("figures")
OUT.mkdir(exist_ok=True)


def n_eff(N):
    return N * (1.0 - BETA_NOTE)


def xdot(X, N, G, C, E, p):
    Ne = n_eff(N)
    return X * (1.0 - X) * ((Ne + G - K - E + C - p * Q_VAL) - X * Ne)


def classify(N, G, C, E, p):
    Ne    = n_eff(N)
    alpha = Ne + G - K - E + C - p * Q_VAL
    beta  =      G - K - E + C - p * Q_VAL
    if alpha < 0:
        return "attend"
    if beta > 0:
        return "skip"
    return "mixed"


def interior(N, G, C, E, p):
    Ne = n_eff(N)
    if Ne <= 0:
        return None
    alpha = Ne + G - K - E + C - p * Q_VAL
    Xs = alpha / Ne
    return Xs if 0.0 < Xs < 1.0 else None


def rep_points(N, G):
    return {
        "attend": (max(0.05, K + 1.0 - N - G - 0.3), 1.0),
        "mixed":  (max(0.05, K + 0.5 - G - N / 2),   0.5),
        "skip":   (max(0.05, K + 0.0 - G + 0.3),      0.0),
    }


# ---------------------------------------------------------------------------
# Heatmap animation
# ---------------------------------------------------------------------------

def fig_heatmap_anim(key, N, G, label):
    C_arr = np.linspace(0, 2, 400)
    E_arr = np.linspace(0, 2, 400)
    Cg, Eg = np.meshgrid(C_arr, E_arr)
    Ne = n_eff(N)
    cmap = mcolors.ListedColormap([COL["attend"], COL["mixed"], COL["skip"]])

    fig, ax = plt.subplots(figsize=(5.5, 5.0))

    def update(i):
        p = P_VALUES[i]
        ax.cla()

        Z = np.where(Ne + G - K - Eg + Cg - p * Q_VAL < 0, 0,
            np.where(     G - K - Eg + Cg - p * Q_VAL > 0, 2, 1))
        ax.pcolormesh(Cg, Eg, Z, cmap=cmap, vmin=-0.5, vmax=2.5, shading="auto")

        ax.plot(C_arr, C_arr + (Ne + G - K - p * Q_VAL), color="0.2", ls="--", lw=1.6)
        ax.plot(C_arr, C_arr + (     G - K - p * Q_VAL), color="0.2", ls=":",  lw=1.6)

        ax.text(0.15, 1.80, "Attend ESS", fontsize=9, color="0.15", va="top")
        E_mid = 1.0 + G - K - p * Q_VAL + Ne / 2.0
        if Ne > 0.05 and 0.1 < E_mid < 1.85:
            ax.text(1.0, E_mid, "Mixed ESS", fontsize=9, color="0.15",
                    ha="center", va="center")
        ax.text(1.85, 0.10, "Skip ESS", fontsize=9, color="0.15", ha="right")

        ax.set(xlim=(0, 2), ylim=(0, 2),
               xlabel=r"$C$  (effort cost)",
               ylabel=r"$E$  (attendance value)",
               title=(f"{label}  |  $N={N},\\ G={G}$\n"
                      rf"$p={p:.2f},\quad Q={Q_VAL},\quad \beta_{{\rm note}}={BETA_NOTE}$"))
        ax.legend(handles=[
            Line2D([0], [0], color="0.2", ls="--", lw=1.6, label="Attend/Mixed boundary"),
            Line2D([0], [0], color="0.2", ls=":",  lw=1.6, label="Mixed/Skip boundary"),
        ], fontsize=8, loc="upper right", framealpha=0.9, edgecolor="0.8")

    ani = FuncAnimation(fig, update, frames=len(P_VALUES), interval=1000 / FPS)
    fname = f"heatmap_{key}_anim.gif"
    ani.save(str(OUT / fname), writer=PillowWriter(fps=FPS))
    plt.close(fig)
    print(fname)


# ---------------------------------------------------------------------------
# Phase portrait animation
# ---------------------------------------------------------------------------

def fig_phase_anim(key, N, G, label):
    pts  = rep_points(N, G)
    Xarr = np.linspace(0, 1, 800)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    sup = fig.suptitle("", fontsize=11, y=0.98)

    def update(i):
        p = P_VALUES[i]
        for ax, (etype, (C, E)) in zip(axes, pts.items()):
            ax.cla()
            ess_now = classify(N, G, C, E, p)
            Xd = xdot(Xarr, N, G, C, E, p)

            ax.plot(Xarr, Xd, color=COL_DARK[etype], lw=1.8)
            ax.axhline(0, color="0.6", lw=0.8, ls="--")

            for xa in np.linspace(0.12, 0.88, 5):
                Xd_a = float(xdot(xa, N, G, C, E, p))
                dx   = 0.045 * (1 if Xd_a > 0 else -1)
                ax.annotate("", xy=(xa + dx, float(xdot(xa + dx, N, G, C, E, p))),
                            xytext=(xa, Xd_a),
                            arrowprops=dict(arrowstyle="-|>", color="0.45",
                                            lw=0.7, mutation_scale=9))

            for xv, stable in [(0.0, ess_now == "attend"), (1.0, ess_now == "skip")]:
                ax.scatter([xv], [0], s=60, zorder=6,
                           facecolors=COL_DARK[etype] if stable else "white",
                           edgecolors="0.3", lw=1.4)

            Xs = interior(N, G, C, E, p)
            if Xs is not None:
                ax.scatter([Xs], [0], s=60, zorder=6,
                           facecolors=COL_DARK["mixed"], edgecolors="0.3", lw=1.4)
                ax.axvline(Xs, color=COL["mixed"], ls=":", lw=1.1, alpha=0.7,
                           label=f"$X^*={Xs:.2f}$")
                ax.legend(fontsize=8, loc="upper center", frameon=False)

            ax.set(xlim=(-0.05, 1.05),
                   xlabel=r"$X$  (skip fraction)",
                   ylabel=r"$\dot{X}$  (rate of change)",
                   title=f"{LABEL[etype]}  [{ess_now}]\n$C={C:.2f},\\ E={E:.1f}$")

        sup.set_text(
            rf"Phase Portraits: {label}  |  "
            rf"$p={p:.2f},\quad Q={Q_VAL},\quad \beta_{{\rm note}}={BETA_NOTE}$"
        )

    update(0)
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    ani = FuncAnimation(fig, update, frames=len(P_VALUES), interval=1000 / FPS)
    fname = f"phase_{key}_anim.gif"
    ani.save(str(OUT / fname), writer=PillowWriter(fps=FPS))
    plt.close(fig)
    print(fname)


# ---------------------------------------------------------------------------
# Trajectory animation
# ---------------------------------------------------------------------------

def fig_trajectories_anim(key, N, G, label, n_ic=8):
    pts    = rep_points(N, G)
    X0s    = np.linspace(0.05, 0.95, n_ic)
    T      = 200.0
    t_eval = np.linspace(0, T, 1000)

    print(f"  pre-computing trajectories for {key} ({len(P_VALUES)} p-values × "
          f"{len(pts)} panels × {n_ic} ICs)...")

    def make_ode(C, E, p):
        return lambda t, y: [xdot(np.clip(y[0], 0.0, 1.0), N, G, C, E, p)]

    trajs = {}
    for etype, (C, E) in pts.items():
        trajs[etype] = [
            [solve_ivp(make_ode(C, E, p), (0, T), [X0],
                       t_eval=t_eval, rtol=1e-8, atol=1e-10)
             for X0 in X0s]
            for p in P_VALUES
        ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True)
    sup  = fig.suptitle("", fontsize=11, y=0.98)
    fig.text(0.5, 0.01,
             "Note: time axes differ as each panel is cropped to when trajectories settle",
             ha="center", fontsize=8, color="0.4")

    def update(i):
        p = P_VALUES[i]
        for ax, (etype, (C, E)) in zip(axes, pts.items()):
            ax.cla()
            sols = trajs[etype][i]

            for sol in sols:
                ax.plot(sol.t, sol.y[0], color="0.55", lw=1.1, alpha=0.75)

            Xs      = interior(N, G, C, E, p)
            ess_now = classify(N, G, C, E, p)
            eq_x    = {"attend": 0.0,
                       "skip":   1.0,
                       "mixed":  Xs if Xs is not None else 0.5}[ess_now]

            settled = []
            for sol in sols:
                close = np.where(np.abs(sol.y[0] - eq_x) < 0.05)[0]
                if len(close):
                    settled.append(sol.t[close[0]])
            xlim_max = min(T, max(settled) * 1.4) if settled else T
            ax.set_xlim(0, xlim_max)
            ax.set_ylim(-0.05, 1.05)

            eq_s = {"attend": r"$X = 0$  (Attend ESS)",
                    "skip":   r"$X = 1$  (Skip ESS)",
                    "mixed":  rf"$X^* = {eq_x:.2f}$  (Mixed ESS)"}[ess_now]
            ax.axhline(eq_x, color=COL_DARK[etype], ls="--", lw=1.8, label=eq_s)
            ax.set(xlabel="Time",
                   title=f"{LABEL[etype]}  [{ess_now}]\n$C={C:.2f},\\ E={E:.1f}$")
            if ax is axes[0]:
                ax.set_ylabel(r"$X$  (fraction skipping)")
            ax.legend(fontsize=9, frameon=False)

        sup.set_text(
            rf"Trajectories: {label}  |  "
            rf"$p={p:.2f},\quad Q={Q_VAL},\quad \beta_{{\rm note}}={BETA_NOTE}$"
        )

    update(0)
    plt.tight_layout(rect=[0, 0.06, 1, 0.90])
    ani = FuncAnimation(fig, update, frames=len(P_VALUES), interval=1000 / FPS)
    fname = f"trajectories_{key}_anim.gif"
    ani.save(str(OUT / fname), writer=PillowWriter(fps=FPS))
    plt.close(fig)
    print(fname)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    for key, sc in SCENARIOS.items():
        fig_heatmap_anim(key, sc["N"], sc["G"], sc["label"])

    sc = SCENARIOS["Highest_Variance"]
    fig_phase_anim("Highest_Variance", sc["N"], sc["G"], sc["label"])
    fig_trajectories_anim("Highest_Variance", sc["N"], sc["G"], sc["label"])


if __name__ == "__main__":
    main()
