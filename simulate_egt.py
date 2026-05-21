#!/usr/bin/env python3

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
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

K = 1.0

SCENARIOS = {
    "Highest_Variance":        {"N": 0.4607, "G": 0.4107, "label": "Highest Variance (Max G)"},
    "Lowest_Variance":         {"N": 0.5898, "G": 0.1767, "label": "Lowest Variance (Min G)"},
    "Highest_Note_Utilization":{"N": 1.1655, "G": 0.1818, "label": "Highest Note Utilization (Max N)"},
    "Lowest_Note_Utilization": {"N": 0.0328, "G": 0.2273, "label": "Lowest Note Utilization (Min N)"},
}

COL      = {"attend": "#C3E7D0", "mixed": "#F9E7C2", "skip": "#F3B7BD"}
COL_DARK = {"attend": "#B7DCC5", "mixed": "#E6AF56", "skip": "#E76D81"}
LABEL    = {"attend": "Attend ESS", "mixed": "Mixed ESS", "skip": "Skip ESS"}

OUT = Path("figures")
OUT.mkdir(exist_ok=True)

def xdot(X, N, G, C, E):
    return X * (1 - X) * ((N + G - K - E + C) - X * N)

def classify(N, G, C, E):
    alpha = N + G - K - E + C
    beta  =     G - K - E + C
    if alpha < 0: return "attend"
    if beta  > 0: return "skip"
    return "mixed"

def interior(N, G, C, E):
    alpha = N + G - K - E + C
    Xs = alpha / N if N > 0 else None
    return Xs if Xs and 0 < Xs < 1 else None

def rep_points(N, G):
    return {
        "attend": (max(0.05, K + 1.0 - N - G - 0.3), 1.0),
        "mixed":  (max(0.05, K + 0.5 - G - N / 2),   0.5),
        "skip":   (max(0.05, K + 0.0 - G + 0.3),      0.0),
    }

def heatmap_bg(ax, Cg, Eg, N, G, alpha=1.0):
    Z = np.where(N+G-K-Eg+Cg < 0, 0,
        np.where(  G-K-Eg+Cg > 0, 2, 1))
    cmap = mcolors.ListedColormap([COL["attend"], COL["mixed"], COL["skip"]])
    ax.pcolormesh(Cg, Eg, Z, cmap=cmap, vmin=-0.5, vmax=2.5,
                  alpha=alpha, shading="auto")

def save(fig, fname):
    fig.savefig(OUT / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(fname)

def fig_heatmap(key, N, G, label):
    C = np.linspace(0, 2, 500); E = np.linspace(0, 2, 500)
    Cg, Eg = np.meshgrid(C, E)
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    heatmap_bg(ax, Cg, Eg, N, G)

    ax.plot(C, C + (N+G-K), color="0.2", ls="--", lw=1.6)
    ax.plot(C, C + (G-K),   color="0.2", ls=":",  lw=1.6)

    ax.text(0.15, 1.80, "Attend ESS", fontsize=9, color="0.15", va="top")
    E_mid = 1.0 + (G + N/2 - K)
    if N > 0.08 and 0.1 < E_mid < 1.85:
        ax.text(1.0, E_mid, "Mixed ESS", fontsize=9, color="0.15",
                ha="center", va="center")
    ax.text(1.85, 0.10, "Skip ESS", fontsize=9, color="0.15", ha="right")

    ax.set(xlim=(0,2), ylim=(0,2),
           xlabel=r"$C$  (effort cost)",
           ylabel=r"$E$  (attendance value)",
           title=f"{label}\n$N={N},\\ G={G}$")
    ax.legend(handles=[
        Line2D([0],[0], color="0.2", ls="--", lw=1.6, label="Attend/Mixed boundary"),
        Line2D([0],[0], color="0.2", ls=":",  lw=1.6, label="Mixed/Skip boundary"),
    ], fontsize=8, loc="upper right", framealpha=0.9, edgecolor="0.8")
    save(fig, f"heatmap_{key}.png")


def fig_phase(key, N, G, label):
    pts = rep_points(N, G)
    X = np.linspace(0, 1, 800)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, (etype, (C, E)) in zip(axes, pts.items()):
        Xd = xdot(X, N, G, C, E)
        ax.plot(X, Xd, color=COL_DARK[etype], lw=1.8)
        ax.axhline(0, color="0.6", lw=0.8, ls="--")

        for xa in np.linspace(0.12, 0.88, 5):
            Xd_a = xdot(xa, N, G, C, E)
            dx = 0.045 * (1 if Xd_a > 0 else -1)
            ax.annotate("", xy=(xa+dx, xdot(xa+dx, N, G, C, E)),
                        xytext=(xa, Xd_a),
                        arrowprops=dict(arrowstyle="-|>", color="0.45",
                                        lw=0.7, mutation_scale=9))

        for xv, stable in [(0.0, etype=="attend"), (1.0, etype=="skip")]:
            ax.scatter([xv], [0], s=60, zorder=6,
                       facecolors=COL_DARK[etype] if stable else "white",
                       edgecolors="0.3", lw=1.4)
        Xs = interior(N, G, C, E)
        if Xs:
            ax.scatter([Xs], [0], s=60, zorder=6,
                       facecolors=COL_DARK["mixed"], edgecolors="0.3", lw=1.4)
            ax.axvline(Xs, color=COL["mixed"], ls=":", lw=1.1, alpha=0.7,
                       label=f"$X^*={Xs:.2f}$")
            ax.legend(fontsize=8, loc="upper center", frameon=False)

        ax.set(xlim=(-0.05, 1.05),
               xlabel=r"$X$  (skip fraction)",
               ylabel=r"$\dot{X}$  (rate of change)",
               title=f"{LABEL[etype]},  $C={C:.2f}$, $E={E:.1f}$")

    fig.suptitle(f"Phase Portraits: {label}", fontsize=11, y=1.02)
    plt.tight_layout()
    save(fig, f"phase_{key}.png")


def fig_trajectories(key, N, G, label, n_ic=8):
    pts = rep_points(N, G)
    X0s = np.linspace(0.05, 0.95, n_ic)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, (etype, (C, E)) in zip(axes, pts.items()):
        T = 200.0
        t_eval = np.linspace(0, T, 1000)
        sols = []
        for X0 in X0s:
            sol = solve_ivp(lambda t, y: [xdot(np.clip(y[0], 0, 1), N, G, C, E)],
                            (0, T), [X0], t_eval=t_eval, rtol=1e-8, atol=1e-10)
            sols.append(sol)
            ax.plot(sol.t, sol.y[0], color="0.55", lw=1.1, alpha=0.75)

        Xs = interior(N, G, C, E)
        eq_x = {"attend": 0.0, "skip": 1.0, "mixed": Xs or 0.5}[etype]
        settled = []
        for sol in sols:
            close = np.where(np.abs(sol.y[0] - eq_x) < 0.05)[0]
            if len(close):
                settled.append(sol.t[close[0]])
        xlim_max = min(T, max(settled) * 1.4) if settled else T
        ax.set_xlim(0, xlim_max)

        eq_s = {"attend": r"$X = 0$  (Attend ESS)",
                "skip":   r"$X = 1$  (Skip ESS)",
                "mixed":  rf"$X^* = {eq_x:.2f}$  (Mixed ESS)"}[etype]
        ax.axhline(eq_x, color=COL_DARK[etype], ls="--", lw=1.8, label=eq_s)
        ax.set(ylim=(-0.05, 1.05), xlabel="Time",
               title=f"{LABEL[etype]},  $C={C:.2f}$, $E={E:.1f}$")
        if ax is axes[0]:
            ax.set_ylabel(r"$X$  (fraction skipping)")
        ax.legend(fontsize=9, frameon=False)

    fig.text(0.5, -0.04,
             "Note: time axes differ as each panel is cropped to when trajectories settle",
             ha="center", fontsize=8, color="0.4")
    fig.suptitle(f"Trajectories: {label}", fontsize=11, y=1.02)
    plt.tight_layout()
    save(fig, f"trajectories_{key}.png")


def main():
    for key, sc in SCENARIOS.items():
        fig_heatmap(key, sc["N"], sc["G"], sc["label"])

    sc = SCENARIOS["Highest_Variance"]
    fig_phase("Highest_Variance", sc["N"], sc["G"], sc["label"])
    fig_trajectories("Highest_Variance", sc["N"], sc["G"], sc["label"])

if __name__ == "__main__":
    main()