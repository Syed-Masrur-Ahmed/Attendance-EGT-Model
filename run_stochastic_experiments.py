#!/usr/bin/env python3
"""
Orchestrator: runs the full stochastic EGT experiment pipeline.

Outputs
-------
outputs/stochastic/phase_grid.csv
outputs/stochastic/replicator_trajectories.csv   (already produced by simulate_replicator.py)
outputs/stochastic/monte_carlo_runs.csv
outputs/stochastic/monte_carlo_summary.csv
outputs/stochastic/trajectory_comparison.png
outputs/stochastic/phase_diagram.png
outputs/stochastic/same_pq_comparison.png
"""

import csv
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from attendance_game import AttendanceParams, classify_ess, interior_equilibrium
from simulate_replicator import (
    DT, T_MAX, X0_VALUES,
    simulate_replicator_fan,
    save_trajectories_csv,
)
from simulate_agent_monte_carlo import (
    run_monte_carlo_batch,
    save_runs_csv,
    save_summary_csv,
    summarize_runs,
)

plt.rcParams.update({
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.size":         10,
    "axes.labelsize":    11,
    "axes.titlesize":    11,
})

OUT = Path("outputs/stochastic")
OUT.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------
# Scenarios
# -------------------------------------------------------------------

SCENARIOS = {
    "Highest_Variance":         {"N": 0.4607, "G": 0.4107, "label": "Highest Variance"},
    "Lowest_Variance":          {"N": 0.5898, "G": 0.1767, "label": "Lowest Variance"},
    "Highest_Note_Utilization": {"N": 1.1655, "G": 0.1818, "label": "Highest Note Util."},
    "Lowest_Note_Utilization":  {"N": 0.0328, "G": 0.2273, "label": "Lowest Note Util."},
}

K_DEFAULT    = 1.0
C_DEFAULT    = 0.8
E_DEFAULT    = 0.6
P_DEFAULT    = 0.25
Q_DEFAULT    = 0.5
BETA_DEFAULT = 0.25

MC_N_POP     = 100
MC_T         = 250
MC_M         = 500
MC_X0        = 0.20
MC_SEL       = 5.0
MC_MUT       = 0.005
MC_SEED      = 123

COL_ESS = {"attend": "#C3E7D0", "mixed": "#F9E7C2", "skip": "#F3B7BD", "neutral": "#CCCCCC"}
COL_DARK = {"attend": "#4CAF50", "mixed": "#E6AF56", "skip": "#E76D81"}


# -------------------------------------------------------------------
# Step 3: Phase-grid CSV
# -------------------------------------------------------------------

def build_phase_grid():
    C_vals       = np.arange(0.0, 1.55, 0.05)
    E_vals       = np.arange(0.0, 1.55, 0.05)
    p_vals       = [0.0, 0.10, 0.25, 0.50, 0.75, 1.0]
    Q_vals       = [0.25, 0.50, 1.0]
    beta_vals    = [0.0, 0.25, 0.50, 0.75]

    path = OUT / "phase_grid.csv"
    fieldnames = [
        "scenario", "N", "G", "K", "C", "E",
        "p", "Q", "beta_note", "N_eff", "x_star", "ess_class",
    ]
    count = 0
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for scenario, sc in SCENARIOS.items():
            for p in p_vals:
                for Q in Q_vals:
                    for beta in beta_vals:
                        for C in C_vals:
                            for E in E_vals:
                                params = AttendanceParams(
                                    K=K_DEFAULT, N=sc["N"], G=sc["G"],
                                    C=float(C), E=float(E),
                                    p=p, Q=Q, beta_note=beta,
                                )
                                N_eff = sc["N"] * (1.0 - beta)
                                xs    = interior_equilibrium(params)
                                ess   = classify_ess(params)
                                writer.writerow({
                                    "scenario": scenario,
                                    "N": sc["N"], "G": sc["G"],
                                    "K": K_DEFAULT,
                                    "C": round(float(C), 4),
                                    "E": round(float(E), 4),
                                    "p": p, "Q": Q, "beta_note": beta,
                                    "N_eff": round(N_eff, 6),
                                    "x_star": round(xs, 6) if xs is not None else "",
                                    "ess_class": ess,
                                })
                                count += 1
    print(f"phase_grid.csv: {count} rows → {path}")


# -------------------------------------------------------------------
# Step 4–5: Monte Carlo batch
# -------------------------------------------------------------------

def run_all_monte_carlo():
    runs_path    = OUT / "monte_carlo_runs.csv"
    summary_path = OUT / "monte_carlo_summary.csv"
    summaries    = []
    first        = True

    for scenario, sc in SCENARIOS.items():
        params = AttendanceParams(
            K=K_DEFAULT, N=sc["N"], G=sc["G"],
            C=C_DEFAULT, E=E_DEFAULT,
            p=P_DEFAULT, Q=Q_DEFAULT, beta_note=BETA_DEFAULT,
        )
        t0 = time.time()
        print(f"  MC {scenario} (M={MC_M})...", end=" ", flush=True)
        all_runs = run_monte_carlo_batch(
            params, M=MC_M, N_pop=MC_N_POP, T=MC_T,
            initial_skip_rate=MC_X0,
            selection_intensity=MC_SEL,
            mutation_rate=MC_MUT,
            base_seed=MC_SEED,
        )
        save_runs_csv(
            all_runs, scenario, params,
            MC_N_POP, MC_X0, MC_SEL, MC_MUT,
            runs_path, write_header=first,
        )
        first = False
        summary = summarize_runs(
            all_runs, scenario, params,
            MC_N_POP, MC_X0, MC_SEL, MC_MUT,
        )
        summaries.append(summary)
        elapsed = time.time() - t0
        print(f"done ({elapsed:.1f}s)  mean_final={summary['mean_final_skip_rate']:.3f}  "
              f"ESS={summary['det_ess_class']}")

    save_summary_csv(summaries, summary_path)
    print(f"Saved {runs_path}  and  {summary_path}")
    return summaries


# -------------------------------------------------------------------
# Step 6a: Trajectory comparison plot
# -------------------------------------------------------------------

def plot_trajectory_comparison():
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    axes = axes.flatten()

    for ax, (scenario, sc) in zip(axes, SCENARIOS.items()):
        params = AttendanceParams(
            K=K_DEFAULT, N=sc["N"], G=sc["G"],
            C=C_DEFAULT, E=E_DEFAULT,
            p=P_DEFAULT, Q=Q_DEFAULT, beta_note=BETA_DEFAULT,
        )

        # Deterministic fan
        fan = simulate_replicator_fan(params, X0_VALUES, DT, T_MAX)
        ts_by_x0 = {}
        for row in fan:
            ts_by_x0.setdefault(row["x0"], []).append((row["t"], row["x"]))

        for x0, path_data in ts_by_x0.items():
            ts = [r[0] for r in path_data]
            xs = [r[1] for r in path_data]
            ax.plot(ts, xs, color="0.70", lw=0.9, alpha=0.7)

        # Monte Carlo ensemble (M=100 for speed in plot)
        all_runs = run_monte_carlo_batch(
            params, M=100, N_pop=MC_N_POP, T=MC_T,
            initial_skip_rate=MC_X0,
            selection_intensity=MC_SEL,
            mutation_rate=MC_MUT,
            base_seed=MC_SEED,
        )
        T_steps = len(all_runs[0])
        mc_t    = np.arange(T_steps)
        mc_mat  = np.array([[r["skip_rate"] for r in run] for run in all_runs])
        mc_mean = mc_mat.mean(axis=0)
        mc_p10  = np.percentile(mc_mat, 10, axis=0)
        mc_p90  = np.percentile(mc_mat, 90, axis=0)

        ess = classify_ess(params)
        xs  = interior_equilibrium(params)
        color = COL_DARK.get(ess, "#666666")
        ax.fill_between(mc_t, mc_p10, mc_p90, alpha=0.25, color=color)
        ax.plot(mc_t, mc_mean, color=color, lw=2.0,
                label=f"MC mean (ESS: {ess})")

        if xs is not None:
            ax.axhline(xs, color=color, ls="--", lw=1.2, label=f"$x^*={xs:.2f}$")

        ax.set(xlim=(0, MC_T), ylim=(-0.05, 1.05),
               title=sc["label"],
               xlabel="Time (class days)",
               ylabel=r"Skip fraction $x$" if ax is axes[0] or ax is axes[2] else "")
        ax.legend(fontsize=8, frameon=False)

    fig.suptitle(
        rf"Trajectory Comparison: Replicator (grey) vs MC mean±band  "
        rf"$p={P_DEFAULT},\ Q={Q_DEFAULT},\ \beta={BETA_DEFAULT}$",
        fontsize=11, y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path = OUT / "trajectory_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"trajectory_comparison.png → {path}")


# -------------------------------------------------------------------
# Step 6b: Phase diagram (one per scenario, faceted by p)
# -------------------------------------------------------------------

def plot_phase_diagram():
    C_vals = np.arange(0.0, 1.55, 0.05)
    E_vals = np.arange(0.0, 1.55, 0.05)
    p_vals = [0.0, 0.25, 0.50, 0.75, 1.0]
    Cg, Eg = np.meshgrid(C_vals, E_vals)

    import matplotlib.colors as mcolors
    cmap = mcolors.ListedColormap([COL_ESS["attend"], COL_ESS["mixed"], COL_ESS["skip"]])

    fig, axes = plt.subplots(
        len(SCENARIOS), len(p_vals),
        figsize=(3.5 * len(p_vals), 3.2 * len(SCENARIOS)),
        sharex=True, sharey=True,
    )

    for row_i, (scenario, sc) in enumerate(SCENARIOS.items()):
        for col_i, p in enumerate(p_vals):
            ax = axes[row_i][col_i]
            Z = np.zeros_like(Cg, dtype=int)
            for ri in range(len(E_vals)):
                for ci in range(len(C_vals)):
                    params = AttendanceParams(
                        K=K_DEFAULT, N=sc["N"], G=sc["G"],
                        C=float(C_vals[ci]), E=float(E_vals[ri]),
                        p=p, Q=Q_DEFAULT, beta_note=BETA_DEFAULT,
                    )
                    ess = classify_ess(params)
                    Z[ri, ci] = {"attend": 0, "mixed": 1, "skip": 2}.get(ess, 1)

            ax.pcolormesh(Cg, Eg, Z, cmap=cmap, vmin=-0.5, vmax=2.5, shading="auto")
            if row_i == 0:
                ax.set_title(f"$p={p}$", fontsize=10)
            if col_i == 0:
                ax.set_ylabel(f"{sc['label']}\n$E$ (engagement)", fontsize=9)
            if row_i == len(SCENARIOS) - 1:
                ax.set_xlabel("$C$ (effort cost)", fontsize=9)

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=COL_ESS["attend"], label="Attend ESS"),
        Patch(facecolor=COL_ESS["mixed"],  label="Mixed ESS"),
        Patch(facecolor=COL_ESS["skip"],   label="Skip ESS"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3,
               fontsize=10, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(
        rf"Phase Diagrams: ESS by $(C, E)$  |  $Q={Q_DEFAULT},\ \beta={BETA_DEFAULT}$",
        fontsize=12, y=1.01,
    )
    plt.tight_layout(rect=[0, 0.04, 1, 0.98])
    path = OUT / "phase_diagram.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"phase_diagram.png → {path}")


# -------------------------------------------------------------------
# Step 6c: Same pQ, different variance comparison
# -------------------------------------------------------------------

def plot_same_pq_comparison():
    """Frequent small vs rare large penalty: same pQ=0.20, one scenario."""
    sc = SCENARIOS["Highest_Variance"]
    base = dict(K=K_DEFAULT, N=sc["N"], G=sc["G"],
                C=C_DEFAULT, E=E_DEFAULT, beta_note=BETA_DEFAULT)

    configs = [
        {"p": 0.8,  "Q": 0.25, "label": r"$p=0.8,\ Q=0.25$  (frequent, small)"},
        {"p": 0.2,  "Q": 1.00, "label": r"$p=0.2,\ Q=1.00$  (rare, large)"},
        {"p": 0.0,  "Q": 0.0,  "label": r"$p=0,\ Q=0$  (no intervention)"},
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)

    for ax, cfg in zip(axes, configs):
        params = AttendanceParams(**base, p=cfg["p"], Q=cfg["Q"])
        all_runs = run_monte_carlo_batch(
            params, M=200, N_pop=MC_N_POP, T=MC_T,
            initial_skip_rate=MC_X0,
            selection_intensity=MC_SEL,
            mutation_rate=MC_MUT,
            base_seed=MC_SEED,
        )
        mc_mat  = np.array([[r["skip_rate"] for r in run] for run in all_runs])
        mc_mean = mc_mat.mean(axis=0)
        mc_p10  = np.percentile(mc_mat, 10, axis=0)
        mc_p90  = np.percentile(mc_mat, 90, axis=0)
        mc_t    = np.arange(len(all_runs[0]))

        ess   = classify_ess(params)
        color = COL_DARK.get(ess, "#666666")
        ax.fill_between(mc_t, mc_p10, mc_p90, alpha=0.25, color=color)
        ax.plot(mc_t, mc_mean, color=color, lw=2.0)

        # Deterministic overlay
        fan = simulate_replicator_fan(params, [MC_X0], DT, T_MAX)
        ts  = [r["t"] for r in fan if r["x0"] == MC_X0]
        xs  = [r["x"] for r in fan if r["x0"] == MC_X0]
        ax.plot(ts, xs, color="0.3", lw=1.5, ls="--", label="Deterministic")

        pq = cfg["p"] * cfg["Q"]
        ax.set(xlim=(0, MC_T), ylim=(-0.05, 1.05),
               title=f"{cfg['label']}\n$pQ={pq:.2f}$, ESS: {ess}",
               xlabel="Time (class days)")
        if ax is axes[0]:
            ax.set_ylabel(r"Skip fraction $x$")
        ax.legend(fontsize=8, frameon=False)

    fig.suptitle(
        "Same Expected Penalty ($pQ=0.20$) — Stochastic vs Deterministic\n"
        f"Highest Variance scenario: $N={sc['N']},\\ G={sc['G']}$",
        fontsize=11, y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    path = OUT / "same_pq_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"same_pq_comparison.png → {path}")


# -------------------------------------------------------------------
# Step 7: Empirical weekly C from StudentLife
# -------------------------------------------------------------------

STUDENTLIFE_WEEKLY = Path("outputs/studentlife/studentlife_C_weekly_summary.csv")
DAYS_PER_WEEK = 5


def load_empirical_c_sequence() -> tuple[list[float], list[str]]:
    """Return (day-level C values, week_start labels) from StudentLife weekly summary."""
    import csv as _csv
    weeks, c_means = [], []
    with open(STUDENTLIFE_WEEKLY, newline="") as f:
        for row in _csv.DictReader(f):
            weeks.append(row["week_start"])
            c_means.append(float(row["C_mean"]))
    # Expand to one value per class day
    c_sequence = [c for c in c_means for _ in range(DAYS_PER_WEEK)]
    return c_sequence, weeks


def plot_empirical_c_experiment():
    c_sequence, week_labels = load_empirical_c_sequence()
    T = len(c_sequence) - 1          # 54 days (11 weeks × 5)
    week_ticks = list(range(0, T + 1, DAYS_PER_WEEK))
    short_labels = [w[5:] for w in week_labels]  # "MM-DD" from "YYYY-MM-DD"

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes = axes.flatten()

    for ax, (scenario, sc) in zip(axes, SCENARIOS.items()):
        # Use p=0.25, Q=0.5, beta=0.25 with base C ignored (overridden by c_sequence)
        params = AttendanceParams(
            K=K_DEFAULT, N=sc["N"], G=sc["G"],
            C=C_DEFAULT, E=E_DEFAULT,
            p=P_DEFAULT, Q=Q_DEFAULT, beta_note=BETA_DEFAULT,
        )
        all_runs = run_monte_carlo_batch(
            params, M=MC_M, N_pop=MC_N_POP, T=T,
            initial_skip_rate=MC_X0,
            selection_intensity=MC_SEL,
            mutation_rate=MC_MUT,
            base_seed=MC_SEED,
            c_sequence=c_sequence,
        )

        mc_mat  = np.array([[r["skip_rate"] for r in run] for run in all_runs])
        mc_mean = mc_mat.mean(axis=0)
        mc_p10  = np.percentile(mc_mat, 10, axis=0)
        mc_p90  = np.percentile(mc_mat, 90, axis=0)
        mc_t    = np.arange(T + 1)

        # Left axis: skip fraction
        color = COL_DARK.get("attend", "#4CAF50")
        ax.fill_between(mc_t, mc_p10, mc_p90, alpha=0.25, color="#4CAF50")
        ax.plot(mc_t, mc_mean, color="#4CAF50", lw=2.0, label="MC mean skip rate")
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel(r"Skip fraction $x$", color="#2e7d32")
        ax.tick_params(axis="y", labelcolor="#2e7d32")
        ax.set_xticks(week_ticks)
        ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=7)
        ax.set_xlabel("Week start")
        ax.set_title(f"{sc['label']}\n$N={sc['N']},\\ G={sc['G']}$", fontsize=10)

        # Right axis: empirical C overlaid as a step line
        ax2 = ax.twinx()
        c_day = np.array(c_sequence)
        ax2.step(np.arange(len(c_day)), c_day, where="post",
                 color="#b5651d", lw=1.5, ls="--", alpha=0.8, label="Weekly $C$ (StudentLife)")
        ax2.set_ylim(0, 1.05)
        ax2.set_ylabel("Effort cost $C$", color="#b5651d")
        ax2.tick_params(axis="y", labelcolor="#b5651d")
        ax2.spines["right"].set_visible(True)

        # Combined legend
        lines1, labs1 = ax.get_legend_handles_labels()
        lines2, labs2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labs1 + labs2, fontsize=7, frameon=False, loc="upper left")

    fig.suptitle(
        rf"Empirical Weekly $C$ (StudentLife) — MC Skip Fraction over Semester"
        rf"  |  $p={P_DEFAULT},\ Q={Q_DEFAULT},\ \beta={BETA_DEFAULT}$",
        fontsize=11, y=0.995,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = OUT / "empirical_c_experiment.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"empirical_c_experiment.png → {path}")


# -------------------------------------------------------------------
# Step 7b: Empirical C — p=0 vs p=0.25 comparison
# -------------------------------------------------------------------

def plot_empirical_c_no_intervention():
    c_sequence, week_labels = load_empirical_c_sequence()
    T = len(c_sequence) - 1
    week_ticks  = list(range(0, T + 1, DAYS_PER_WEEK))
    short_labels = [w[5:] for w in week_labels]

    configs = [
        {"p": 0.0,       "Q": 0.0,      "color": "#E76D81", "label": "No intervention ($p=0$)"},
        {"p": P_DEFAULT, "Q": Q_DEFAULT, "color": "#4CAF50", "label": f"Enforcement ($p={P_DEFAULT}$)"},
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes = axes.flatten()

    for ax, (scenario, sc) in zip(axes, SCENARIOS.items()):
        for cfg in configs:
            params = AttendanceParams(
                K=K_DEFAULT, N=sc["N"], G=sc["G"],
                C=C_DEFAULT, E=E_DEFAULT,
                p=cfg["p"], Q=cfg["Q"], beta_note=BETA_DEFAULT,
            )
            all_runs = run_monte_carlo_batch(
                params, M=MC_M, N_pop=MC_N_POP, T=T,
                initial_skip_rate=MC_X0,
                selection_intensity=MC_SEL,
                mutation_rate=MC_MUT,
                base_seed=MC_SEED,
                c_sequence=c_sequence,
            )
            mc_mat  = np.array([[r["skip_rate"] for r in run] for run in all_runs])
            mc_mean = mc_mat.mean(axis=0)
            mc_p10  = np.percentile(mc_mat, 10, axis=0)
            mc_p90  = np.percentile(mc_mat, 90, axis=0)
            mc_t    = np.arange(T + 1)

            ax.fill_between(mc_t, mc_p10, mc_p90, alpha=0.18, color=cfg["color"])
            ax.plot(mc_t, mc_mean, color=cfg["color"], lw=2.0, label=cfg["label"])

        # Empirical C on right axis
        ax2 = ax.twinx()
        ax2.step(np.arange(len(c_sequence)), c_sequence, where="post",
                 color="#b5651d", lw=1.4, ls=":", alpha=0.7, label="Weekly $C$")
        ax2.set_ylim(0, 1.05)
        ax2.set_ylabel("Effort cost $C$", color="#b5651d", fontsize=9)
        ax2.tick_params(axis="y", labelcolor="#b5651d")
        ax2.spines["right"].set_visible(True)

        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel(r"Skip fraction $x$")
        ax.set_xticks(week_ticks)
        ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=7)
        ax.set_xlabel("Week start")
        ax.set_title(f"{sc['label']}  $N={sc['N']},\\ G={sc['G']}$", fontsize=10)

        lines1, labs1 = ax.get_legend_handles_labels()
        lines2, labs2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labs1 + labs2, fontsize=7, frameon=False, loc="upper right")

    fig.suptitle(
        rf"Empirical Weekly $C$ — No Intervention vs Enforcement  "
        rf"|  $Q={Q_DEFAULT},\ \beta={BETA_DEFAULT}$",
        fontsize=11, y=0.995,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = OUT / "empirical_c_no_intervention.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"empirical_c_no_intervention.png → {path}")


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    print("=== Step 3: Phase grid ===")
    build_phase_grid()

    print("\n=== Step 2: Replicator trajectories CSV ===")
    rep_path = OUT / "replicator_trajectories.csv"
    first = True
    for scenario, sc in SCENARIOS.items():
        params = AttendanceParams(
            K=K_DEFAULT, N=sc["N"], G=sc["G"],
            C=C_DEFAULT, E=E_DEFAULT,
            p=P_DEFAULT, Q=Q_DEFAULT, beta_note=BETA_DEFAULT,
        )
        rows = simulate_replicator_fan(params, X0_VALUES, DT, T_MAX)
        save_trajectories_csv(rows, scenario, params, rep_path, write_header=first)
        first = False
    print(f"replicator_trajectories.csv → {rep_path}")

    print("\n=== Steps 4–5: Monte Carlo ===")
    run_all_monte_carlo()

    print("\n=== Step 6: Plots ===")
    plot_trajectory_comparison()
    plot_phase_diagram()
    plot_same_pq_comparison()

    print("\n=== Step 7: Empirical C experiment ===")
    plot_empirical_c_experiment()

    print("\nDone. All outputs in outputs/stochastic/")


if __name__ == "__main__":
    main()
