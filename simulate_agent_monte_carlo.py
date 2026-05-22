#!/usr/bin/env python3
"""Finite-population stochastic agent-based simulation of the attendance game."""

import csv
from pathlib import Path

import numpy as np

from attendance_game import AttendanceParams, classify_ess, interior_equilibrium

OUT = Path("outputs/stochastic")
OUT.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------
# Pairwise payoff matrix (non-expected — intervention applied separately)
# -------------------------------------------------------------------
#
#                    Opponent skips         Opponent attends
# You skip           0                      G + N*(1-beta_note)
# You attend         K + E - C - G          K + E - C
#

def _payoff_skip_vs_skip(params: AttendanceParams) -> float:
    return 0.0

def _payoff_skip_vs_attend(params: AttendanceParams) -> float:
    return params.G + params.N * (1.0 - params.beta_note)

def _payoff_attend_vs_skip(params: AttendanceParams) -> float:
    return params.K + params.E - params.C - params.G

def _payoff_attend_vs_attend(params: AttendanceParams) -> float:
    return params.K + params.E - params.C


def run_one_simulation(
    params: AttendanceParams,
    N_pop: int = 100,
    T: int = 250,
    initial_skip_rate: float = 0.20,
    selection_intensity: float = 5.0,
    mutation_rate: float = 0.005,
    seed: int = 123,
) -> list[dict]:
    """
    Returns a list of dicts with one entry per time step:
      t, skip_rate, intervention_happened
    """
    rng = np.random.default_rng(seed)

    # 0 = attend, 1 = skip
    n_skip = int(round(initial_skip_rate * N_pop))
    strategies = np.array([1] * n_skip + [0] * (N_pop - n_skip), dtype=np.int8)

    # Precompute payoff lookup
    pss  = _payoff_skip_vs_skip(params)
    psa  = _payoff_skip_vs_attend(params)
    pas  = _payoff_attend_vs_skip(params)
    paa  = _payoff_attend_vs_attend(params)

    records = []

    for t in range(T + 1):
        skip_rate = strategies.mean()
        intervention = bool(rng.random() < params.p)
        records.append({
            "t": t,
            "skip_rate": float(skip_rate),
            "intervention_happened": int(intervention),
        })

        if t == T:
            break

        # --- compute pairwise payoffs ---
        order = rng.permutation(N_pop)
        payoffs = np.zeros(N_pop)

        for k in range(0, N_pop - 1, 2):
            i, j = order[k], order[k + 1]
            si, sj = strategies[i], strategies[j]
            if si == 1 and sj == 1:
                pi, pj = pss, pss
            elif si == 1 and sj == 0:
                pi, pj = psa, pas
            elif si == 0 and sj == 1:
                pi, pj = pas, psa
            else:
                pi, pj = paa, paa
            payoffs[i] = pi
            payoffs[j] = pj

        # If N_pop is odd, the unpaired student gets average payoff
        if N_pop % 2 == 1:
            last = order[-1]
            payoffs[last] = payoffs[order[:-1]].mean()

        # --- apply intervention (no double-counting: only here, not in base payoff) ---
        if intervention:
            payoffs[strategies == 1] -= params.Q

        # --- Fermi imitation ---
        new_strategies = strategies.copy()
        for i in range(N_pop):
            j = rng.integers(0, N_pop)
            while j == i:
                j = rng.integers(0, N_pop)
            diff = payoffs[j] - payoffs[i]
            prob = 1.0 / (1.0 + np.exp(-selection_intensity * diff))
            if rng.random() < prob:
                new_strategies[i] = strategies[j]

        # --- mutation ---
        flip_mask = rng.random(N_pop) < mutation_rate
        new_strategies[flip_mask] = 1 - new_strategies[flip_mask]

        strategies = new_strategies

    return records


def run_monte_carlo_batch(
    params: AttendanceParams,
    M: int = 500,
    N_pop: int = 100,
    T: int = 250,
    initial_skip_rate: float = 0.20,
    selection_intensity: float = 5.0,
    mutation_rate: float = 0.005,
    base_seed: int = 123,
) -> list[list[dict]]:
    """Run M independent simulations, returning a list of per-run record lists."""
    return [
        run_one_simulation(
            params, N_pop, T, initial_skip_rate,
            selection_intensity, mutation_rate,
            seed=base_seed + run_id,
        )
        for run_id in range(M)
    ]


def save_runs_csv(
    all_runs: list[list[dict]],
    scenario: str,
    params: AttendanceParams,
    N_pop: int,
    initial_skip_rate: float,
    selection_intensity: float,
    mutation_rate: float,
    path: Path,
    write_header: bool = True,
):
    fieldnames = [
        "scenario", "run_id", "t",
        "K", "N", "G", "C", "E", "p", "Q", "beta_note",
        "N_pop", "initial_skip_rate", "selection_intensity", "mutation_rate",
        "intervention_happened", "skip_rate",
    ]
    mode = "w" if write_header else "a"
    with open(path, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for run_id, records in enumerate(all_runs):
            for rec in records:
                writer.writerow({
                    "scenario": scenario,
                    "run_id": run_id,
                    "t": rec["t"],
                    "K": params.K, "N": params.N, "G": params.G,
                    "C": params.C, "E": params.E,
                    "p": params.p, "Q": params.Q, "beta_note": params.beta_note,
                    "N_pop": N_pop,
                    "initial_skip_rate": initial_skip_rate,
                    "selection_intensity": selection_intensity,
                    "mutation_rate": mutation_rate,
                    "intervention_happened": rec["intervention_happened"],
                    "skip_rate": rec["skip_rate"],
                })


def summarize_runs(
    all_runs: list[list[dict]],
    scenario: str,
    params: AttendanceParams,
    N_pop: int,
    initial_skip_rate: float,
    selection_intensity: float,
    mutation_rate: float,
) -> dict:
    M = len(all_runs)
    T = len(all_runs[0]) - 1
    last_50_start = max(0, T - 49)

    final_rates = np.array([r[-1]["skip_rate"] for r in all_runs])
    last_50_rates = np.array([
        np.mean([r["skip_rate"] for r in run[last_50_start:]])
        for run in all_runs
    ])

    return {
        "scenario": scenario,
        "K": params.K, "N": params.N, "G": params.G,
        "C": params.C, "E": params.E,
        "p": params.p, "Q": params.Q, "beta_note": params.beta_note,
        "N_pop": N_pop, "initial_skip_rate": initial_skip_rate,
        "selection_intensity": selection_intensity,
        "mutation_rate": mutation_rate,
        "M": M, "T": T,
        "mean_final_skip_rate":  float(final_rates.mean()),
        "sd_final_skip_rate":    float(final_rates.std()),
        "p_attend_dominates":    float((final_rates < 0.10).mean()),
        "p_skip_dominates":      float((final_rates > 0.90).mean()),
        "p_mixed":               float(((final_rates >= 0.10) & (final_rates <= 0.90)).mean()),
        "mean_last_50_skip_rate": float(last_50_rates.mean()),
        "sd_last_50_skip_rate":   float(last_50_rates.std()),
        "det_ess_class":          classify_ess(params),
        "det_x_star":             interior_equilibrium(params) or "",
    }


def save_summary_csv(summaries: list[dict], path: Path):
    if not summaries:
        return
    fieldnames = list(summaries[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)


# -------------------------------------------------------------------
# Quick smoke test (single run, small params)
# -------------------------------------------------------------------

def main():
    params = AttendanceParams(
        K=1.0, N=0.7, G=0.2, C=0.8, E=0.6,
        p=0.0, Q=0.0, beta_note=0.0,
    )
    print("Smoke test: single run, p=0 (no intervention)...")
    records = run_one_simulation(params, N_pop=100, T=250, seed=42)
    final = records[-1]["skip_rate"]
    print(f"  Final skip rate: {final:.3f}  (det ESS: {classify_ess(params)})")

    params_iv = AttendanceParams(
        K=1.0, N=0.7, G=0.2, C=0.8, E=0.6,
        p=0.25, Q=0.5, beta_note=0.25,
    )
    print("Smoke test: single run, p=0.25...")
    records_iv = run_one_simulation(params_iv, N_pop=100, T=250, seed=42)
    final_iv = records_iv[-1]["skip_rate"]
    print(f"  Final skip rate: {final_iv:.3f}  (det ESS: {classify_ess(params_iv)})")


if __name__ == "__main__":
    main()
