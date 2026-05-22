#!/usr/bin/env python3
"""Deterministic replicator trajectory integration with CSV output."""

import csv
from pathlib import Path

from attendance_game import (
    AttendanceParams,
    classify_ess,
    dxdt,
    interior_equilibrium,
)

OUT = Path("outputs/stochastic")
OUT.mkdir(parents=True, exist_ok=True)

SCENARIOS = {
    "Highest_Variance":         {"N": 0.4607, "G": 0.4107},
    "Lowest_Variance":          {"N": 0.5898, "G": 0.1767},
    "Highest_Note_Utilization": {"N": 1.1655, "G": 0.1818},
    "Lowest_Note_Utilization":  {"N": 0.0328, "G": 0.2273},
}

K_DEFAULT      = 1.0
C_DEFAULT      = 0.8
E_DEFAULT      = 0.6
P_DEFAULT      = 0.25
Q_DEFAULT      = 0.5
BETA_DEFAULT   = 0.25

DT             = 0.05
T_MAX          = 200.0
X0_VALUES      = [0.01, 0.10, 0.20, 0.50, 0.80, 0.90, 0.99]


def simulate_replicator_path(
    params: AttendanceParams,
    x0: float,
    dt: float = DT,
    T: float = T_MAX,
) -> list[dict]:
    rows = []
    x = float(x0)
    ess = classify_ess(params)
    xs  = interior_equilibrium(params)
    t   = 0.0
    while t <= T + 1e-12:
        rows.append({
            "t":        round(t, 6),
            "x":        x,
            "dxdt":     dxdt(x, params),
            "ess_class": ess,
            "x_star":   xs if xs is not None else "",
        })
        dx = dxdt(x, params) * dt
        x  = max(0.0, min(1.0, x + dx))
        t += dt
    return rows


def simulate_replicator_fan(
    params: AttendanceParams,
    x0_values: list[float] = X0_VALUES,
    dt: float = DT,
    T: float = T_MAX,
) -> list[dict]:
    all_rows = []
    for x0 in x0_values:
        for row in simulate_replicator_path(params, x0, dt, T):
            row["x0"] = x0
            all_rows.append(row)
    return all_rows


def save_trajectories_csv(
    all_rows: list[dict],
    scenario: str,
    params: AttendanceParams,
    path: Path,
    write_header: bool = True,
):
    fieldnames = [
        "scenario", "K", "N", "G", "C", "E",
        "p", "Q", "beta_note",
        "x0", "t", "x", "dxdt", "ess_class", "x_star",
    ]
    mode = "w" if write_header else "a"
    with open(path, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for row in all_rows:
            writer.writerow({
                "scenario":  scenario,
                "K":         params.K,
                "N":         params.N,
                "G":         params.G,
                "C":         params.C,
                "E":         params.E,
                "p":         params.p,
                "Q":         params.Q,
                "beta_note": params.beta_note,
                **row,
            })


def main():
    out_path = OUT / "replicator_trajectories.csv"
    first = True
    for scenario, sc in SCENARIOS.items():
        params = AttendanceParams(
            K=K_DEFAULT, N=sc["N"], G=sc["G"],
            C=C_DEFAULT, E=E_DEFAULT,
            p=P_DEFAULT, Q=Q_DEFAULT, beta_note=BETA_DEFAULT,
        )
        rows = simulate_replicator_fan(params)
        save_trajectories_csv(rows, scenario, params, out_path, write_header=first)
        first = False
        ess = classify_ess(params)
        xs  = interior_equilibrium(params)
        xs_str = f"{xs:.4f}" if xs is not None else "N/A"
        print(f"{scenario}: ESS={ess}, x*={xs_str}")

    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
