#!/usr/bin/env python3
"""Analytical backbone for the stochastic attendance EGT model."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AttendanceParams:
    K: float          # baseline knowledge value from attending
    N: float          # note-sharing / async material value
    G: float          # grading-curve / competitive penalty
    C: float          # effort cost of attending
    E: float          # professor engagement value
    p: float = 0.0   # intervention probability
    Q: float = 0.0   # penalty when intervention happens
    beta_note: float = 0.0  # fraction of N removed by intervention


def effective_note_value(params: AttendanceParams) -> float:
    return params.N * (1.0 - params.beta_note)


def fitness_skip(x: float, params: AttendanceParams) -> float:
    """Expected payoff for a skipper when skip fraction is x."""
    N_eff = effective_note_value(params)
    return -params.p * params.Q + (1.0 - x) * (params.G + N_eff)


def fitness_attend(x: float, params: AttendanceParams) -> float:
    """Expected payoff for an attender when skip fraction is x."""
    return params.K + params.E - params.C - x * params.G


def fitness_difference(x: float, params: AttendanceParams) -> float:
    return fitness_skip(x, params) - fitness_attend(x, params)


def dxdt(x: float, params: AttendanceParams) -> float:
    """Replicator equation: rate of change of skip fraction."""
    return x * (1.0 - x) * fitness_difference(x, params)


def interior_equilibrium(params: AttendanceParams) -> float | None:
    """x* = 1 - A/N_eff where A = pQ - G + K + E - C."""
    N_eff = effective_note_value(params)
    if N_eff <= 0.0:
        return None
    A = params.p * params.Q - params.G + params.K + params.E - params.C
    x_star = 1.0 - A / N_eff
    return x_star if 0.0 < x_star < 1.0 else None


def classify_ess(params: AttendanceParams) -> str:
    """Return 'attend', 'mixed', 'skip', or 'neutral'."""
    N_eff = effective_note_value(params)
    if N_eff <= 0.0:
        # No x-dependence in fitness diff; compare at x=0.5
        fd = fitness_difference(0.5, params)
        if abs(fd) < 1e-10:
            return "neutral"
        return "skip" if fd > 0 else "attend"
    x_star = interior_equilibrium(params)
    if x_star is None:
        # Check which boundary was hit
        A = params.p * params.Q - params.G + params.K + params.E - params.C
        x_raw = 1.0 - A / N_eff
        return "attend" if x_raw <= 0.0 else "skip"
    return "mixed"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate():
    # Baseline: no intervention — should match simulate_egt.py
    base = AttendanceParams(K=1.0, N=0.7, G=0.2, C=0.8, E=0.6)
    N_eff = effective_note_value(base)
    assert N_eff == 0.7, N_eff
    x_star = interior_equilibrium(base)
    # alpha = N_eff+G-K-E+C = 0.7+0.2-1-0.6+0.8 = 0.1; x* = 0.1/0.7
    expected = 0.1 / 0.7
    assert x_star is not None and abs(x_star - expected) < 1e-10, x_star
    assert classify_ess(base) == "mixed"

    # With intervention — A=0.85, N_eff=0.35, so x*=1-0.85/0.35 < 0 => attend ESS
    iv = AttendanceParams(K=1.0, N=0.7, G=0.2, C=0.8, E=0.6, p=0.5, Q=0.5, beta_note=0.5)
    N_eff_iv = effective_note_value(iv)
    assert N_eff_iv == 0.35, N_eff_iv
    assert interior_equilibrium(iv) is None
    assert classify_ess(iv) == "attend"

    # Intervention case where x* stays in (0,1): small p, small Q
    iv2 = AttendanceParams(K=1.0, N=0.7, G=0.2, C=0.8, E=0.6, p=0.1, Q=0.1, beta_note=0.1)
    N_eff_iv2 = effective_note_value(iv2)   # 0.7 * 0.9 = 0.63
    A2 = 0.1*0.1 - 0.2 + 1.0 + 0.6 - 0.8  # 0.01 + 0.6 = 0.61
    x_star_iv2 = 1.0 - A2 / N_eff_iv2
    xs2 = interior_equilibrium(iv2)
    assert xs2 is not None and abs(xs2 - x_star_iv2) < 1e-10, xs2
    assert classify_ess(iv2) == "mixed"

    # p=0, Q=0, beta_note=0 must reduce to original
    same = AttendanceParams(K=1.0, N=0.7, G=0.2, C=0.8, E=0.6, p=0.0, Q=0.0, beta_note=0.0)
    assert interior_equilibrium(same) == interior_equilibrium(base)

    print("attendance_game.py: all validation checks passed.")


if __name__ == "__main__":
    _validate()
