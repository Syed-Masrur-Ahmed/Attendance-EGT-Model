#!/usr/bin/env python3
"""
Extract Evolutionary Game Theory (EGT) model parameters from the
Open University Learning Analytics Dataset (OULAD).

The script derives two parameters PER COURSE-PRESENTATION, applies locked
scaling constants, and reports the empirical extremes for comparative
analysis:

  N_raw   -- "Note-Sharing Value / Free-Rider Extraction"
             Per-course mean of clicks on static asynchronous material
             (resource, oucontent) by students who PASSED but NEVER attended
             a synchronous live virtual class -- the "successful skippers".

  G_proxy -- "Grading Curve Effect / Sucker's Penalty"
             Per-course Coefficient of Variation (sigma / mu) of assessment
             scores, a proxy for how steep / competitive the curve is.

Stage 3 merges the two per-course tables, applies the locked scalars
(ALPHA, GAMMA) to produce N_final / G_final, and prints the four extreme
course-presentations (highest/lowest curve, highest/lowest VLE reliance)
ready to plug into the differential-equation solver.

Run from inside the Attendance-EGT-Model/ directory:

    python3 extract_egt_params.py
"""

from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
# Filepath for the directory containing the CSVs, modify for your setup
DATA_DIR = Path(__file__).resolve().parent.parent / "dataset"

# A course-presentation is identified by these two columns.
COURSE_KEYS = ["code_module", "code_presentation"]
# A student within a specific course-presentation (a student may appear in
# several course-presentations, with a different role/result in each).
STUDENT_KEYS = ["id_student", "code_module", "code_presentation"]

# Synchronous live web-conferencing activity types. A student who logged a
# click on EITHER tool counts as having "attended" a live class, so they are
# NOT a true skipper.
LIVE_TYPES = {"oucollaborate", "ouelluminate"}

# Static asynchronous "note" material -- what a free-rider consumes instead
# of attending the live class.
NOTE_TYPES = {"resource", "oucontent"}

# Outcomes that count as having passed the course.
SUCCESS = {"Pass", "Distinction"}

# Locked scaling constants -- DO NOT MODIFY. These map the empirical OULAD
# measurements onto the input range expected by the EGT solver.
ALPHA = 0.001  # scales N_raw  -> N_final
GAMMA = 1.0    # scales G_proxy -> G_final


def banner(title: str) -> None:
    """Print a clearly delimited section header."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# --------------------------------------------------------------------------
# Pipeline 1 -- N_raw per course (Note-Sharing Value / Free-Rider Extraction)
# --------------------------------------------------------------------------
def extract_n() -> pd.DataFrame:
    banner("PIPELINE 1 -- N_raw per course (Free-Rider Extraction)")

    # -- Step 1: load the VLE material catalogue (small: ~6.4k rows). --------
    # id_site is globally unique here, but we keep the course keys so the
    # merge below matches the requested join semantics exactly.
    vle = pd.read_csv(
        DATA_DIR / "vle.csv",
        usecols=["id_site", "code_module", "code_presentation", "activity_type"],
    )

    # -- Step 2: load the VLE interaction log (~10.6M rows). ----------------
    # Downcast dtypes so the whole file fits comfortably in memory:
    #   - integer ids / click counts as int32 instead of int64
    #   - low-cardinality string columns as 'category'
    student_vle = pd.read_csv(
        DATA_DIR / "studentVle.csv",
        usecols=[
            "id_student",
            "id_site",
            "code_module",
            "code_presentation",
            "sum_click",
        ],
        dtype={
            "id_student": "int32",
            "id_site": "int32",
            "sum_click": "int32",
            "code_module": "category",
            "code_presentation": "category",
        },
    )
    print(f"Loaded studentVle.csv : {len(student_vle):,} click rows")

    # -- Step 3: join clicks -> activity_type. ------------------------------
    # Left merge keeps every click; each row gains the activity_type of the
    # material it touched.
    clicks = student_vle.merge(
        vle,
        on=["id_site", "code_module", "code_presentation"],
        how="left",
    )
    del student_vle  # free the large intermediate frame

    # -- Step 4: identify the "Live Attenders". -----------------------------
    # Any (student, course) pair with >= 1 click on a synchronous live tool.
    live_mask = clicks["activity_type"].isin(LIVE_TYPES)
    live_attenders = (
        clicks.loc[live_mask, STUDENT_KEYS]
        .drop_duplicates()
        .set_index(STUDENT_KEYS)
        .index
    )
    print(f"Live attenders        : {len(live_attenders):,} (student, course) pairs")

    # -- Step 5: identify successful students. ------------------------------
    student_info = pd.read_csv(
        DATA_DIR / "studentInfo.csv",
        usecols=["id_student", "code_module", "code_presentation", "final_result"],
    )
    successful = student_info[student_info["final_result"].isin(SUCCESS)]
    successful_idx = successful.set_index(STUDENT_KEYS).index
    print(f"Successful students   : {len(successful_idx):,} (Pass / Distinction)")

    # -- Step 6: master frame of "Successful Skippers" = successful MINUS ----
    # live attenders. The MultiIndex difference performs the anti-join while
    # preserving each student's per-course identity; we materialise it as a
    # DataFrame so it can serve as the master frame for the left join below.
    skippers = (
        successful_idx.difference(live_attenders)
        .to_frame(index=False)[STUDENT_KEYS]
    )
    print(f"Successful skippers   : {len(skippers):,} (passed, never attended live)")

    # -- Step 7: total note-material clicks per skipper. --------------------
    # Keep only clicks on static async material, then sum sum_click per
    # (id_student, code_module, code_presentation).
    note_clicks = clicks.loc[clicks["activity_type"].isin(NOTE_TYPES)]
    note_sums = (
        note_clicks.groupby(STUDENT_KEYS, observed=True)["sum_click"]
        .sum()
        .reset_index()
    )

    # -- Step 8: LEFT JOIN aggregated clicks onto the skipper master frame. -
    # Skippers with no resource/oucontent rows get NaN -> filled with 0, so
    # every successful skipper contributes to the per-course mean.
    skipper_clicks = skippers.merge(note_sums, on=STUDENT_KEYS, how="left")
    skipper_clicks["sum_click"] = skipper_clicks["sum_click"].fillna(0)
    n_zero = int((skipper_clicks["sum_click"] == 0).sum())
    print(f"Skippers with 0 clicks: {n_zero:,} (kept via left join, counted as 0)")

    # -- Step 9: per-course mean -> N_raw for each course-presentation. -----
    n_by_course = (
        skipper_clicks.groupby(COURSE_KEYS, observed=True)["sum_click"]
        .mean()
        .reset_index()
        .rename(columns={"sum_click": "N_raw"})
    )
    print(f"\n>>> N_raw computed for {len(n_by_course)} course-presentations.")
    return n_by_course


# --------------------------------------------------------------------------
# Pipeline 2 -- G_proxy per course (Grading Curve Effect / Sucker's Penalty)
# --------------------------------------------------------------------------
def extract_g() -> pd.DataFrame:
    banner("PIPELINE 2 -- G_proxy per course (Grading Curve / CV)")

    # -- Step 1: load assessment scores. ------------------------------------
    # Some scores are blank ("?"); coerce to numeric and drop the NaNs so the
    # mean / std are computed only over valid marks.
    student_assessment = pd.read_csv(
        DATA_DIR / "studentAssessment.csv",
        usecols=["id_assessment", "score"],
    )
    n_before = len(student_assessment)
    student_assessment["score"] = pd.to_numeric(
        student_assessment["score"], errors="coerce"
    )
    student_assessment = student_assessment.dropna(subset=["score"])
    print(
        f"Assessment scores     : {len(student_assessment):,} valid "
        f"({n_before - len(student_assessment):,} blank scores dropped)"
    )

    # -- Step 2: map each score to its course. ------------------------------
    assessments = pd.read_csv(
        DATA_DIR / "assessments.csv",
        usecols=["id_assessment", "code_module", "code_presentation"],
    )
    scored = student_assessment.merge(assessments, on="id_assessment", how="inner")

    # -- Step 3 & 4: mean (mu) and std (sigma) of score per course. ---------
    # std uses the default sample standard deviation (ddof=1).
    stats = (
        scored.groupby(COURSE_KEYS)["score"]
        .agg(mean="mean", std="std")
        .reset_index()
    )

    # -- Step 5: Coefficient of Variation = sigma / mu -> G_proxy. ----------
    stats["G_proxy"] = stats["std"] / stats["mean"]
    stats = stats.sort_values("G_proxy", ascending=False).reset_index(drop=True)

    print(f"\nPer-course grading-curve dispersion ({len(stats)} course-presentations):\n")
    with pd.option_context(
        "display.max_rows", None,
        "display.width", 100,
        "display.float_format", "{:.4f}".format,
    ):
        print(stats.to_string(index=False))

    # Return the clean per-course frame (code_module, code_presentation,
    # mean, std, G_proxy) for the integration engine.
    return stats


# --------------------------------------------------------------------------
# Stage 3 -- Integration and Scaling Engine
# --------------------------------------------------------------------------
def integrate_and_scale(
    n_by_course: pd.DataFrame, g_by_course: pd.DataFrame
) -> pd.DataFrame:
    banner("INTEGRATION & SCALING ENGINE")

    # -- Merge the two per-course parameter tables on the course identifier. -
    matrix = n_by_course.merge(
        g_by_course[COURSE_KEYS + ["G_proxy"]],
        on=COURSE_KEYS,
        how="inner",
    )

    # -- Apply the locked scalars to produce the final solver inputs. -------
    matrix["N_final"] = matrix["N_raw"] * ALPHA
    matrix["G_final"] = matrix["G_proxy"] * GAMMA

    print(f"Locked scalars        : ALPHA = {ALPHA}, GAMMA = {GAMMA}")
    print(f"Course-presentations  : {len(matrix)}\n")
    with pd.option_context(
        "display.max_rows", None,
        "display.width", 100,
        "display.float_format", "{:.4f}".format,
    ):
        print(
            matrix[COURSE_KEYS + ["N_raw", "G_proxy", "N_final", "G_final"]]
            .to_string(index=False)
        )

    # -- Isolate the four empirical extremes via idxmax / idxmin. -----------
    edges = [
        ("HIGHEST CURVE        (max G_final)", matrix["G_final"].idxmax()),
        ("LOWEST CURVE         (min G_final)", matrix["G_final"].idxmin()),
        ("HIGHEST VLE RELIANCE (max N_final)", matrix["N_final"].idxmax()),
        ("LOWEST VLE RELIANCE  (min N_final)", matrix["N_final"].idxmin()),
    ]

    banner("EMPIRICAL EXTREMES -- SOLVER INPUT PAIRS")
    for label, idx in edges:
        row = matrix.loc[idx]
        print(f"\n{label}")
        print(f"  Course   : {row['code_module']} {row['code_presentation']}")
        print(f"  N_final  : {row['N_final']:.6f}")
        print(f"  G_final  : {row['G_final']:.6f}")
    print()
    return matrix


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def main() -> None:
    if not DATA_DIR.is_dir():
        raise SystemExit(
            f"Dataset directory not found: {DATA_DIR}\n"
            "Edit DATA_DIR at the top of this script to point at the OULAD CSVs."
        )

    n_by_course = extract_n()
    g_by_course = extract_g()
    integrate_and_scale(n_by_course, g_by_course)


if __name__ == "__main__":
    main()
