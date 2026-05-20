#!/usr/bin/env python3
"""
Extract Evolutionary Game Theory (EGT) model parameters from the
Open University Learning Analytics Dataset (OULAD).

Two parameters are derived and printed to the console:

  N_raw   -- "Note-Sharing Value / Free-Rider Extraction"
             Average clicks on static asynchronous material (resource,
             oucontent) by students who PASSED but NEVER attended a
             synchronous live virtual class. These are the "successful
             skippers" -- the free-riders the EGT model is built around.

  G_proxy -- "Grading Curve Effect / Sucker's Penalty"
             Coefficient of Variation (sigma / mu) of assessment scores,
             a proxy for how steep / competitive the grading curve is.

Run from inside the Attendance-EGT-Model/ directory:

    python3 extract_egt_params.py

Both values are printed at the end, ready to plug into the
differential-equation solver.
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


def banner(title: str) -> None:
    """Print a clearly delimited section header."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# --------------------------------------------------------------------------
# Pipeline 1 -- N (Note-Sharing Value / Free-Rider Extraction)
# --------------------------------------------------------------------------
def extract_n() -> float:
    banner("PIPELINE 1 -- N (Free-Rider Extraction)")

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

    # -- Step 6: "Successful Skippers" = successful MINUS live attenders. ----
    # MultiIndex difference performs the anti-join while preserving the
    # per-course identity of each student.
    skippers_idx = successful_idx.difference(live_attenders)
    print(f"Successful skippers   : {len(skippers_idx):,} (passed, never attended live)")

    # -- Step 7: total note-material clicks per skipper. --------------------
    # Keep only clicks on static async material, then sum per (student, course).
    note_clicks = clicks.loc[clicks["activity_type"].isin(NOTE_TYPES)]
    note_sums = note_clicks.groupby(STUDENT_KEYS, observed=True)["sum_click"].sum()

    # -- Step 8: restrict to skippers; zero-click skippers count as 0. ------
    # Reindexing onto the full skipper index materialises every skipper;
    # those with no resource/oucontent rows become NaN -> filled with 0.
    skipper_note_sums = note_sums.reindex(skippers_idx).fillna(0)
    n_zero = int((skipper_note_sums == 0).sum())
    print(f"Skippers with 0 clicks: {n_zero:,} (included in the average as 0)")

    # -- Step 9: global average across all successful skippers. -------------
    n_raw = float(skipper_note_sums.mean())
    print(f"\n>>> N_raw = {n_raw:.4f}")
    print("    (mean note-material clicks per successful skipper)")
    return n_raw


# --------------------------------------------------------------------------
# Pipeline 2 -- G (Grading Curve Effect / Sucker's Penalty)
# --------------------------------------------------------------------------
def extract_g() -> pd.DataFrame:
    banner("PIPELINE 2 -- G (Grading Curve / Coefficient of Variation)")

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

    # -- Step 5: Coefficient of Variation = sigma / mu. ---------------------
    stats["G_proxy"] = stats["std"] / stats["mean"]

    # -- Step 6: sort by G_proxy descending. --------------------------------
    stats = stats.sort_values("G_proxy", ascending=False).reset_index(drop=True)

    print(f"\nPer-course grading-curve dispersion ({len(stats)} course-presentations):\n")
    with pd.option_context(
        "display.max_rows", None, "display.width", 100, "display.float_format", "{:.4f}".format
    ):
        print(stats.to_string(index=False))

    # -- Step 7: overall average G_proxy across all courses. ----------------
    g_mean = float(stats["G_proxy"].mean())
    print(f"\n>>> G_proxy = {g_mean:.4f}")
    print(f"    (mean Coefficient of Variation across {len(stats)} course-presentations)")
    return stats


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def main() -> None:
    if not DATA_DIR.is_dir():
        raise SystemExit(
            f"Dataset directory not found: {DATA_DIR}\n"
            "Edit DATA_DIR at the top of this script to point at the OULAD CSVs."
        )

    n_raw = extract_n()
    g_stats = extract_g()
    g_proxy = float(g_stats["G_proxy"].mean())

    # -- Final summary: the two values for the EGT solver. ------------------
    banner("EGT MODEL PARAMETERS")
    print(f"  N_raw   = {n_raw:.4f}")
    print(f"  G_proxy = {g_proxy:.4f}   (mean across {len(g_stats)} courses)")
    print("=" * 70)


if __name__ == "__main__":
    main()
