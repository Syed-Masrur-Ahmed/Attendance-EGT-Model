#!/usr/bin/env python3
"""
Build an empirical StudentLife proxy for C, the effort cost of attending class.

The proxy is a weekly student-level index:

    C_iw = normalized_stress_iw
         + normalized_sleep_loss_iw
         + normalized_deadline_pressure_iw

where sleep_loss = max(0, 8 - sleep_hours). The raw sum is then min-max
rescaled to [0, 1.5] so it can motivate the C range used in the EGT heatmaps.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "dataset" / "student-life"
OUT_DIR = ROOT / "outputs" / "studentlife"

STRESS_DIR = DATA_DIR / "EMA" / "response" / "Stress"
SLEEP_DIR = DATA_DIR / "EMA" / "response" / "Sleep"
DEADLINES_PATH = DATA_DIR / "education" / "deadlines.csv"
GRADES_PATH = DATA_DIR / "education" / "grades.csv"

# Spring-term analysis window used by the StudentLife education files.
DATE_MIN = pd.Timestamp("2013-03-27")
DATE_MAX = pd.Timestamp("2013-06-05")
C_MAX = 1.5
MIN_OBSERVED_WEEKS_FOR_GRADES = 4

# EMA/response/Stress options:
# [1] A little stressed, [2] Definitely stressed, [3] Stressed out,
# [4] Feeling good, [5] Feeling great.
STRESS_TO_SEVERITY = {
    1: 1.0 / 3.0,
    2: 2.0 / 3.0,
    3: 1.0,
    4: 0.0,
    5: 0.0,
}

# EMA/response/Sleep hour options:
# [1]<3, [2]3.5, [3]4, ..., [19]12.
SLEEP_CODE_TO_HOURS = {
    1: 2.5,
    2: 3.5,
    3: 4.0,
    4: 4.5,
    5: 5.0,
    6: 5.5,
    7: 6.0,
    8: 6.5,
    9: 7.0,
    10: 7.5,
    11: 8.0,
    12: 8.5,
    13: 9.0,
    14: 9.5,
    15: 10.0,
    16: 10.5,
    17: 11.0,
    18: 11.5,
    19: 12.0,
}


@dataclass(frozen=True)
class Coverage:
    stress_responses: int
    sleep_responses: int
    deadline_rows: int
    student_weeks: int
    valid_c_weeks: int


def week_start(series: pd.Series) -> pd.Series:
    dates = pd.to_datetime(series).dt.normalize()
    return dates - pd.to_timedelta(dates.dt.weekday, unit="D")


def minmax(series: pd.Series) -> pd.Series:
    valid = series.dropna()
    if valid.empty:
        return pd.Series(pd.NA, index=series.index, dtype="Float64")
    low = valid.min()
    high = valid.max()
    if high == low:
        return pd.Series(0.0, index=series.index)
    return (series - low) / (high - low)


def uid_from_path(path: Path) -> str:
    return path.stem.split("_")[-1]


def load_stress() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted(STRESS_DIR.glob("Stress_*.json")):
        uid = uid_from_path(path)
        with path.open() as file:
            payload = json.load(file)
        for response in payload:
            if "level" not in response:
                continue
            try:
                level = int(response["level"])
            except (TypeError, ValueError):
                continue
            severity = STRESS_TO_SEVERITY.get(level)
            if severity is None:
                continue
            rows.append(
                {
                    "uid": uid,
                    "date": pd.to_datetime(response["resp_time"], unit="s"),
                    "stress_level": level,
                    "stress_score": severity,
                }
            )

    if not rows:
        return pd.DataFrame(columns=["uid", "week_start", "stress_score", "stress_n"])

    df = pd.DataFrame(rows)
    df = df[(df["date"] >= DATE_MIN) & (df["date"] <= DATE_MAX)]
    df["week_start"] = week_start(df["date"])
    return (
        df.groupby(["uid", "week_start"], as_index=False)
        .agg(stress_score=("stress_score", "mean"), stress_n=("stress_score", "size"))
    )


def load_sleep() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted(SLEEP_DIR.glob("Sleep_*.json")):
        uid = uid_from_path(path)
        with path.open() as file:
            payload = json.load(file)
        for response in payload:
            if "hour" not in response:
                continue
            try:
                code = int(response["hour"])
            except (TypeError, ValueError):
                continue
            sleep_hours = SLEEP_CODE_TO_HOURS.get(code)
            if sleep_hours is None:
                continue
            rows.append(
                {
                    "uid": uid,
                    "date": pd.to_datetime(response["resp_time"], unit="s"),
                    "sleep_code": code,
                    "sleep_hours": sleep_hours,
                    "sleep_loss": max(0.0, 8.0 - sleep_hours),
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=["uid", "week_start", "sleep_hours", "sleep_loss", "sleep_n"]
        )

    df = pd.DataFrame(rows)
    df = df[(df["date"] >= DATE_MIN) & (df["date"] <= DATE_MAX)]
    df["week_start"] = week_start(df["date"])
    return (
        df.groupby(["uid", "week_start"], as_index=False)
        .agg(
            sleep_hours=("sleep_hours", "mean"),
            sleep_loss=("sleep_loss", "mean"),
            sleep_n=("sleep_loss", "size"),
        )
    )


def load_deadlines() -> pd.DataFrame:
    raw = pd.read_csv(DEADLINES_PATH)
    date_cols = [col for col in raw.columns if col != "uid"]
    long = raw.melt(
        id_vars="uid",
        value_vars=date_cols,
        var_name="date",
        value_name="deadline_count",
    )
    long["date"] = pd.to_datetime(long["date"], errors="coerce")
    long["deadline_count"] = pd.to_numeric(long["deadline_count"], errors="coerce").fillna(0)
    long = long[(long["date"] >= DATE_MIN) & (long["date"] <= DATE_MAX)]
    long["week_start"] = week_start(long["date"])
    return (
        long.groupby(["uid", "week_start"], as_index=False)
        .agg(deadline_count=("deadline_count", "sum"))
    )


def build_weekly_c() -> tuple[pd.DataFrame, Coverage]:
    stress = load_stress()
    sleep = load_sleep()
    deadlines = load_deadlines()

    weekly = deadlines.merge(stress, on=["uid", "week_start"], how="left")
    weekly = weekly.merge(sleep, on=["uid", "week_start"], how="left")

    weekly["stress_norm"] = weekly["stress_score"]
    weekly["sleep_loss_norm"] = minmax(weekly["sleep_loss"])
    weekly["deadline_pressure_norm"] = minmax(weekly["deadline_count"])

    components = ["stress_norm", "sleep_loss_norm", "deadline_pressure_norm"]
    weekly["component_count"] = weekly[components].notna().sum(axis=1)
    weekly["C_raw"] = weekly[components].sum(axis=1, min_count=len(components))
    weekly["C_scaled"] = minmax(weekly["C_raw"]) * C_MAX

    ordered = [
        "uid",
        "week_start",
        "C_scaled",
        "C_raw",
        "stress_norm",
        "sleep_loss_norm",
        "deadline_pressure_norm",
        "stress_score",
        "sleep_hours",
        "sleep_loss",
        "deadline_count",
        "stress_n",
        "sleep_n",
        "component_count",
    ]
    weekly = weekly[ordered].sort_values(["uid", "week_start"]).reset_index(drop=True)

    coverage = Coverage(
        stress_responses=int(stress["stress_n"].sum()) if not stress.empty else 0,
        sleep_responses=int(sleep["sleep_n"].sum()) if not sleep.empty else 0,
        deadline_rows=len(deadlines),
        student_weeks=len(weekly),
        valid_c_weeks=int(weekly["C_scaled"].notna().sum()),
    )
    return weekly, coverage


def build_student_summary(weekly: pd.DataFrame) -> pd.DataFrame:
    student = (
        weekly.groupby("uid", as_index=False)
        .agg(
            C_mean=("C_scaled", "mean"),
            C_median=("C_scaled", "median"),
            C_max=("C_scaled", "max"),
            stress_mean=("stress_norm", "mean"),
            sleep_loss_mean=("sleep_loss", "mean"),
            deadline_count_mean=("deadline_count", "mean"),
            observed_weeks=("C_scaled", "count"),
            total_weeks=("C_scaled", "size"),
            weeks_with_stress=("stress_norm", "count"),
            weeks_with_sleep=("sleep_loss", "count"),
        )
        .sort_values("C_mean", ascending=False)
    )

    if GRADES_PATH.exists():
        grades = pd.read_csv(GRADES_PATH)
        grades.columns = [col.strip() for col in grades.columns]
        student = student.merge(grades, on="uid", how="left")

    return student


def build_grade_correlations(student: pd.DataFrame) -> pd.DataFrame:
    candidates = ["gpa all", "gpa 13s", "cs 65"]
    rows = []
    for grade_col in candidates:
        if grade_col not in student.columns:
            continue
        pair = student.loc[
            student["observed_weeks"] >= MIN_OBSERVED_WEEKS_FOR_GRADES,
            ["C_mean", grade_col],
        ].dropna()
        if len(pair) < 3:
            continue
        rows.append(
            {
                "outcome": grade_col,
                "n_students": len(pair),
                "min_observed_c_weeks": MIN_OBSERVED_WEEKS_FOR_GRADES,
                "pearson_corr_with_C_mean": pair["C_mean"].corr(pair[grade_col]),
            }
        )
    return pd.DataFrame(rows)


def build_weekly_summary(weekly: pd.DataFrame) -> pd.DataFrame:
    return (
        weekly.groupby("week_start", as_index=False)
        .agg(
            student_weeks=("uid", "size"),
            valid_c_weeks=("C_scaled", "count"),
            C_mean=("C_scaled", "mean"),
            C_median=("C_scaled", "median"),
            C_q25=("C_scaled", lambda x: x.quantile(0.25)),
            C_q75=("C_scaled", lambda x: x.quantile(0.75)),
            deadline_mean=("deadline_count", "mean"),
            sleep_loss_mean=("sleep_loss", "mean"),
            stress_mean=("stress_norm", "mean"),
        )
        .sort_values("week_start")
    )


def plot_weekly(weekly_summary: pd.DataFrame) -> None:
    width, height = 1200, 700
    left, right, top, bottom = 110, 45, 85, 115
    plot_w = width - left - right
    plot_h = height - top - bottom

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    title = "StudentLife Weekly Effort-Cost Proxy"
    subtitle = "Mean C by week, rescaled to [0, 1.5]"
    draw.text((left, 28), title, fill="#1f2933", font=font)
    draw.text((left, 52), subtitle, fill="#52606d", font=font)

    # Axes and horizontal grid.
    axis_color = "#52606d"
    grid_color = "#d9e2ec"
    for tick in [0.0, 0.375, 0.75, 1.125, 1.5]:
        y = top + plot_h - (tick / C_MAX) * plot_h
        draw.line((left, y, width - right, y), fill=grid_color, width=1)
        draw.text((30, y - 7), f"{tick:.2f}", fill=axis_color, font=font)
    draw.line((left, top, left, height - bottom), fill=axis_color, width=2)
    draw.line((left, height - bottom, width - right, height - bottom), fill=axis_color, width=2)

    n = len(weekly_summary)
    if n == 1:
        xs = [left + plot_w / 2]
    else:
        xs = [left + i * plot_w / (n - 1) for i in range(n)]

    def y_for(value: float) -> float:
        return top + plot_h - (float(value) / C_MAX) * plot_h

    points = list(zip(xs, [y_for(v) for v in weekly_summary["C_mean"]]))
    q25 = [y_for(v) for v in weekly_summary["C_q25"]]
    q75 = [y_for(v) for v in weekly_summary["C_q75"]]

    # Interquartile range as vertical bands, then mean C line.
    for x, low_y, high_y in zip(xs, q25, q75):
        draw.line((x, high_y, x, low_y), fill="#9fbfb3", width=7)
    if len(points) > 1:
        draw.line(points, fill="#2f5d50", width=4, joint="curve")
    for x, y in points:
        draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill="#2f5d50", outline="white", width=2)

    for x, date in zip(xs, weekly_summary["week_start"]):
        label = pd.to_datetime(date).strftime("%m/%d")
        draw.text((x - 18, height - bottom + 18), label, fill=axis_color, font=font)

    draw.text((left, height - 42), "Week starting", fill=axis_color, font=font)
    draw.text((26, top - 25), "C", fill=axis_color, font=font)
    draw.rectangle((width - 255, 28, width - 55, 72), outline="#d9e2ec", width=1)
    draw.line((width - 238, 48, width - 202, 48), fill="#2f5d50", width=4)
    draw.text((width - 190, 41), "Mean C", fill="#1f2933", font=font)
    draw.line((width - 238, 63, width - 202, 63), fill="#9fbfb3", width=7)
    draw.text((width - 190, 56), "IQR", fill="#1f2933", font=font)

    image.save(OUT_DIR / "studentlife_C_weekly_plot.png")


def main() -> None:
    if not DATA_DIR.is_dir():
        raise SystemExit(f"StudentLife directory not found: {DATA_DIR}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    weekly, coverage = build_weekly_c()
    weekly_summary = build_weekly_summary(weekly)
    student = build_student_summary(weekly)
    correlations = build_grade_correlations(student)

    weekly.to_csv(OUT_DIR / "studentlife_C_by_week.csv", index=False)
    weekly_summary.to_csv(OUT_DIR / "studentlife_C_weekly_summary.csv", index=False)
    student.to_csv(OUT_DIR / "studentlife_C_by_student.csv", index=False)
    correlations.to_csv(OUT_DIR / "studentlife_C_grade_correlations.csv", index=False)
    plot_weekly(weekly_summary)

    print("StudentLife C proxy generated")
    print(f"  stress responses used : {coverage.stress_responses}")
    print(f"  sleep responses used  : {coverage.sleep_responses}")
    print(f"  student-week rows     : {coverage.student_weeks}")
    print(f"  valid C rows          : {coverage.valid_c_weeks}")
    print(f"  output directory      : {OUT_DIR}")
    if not correlations.empty:
        print("\nDescriptive grade correlations:")
        print(correlations.to_string(index=False))


if __name__ == "__main__":
    main()
