# Attendance-EGT-Model

An Evolutionary Game Theory (EGT) study of classroom attendance modelled as a
public-goods game. This repository holds empirical preprocessing scripts that
support the payoff parameters used by an EGT differential-equation solver.

The current empirical layers are:

- **OULAD**: estimates `N`, the note-sharing / asynchronous academic value,
  and `G`, a grading-curve / assessment-dispersion proxy.
- **StudentLife**: builds a proxy for `C`, the effort cost of attending class,
  from weekly stress, sleep loss, and deadline pressure.

## What `extract_egt_params.py` does

The script reads the OULAD CSVs with `pandas` and runs three stages: it
derives `N_raw` and `G_proxy` **per course-presentation**, applies locked
scaling constants, and reports the empirical extremes — all printed to the
console ready to paste into the solver.

### Stage 1 — `N_raw`: Note-Sharing Value / Free-Rider Extraction

The **per-course** mean number of clicks on **static asynchronous material**
(`resource`, `oucontent`) made by **successful skippers** — students who
passed the course (`Pass` or `Distinction`) but **never** attended a
synchronous live virtual class. The pipeline:

1. Joins the 10.6M-row VLE interaction log (`studentVle.csv`) to the VLE
   material catalogue (`vle.csv`) to label every click with its
   `activity_type`. The log is loaded with downcast dtypes (`int32`,
   `category`) so it fits comfortably in memory.
2. Flags **Live Attenders** — any `(student, course)` pair with at least one
   click on a live web-conferencing tool (`oucollaborate` or `ouelluminate`).
3. Takes successful students from `studentInfo.csv` and removes the Live
   Attenders, leaving the **Successful Skippers** master frame.
4. Sums each skipper's clicks on `resource`/`oucontent` material, then
   **left-joins** those sums back onto the master frame so skippers with zero
   such clicks are kept and counted as `0`.
5. Groups by course and takes the mean — `N_raw` for each course-presentation.

### Stage 2 — `G_proxy`: Grading Curve Effect / Sucker's Penalty

The **per-course** **Coefficient of Variation** (`CV = sigma / mu`) of
assessment scores, a proxy for how steep and competitive the grading curve
is. The pipeline:

1. Joins `studentAssessment.csv` to `assessments.csv` to attach each score to
   its course-presentation; blank/invalid scores are coerced to `NaN` and
   dropped.
2. Computes the mean (`mu`) and sample standard deviation (`sigma`) of the
   `score` column per course-presentation.
3. `G_proxy` per course is `sigma / mu`; the script prints the full per-course
   table sorted descending.

### Stage 3 — Integration & Scaling Engine

1. Merges the per-course `N_raw` and `G_proxy` tables on
   `code_module` / `code_presentation`.
2. Applies two **locked constants** — `ALPHA = 0.001` and `GAMMA = 1.0` — to
   produce the final solver inputs: `N_final = N_raw × ALPHA` and
   `G_final = G_proxy × GAMMA`.
3. Uses `idxmax` / `idxmin` to isolate and print the four empirical extremes —
   highest/lowest grading curve and highest/lowest VLE reliance — each with
   its course identifier and `(N_final, G_final)` pair.

## What `build_studentlife_c_proxy.py` does

The StudentLife script builds an empirical proxy for `C`, the effort cost of
attending class. This is a separate preprocessing step from the OULAD script:
it does **not** replace the EGT simulation, and it does not estimate `N` or
`G`. Instead, it gives the model an evidence-backed range for the attendance
cost parameter.

For each student-week, the script computes:

```text
C_i,w = normalized stress_i,w
      + normalized sleep loss_i,w
      + normalized deadline pressure_i,w
```

where:

```text
sleep loss_i,w = max(0, 8 - sleep hours_i,w)
```

The raw weekly index is then min-max rescaled to:

```text
C_i,w in [0, 1.5]
```

This makes the heatmap range for `C` defensible: instead of treating `C` as a
single fixed constant, the model can say that StudentLife shows effort cost
varies across the term as stress, sleep loss, and deadline pressure change.

### StudentLife inputs

The script uses:

- `dataset/student-life/EMA/response/Stress/` for stress self-reports.
- `dataset/student-life/EMA/response/Sleep/` for sleep-duration self-reports.
- `dataset/student-life/education/deadlines.csv` for weekly deadline load.
- `dataset/student-life/education/grades.csv` for a descriptive, non-causal
  grade-correlation check.

### StudentLife outputs

Running the script creates:

```text
outputs/studentlife/
├── studentlife_C_by_week.csv
├── studentlife_C_weekly_summary.csv
├── studentlife_C_by_student.csv
├── studentlife_C_grade_correlations.csv
└── studentlife_C_weekly_plot.png
```

The student-week CSV is the main model-facing output. The weekly summary and
plot describe the term-level pattern. The student summary and grade
correlations are descriptive diagnostics only; they require at least four valid
`C` weeks per student and should not be interpreted as causal evidence that
effort cost changes grades.

## Dataset

This project uses two external datasets.

### Open University Learning Analytics Dataset

The OULAD preprocessing stage uses the **Open University Learning Analytics
Dataset (OULAD)**.

> Kuzilek, J., Hlosta, M., & Zdrahal, Z. (2017). Open University Learning
> Analytics dataset. *Scientific Data*, 4, 170171.
> https://doi.org/10.1038/sdata.2017.171

UCI Machine Learning Repository entry:
https://archive.ics.uci.edu/dataset/349/open+university+learning+analytics+dataset

The dataset is **not included** in this repository — download it separately
(see below). It is distributed under the
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) licence.

### StudentLife

The StudentLife preprocessing stage uses the Dartmouth StudentLife dataset.
Place it at:

```text
dataset/student-life/
```

The relevant folders/files are:

```text
dataset/student-life/
├── EMA/response/Stress/
├── EMA/response/Sleep/
└── education/
    ├── deadlines.csv
    ├── grades.csv
    └── piazza.csv
```

## Setup

### 1. Requirements

- Python 3.9+
- `pandas`
- `pillow`

```bash
pip install pandas pillow
```

### 2. Download and unzip the dataset

1. Open the UCI page:
   https://archive.ics.uci.edu/dataset/349/open+university+learning+analytics+dataset
2. Click **Download** to get the dataset archive (a `.zip`).
3. Unzip it into a folder — e.g. a `dataset/` directory next to this repo:

   ```bash
   # macOS / Linux
   unzip ~/Downloads/open+university+learning+analytics+dataset.zip -d dataset
   ```

After unzipping, the folder should contain these CSVs:

```
dataset/
├── assessments.csv
├── courses.csv
├── studentAssessment.csv
├── studentInfo.csv
├── studentRegistration.csv
├── studentVle.csv      (~433 MB)
└── vle.csv
```

### 3. Configure the dataset path

The script reads the path from the `DATA_DIR` constant near the top of
`extract_egt_params.py`:

```python
# Filepath for the directory containing the CSVs, modify for your setup
DATA_DIR = Path(__file__).resolve().parent.parent / "dataset"
```

By default it points at a `dataset/` folder **beside this repository**
(i.e. sharing the same parent directory). If your CSVs live elsewhere, edit
this line, for example:

```python
DATA_DIR = Path("/absolute/path/to/dataset")
```

## Usage

### OULAD extraction for `N` and `G`

```bash
cd Attendance-EGT-Model
python3 extract_egt_params.py
```

The script prints each stage's diagnostics, the full per-course parameter
matrix (`N_raw`, `G_proxy`, `N_final`, `G_final` for all 22
course-presentations), and ends with the four extreme courses:

```
======================================================================
EMPIRICAL EXTREMES -- SOLVER INPUT PAIRS
======================================================================

HIGHEST CURVE        (max G_final)
  Course   : BBB 2014J
  N_final  : 0.460719
  G_final  : 0.410651

LOWEST CURVE         (min G_final)
  Course   : EEE 2014J
  N_final  : 0.589751
  G_final  : 0.176682

HIGHEST VLE RELIANCE (max N_final)
  Course   : FFF 2014J
  N_final  : 1.165468
  G_final  : 0.181847

LOWEST VLE RELIANCE  (min N_final)
  Course   : BBB 2013B
  N_final  : 0.032826
  G_final  : 0.227268
```

### StudentLife extraction for `C`

If using the repo-local virtual environment:

```bash
source .venv/bin/activate
python build_studentlife_c_proxy.py
```

Or run it directly:

```bash
.venv/bin/python build_studentlife_c_proxy.py
```

Expected console summary:

```text
StudentLife C proxy generated
  stress responses used : 2154
  sleep responses used  : 1372
  student-week rows     : 484
  valid C rows          : 268
  output directory      : .../outputs/studentlife
```

The important modeling output is `studentlife_C_by_week.csv`, which gives a
student-week-level `C_scaled` value in `[0, 1.5]` when stress, sleep, and
deadline components are all available.
