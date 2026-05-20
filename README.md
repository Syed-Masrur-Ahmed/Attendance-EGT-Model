# Attendance-EGT-Model

An Evolutionary Game Theory (EGT) study of classroom attendance modelled as a
public-goods game. This repository holds the data-extraction stage: deriving
empirical parameters from the Open University Learning Analytics Dataset
(OULAD) to feed an EGT differential-equation solver.

## What `extract_egt_params.py` does

The script reads the OULAD CSVs with `pandas` and computes two scalar
parameters, printing them to the console ready to paste into the solver.

### `N_raw` — Note-Sharing Value / Free-Rider Extraction

The average number of clicks on **static asynchronous material**
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
   Attenders, leaving the **Successful Skippers**.
4. Sums each skipper's clicks on `resource`/`oucontent` material; skippers
   with zero such clicks count as `0`.
5. `N_raw` is the global mean of those per-skipper sums.

### `G_proxy` — Grading Curve Effect / Sucker's Penalty

The **Coefficient of Variation** (`CV = sigma / mu`) of assessment scores,
a proxy for how steep and competitive the grading curve is. The pipeline:

1. Joins `studentAssessment.csv` to `assessments.csv` to attach each score to
   its course-presentation; blank/invalid scores are coerced to `NaN` and
   dropped.
2. Computes the mean (`mu`) and sample standard deviation (`sigma`) of the
   `score` column per course-presentation.
3. `G_proxy` per course is `sigma / mu`; the script prints the full per-course
   table sorted descending, plus the overall mean across all courses.

## Dataset

This project uses the **Open University Learning Analytics Dataset (OULAD)**.

> Kuzilek, J., Hlosta, M., & Zdrahal, Z. (2017). Open University Learning
> Analytics dataset. *Scientific Data*, 4, 170171.
> https://doi.org/10.1038/sdata.2017.171

UCI Machine Learning Repository entry:
https://archive.ics.uci.edu/dataset/349/open+university+learning+analytics+dataset

The dataset is **not included** in this repository — download it separately
(see below). It is distributed under the
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) licence.

## Setup

### 1. Requirements

- Python 3.9+
- `pandas` (developed against 2.2.3)

```bash
pip install pandas
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

```bash
cd Attendance-EGT-Model
python3 extract_egt_params.py
```

The script prints both pipelines' diagnostics and ends with the two
parameters:

```
======================================================================
EGT MODEL PARAMETERS
======================================================================
  N_raw   = 392.4109
  G_proxy = 0.2395   (mean across 22 courses)
======================================================================
```
