# Stochastic Attendance Simulation Plan

This document explains how to extend the current Attendance-EGT project from
empirical parameter extraction into a full stochastic simulation pipeline.

The existing repository estimates empirical inputs:

- `N`: note-sharing / asynchronous-material value from OULAD.
- `G`: grading-curve / assessment-dispersion proxy from OULAD.
- `C`: effort-cost proxy from StudentLife stress, sleep loss, and deadlines.

The stochastic extension should consume those inputs and simulate how a finite
population of students evolves between two strategies:

- `Skip`
- `Attend`

The extension has two connected parts:

1. A deterministic expected-value replicator model using a stochastic
   professor-intervention payoff matrix.
2. A Monte Carlo agent-based simulation where intervention events actually
   happen randomly in finite populations.

These two parts answer different questions. The deterministic model gives the
clean mathematical phase boundaries. The Monte Carlo model shows finite-size
randomness, path dependence, and differences between frequent small penalties
and rare large penalties.

## 1. Model Variables

Use the same core attendance game from the presentation, then add stochastic
professor controls.

### Student-side parameters

```text
K = baseline knowledge value from attending
N = note-sharing / asynchronous-material value available to skippers
G = grading-curve or competitive penalty effect
C = effort cost of attending
E = professor engagement / classroom value added to attenders
```

Current empirical sources:

```text
N, G -> extract_egt_params.py using OULAD
C    -> build_studentlife_c_proxy.py using StudentLife
K, E -> controlled model parameters
```

### Professor-intervention parameters

```text
p = probability an enforcement/intervention event happens on a class day
Q = penalty cost paid by skippers when the event happens
beta_note = fraction of note-sharing value removed by the intervention
```

Examples:

```text
p = 0.10       rare intervention
Q = 1.00       high penalty when caught
beta_note = 0.50  notes are half as useful because class is more interactive
```

Use `beta_note` in code rather than just `beta`, because the Monte Carlo update
rule will also need a selection-intensity parameter. Keeping the names separate
prevents confusion.

## 2. Deterministic Expected Payoff Model

Let:

```text
x = fraction of the population currently skipping
1 - x = fraction currently attending
```

The expected stochastic-intervention payoff matrix is:

```text
                         Opponent skips              Opponent attends
You skip                 -pQ                         G + N(1 - beta_note) - pQ
You attend               K + E - C - G               K + E - C
```

This matrix uses the expected penalty `pQ`. It is appropriate for the
deterministic replicator model because the deterministic model averages over
many possible intervention outcomes.

The expected fitness functions are:

```text
f_skip(x) = -pQ + (1 - x)(G + N(1 - beta_note))

f_attend(x) = x(K + E - C - G) + (1 - x)(K + E - C)
            = K + E - C - xG
```

The fitness difference is:

```text
f_skip(x) - f_attend(x)
  = -pQ + G + N(1 - beta_note)(1 - x) - K - E + C
```

The replicator equation is:

```text
x' = x(1 - x)(f_skip(x) - f_attend(x))
```

Interpretation:

```text
x' > 0  -> skipping spreads
x' = 0  -> equilibrium
x' < 0  -> attending spreads
```

## 3. Equilibrium Classification

Define:

```text
N_eff = N(1 - beta_note)
A = pQ - G + K + E - C
```

When `N_eff > 0`, the interior equilibrium is:

```text
x_star = 1 - A / N_eff
```

Classification:

```text
Attend ESS if x_star < 0
Mixed ESS  if 0 <= x_star <= 1
Skip ESS   if x_star > 1
```

Equivalent boundary conditions:

```text
Attend ESS if K + E - C + pQ > G + N_eff
Skip ESS   if K + E - C + pQ < G
Mixed ESS  otherwise
```

Special case:

```text
If N_eff = 0, the free-rider note value has been fully removed.
Then f_skip - f_attend no longer depends on x.
The system goes to whichever pure strategy has higher fitness.
```

This equilibrium logic should live in one shared module so the phase diagrams,
trajectory integrations, and Monte Carlo comparisons all use the same model
definitions.

## 4. Important Double-Counting Rule

The deterministic and Monte Carlo simulations handle the penalty differently.

### Deterministic replicator model

Use the expected penalty:

```text
-pQ
```

This belongs in the deterministic expected payoff matrix.

### Monte Carlo agent-based model

Do not use `-pQ` in each skipper's base payoff.

Instead:

1. Draw an intervention event with probability `p`.
2. If the event happens, subtract `Q` from skippers.
3. If the event does not happen, subtract nothing.

This distinction matters. If the Monte Carlo model includes `-pQ` and also
randomly applies `-Q`, the penalty is counted twice.

## 5. Proposed Repository Structure

Add a new simulation layer without mixing it into the existing preprocessing
scripts.

```text
Attendance-EGT-Model/
├── README.md
├── STOCHASTIC_SIMULATION_README.md
├── extract_egt_params.py
├── build_studentlife_c_proxy.py
├── attendance_game.py
├── simulate_replicator.py
├── simulate_agent_monte_carlo.py
├── run_stochastic_experiments.py
├── outputs/
│   ├── studentlife/
│   │   ├── studentlife_C_by_week.csv
│   │   ├── studentlife_C_weekly_summary.csv
│   │   ├── studentlife_C_by_student.csv
│   │   ├── studentlife_C_grade_correlations.csv
│   │   └── studentlife_C_weekly_plot.png
│   └── stochastic/
│       ├── replicator_trajectories.csv
│       ├── phase_grid.csv
│       ├── monte_carlo_runs.csv
│       ├── monte_carlo_summary.csv
│       ├── trajectory_comparison.png
│       └── phase_diagram.png
└── dataset/
    └── student-life/
```

The current scripts remain responsible for empirical preprocessing:

```text
extract_egt_params.py          -> builds N and G from OULAD
build_studentlife_c_proxy.py   -> builds C from StudentLife
```

The new scripts should be responsible only for model simulation:

```text
attendance_game.py             -> formulas and ESS classification
simulate_replicator.py         -> deterministic expected-value trajectories
simulate_agent_monte_carlo.py  -> finite-population stochastic simulation
run_stochastic_experiments.py  -> parameter sweeps and output generation
```

## 6. Module 1: `attendance_game.py`

This module is the analytical backbone.

### Responsibilities

It should define the model parameters, payoff functions, equilibrium formula,
and ESS classification.

### Suggested objects and functions

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class AttendanceParams:
    K: float
    N: float
    G: float
    C: float
    E: float
    p: float = 0.0
    Q: float = 0.0
    beta_note: float = 0.0


def effective_note_value(params: AttendanceParams) -> float:
    ...


def fitness_skip(x: float, params: AttendanceParams) -> float:
    ...


def fitness_attend(x: float, params: AttendanceParams) -> float:
    ...


def fitness_difference(x: float, params: AttendanceParams) -> float:
    ...


def dxdt(x: float, params: AttendanceParams) -> float:
    ...


def interior_equilibrium(params: AttendanceParams) -> float | None:
    ...


def classify_ess(params: AttendanceParams) -> str:
    ...
```

### Expected classification labels

Use stable, machine-readable labels:

```text
attend
mixed
skip
neutral
```

`neutral` is only for edge cases where the payoff difference is zero or nearly
zero across the whole state space.

### Validation checks

Before building the full simulation, test this module with the presentation
example:

```text
K = 1.0
N = 0.7
G = 0.2
C = 0.8
E = 0.6
p = 0.0
Q = 0.0
beta_note = 0.0
```

Then test one intervention case:

```text
p = 0.5
Q = 0.5
beta_note = 0.5
```

Confirm manually that:

```text
N_eff = N(1 - beta_note)
x_star = 1 - (pQ - G + K + E - C) / N_eff
```

matches the code output.

## 7. Module 2: `simulate_replicator.py`

This module numerically integrates the deterministic replicator equation:

```text
x' = x(1 - x)(f_skip(x) - f_attend(x))
```

### First implementation

Start with Euler integration. It is transparent and enough for the first pass.

```text
x_next = x_current + dt * dxdt(x_current)
```

Clamp `x_next` to `[0, 1]` after each update to avoid tiny numerical drift.

Suggested defaults:

```text
dt = 0.05
T = 100
initial_x_values = [0.01, 0.10, 0.20, 0.50, 0.80, 0.90, 0.99]
```

### Later improvement

After the Euler version works, optionally add `scipy.integrate.solve_ivp`.
Do not make SciPy required in the first pass unless the project already depends
on it.

### Output schema

Save one row per time step:

```text
scenario,K,N,G,C,E,p,Q,beta_note,x0,t,x,dxdt,ess_class,x_star
```

Write to:

```text
outputs/stochastic/replicator_trajectories.csv
```

### What this output tells us

This answers:

```text
For a given parameter setting, does skipping grow or shrink?
Do all starting conditions converge to the same equilibrium?
How does changing p, Q, or beta_note move the equilibrium?
```

## 8. Module 3: Phase Boundary Computation

The phase boundary computation sweeps a grid of `C` and `E` values and classifies
the ESS regime at each point.

### Why this matters

This reproduces the presentation's phase-diagram logic, but now with stochastic
professor controls:

```text
p, Q, beta_note
```

Instead of asking only "what happens for one parameter setting?", it asks:

```text
Across possible effort costs and engagement levels, where does attendance win,
where does skipping win, and where is the stable mixed equilibrium?
```

### Grid

Use the StudentLife scaling range for `C`:

```text
C in [0.0, 1.5]
```

Use a similar range for `E`:

```text
E in [0.0, 1.5]
```

Suggested first grid:

```text
C_values = 0.00, 0.05, 0.10, ..., 1.50
E_values = 0.00, 0.05, 0.10, ..., 1.50
```

### OULAD scenarios

Use the four scenarios already documented in the main README:

```text
Highest curve:
  N = 0.460719
  G = 0.410651

Lowest curve:
  N = 0.589751
  G = 0.176682

Highest VLE reliance:
  N = 1.165468
  G = 0.181847

Lowest VLE reliance:
  N = 0.032826
  G = 0.227268
```

### Professor-control slices

Start with a small grid:

```text
p_values = [0.0, 0.10, 0.25, 0.50, 0.75, 1.0]
Q_values = [0.0, 0.25, 0.50, 1.0]
beta_note_values = [0.0, 0.25, 0.50, 0.75]
```

Do not start with every possible combination if plotting becomes too noisy.
Begin with fixed `Q = 0.5` and compare `p` slices, then add the full grid.

### Output schema

Save one row per grid point:

```text
scenario,N,G,K,C,E,p,Q,beta_note,N_eff,x_star,ess_class
```

Write to:

```text
outputs/stochastic/phase_grid.csv
```

### Plots

For each OULAD scenario and each selected intervention slice, make a heatmap:

```text
x-axis: E
y-axis: C
color: ESS class
```

Suggested colors:

```text
attend -> green
mixed  -> yellow / neutral
skip   -> red
```

Write to:

```text
outputs/stochastic/phase_diagram.png
```

If there are many scenarios, save one image per scenario:

```text
outputs/stochastic/phase_highest_curve.png
outputs/stochastic/phase_lowest_curve.png
outputs/stochastic/phase_highest_vle_reliance.png
outputs/stochastic/phase_lowest_vle_reliance.png
```

## 9. Module 4: `simulate_agent_monte_carlo.py`

This is the genuinely stochastic simulation.

The deterministic replicator model uses expected penalties. The Monte Carlo
model simulates actual finite students and actual random intervention events.

### Core setup

```text
N_pop = number of students
T = number of class days / generations
M = number of repeated simulation runs
initial_skip_rate = initial fraction of skippers
selection_intensity = strength of payoff-based imitation
mutation_rate = probability of random strategy flip
seed = random seed for reproducibility
```

Suggested first defaults:

```text
N_pop = 100
T = 250
M = 500
initial_skip_rate = 0.20
selection_intensity = 5.0
mutation_rate = 0.005
seed = 123
```

### Strategy representation

Use integers:

```text
0 = attend
1 = skip
```

This makes arrays simple and fast.

### Round structure

Each round represents one class day.

For every round:

1. Randomly shuffle students.
2. Pair them into random pairs.
3. Compute pairwise payoffs.
4. Draw the intervention event:

   ```text
   intervention_happens = random_uniform() < p
   ```

5. If intervention happens, subtract `Q` from every skipper's payoff.
6. Update strategies using Fermi imitation.
7. Apply rare mutation/random experimentation.
8. Record the skip fraction `x(t)`.

### Pairwise non-expected payoff matrix

In the Monte Carlo model, use the non-expected matrix before intervention:

```text
                         Opponent skips              Opponent attends
You skip                 0                            G + N(1 - beta_note)
You attend               K + E - C - G                K + E - C
```

Then apply stochastic intervention separately:

```text
if intervention_happens:
    payoff[students_who_skip] -= Q
```

### Fermi imitation update

For each student `i`, compare them to another student `j`.

```text
P(i adopts j's strategy)
  = 1 / (1 + exp(-selection_intensity * (payoff_j - payoff_i)))
```

Interpretation:

```text
If j has much higher payoff, i is likely to copy j.
If j has lower payoff, i might still copy j by chance, but less often.
```

### Mutation

After imitation, flip a student's strategy with small probability:

```text
mutation_rate = 0.005
```

This prevents the population from becoming completely frozen too early and
represents experimentation, absence for unrelated reasons, or students changing
behavior for non-payoff reasons.

### Output schema: raw runs

Save one row per run and time point:

```text
scenario,run_id,t,K,N,G,C,E,p,Q,beta_note,N_pop,initial_skip_rate,
selection_intensity,mutation_rate,intervention_happened,skip_rate
```

Write to:

```text
outputs/stochastic/monte_carlo_runs.csv
```

### Output schema: summary

Save one row per parameter setting:

```text
scenario,K,N,G,C,E,p,Q,beta_note,N_pop,initial_skip_rate,
selection_intensity,mutation_rate,M,T,mean_final_skip_rate,
sd_final_skip_rate,p_attend_dominates,p_skip_dominates,p_mixed,
mean_last_50_skip_rate,sd_last_50_skip_rate
```

Write to:

```text
outputs/stochastic/monte_carlo_summary.csv
```

Suggested outcome definitions:

```text
attend_dominates if final_skip_rate < 0.10
skip_dominates   if final_skip_rate > 0.90
mixed            otherwise
```

## 10. Module 5: StudentLife `C` Integration

There are three ways to use the existing StudentLife `C` output.

### Option A: fixed C

Use a constant:

```text
C = 0.6
```

This is best for debugging.

### Option B: grid C

Sweep:

```text
C = 0.0, 0.05, 0.10, ..., 1.5
```

This is best for phase diagrams and clean theory comparisons.

### Option C: empirical weekly C

Use:

```text
outputs/studentlife/studentlife_C_weekly_summary.csv
```

Set `C` equal to the weekly `C_mean` for each simulated week.

This asks:

```text
Does skipping become more likely during high-stress / high-deadline weeks?
```

### Option D: empirical student-week C sampling

Use:

```text
outputs/studentlife/studentlife_C_by_week.csv
```

Sample from non-null `C_scaled` values.

Two versions are possible:

```text
Population-level sampling:
  one C value per class day, shared by all students

Student-level sampling:
  each student receives their own C value each day
```

Start with population-level sampling because it is easier to interpret.

## 11. `run_stochastic_experiments.py`

This should be the main script that runs the full simulation pipeline.

### Responsibilities

1. Define OULAD scenarios.
2. Define model parameter grids.
3. Run deterministic phase classification.
4. Run deterministic trajectory simulations.
5. Run Monte Carlo simulations.
6. Save CSV outputs.
7. Save plots.
8. Print a short console summary.

### Suggested first run

Start small:

```text
scenario = presentation_example
K = 1.0
N = 0.7
G = 0.2
C = 0.8
E = 0.6
p = 0.0
Q = 0.0
beta_note = 0.0
N_pop = 100
T = 250
M = 100
initial_skip_rate = 0.20
selection_intensity = 5.0
mutation_rate = 0.005
```

Then add intervention:

```text
p = 0.25
Q = 0.5
beta_note = 0.5
```

Then run the four OULAD scenarios.

### Full experiment grid

After the small run works:

```text
scenarios = [
  presentation_example,
  highest_curve,
  lowest_curve,
  highest_vle_reliance,
  lowest_vle_reliance,
]

C_values = [0.0, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50]
E_values = [0.0, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50]
p_values = [0.0, 0.10, 0.25, 0.50, 0.75, 1.0]
Q_values = [0.25, 0.50, 1.0]
beta_note_values = [0.0, 0.25, 0.50, 0.75]
```

Do not run the full grid until the small run is validated. The full grid can
become large quickly.

## 12. Visualizations

The final simulation layer should produce five main plot types.

### 1. Phase diagram

```text
x-axis: E
y-axis: C
color: attend / mixed / skip ESS
facet: p, Q, or beta_note
```

Purpose:

```text
Shows how intervention moves the theoretical attendance boundary.
```

### 2. Replicator trajectory fan

```text
x-axis: time
y-axis: skip fraction x
line: one initial condition x0
```

Purpose:

```text
Shows whether multiple starting conditions converge to the same equilibrium.
```

### 3. Phase portrait

```text
x-axis: skip fraction x
y-axis: x'
```

Purpose:

```text
Shows where skipping grows or shrinks and marks stable/unstable equilibria.
```

### 4. Monte Carlo ensemble

```text
x-axis: time
y-axis: skip fraction
line: mean across Monte Carlo runs
band: 10th to 90th percentile or standard deviation
overlay: deterministic replicator trajectory
```

Purpose:

```text
Compares finite-population stochastic behavior to deterministic prediction.
```

### 5. Same expected penalty, different variance comparison

Compare cases with the same `pQ`:

```text
frequent low penalty:
  p = 0.8
  Q = 0.25
  pQ = 0.20

rare high penalty:
  p = 0.2
  Q = 1.00
  pQ = 0.20
```

Purpose:

```text
The deterministic model treats these as equivalent.
The Monte Carlo model may not.
```

This is likely one of the most interesting results.

## 13. Validation Checklist

Before trusting results, verify these checks.

### Analytical checks

```text
fitness_difference(x) matches the hand-derived formula.
x_star matches the hand-derived equilibrium.
ESS classification matches the boundary inequalities.
When p = 0, Q = 0, beta_note = 0, the model reduces to the original game.
When beta_note = 1, note-sharing value N_eff becomes 0.
```

### Deterministic simulation checks

```text
x always remains in [0, 1].
If dxdt is positive at x0, the next x increases.
If dxdt is negative at x0, the next x decreases.
Trajectories converge toward the classified equilibrium.
Smaller dt gives nearly the same trajectory as larger dt.
```

### Monte Carlo checks

```text
With p = 0, no intervention penalties are applied.
With p = 1, skippers are penalized every round.
With Q = 0, intervention events have no payoff effect.
With beta_note = 1, skippers get no note-sharing benefit against attenders.
With mutation_rate = 0, all-attend and all-skip states can become absorbing.
With a fixed random seed, results are reproducible.
```

### Deterministic vs Monte Carlo checks

```text
For large N_pop and many runs, Monte Carlo mean should roughly follow the
deterministic expected trajectory.

For small N_pop, Monte Carlo should show more variance and occasional fixation
by chance.

Cases with the same pQ should match in the deterministic model but may differ
in Monte Carlo due to penalty variance.
```

## 14. Recommended Implementation Order

Do the work in this order.

### Step 1: Build `attendance_game.py`

Implement:

```text
AttendanceParams
fitness_skip
fitness_attend
fitness_difference
dxdt
interior_equilibrium
classify_ess
```

Do not move on until the formulas match hand calculations.

### Step 2: Build deterministic trajectory integration

Implement:

```text
simulate_replicator_path(params, x0, dt, T)
simulate_replicator_fan(params, x0_values, dt, T)
```

Save:

```text
outputs/stochastic/replicator_trajectories.csv
```

### Step 3: Build phase-grid classification

Implement:

```text
build_phase_grid(scenarios, C_values, E_values, p_values, Q_values, beta_note_values)
```

Save:

```text
outputs/stochastic/phase_grid.csv
```

### Step 4: Build the first Monte Carlo simulation

Implement one run:

```text
run_one_agent_simulation(params, N_pop, T, initial_skip_rate, selection_intensity,
                         mutation_rate, seed)
```

Save one trajectory and inspect it.

### Step 5: Add repeated Monte Carlo runs

Implement:

```text
run_monte_carlo_batch(params, M, ...)
summarize_monte_carlo_runs(...)
```

Save:

```text
outputs/stochastic/monte_carlo_runs.csv
outputs/stochastic/monte_carlo_summary.csv
```

### Step 6: Add plots

Start with:

```text
trajectory_comparison.png
phase_diagram.png
```

Only add animation or complicated faceting after the basic plots are correct.

### Step 7: Add StudentLife C modes

Add:

```text
fixed C
grid C
weekly empirical C
sampled empirical C
```

Start with fixed and grid C. Add empirical C after the core simulation works.

## 15. What Results To Report

The final writeup should report these findings.

### Deterministic findings

```text
How p changes the attend / mixed / skip regions.
How Q changes the attend / mixed / skip regions.
How beta_note changes the attend / mixed / skip regions.
Which OULAD scenario is most vulnerable to skipping.
Which scenario is easiest to shift toward attendance.
```

### Monte Carlo findings

```text
How often attendance dominates.
How often skipping dominates.
How often the population stays mixed.
How much variance exists across runs.
How long it takes to approach equilibrium.
Whether rare large penalties differ from frequent small penalties.
Whether high-C StudentLife weeks increase skipping risk.
```

### Important caution

Do not claim that StudentLife `C` causally determines grades. The existing
grade correlations are descriptive diagnostics only. The simulation can say
that higher modeled effort cost makes skipping more attractive inside the EGT
model. It cannot prove that stress, sleep loss, or deadlines caused real
students to skip class or earn different grades.

## 16. Minimum Complete Version

A minimum complete stochastic extension is done when the repo can produce:

```text
outputs/stochastic/replicator_trajectories.csv
outputs/stochastic/phase_grid.csv
outputs/stochastic/monte_carlo_runs.csv
outputs/stochastic/monte_carlo_summary.csv
outputs/stochastic/trajectory_comparison.png
outputs/stochastic/phase_diagram.png
```

and the console can answer:

```text
For each scenario, what is the deterministic ESS class?
For each scenario, what is the Monte Carlo mean final skip rate?
How often does attendance dominate, skipping dominate, or remain mixed?
How do p, Q, and beta_note change the outcome?
```

That is the clean bridge from the current empirical preprocessing repository to
a full stochastic EGT attendance simulation.
