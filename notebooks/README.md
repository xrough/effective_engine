# Volatility research notebooks

## Volatility project workflow

Start here for a four-to-six-hour, Kaggle-style data exercise. These three notebooks are
short, linear, standalone worked examples: their core cells use ordinary pandas,
scikit-learn and matplotlib, with LightGBM when it is installed and an explicitly labelled
scikit-learn fallback when it is not.

| Notebook | Use it for | Finished deliverable |
|---|---|---|
| [Data and joins](volatility_project_01_data_and_joins.ipynb) | unfamiliar files, key audits, `1:1`/`m:1` merges and backward as-of joins | one modelling table plus an issue and availability log |
| [One time series](volatility_project_02_time_series.ipynb) | causal lags, persistence baselines and expanding validation | OOF comparison, sealed-tail result and submission |
| [Financial panel](volatility_project_03_panel.ipynb) | many entities per date, group baselines, Ridge and LightGBM | date-safe OOF ledger, rank diagnostics and submission |

A practical clock:

| Elapsed | Work |
|---:|---|
| 0:00–0:35 | Define the row, target, metric and prediction-time information set; inventory and load the tables. |
| 0:35–1:15 | Check keys, time coverage, missingness and joins; write down unresolved data risks. |
| 1:15–2:00 | Build causal features, seal the final tail and prove the folds respect time. |
| 2:00–3:15 | Score the mean/lag/group baselines, Ridge and LightGBM on the same OOF rows. |
| 3:15–4:15 | Diagnose fold stability, time or group failures, tails, drift and residual dependence. |
| 4:15–5:00 | Open the sealed tail once, refit, validate the submission and prepare the result summary. |
| 5:00–6:00 | Contingency: fix data issues or test one justified extension, not a broad search. |

The only cell intended to be replaced is tagged `swap-data`. With `USE_DEMO=True`, each
notebook generates deterministic financial-style inputs offline. With `USE_DEMO=False`,
missing real inputs raise immediately; the notebook never silently substitutes demo data.
The default target-lag baseline is disabled unless the prediction contract says historical
labels are actually available.

Minimal environment:

```bash
python -m pip install numpy pandas matplotlib scikit-learn jupyterlab
python -m pip install lightgbm  # preferred nonlinear model; optional
python -m jupyter lab notebooks
```

For depth after the volatility project workflow, use [Forecasting the Smile](forecasting_the_smile.ipynb)
for temporal forecasting, [The Cross-Section](the_cross_section.ipynb) for panel research,
[Seeing and Joining Evidence](seeing_and_joining_evidence.ipynb#r7-32) for join variants,
and the [Competition Workbench](competition_workbench.ipynb) for stricter experiment and
artifact contracts.

> **Stored-output provenance:** an output describes the data and code printed in that stored
> run, not whatever files happen to be on this machine now. Rerun a notebook top to bottom
> before relying on its results in an interview.

---

## Advanced research reference

The other nine notebooks use volatility to show what a model claims, how its parameters are
estimated, and how you decide whether it is telling you anything; the competition workbench
turns that method into a reusable structured-data controller. The subject is volatility; the
through-line is **methodology**, because in empirical work the order in which questions get
asked is what makes an answer trustworthy.

The series is organised as a research programme in four parts, not as a list of procedures.

**Part I — a volatility model, and whether it holds.** One hypothesis taken all the
way to a verdict. State what the model claims, estimate the parameter it turns on, build
features that do not leak, establish a benchmark and a validation scheme that respects time,
score it, ask what the model actually used, ask whether the result holds anywhere, and finish
with an economic test rather than a statistical one. The worked hypothesis is *rough
volatility* — that volatility is driven by fractional Brownian motion with Hurst exponent
H ≈ 0.1 rather than the H = 0.5 of a diffusion. Its raw maturity-scaling claim fails, while
a nested carry-conditioned correction can recover modest out-of-fold forecast skill. That
mixed verdict is the point: the machinery must distinguish a structural claim, a predictive
model and a trading result instead of forcing all three into one headline.

**Part II — the cross-sectional design.** The same methodology under a different
research design: many entities on one date rather than one series through time. The estimand
becomes a ranking, the validation becomes purged by date, the metric becomes rank IC. What
transfers is the sequence; what changes is every piece of machinery inside it — which is the
lesson. Point it at your own panel data by setting one environment variable.

**Part III — method in depth.** Each technique the first two parts leaned on, treated
on its own terms and pushed further than either needed: where numbers come from and how to
read a repository of them, what an audit can establish and the measurement error it cannot
remove, how evidence is displayed, how sources combine without corrupting each other, the
volatility estimators in their own right, how a model becomes a position with costs attached,
and the inference neither earlier part had to reach for.

**Part IV — the competition workbench.** One thin control notebook connects those recipes
into an executable spine for IID, grouped, temporal and panel data; regression, imbalanced
binary classification and ranking; missing values, unseen categories and train/test drift.
An immutable `ProblemSpec` chooses the validation design, every prediction is recorded out of
fold, and inference ends at a sample-validated, fresh-process-tested, atomically published
evidence bundle.

### Also a reference

Every notebook is built from numbered **recipes**: small, self-contained, copy-pasteable
blocks, each labelled with the technique it demonstrates and each verified to run on its own.
Read the parts in order and you get the method; search the [index](#recipe-index) and you get
the ingredient. When a task lands and the clock is running, the index is the faster door.

---

## Running them

From the repository root, create or activate an environment and install the complete notebook
extra:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[notebooks,dev]"
python -m jupyter lab notebooks
```

The nine advanced notebooks first import an installed `rvlab` and can also bootstrap it from
`src/` in a source checkout. Optional local `roughvol` and research modules are detected at
runtime; the documented notebook extra contains every package used by those paths. The three
volatility project notebooks deliberately do not import `rvlab`.

To re-execute the whole series, or to prove it runs with none of the local market data:

```bash
bash scripts/run_notebooks.sh
```

```bash
RVLAB_FORCE_SYNTHETIC=1 bash scripts/run_notebooks.sh
```

Set `RVLAB_PYTHON` when the runner should use a particular environment. By default it creates
a temporary kernelspec pinned to that exact interpreter, then removes it on exit. You may set
`RVLAB_KERNEL` to use an existing Python kernelspec instead; the runner resolves its command
and refuses to start unless it is the same interpreter it preflighted. This prevents a common
PATH mismatch where the shell checks one Python and Jupyter silently launches another.

`run_notebooks.sh` updates committed notebook outputs in place by default. For a read-only
checkout or a repository attached under `/kaggle/input`, write executed copies elsewhere:

```bash
RVLAB_OUTPUT_DIR=/kaggle/working/rvlab_output \
  bash scripts/run_notebooks.sh --output-dir /kaggle/working/executed-notebooks
```

`RVLAB_OUTPUT_DIR` holds figures, models and manifests created by cells; `--output-dir` (or
`RVLAB_EXECUTED_NOTEBOOK_DIR`) holds the executed `.ipynb` copies. Install non-editably from a
read-only input mount, or copy the repository to `/kaggle/working` before using the editable
installation command above.

## Data

**Part I** uses the SPY smile panel under `demo/`, which is gitignored. **Part III** reads the
repository itself — a live census across every format. **Part II** uses a (date × entity) equity
panel: real panel data if you have it, synthetic otherwise.

```bash
export RVLAB_PANEL_ROOT=/path/to/unzipped-competition   # containing train_files/stock_prices.csv
```

Every loader falls back to synthetic data with the same schema and stamps `synthetic: True`
on the frame, so the notebooks always execute — they just say which they used.
`the_volatility_surface.ipynb` reports what this machine has.

The reusable [Competition Workbench](competition_workbench.ipynb) is deliberately stricter:
its demo data is enabled by an explicit `USE_DEMO=True`. A missing real path must raise rather
than silently turn a competition run into a synthetic one.

Part II's synthetic panel has a **short-term reversal effect planted in it** by construction.
That is deliberate: a correct pipeline must find it, which makes Part II a test of the method
rather than a test of a market — a positive control to set against Part I's negative result.
`reversal_strength=0.0` gives a null panel for the opposite check.

## Where the code lives

Reusable logic is in [`src/rvlab/`](../src/rvlab); the notebooks import, call, plot and
narrate. Small local functions may make a single recipe readable, while anything shared
between recipes belongs in the package.

| Module | What it holds |
|---|---|
| `rvlab.data` | loaders, schema contracts, synthetic fallbacks |
| `rvlab.data.catalog` | repo-wide discovery, load-any-format, run-tree concatenation, parquet caching |
| `rvlab.data.quality` | health report, datetime audit, outliers three ways, KS/PSI drift, leakage scan |
| `rvlab.data.panel` | dataset discovery, train/supplemental concat, panel health audit |
| `rvlab.features` | smile, realized-vol, time-series and roughness builders |
| `rvlab.features.panel` | calendar, price geometry, event-masked returns, cross-sectional and group-relative transforms |
| `rvlab.features.joins` | validated merges, as-of and interval joins, aggregate-and-join, reshaping |
| `rvlab.features.encoding` | fold-safe target encoding, rare-level grouping, frequency and cyclical encoders |
| `rvlab.models` | Black-Scholes, the rough closed form, forecasting baselines |
| `rvlab.models.volatility` | five realized estimators, EWMA, HAR, GARCH(1,1) by MLE, term structure, VRP |
| `rvlab.pipelines` | sklearn transformers, leakage-safe splitters, pipeline factories |
| `rvlab.pipelines.inference` | schema alignment, rolling history buffer, submission ranking |
| `rvlab.evaluate` | metrics, statistical tests, parameter sweeps, hedge backtest |
| `rvlab.evaluate.ranking` | rank IC / ICIR / top-k spread Sharpe, blending |
| `rvlab.evaluate.classification` | imbalance, threshold sweeps, calibration slope, decile lift |
| `rvlab.strategy` | signal → position → costs → P&L, vol targeting, tearsheet |
| `rvlab.plotting` | one house style; 48 charts across matplotlib and seaborn |
| `rvlab.competition` | immutable problem specs, temporal and group-safe split routing, task-oriented ranking, OOF ledgers, adversarial drift, atomic checksummed evidence bundles |

```bash
python -m pytest src/rvlab/tests -q
```

---

## Reading order

Twelve notebooks are listed in `notebooks/.order`: the nine-notebook research programme below,
followed by the three-notebook volatility project workflow linked at the top of this README. Each advanced
notebook covers several sections, listed here so you can see where a topic lives without opening
the file.

### Part I — a volatility model, and whether it holds

| Notebook | Sections | The question it settles | What it found |
|---|---|---|---|
| [The Volatility Surface](the_volatility_surface.ipynb) | orientation · what is being observed · the first look | what is being measured, and how much of it is real? | ~12 expiries per instant over 7–60 days; clean quotes; skew negative in **100%** of observations |
| [What Roughness Claims](what_roughness_claims.ipynb) | volatility models · estimation and inference | does the smile scale the way the model predicts? | **no** — fitted slope **+0.21** against the predicted −0.40, wrong sign |
| [Forecasting the Smile](forecasting_the_smile.ipynb) | design matrix · benchmark and validation · the scoreboard | can that structure forecast anything? | carry is the strong baseline; the raw rough correction loses to it, while a nested Ridge correction shows modest positive OOF skill in the active run |
| [When a Result Is Real](when_a_result_is_real.ipynb) | attribution · robustness · the hedging test | does the result survive being attacked? | broad parameter and tenor sweeps find no robust raw effect; the conditioned forecast remains genuinely OOF, while the hedge still fails its economic comparison |

### Part II — the cross-sectional design

| Notebook | Sections | The question it settles | What it found |
|---|---|---|---|
| [The Cross-Section](the_cross_section.ipynb) | the design · ranking and purged validation · combining and shipping | what changes when the unit of analysis is the cross-section? | one best-supported individual model is selected dynamically and reused for refit and serving; RMSE can still disagree sharply with equivalent rankings |

### Part III — method in depth

| Notebook | Sections | The question it settles | What it found |
|---|---|---|---|
| [Where the Numbers Come From](where_the_numbers_come_from.ipynb) | reading a repository · auditing a dataset | what are the inputs, and what can be established about them? | 88% of the run tree is byte-duplicated — 16 rows, 2 distinct observations; two extractions of the same smile disagree, and that gap is a measurement uncertainty |
| [Seeing and Joining Evidence](seeing_and_joining_evidence.ipynb) | thirty-one charts · joins and reshaping | how is evidence displayed, and combined, without inventing it? | key dtypes and actual overlap are audited separately; deterministic as-of examples expose how join direction can leak future observations |
| [Measuring Volatility, and Trading It](measuring_and_trading_volatility.ipynb) | estimators and forecasts · model to position · inference | how is volatility measured, traded, and honestly reported? | the estimator choice moves the answer 25%; break-even cost is a few basis points at a realistic rank IC of 0.017 |

### Part IV — from case study to competition template

| Notebook | Sections | The question it settles | What it produces |
|---|---|---|---|
| [Competition Workbench](competition_workbench.ipynb) | contract and data boundary · audit and validation · OOF evidence · refit and submission | which analysis path is valid for this data geometry? | a geometry-safe final holdout, task-oriented OOF evidence, a checked submission, fitted pipeline and atomically published checksummed bundle |

## Advanced competition quickstart

Open [Competition Workbench](competition_workbench.ipynb), change the parameters cell, then
replace its single `swap-data` cell with loaders for `train`, `test`, and the organizer's
sample submission. Pick the geometry from the unit that must remain independent:

| Your rows represent | Geometry | Supported tasks | Validation invariant |
|---|---|---|---|
| independent observations | `iid` | regression, binary | shuffled folds; stratify a binary target |
| repeated people, sites, patients or devices | `grouped` | regression, binary, ranking | no group appears in both train and validation |
| observations used to predict a later time | `time` | regression, binary | every training time precedes validation; purge the forward label horizon and declare the train/test time relation |
| many entities observed at each time | `panel` | regression, binary, ranking | split whole times; entities may repeat; ranking additionally declares whether higher or lower targets are better |

The contract checks the metric and its direction rather than inferring them: regression
supports RMSE, MAE and R²; binary classification supports ROC-AUC, average precision, log
loss and Brier score; ranking uses mean within-query Spearman correlation. When adapting the
swap-data cell, seal the final holdout before displaying target values and replace the demo
loader identity with an immutable dataset and loader-code version.

The demo profiles deliberately include missing numerics, missing categories, unseen test
levels, imbalance and multivariate drift. The final condition-matrix recipe executes all ten
valid geometry/task pairings, so adapting one path cannot silently break another. For deeper
treatment, follow the workbench's links into the relevant volatility recipe rather than
copying more code into the control notebook.

### The single most transferable result

Each part turns on the same methodological point: **a result you have not held out is not a
result.** The failure mode changes with the design, so the defence must change with it.

Part I's [R4.12](when_a_result_is_real.ipynb#r4-12) estimates regime thresholds only from
preceding folds, then scores the conditioned forecast on unseen rows. The active-run estimate
remains genuinely out of fold and is allowed to be null; the notebook does not manufacture a
regime win by choosing thresholds on the rows it later reports.

Part II's counterpart is [R5.19](the_cross_section.ipynb#r5-19): two predictions that
are the *same trading strategy* — one is `100x + 7` of the other — with identical rank IC and
spread Sharpe and RMSE differing by a factor of 191. Any hyperparameter search scored on RMSE
would confidently prefer one over the other.

Part III's is [R8.25](measuring_and_trading_volatility.ipynb#r8-25): target-encode a **random** target and
the naive full-sample version still returns a confident positive cross-validated R². The
out-of-fold encoder returns approximately zero, which is the right answer. Nothing about the
leaky version looks wrong from the outside.

Part IV makes the discipline operational. [R9.8](competition_workbench.ipynb#r9-8) reserves
a geometry-safe final holdout before target-based development checks, and
[R9.17](competition_workbench.ipynb#r9-17) locks the selected configuration before the single
refit and grader-facing transformation. The exported bundle records the exact folds, rows,
predictions, artifacts and hashes.

Four designs, one discipline. That is what the series is arguing for.

---

## Recipe index

Every recipe is self-contained relative to its section: it uses the section's tagged data
context, the parameters cell, explicitly declared prerequisites, and names it defines itself;
it also carries its own imports. That is enforced, not merely intended:

```bash
python scripts/lint_notebooks.py --copy-paste-test
```

The copy-paste check forces bounded synthetic fixtures, redirects artifacts to a temporary
directory, caps numerical libraries at one worker, closes recipe figures eagerly, and runs
each notebook in its own subprocess. That makes the check reproducible on CI and constrained
competition runtimes, and identifies the notebook cleanly if a native dependency crashes or
hangs. Each notebook gets 15 minutes by default; set `RVLAB_COPY_PASTE_TIMEOUT_S` to change
that ceiling.

<!-- BEGIN RECIPE INDEX -->

_265 recipes across 12 notebooks. Generated by `scripts/build_notebook_index.py` — do not edit by hand._

[Sources and provenance](#sources-and-provenance) 19 · [Measurement and data quality](#measurement-and-data-quality) 29 · [Combining sources](#combining-sources) 19 · [Model inputs and targets](#model-inputs-and-targets) 20 · [Exploration and evidence](#exploration-and-evidence) 34 · [Volatility models and estimators](#volatility-models-and-estimators) 17 · [Predictive models](#predictive-models) 31 · [Validation and inference](#validation-and-inference) 34 · [Diagnostics and interpretation](#diagnostics-and-interpretation) 35 · [From model to position](#from-model-to-position) 13 · [Research practice](#research-practice) 14

### Sources and provenance

| Topic | Recipe | Use when | Go |
|---|---|---|---|
| Archives | Read inside an archive without extracting it | the data is zipped and the archive is large. `ZipFile` streams one member at a time, and listing the contents  | [R6.8](where_the_numbers_come_from.ipynb#r6-8) |
| Arrays | Load arrays, and find the metadata that is not in them | the tree contains `.npy` tensors. Memory-map them to inspect shape and dtype for free, then go looking for wha | [R6.6](where_the_numbers_come_from.ipynb#r6-6) |
| Caching | Cache to parquet, not CSV | you have paid to parse something once and will read it repeatedly. Parquet round-trips dtypes, compresses seve | [R6.12](where_the_numbers_come_from.ipynb#r6-12) |
| Data catalog | Build a catalogue and let it find the join keys | you have dozens of files and need to know what connects to what. Profile their schemas, then look for column n | [R6.11](where_the_numbers_come_from.ipynb#r6-11) |
| Data discovery | See which datasets are present, and what happens when they are not | before running the series on a new machine. The SPY market data is gitignored, so a fresh clone has none of it | [R1.2](the_volatility_surface.ipynb#r1-2) |
| Data discovery | Find out what data this machine actually has | you are opening a project on a new machine, or a loader just silently returned something unexpected. Check ava | [R1.6](the_volatility_surface.ipynb#r1-6) |
| Data discovery | Find the dataset wherever it happens to live | the data is in a different place on every machine — a Kaggle input mount, a Downloads folder, a sibling direct | [R5.1](the_cross_section.ipynb#r5-1) |
| File discovery | Census the tree by file type | you open an unfamiliar repository. Counts and sizes by extension tell you what kind of project it is and where | [R6.1](where_the_numbers_come_from.ipynb#r6-1) |
| File discovery | Find where the data actually lives | the extension census says *what*; this says *where*. Directories holding many files of one type are usually ge | [R6.2](where_the_numbers_come_from.ipynb#r6-2) |
| File I/O | Read a CSV without letting pandas guess | loading any delimited file. Left to itself pandas will infer dtypes from the first rows, read dates as strings | [R1.8](the_volatility_surface.ipynb#r1-8) |
| File I/O | Concatenate train and supplemental files, then de-duplicate | the competition ships a frozen `train_files/` plus a rolling `supplemental_files/` that overlaps it. Concatena | [R5.2](the_cross_section.ipynb#r5-2) |
| File I/O | Load anything, dispatching on extension | you have a mixed tree and do not want thirteen branches at every call site. One dispatcher with the right defa | [R6.3](where_the_numbers_come_from.ipynb#r6-3) |
| File I/O | Read a CSV without letting pandas guess | every delimited file. Left alone, pandas infers dtypes from the first rows, reads dates as strings, and promot | [R6.4](where_the_numbers_come_from.ipynb#r6-4) |
| File I/O | Flatten nested JSON into a frame | configuration, registries and API dumps arrive as nested JSON. `json_normalize` turns a list of nested records | [R6.7](where_the_numbers_come_from.ipynb#r6-7) |
| Large files | Stream a file too large to hold | the file does not fit, or you only need an aggregate. `chunksize` turns `read_csv` into an iterator of frames, | [R6.5](where_the_numbers_come_from.ipynb#r6-5) |
| Many small files | Concatenate a run tree, lifting metadata out of the paths | an experiment directory: many small files whose identity is encoded in their location. `runs/<run_id>/windows/ | [R6.9](where_the_numbers_come_from.ipynb#r6-9) |
| Memory | Shrink a frame without losing anything | a frame is large enough that operations feel slow, or it will not fit in memory. Downcasting floats and conver | [R1.9](the_volatility_surface.ipynb#r1-9) |
| Memory | Reduce memory before it becomes the bottleneck | the panel is large. Competition panels reach tens of millions of rows; downcasting floats and converting low-c | [R5.7](the_cross_section.ipynb#r5-7) |
| Performance | When pandas is the bottleneck | the file is tens of millions of rows and a groupby takes minutes. Polars reads and aggregates lazily, in paral | [R6.13](where_the_numbers_come_from.ipynb#r6-13) |

### Measurement and data quality

| Topic | Recipe | Use when | Go |
|---|---|---|---|
| Categorical data | Check categorical cardinality before encoding | before one-hot encoding anything. Cardinality and the share of mass in rare levels decide whether encoding is  | [R6.19](where_the_numbers_come_from.ipynb#r6-19) |
| Cross-source validation | Reconcile two sources that should agree | you have two measurements of the same quantity. Whether they agree — and by how much — is a fact about your da | [R6.23](where_the_numbers_come_from.ipynb#r6-23) |
| Data contracts | State a schema contract and check it | always, immediately after loading, and again after any pipeline that reshapes data. A contract turns 'the anal | [R1.11](the_volatility_surface.ipynb#r1-11) |
| Data contracts | Write a contract for your own data | your frame is not a smile panel. `SchemaSpec` is declarative: name the columns, their kinds, the ones that mus | [R1.12](the_volatility_surface.ipynb#r1-12) |
| Data contracts | Turn the checks into a contract you can re-run | once you know what good looks like. A `SchemaSpec` states required columns, kinds, non-null rules, plausible r | [R6.24](where_the_numbers_come_from.ipynb#r6-24) |
| Data quality | Check quote sanity before trusting a derived value | any dataset with bid/ask. `bid <= mid <= ask` and a plausible spread are the two cheapest checks in derivative | [R1.18](the_volatility_surface.ipynb#r1-18) |
| Data quality | Detect stale quotes | intraday data where a price can simply stop updating. A repeated value is not necessarily stale — but a long r | [R1.19](the_volatility_surface.ipynb#r1-19) |
| Data quality | Inventory every source at its declared grain | opening unfamiliar files. This table makes row counts, key uniqueness, missingness, memory and time coverage c | [R10.1](volatility_project_01_data_and_joins.ipynb#r10-1) |
| Data quality | Resolve a corrected duplicate and start the issue log | a feed publishes corrected records under the same business key. Preserve the evidence, state the version polic | [R10.3](volatility_project_01_data_and_joins.ipynb#r10-3) |
| Data quality | Inspect missingness, drift, and the development target | you need a ten-minute diagnosis that changes modelling choices rather than a decorative EDA tour. | [R11.2](volatility_project_02_time_series.ipynb#r11-2) |
| Data quality | Seal the final dates, then diagnose development | the joined panel is ready and before target-aware exploration or model choices. | [R12.2](volatility_project_03_panel.ipynb#r12-2) |
| Data quality | Check whether all those files are actually different | immediately after concatenating a tree. Many pipelines write the same artefact into every run directory, and a | [R6.10](where_the_numbers_come_from.ipynb#r6-10) |
| Datetime | Audit the time axis | any time series or panel. Span, timezone, monotonicity, duplicates and — the one that matters — the largest ga | [R6.18](where_the_numbers_come_from.ipynb#r6-18) |
| Drift | Detect drift between train and test | before trusting any validation score. Two measures, because they see different things: the KS test is sensitiv | [R6.20](where_the_numbers_come_from.ipynb#r6-20) |
| Drift | Compare train and test one feature at a time | test inputs are visible. Compare numeric distributions, missingness and unseen categorical levels; these descr | [R9.6](competition_workbench.ipynb#r9-6) |
| Drift | Detect multivariate drift adversarially | individual features look modestly different but their combination may identify train versus test. Cross-valida | [R9.7](competition_workbench.ipynb#r9-7) |
| Health check | Audit future drift, re-check integrity, and publish the issue log | handing the modeling table to cross-validation. Seal a future block, inspect drift, repeat all join invariants | [R10.8](volatility_project_01_data_and_joins.ipynb#r10-8) |
| Health check | Audit the panel before touching a model | immediately after loading, always. Each check here catches a failure that produces a *plausible wrong answer*  | [R5.3](the_cross_section.ipynb#r5-3) |
| Health check | Plot coverage over time | any panel. The entity count per date is the single most informative plot in a panel audit: a step down is a de | [R5.4](the_cross_section.ipynb#r5-4) |
| Health check | Run one audit that catches the common defects | immediately after loading, and again after any pipeline that reshapes data. Dtypes, nulls, duplicates, constan | [R6.14](where_the_numbers_come_from.ipynb#r6-14) |
| Health check | Check numeric ranges, zeros and infinities | before any arithmetic. The three columns people skip: `n_inf` (one infinity turns every downstream mean into N | [R6.15](where_the_numbers_come_from.ipynb#r6-15) |
| Health check | Run integrity gates before EDA | immediately after loading. Duplicated IDs, infinities, all-null columns and silent schema differences should s | [R9.5](competition_workbench.ipynb#r9-5) |
| Missing data | Map the missing values before deciding what to do about them | any frame with gaps. The *pattern* of missingness decides the fix: a column that fails everywhere is a broken  | [R1.15](the_volatility_surface.ipynb#r1-15) |
| Missing data | Choose an imputation strategy on purpose | you have gaps and need numbers. The three sklearn imputers make different assumptions, and the right one depen | [R1.16](the_volatility_surface.ipynb#r1-16) |
| Missing data | Map the missing values before deciding what to do | any frame with gaps. The *pattern* decides the fix: a column failing everywhere is a broken feature; whole row | [R6.17](where_the_numbers_come_from.ipynb#r6-17) |
| Outliers | Winsorize rather than delete | a column has extreme values that are more likely bad data than real events. Clipping to quantiles keeps the ob | [R1.17](the_volatility_surface.ipynb#r1-17) |
| Outliers | Count outliers three ways and read the disagreement | deciding what to clip. The three methods disagree, and the size of the disagreement is itself the diagnostic. | [R6.16](where_the_numbers_come_from.ipynb#r6-16) |
| Target quality | Audit the label itself | a target problem invalidates everything downstream, so check it separately and early: missingness, balance, ta | [R6.22](where_the_numbers_come_from.ipynb#r6-22) |
| Timezones | Get timezones right once, at the boundary | any intraday data. Convert to UTC at load time and keep it UTC internally; convert to a local zone only for di | [R1.10](the_volatility_surface.ipynb#r1-10) |

### Combining sources

| Topic | Recipe | Use when | Go |
|---|---|---|---|
| As-of joins | Join timestamped macro releases backward in time | an event feature is released irregularly and remains in force until the next release. Each panel row receives  | [R10.6](volatility_project_01_data_and_joins.ipynb#r10-6) |
| As-of joins | Match each row to the most recent prior observation | two series on different clocks — a minute-level panel against a daily series, a trade against the prevailing q | [R7.37](seeing_and_joining_evidence.ipynb#r7-37) |
| Cross-source joins | Compare two sources that measure the same thing | you have two pipelines producing the same quantity. Join them and difference; when no composite keys overlap,  | [R7.35](seeing_and_joining_evidence.ipynb#r7-35) |
| Interval joins | Join through a date-ranged link table | the mapping between two identifiers is only valid between two dates. Security reclassifications, customer segm | [R7.38](seeing_and_joining_evidence.ipynb#r7-38) |
| Joins | Join two time series on time, correctly | combining series sampled on different clocks. `merge_asof` matches each left row to the most recent right row  | [R1.21](the_volatility_surface.ipynb#r1-21) |
| Joins | Join panel labels one-to-one | each observation may have at most one label at the same panel key. `validate="one_to_one"` protects both sides | [R10.4](volatility_project_01_data_and_joins.ipynb#r10-4) |
| Joins | Join static attributes many-to-one | many dated observations map to one stable security record. The right table must be unique on `asset`; a duplic | [R10.5](volatility_project_01_data_and_joins.ipynb#r10-5) |
| Joins | Join the tables without changing the question | raw observations, static attributes and dated releases arrive separately. | [R12.1](volatility_project_03_panel.ipynb#r12-1) |
| Joins | Diagnose a join before you run it | always, on any join you have not run before. Key overlap, dtype agreement, duplicate keys on each side, and th | [R7.32](seeing_and_joining_evidence.ipynb#r7-32) |
| Joins | Refuse a join that changes the row count unexpectedly | every enrichment join. `safe_merge` states the contract — same row count, or fewer, or a fan-out you explicitl | [R7.33](seeing_and_joining_evidence.ipynb#r7-33) |
| Joins | Find the dtype trap that returns an empty frame | a join that should match returns nothing, or almost nothing. The cause is nearly always a key whose dtype diff | [R7.34](seeing_and_joining_evidence.ipynb#r7-34) |
| Joins | Establish that a join is impossible, in sixty seconds | two files share a name and look joinable. Before writing any merge, check that a key actually exists, that the | [R7.36](seeing_and_joining_evidence.ipynb#r7-36) |
| Joins | Aggregate first, then join | the right frame has many rows per key and you want a summary, not a cross product. This is the fix for almost  | [R7.39](seeing_and_joining_evidence.ipynb#r7-39) |
| Joins | Reconcile row counts across a chain of merges | any pipeline with more than one join. Row counts drift a few at a time and nobody notices until the totals are | [R7.40](seeing_and_joining_evidence.ipynb#r7-40) |
| Resampling | Resample an irregular panel without inventing observations | you want a coarser frequency. Subsampling keeps real observations; aggregating creates an average smile that n | [R1.20](the_volatility_surface.ipynb#r1-20) |
| Reshaping | Melt a frame whose columns are dates | the single most common untidy shape in delivered data: one column per period. Every time operation is impossib | [R7.41](seeing_and_joining_evidence.ipynb#r7-41) |
| Reshaping | Pivot long back to wide, safely | you need a matrix — one row per entity, one column per period — for a heatmap, a correlation, or a model that  | [R7.42](seeing_and_joining_evidence.ipynb#r7-42) |
| Reshaping | Stack many files into one frame with a label column | several files share a schema and differ only in which variant produced them. Concatenate with a label, and the | [R7.43](seeing_and_joining_evidence.ipynb#r7-43) |
| Reshaping | Handle the MultiIndex a groupby hands you | any `groupby().agg()` with several statistics returns MultiIndex columns, and most downstream tools are happie | [R7.44](seeing_and_joining_evidence.ipynb#r7-44) |

### Model inputs and targets

| Topic | Recipe | Use when | Go |
|---|---|---|---|
| Cross-sectional features | Cross-sectional rank and de-mean — the panel-specific transform | always, on a ranking problem. Rank and de-mean each feature *within its date*. This is the single highest-valu | [R5.11](the_cross_section.ipynb#r5-11) |
| Cross-sectional features | Subtract the group's mean — sector-relative features | you know a grouping that drives common variation. Subtracting the sector's same-date mean is a sharper control | [R5.12](the_cross_section.ipynb#r5-12) |
| Encoding | Target-encode inside the folds, never across them | you have a high-cardinality categorical. Target encoding is the strongest tool for it and the easiest to get c | [R8.25](measuring_and_trading_volatility.ipynb#r8-25) |
| Encoding | Handle rare and unseen categories | before one-hot encoding anything high-cardinality. Levels seen a handful of times produce near-empty dummies,  | [R8.26](measuring_and_trading_volatility.ipynb#r8-26) |
| Encoding | Encode periodic features as a sine/cosine pair | month, weekday, hour — anything circular. An integer encoding tells the model December and January are eleven  | [R8.27](measuring_and_trading_volatility.ipynb#r8-27) |
| Feature engineering | Build lags and rolling state from observable history | the target is future-facing but current and past market variables are observable. | [R11.3](volatility_project_02_time_series.ipynb#r11-3) |
| Feature engineering | Build only features available at the forecast origin | rows repeat by entity and lag or rolling operations could cross entity boundaries. | [R12.3](volatility_project_03_panel.ipynb#r12-3) |
| Feature engineering | Build the rough features the model suggests | you want a model's structural quantities as columns. Dividing the maturity scaling out of skew and curvature g | [R3.1](forecasting_the_smile.ipynb#r3-1) |
| Feature engineering | Lag, difference and roll — grouped, so they mean what you think | building backward-looking features on a panel. Every one of these must be computed *within* a group, or it cro | [R3.3](forecasting_the_smile.ipynb#r3-3) |
| Feature engineering | Rolling statistics and z-scores, within each entity | you want trend and volatility context. Rolling mean/std of returns, and a rolling z-score of volume. | [R5.10](the_cross_section.ipynb#r5-10) |
| Feature engineering | Build calendar and same-bar price geometry | the first feature block. Calendar effects are cheap and occasionally real; same-bar OHLCV shape is fully obser | [R5.8](the_cross_section.ipynb#r5-8) |
| Feature engineering | Multi-horizon returns, masked where corporate actions intrude | you need momentum and reversal features at several lookbacks. Computed per entity, with any window containing  | [R5.9](the_cross_section.ipynb#r5-9) |
| Feature selection | Separate identifiers, features and labels — once | before any fit. Decide explicitly which columns are identifiers, which are labels and which are features, and  | [R5.15](the_cross_section.ipynb#r5-15) |
| Preprocessing | Declare roles and inspect the final design | the feature builder is complete and before any cross-validation. | [R12.4](volatility_project_03_panel.ipynb#r12-4) |
| Preprocessing | Put imputation and scaling inside the pipeline | always. Any step that *learns* something from data — a median, a quantile, a mean and standard deviation — mus | [R3.4](forecasting_the_smile.ipynb#r3-4) |
| Preprocessing | Write a transformer when a step has fitted state | you need a preprocessing step sklearn does not provide. Implementing `BaseEstimator, TransformerMixin` makes i | [R3.5](forecasting_the_smile.ipynb#r3-5) |
| Preprocessing | Combine numeric and categorical branches | your features are not all numeric. `ColumnTransformer` routes each dtype through its own preprocessing and con | [R3.8](forecasting_the_smile.ipynb#r3-8) |
| Target construction | Build the target, and check it is in the future | the last step before modelling. The target is `shift(-horizon)` — the only place a negative shift is correct — | [R3.7](forecasting_the_smile.ipynb#r3-7) |
| Target construction | De-mean the target within each date | the objective is ranking rather than forecasting. The market's daily move shifts every entity's target togethe | [R5.13](the_cross_section.ipynb#r5-13) |
| Vectorisation | Factorise once, draw many — vectorising a correlated simulation | simulating many paths from a Gaussian process. The covariance factorisation is `O(n_steps^3)` and *does not de | [R2.2](what_roughness_claims.ipynb#r2-2) |

### Exploration and evidence

| Topic | Recipe | Use when | Go |
|---|---|---|---|
| Categorical | Small multiples instead of overplotting | more than about seven groups. Faceting keeps every group readable and makes differences in *shape* obvious, wh | [R7.18](seeing_and_joining_evidence.ipynb#r7-18) |
| Categorical | Bar chart with error bars and value labels | comparing a handful of measured quantities. If you have the uncertainty, plot it. | [R7.19](seeing_and_joining_evidence.ipynb#r7-19) |
| Conditional analysis | Condition on a regime — and control for the regime itself | the hypothesis is that a signal works only in particular states. Split the sample by a regime indicator and sc | [R4.12](when_a_result_is_real.ipynb#r4-12) |
| Correlation | Read a correlation matrix with the right correlation | before feature selection or any linear model. Spearman ranks are robust to the heavy tails that dominate volat | [R1.22](the_volatility_surface.ipynb#r1-22) |
| Craft | Check the palette in greyscale | before any figure goes into a document. Print, photocopy and many screens destroy colour distinctions your mon | [R7.28](seeing_and_joining_evidence.ipynb#r7-28) |
| Craft | Simulate colour blindness | any figure distinguishing series by colour. Roughly 8% of men have reduced red-green vision — precisely the ax | [R7.29](seeing_and_joining_evidence.ipynb#r7-29) |
| Craft | Annotate the finding | every figure that will be read by someone other than you. One arrow and six words is usually the difference be | [R7.30](seeing_and_joining_evidence.ipynb#r7-30) |
| Craft | Export at publication quality | the figure is going into a document, a slide or a PDF. | [R7.31](seeing_and_joining_evidence.ipynb#r7-31) |
| Dimensionality | Decompose the smile into level, slope and curvature with PCA | you want the data's own structure, with no target involved. PCA on a term structure almost always recovers lev | [R4.6](when_a_result_is_real.ipynb#r4-6) |
| Distributions | Histogram with a density overlay | the first look at one variable. The KDE smooths what the bin width arbitrarily discretises, so plotting both s | [R7.1](seeing_and_joining_evidence.ipynb#r7-1) |
| Distributions | ECDF to compare several distributions | you have more than two groups. Overlaid histograms occlude one another and change shape with the bin width; EC | [R7.2](seeing_and_joining_evidence.ipynb#r7-2) |
| Distributions | QQ plot to judge the tails | you need to know whether a normal assumption is defensible. A histogram spends its resolution on the middle; a | [R7.3](seeing_and_joining_evidence.ipynb#r7-3) |
| Distributions | Violin with the observations on top | comparing distribution *shape* across a handful of groups. The violin shows the shape; the strip shows how man | [R7.4](seeing_and_joining_evidence.ipynb#r7-4) |
| Distributions | Ridgeline for many distributions at once | five to fifteen groups whose shapes you want to compare. Stacked densities fit far more groups into one panel  | [R7.5](seeing_and_joining_evidence.ipynb#r7-5) |
| Exploratory analysis | Look at distributions before looking at relationships | immediately after loading. `describe()` on the columns you will actually model, with the percentiles that matt | [R1.14](the_volatility_surface.ipynb#r1-14) |
| Exploratory analysis | Plot coverage and the future target before joining | checking whether the panel is balanced, whether the future-return label has difficult tails, and whether its c | [R10.2](volatility_project_01_data_and_joins.ipynb#r10-2) |
| Exploratory analysis | Look at the target distribution before modelling it | before choosing a loss. Financial targets are near-zero-mean with very heavy tails, which is why squared loss  | [R5.5](the_cross_section.ipynb#r5-5) |
| Exploratory analysis | Compare distributions across groups | the panel has a categorical dimension — sector, region, product line. A boxplot per group shows whether the gr | [R5.6](the_cross_section.ipynb#r5-6) |
| Exploratory summary | Describe a panel in one look | you have just loaded something and want the shape of it in one line rather than five `print` statements. Shape | [R1.13](the_volatility_surface.ipynb#r1-13) |
| Relationships | Pearson against Spearman, and their difference | before any linear model. The third panel is the point: where the two disagree, the relationship is monotone bu | [R7.10](seeing_and_joining_evidence.ipynb#r7-10) |
| Relationships | Clustermap to find redundant feature blocks | deciding what to drop. Reordering the correlation matrix by similarity groups features that move together, whi | [R7.11](seeing_and_joining_evidence.ipynb#r7-11) |
| Relationships | Scatter with a fit and a confidence band | showing a bivariate relationship. The band is the part that matters — it distinguishes a real slope from a lin | [R7.6](seeing_and_joining_evidence.ipynb#r7-6) |
| Relationships | Hexbin when the points overlap | above roughly 5,000 points. A saturated scatter shows you the *outline* of the data and nothing about where th | [R7.7](seeing_and_joining_evidence.ipynb#r7-7) |
| Relationships | Joint plot with marginals attached | the marginal distributions matter as much as the relationship. Catches the case where a correlation is produce | [R7.8](seeing_and_joining_evidence.ipynb#r7-8) |
| Relationships | Pair grid for every relationship at once | the fastest way to see structure in a new feature set — every pairwise scatter and every marginal, in one figu | [R7.9](seeing_and_joining_evidence.ipynb#r7-9) |
| Time series | Line with a rolling volatility band | any level that moves. The band makes the line interpretable: a move of 0.02 means something different when the | [R7.12](seeing_and_joining_evidence.ipynb#r7-12) |
| Time series | Calendar heatmap for periodic effects | you suspect a day-of-week or turn-of-month effect. These are periodic at a scale a line chart compresses away  | [R7.13](seeing_and_joining_evidence.ipynb#r7-13) |
| Time series | ACF and PACF for serial dependence | before modelling a time series, and again on the residuals afterwards. ACF shows total dependence at each lag; | [R7.14](seeing_and_joining_evidence.ipynb#r7-14) |
| Time series | Cumulative performance with drawdown underneath | evaluating anything cumulative. A Sharpe ratio says nothing about the path; the underwater panel answers the q | [R7.15](seeing_and_joining_evidence.ipynb#r7-15) |
| Time series | Event study around a set of dates | you want the average behaviour around events — earnings, shocks, regime changes. Align every event at t=0, re- | [R7.16](seeing_and_joining_evidence.ipynb#r7-16) |
| Time series | Several series, rebased to a common start | comparing series with different scales. Rebasing to 1.0 makes them comparable on one axis. | [R7.17](seeing_and_joining_evidence.ipynb#r7-17) |
| Unsupervised | Cluster observations into regimes | you suspect the data contains distinct states and want to label them without supervision. K-means is the blunt | [R4.7](when_a_result_is_real.ipynb#r4-7) |
| Visualisation | Plot the object you are actually modelling | before any modelling. For this series that is the smile: implied vol against maturity, at one instant, across  | [R1.23](the_volatility_surface.ipynb#r1-23) |
| Visualisation | Turn a sweep into a matrix and look at it | you have a tidy sweep and want to see the shape of the result. Pivot to a matrix and plot it diverging around  | [R4.10](when_a_result_is_real.ipynb#r4-10) |

### Volatility models and estimators

| Topic | Recipe | Use when | Go |
|---|---|---|---|
| Conditional volatility | EWMA — one parameter, no fitting, a strong baseline | you need a conditional volatility estimate today and do not want to fit anything. RiskMetrics' exponentially w | [R8.4](measuring_and_trading_volatility.ipynb#r8-4) |
| Conditional volatility | HAR — long memory from three short-memory terms | forecasting realized variance. Corsi's HAR regresses tomorrow's variance on its own averages over a day, a wee | [R8.5](measuring_and_trading_volatility.ipynb#r8-5) |
| Conditional volatility | GARCH(1,1) by maximum likelihood | you want a conditional variance model with an explicit likelihood, mean reversion and a forecastable path. The | [R8.6](measuring_and_trading_volatility.ipynb#r8-6) |
| Conditional volatility | Forecast, and read the half-life | you need a volatility forecast over a horizon. GARCH decays geometrically toward its long-run level at rate `p | [R8.7](measuring_and_trading_volatility.ipynb#r8-7) |
| Estimation | Measure roughness with the structure function | you have a volatility path and want H back out of it. The structure function `E|X(t+lag) - X(t)| ~ lag^H` is t | [R2.4](what_roughness_claims.ipynb#r2-4) |
| Estimator bias | Measure your estimator's bias instead of assuming it away | before believing any estimate. Run the estimator on synthetic data with a *known* answer, then again with obse | [R2.12](what_roughness_claims.ipynb#r2-12) |
| Forecast evaluation | Score the forecasts against a random walk | before believing any conditional model. The benchmark is 'tomorrow's volatility equals today's' — and it is mu | [R8.8](measuring_and_trading_volatility.ipynb#r8-8) |
| Implied volatility | Measure the variance risk premium | testing whether implied variance exceeds subsequent realized variance. The difference is a candidate variance  | [R8.10](measuring_and_trading_volatility.ipynb#r8-10) |
| Implied volatility | Read the implied-volatility term structure | you have an option panel. Average implied volatility by maturity — upward sloping in calm markets, inverted in | [R8.9](measuring_and_trading_volatility.ipynb#r8-9) |
| Model implementation | Turn the model into a smile | you need the observable consequence of a latent model. The Bergomi-Guyon second-order expansion maps rough par | [R2.5](what_roughness_claims.ipynb#r2-5) |
| Model implementation | Read the two scaling laws off the model | you want the quantitative prediction, not the picture. `psi(T)` is the skew coefficient and scales as `T^(H-1/ | [R2.6](what_roughness_claims.ipynb#r2-6) |
| Parameter sweeps | See what the Hurst exponent does | you want to know whether a parameter matters before fitting it. Sweep it, plot the family, and read the effect | [R2.3](what_roughness_claims.ipynb#r2-3) |
| Realized volatility | Compare the five realized-volatility estimators | you need a volatility number from price history and want to know how much the choice matters. It matters by 25 | [R8.1](measuring_and_trading_volatility.ipynb#r8-1) |
| Realized volatility | Measure the efficiency gain properly | you have heard that Parkinson is five times more efficient and want to check it. The only valid experiment is  | [R8.2](measuring_and_trading_volatility.ipynb#r8-2) |
| Realized volatility | See the discretisation bias | you want to know why published range estimates run low. A daily bar is built from finitely many observations a | [R8.3](measuring_and_trading_volatility.ipynb#r8-3) |
| Regimes | Measure volatility clustering, against a control | before fitting any conditional model. Clustering is the property those models exist to exploit — so measure wh | [R8.11](measuring_and_trading_volatility.ipynb#r8-11) |
| Simulation | Simulate the full model, not just its expansion | the closed-form expansion is not enough — you need paths, for a hedging experiment or a payoff that has no for | [R2.7](what_roughness_claims.ipynb#r2-7) |

### Predictive models

| Topic | Recipe | Use when | Go |
|---|---|---|---|
| Baselines | Put the naive forecasts on the exact OOF rows | a model score needs an economically meaningful bar rather than comparison with zero. | [R11.5](volatility_project_02_time_series.ipynb#r11-5) |
| Baselines | Make simple forecasts obey the folds | defining what a learned model must beat. | [R12.6](volatility_project_03_panel.ipynb#r12-6) |
| Baselines | Define the baseline before the model | always, and first. A score is meaningless without the number it has to beat. For a time series that is almost  | [R3.10](forecasting_the_smile.ipynb#r3-10) |
| Baselines | Fit the trivial baseline inside each fold | before fitting a learned model. Regression predicts the fold-training mean, binary classification the fold-tra | [R9.11](competition_workbench.ipynb#r9-11) |
| Calibration | Check calibration, not just ranking | you will act on the predicted probability rather than the ordering — sizing a position, computing an expected  | [R8.24](measuring_and_trading_volatility.ipynb#r8-24) |
| Class imbalance | Establish what a trivial classifier would score | before reporting any classification metric. On imbalanced data the majority-class baseline is high, and any ac | [R8.22](measuring_and_trading_volatility.ipynb#r8-22) |
| Classification | Do the same for a classification target | the question is directional rather than numerical: will curvature rise or fall? Same pipeline shape, different | [R3.16](forecasting_the_smile.ipynb#r3-16) |
| Ensembling | Put predictions on a common scale before averaging them | blending any two models. Ridge and a gradient booster can produce predictions with different spreads, so a pla | [R5.25](the_cross_section.ipynb#r5-25) |
| Ensembling | Choose blend weights out of sample | you want more than an equal-weight average. Weights fitted on the same data you evaluate on will always look b | [R5.26](the_cross_section.ipynb#r5-26) |
| Ensembling | Generate out-of-fold predictions for stacking | you want a meta-model, or honest blend weights. Every row gets a prediction from a model that never saw it, so | [R5.27](the_cross_section.ipynb#r5-27) |
| Hyperparameters | Make a model parameter tunable instead of assumed | a 'constant' in your model is actually a choice. Wrapping it in a transformer turns it into a hyperparameter t | [R3.6](forecasting_the_smile.ipynb#r3-6) |
| Model selection | Compare every candidate on one OOF ledger | choosing the final forecasting rule. | [R12.8](volatility_project_03_panel.ipynb#r12-8) |
| Model selection | Tune hyperparameters inside cross-validation | you have a parameter to choose. `GridSearchCV` refits on each fold and scores out of sample, so the chosen val | [R3.13](forecasting_the_smile.ipynb#r3-13) |
| Model selection | Search a large space without a full grid | the grid is too big to enumerate. `RandomizedSearchCV` samples it; `HalvingGridSearchCV` runs a tournament, gi | [R3.14](forecasting_the_smile.ipynb#r3-14) |
| Model selection | Trace a regularisation path | you want to know which features survive as a penalty tightens. The order in which coefficients are driven to z | [R4.5](when_a_result_is_real.ipynb#r4-5) |
| Model selection | Tune with GridSearchCV on the purged folds | you have hyperparameters to choose. Pass the fold list from R5.16 straight to `cv=`, so tuning happens inside  | [R5.21](the_cross_section.ipynb#r5-21) |
| Model selection | Refit on everything once selection is finished | the very last fitting step. Model choice, hyperparameters and blend weights are all decided; now use every lab | [R5.31](the_cross_section.ipynb#r5-31) |
| Model selection | Nest the cross-validation when you tune | you tune hyperparameters and then report the cross-validated score. A single CV loop that does both reports an | [R8.28](measuring_and_trading_volatility.ipynb#r8-28) |
| Model selection | Search a large space with Optuna | the grid has more than two or three dimensions. A grid's cost is multiplicative; a sampler's is not, and Optun | [R8.29](measuring_and_trading_volatility.ipynb#r8-29) |
| Pipelines | Keep fitted preprocessing inside Ridge and booster pipelines | missing-value statistics and scaling must be learned separately in every fold. | [R11.6](volatility_project_02_time_series.ipynb#r11-6) |
| Pipelines | Build LightGBM or the clearly labeled sklearn fallback | numeric and categorical columns contain missing or unseen values. | [R12.7](volatility_project_03_panel.ipynb#r12-7) |
| Pipelines | Assemble a pipeline that cannot leak | every supervised model. A `Pipeline` is not tidiness — it is the mechanism that guarantees preprocessing is fi | [R3.9](forecasting_the_smile.ipynb#r3-9) |
| Pipelines | Build a mixed-dtype pipeline for a panel | your features are numeric and categorical. A `ColumnTransformer` routes each branch through its own preprocess | [R5.20](the_cross_section.ipynb#r5-20) |
| Pipelines | Keep every fitted transform inside a pipeline | features mix numbers, categories and missing values. Each cloned fold learns imputation, encoding and scaling  | [R9.10](competition_workbench.ipynb#r9-10) |
| Regression | Run the same regression at every instant | one fit is an anecdote. Running the cross-sectional regression at every timestamp gives a *distribution* of sl | [R2.10](what_roughness_claims.ipynb#r2-10) |
| Regression | Use statsmodels when you need the inference, not just the fit | numpy's `polyfit` gives you coefficients. When you need standard errors, t-statistics and residual diagnostics | [R2.14](what_roughness_claims.ipynb#r2-14) |
| Regression | Estimate H from a volatility time series | you have a volatility path (realized or implied) and want its roughness. Fit `log E|X(t+lag) - X(t)|` against  | [R2.8](what_roughness_claims.ipynb#r2-8) |
| Regression | Fit a power law across a cross-section | you have several maturities at one instant and a model predicting a power law in T. Regress in logs: `log|y| = | [R2.9](what_roughness_claims.ipynb#r2-9) |
| Thresholds | Choose the threshold from what errors cost | you must turn probabilities into decisions. 0.5 is a default, not a choice — the right cut depends on the rela | [R8.23](measuring_and_trading_volatility.ipynb#r8-23) |
| Uncertainty | Predict an interval, not a point | a point forecast is not enough — you need to size a position, set a reserve, or say how confident the model is | [R8.30](measuring_and_trading_volatility.ipynb#r8-30) |
| Uncertainty | Know which uncertainty you measured | reporting any interval. There are two, they answer different questions, and conflating them is the most common | [R8.31](measuring_and_trading_volatility.ipynb#r8-31) |

### Validation and inference

| Topic | Recipe | Use when | Go |
|---|---|---|---|
| Cross-validation | Make purged expanding folds inside the development period | every training observation must precede its validation observation. | [R11.4](volatility_project_02_time_series.ipynb#r11-4) |
| Cross-validation | Split unique dates forward and purge the label horizon | many entities share each timestamp and the label looks forward. | [R12.5](volatility_project_03_panel.ipynb#r12-5) |
| Cross-validation | Choose a splitter that matches the data's structure | any temporal or panel data. Default `KFold` does not shuffle regression rows, but its training folds still inc | [R3.11](forecasting_the_smile.ipynb#r3-11) |
| Cross-validation | Purge and embargo, and see what it costs | labels look forward or observations represent intervals. Purge every training row whose label-information inte | [R3.12](forecasting_the_smile.ipynb#r3-12) |
| Cross-validation | Split dates, not rows — then map back to row indices | every panel problem. Split the **unique dates**, apply a gap for the label horizon, then convert each fold's d | [R5.16](the_cross_section.ipynb#r5-16) |
| Cross-validation | Audit the sealed development set, then build folds | the swap-data cell has already sealed a geometry-safe final holdout before showing any target-bearing output.  | [R9.8](competition_workbench.ipynb#r9-8) |
| Experiment design | Write the cell function, then the grid | any parameter study. Write one function that evaluates a *single* cell and returns a dict of scalars. The grid | [R4.8](when_a_result_is_real.ipynb#r4-8) |
| Experiment design | Make the problem contract immutable | starting any supervised analysis. The contract names the unit of independence, target, IDs, prediction geometr | [R9.1](competition_workbench.ipynb#r9-1) |
| Inference | Open the tail once, then refit and shape the submission | model selection is finished and one untouched chronological estimate is required before delivery. | [R11.10](volatility_project_02_time_series.ipynb#r11-10) |
| Inference | Put a confidence interval on a dependent statistic | you want error bars on a statistic computed from autocorrelated data. Resampling individual observations destr | [R2.13](what_roughness_claims.ipynb#r2-13) |
| Inference | Put an interval on the skill score | a point estimate of skill is not enough. The block bootstrap gives a confidence interval that respects the ser | [R3.19](forecasting_the_smile.ipynb#r3-19) |
| Inference | Check the inference schema before predicting | every time test data arrives. Columns go missing, arrive extra, or change order, and a fitted `ColumnTransform | [R5.32](the_cross_section.ipynb#r5-32) |
| Inference | Carry a rolling history buffer for streaming inference | the competition hands you one day at a time and wants a prediction before showing the next. Rolling features n | [R5.33](the_cross_section.ipynb#r5-33) |
| Inference | Turn predictions into a valid submission | the final step. Predictions become unique integer ranks within each date, in the row order the grader handed y | [R5.34](the_cross_section.ipynb#r5-34) |
| Inference | Run the streaming loop end to end before you trust it | before submitting. Replay the last few days one at a time through the exact code path the live API will use, a | [R5.35](the_cross_section.ipynb#r5-35) |
| Inference | Lock, open the final holdout once, then refit | model selection is finished. Score the reserved geometry-safe holdout once, then refit the unchanged pipeline  | [R9.17](competition_workbench.ipynb#r9-17) |
| Inference | Validate against the sample submission | producing the grader-facing file. The sample owns column/ID order; the declared task-aware transform converts  | [R9.18](competition_workbench.ipynb#r9-18) |
| Leakage | Separate features, metadata and future labels by availability | converting a joined data frame into a modeling contract. Every column gets a role, an availability rule, and a | [R10.7](volatility_project_01_data_and_joins.ipynb#r10-7) |
| Leakage | Know the four ways a feature leaks | before building anything. Leakage is not one mistake, it is four, and they need different defences. This is th | [R3.2](forecasting_the_smile.ipynb#r3-2) |
| Leakage | Audit for leakage by name, and assert on it | before every fit. A name-based scan for label-shaped columns costs nothing, runs in CI, and catches the most d | [R5.14](the_cross_section.ipynb#r5-14) |
| Leakage | Scan for leakage three ways | before every fit. A name check, a correlation check and a monotone-relationship check — each catches leaks the | [R6.21](where_the_numbers_come_from.ipynb#r6-21) |
| Leakage | Demand that a permuted target fails | the pipeline first works and after major feature changes. With labels permuted, performance must collapse towa | [R9.14](competition_workbench.ipynb#r9-14) |
| Leakage | Separate roles from features and state availability | turning raw columns into a design matrix. IDs, time, groups, metadata and the target are roles; only columns a | [R9.3](competition_workbench.ipynb#r9-3) |
| Multiplicity | Correct for how many things you tried | you tested more than one hypothesis — which is always. Testing 30 cells at 5% produces about 1.5 false passes  | [R3.21](forecasting_the_smile.ipynb#r3-21) |
| Multiplicity | Report a pass rate honestly | the sweep is done and some cells look good. 'k of n cells passed' is a descriptive statistic, not evidence — t | [R4.11](when_a_result_is_real.ipynb#r4-11) |
| Robustness | Check sensitivity to the choices you did not think about | your result depends on decisions you made without much thought — a tenor filter, a winsorizing quantile, a fol | [R4.13](when_a_result_is_real.ipynb#r4-13) |
| Significance | Test whether one forecast really beats another | two models differ in RMSE and you need to know if that is signal. Diebold-Mariano tests equal predictive accur | [R3.18](forecasting_the_smile.ipynb#r3-18) |
| Validation | Walk the strategy forward | the evaluation closest to how a strategy is actually run: refit on a rolling window, trade the block that foll | [R8.20](measuring_and_trading_volatility.ipynb#r8-20) |
| Validation | The four ways this goes wrong | before reporting any result. Each of these produces a good-looking backtest and none of them raises. | [R8.21](measuring_and_trading_volatility.ipynb#r8-21) |
| Validation design | Validate the clock and seal the last 20% | observations have a natural order and the final check must resemble deployment. | [R11.1](volatility_project_02_time_series.ipynb#r11-1) |
| Validation design | Set aside a block of dates, purged at its boundary | you will compare several models on one common block. Take the last dates and drop the label-horizon dates imme | [R5.17](the_cross_section.ipynb#r5-17) |
| Validation design | Route by data geometry, not model preference | deciding how rows may cross a validation boundary. The question is which observations share information, not w | [R9.2](competition_workbench.ipynb#r9-2) |
| Validation design | Smoke-test every supported geometry/task condition | changing the router or adapting the template. The matrix exercises boundary gates, final-holdout routing, fold | [R9.20](competition_workbench.ipynb#r9-20) |
| Validation design | Prove the folds do not leak | before passing `cv=` anywhere. Print sizes, target balance, shared groups/times, chronological spans and the r | [R9.9](competition_workbench.ipynb#r9-9) |

### Diagnostics and interpretation

| Topic | Recipe | Use when | Go |
|---|---|---|---|
| Classification metrics | Score a classifier without letting accuracy fool you | the target is a class. Accuracy is misleading whenever classes are unbalanced; ROC-AUC and average precision a | [R3.20](forecasting_the_smile.ipynb#r3-20) |
| Diagnostics | Diagnose the out-of-fold errors as a panel | the OOF winner is locked but before opening the final holdout. | [R12.9](volatility_project_03_panel.ipynb#r12-9) |
| Diagnostics | Ask whether a well-fitting parameter is actually stable | a model fits well at every point in time. High R² says the *shape* is right; it says nothing about whether the | [R2.11](what_roughness_claims.ipynb#r2-11) |
| Diagnostics | Look at when a model earned its score | aggregate metrics hide timing. The cumulative squared-error curve shows whether a model won steadily or won on | [R3.22](forecasting_the_smile.ipynb#r3-22) |
| Diagnostics | Look at the IC path, not just its mean | a mean IC hides everything about reliability. The daily series with a rolling mean shows whether the signal is | [R5.23](the_cross_section.ipynb#r5-23) |
| Diagnostics | Plot what the portfolio would actually have earned | the end of every evaluation. Rank IC measures accuracy; the top-k spread return measures the thing you are sco | [R5.24](the_cross_section.ipynb#r5-24) |
| Diagnostics | Read the diagnostic for this condition | CV evidence is known. Compose task diagnostics with geometry diagnostics: a grouped binary problem needs both  | [R9.15](competition_workbench.ipynb#r9-15) |
| Interpretability | Rank features by permutation importance | you want to know which inputs a fitted model relies on. Shuffle one column at a time in the *test* set and mea | [R4.1](when_a_result_is_real.ipynb#r4-1) |
| Interpretability | Read the coefficients of a linear model correctly | the model is linear. Coefficients are directly interpretable — but only after scaling, because a coefficient's | [R4.2](when_a_result_is_real.ipynb#r4-2) |
| Interpretability | Group correlated features before attributing credit | your features are collinear, so individual importances understate everything. Permuting a whole *block* togeth | [R4.3](when_a_result_is_real.ipynb#r4-3) |
| Interpretability | Plot partial dependence to see the shape of an effect | importance says *how much*; partial dependence says *in which direction and what shape*. ICE curves add the pe | [R4.4](when_a_result_is_real.ipynb#r4-4) |
| Interpretability | Read importance out of a fitted pipeline | after fitting. A `ColumnTransformer` turns your columns into a few thousand anonymous ones; `get_feature_names | [R5.28](the_cross_section.ipynb#r5-28) |
| Interpretability | Aggregate one-hot importances back to their source column | you have a high-cardinality categorical. A sector with ten levels contributes ten rows of small importance tha | [R5.29](the_cross_section.ipynb#r5-29) |
| Interpretability | Use permutation importance when the others disagree | gain importance and coefficients tell different stories. Permutation importance on the reused *block-validatio | [R5.30](the_cross_section.ipynb#r5-30) |
| Interpretability | Explain one held-out validation fold safely | asking which raw columns the fitted pipeline relies on. Permute columns in one held-out CV fold and measure de | [R9.16](competition_workbench.ipynb#r9-16) |
| Metrics | Score against the baseline, not in absolute terms | reporting any forecast result. `skill = 1 - RMSE(model)/RMSE(baseline)`: positive beats the baseline, zero tie | [R3.17](forecasting_the_smile.ipynb#r3-17) |
| Metrics | Score the exact objective and its worst fold | comparing candidates. The checked metric and direction in `ProblemSpec` are applied fold by fold; mean, disper | [R9.13](competition_workbench.ipynb#r9-13) |
| Model comparison | Fit every learned model on the same expanding folds | model comparisons must differ only by estimator, not by validation rows. | [R11.7](volatility_project_02_time_series.ipynb#r11-7) |
| Model comparison | Run the contest — baselines against learned models | the actual question. Every model gets the same folds, the same rows and the same scoring, and the baselines ru | [R3.15](forecasting_the_smile.ipynb#r3-15) |
| Model comparison | Fit every model on the same folds and compare on the objective | the actual contest. Same features, same purged training block, same block validation, scored on the ranking me | [R5.22](the_cross_section.ipynb#r5-22) |
| Model comparison | Make one OOF ledger the source of truth | fitting any candidate. The ledger records source row, fold, IDs, time/group roles, truth and prediction so eve | [R9.12](competition_workbench.ipynb#r9-12) |
| Model diagnostics | Ask where the OOF errors came from | the headline metric is known and you need to find bias, tails, regimes, or a failing period. | [R11.9](volatility_project_02_time_series.ipynb#r11-9) |
| Model diagnostics | Divide out the maturity scaling and see what is left | the sharpest form of the test. If H is right, dividing the scaling out leaves a coefficient with no systematic | [R2.15](what_roughness_claims.ipynb#r2-15) |
| Model diagnostics | Compare the two deltas before backtesting either | you are about to compare hedging policies. Look at what actually differs between them first — if the deltas ar | [R4.15](when_a_result_is_real.ipynb#r4-15) |
| Model diagnostics | The residual four-panel | after every regression fit. Four panels, four assumptions: linearity, normality, homoskedasticity, and indepen | [R7.21](seeing_and_joining_evidence.ipynb#r7-21) |
| Model diagnostics | Learning curve — more data, or a bigger model? | the model underperforms and you must decide what to change. The shape of the two curves gives the diagnosis di | [R7.22](seeing_and_joining_evidence.ipynb#r7-22) |
| Model diagnostics | Validation curve over one hyperparameter | you are about to tune something and want to know whether it matters at all. | [R7.23](seeing_and_joining_evidence.ipynb#r7-23) |
| Model diagnostics | Calibration curve for a classifier | you will act on the predicted *probability* rather than the ordering — position sizing, expected value, a thre | [R7.24](seeing_and_joining_evidence.ipynb#r7-24) |
| Model diagnostics | ROC and precision-recall together | any classification result. Always both, never ROC alone. | [R7.25](seeing_and_joining_evidence.ipynb#r7-25) |
| Model diagnostics | Confusion matrix, normalised by true class | reporting a classification result to someone who will act on it. Row-normalising turns each row into per-class | [R7.26](seeing_and_joining_evidence.ipynb#r7-26) |
| Model diagnostics | Partial dependence for several features | you have importance rankings and want the *shape* of each effect, not just its size. | [R7.27](seeing_and_joining_evidence.ipynb#r7-27) |
| Model evaluation | Score RMSE, MAE, skill, and the worst fold | a single average score could conceal instability or a naive forecast is already strong. | [R11.8](volatility_project_02_time_series.ipynb#r11-8) |
| Model evaluation | Decile lift — does the ordering actually work? | the most direct answer to 'is this model useful'. Bin by predicted score and plot the mean outcome per bin. | [R7.20](seeing_and_joining_evidence.ipynb#r7-20) |
| Ranking metrics | Score the objective, not a convenient proxy | the competition ranks and then trades the extremes. Report daily rank IC, its information ratio, and the top-k | [R5.18](the_cross_section.ipynb#r5-18) |
| Ranking metrics | Watch RMSE disagree with the ranking metrics | you are tempted to tune on RMSE because sklearn scorers are built for it. This shows what that costs: two sign | [R5.19](the_cross_section.ipynb#r5-19) |

### From model to position

| Topic | Recipe | Use when | Go |
|---|---|---|---|
| Attribution | Attribute P&L to its sources | you have a P&L and need to know *where* it came from. A total is not an explanation, and a strategy earning it | [R8.19](measuring_and_trading_volatility.ipynb#r8-19) |
| Backtesting | Backtest policies on identical paths | comparing any two strategies. Every policy must see the same paths, rebalance at the same times, and pay the s | [R4.16](when_a_result_is_real.ipynb#r4-16) |
| Backtesting | Sweep transaction costs instead of assuming one | any backtest with trading in it. A policy that rebalances harder can win at zero cost and lose at five basis p | [R4.17](when_a_result_is_real.ipynb#r4-17) |
| Backtesting | Account for turnover, not just error | a hedge that is slightly more accurate but trades twice as much is not obviously better. Turnover is the quant | [R4.19](when_a_result_is_real.ipynb#r4-19) |
| Backtesting | Walk forward instead of splitting once | you want the evaluation to look like production. Retrain on a fixed window of distinct timestamps, act on the  | [R4.20](when_a_result_is_real.ipynb#r4-20) |
| Costs | Cost is charged on the change, not the level | always. Holding a position is free; changing it is not. This single fact decides which signals are tradeable. | [R8.15](measuring_and_trading_volatility.ipynb#r8-15) |
| Costs | Find the break-even cost | before claiming a strategy works. The number that matters is not the Sharpe at your assumed cost — it is the c | [R8.16](measuring_and_trading_volatility.ipynb#r8-16) |
| Evaluation | Report three ratios, not one | summarising any P&L. Sharpe, Sortino and Calmar penalise different things, and the gaps between them are infor | [R8.17](measuring_and_trading_volatility.ipynb#r8-17) |
| Evaluation | Read the tearsheet | the standard four-panel summary: cumulative net P&L, drawdown, gross against net, and turnover. | [R8.18](measuring_and_trading_volatility.ipynb#r8-18) |
| Position construction | Turn a score into a position | the first decision after the model. Three methods, and the choice changes the risk profile more than the model | [R8.12](measuring_and_trading_volatility.ipynb#r8-12) |
| Position construction | Standardise within the date, not across the panel | any cross-sectional book. The scores must be compared against *the other names on the same day*, not against t | [R8.13](measuring_and_trading_volatility.ipynb#r8-13) |
| Risk metrics | Report the tail, not just the average | evaluating a hedge. A hedge is bought for its behaviour in bad states, so RMSE alone is the wrong summary. CVa | [R4.18](when_a_result_is_real.ipynb#r4-18) |
| Risk scaling | Target a volatility instead of a position size | the strategy's realised volatility drifts with the market's. Scaling by a trailing estimate of the strategy's  | [R8.14](measuring_and_trading_volatility.ipynb#r8-14) |

### Research practice

| Topic | Recipe | Use when | Go |
|---|---|---|---|
| Environment | Check the environment before blaming the analysis | the first cell of any session. `setup_notebook()` returns a report of what this machine has: library versions, | [R1.1](the_volatility_surface.ipynb#r1-1) |
| Experiment infrastructure | Run the grid in parallel, with results cached to disk | the grid is large or slow. `run_grid` builds the Cartesian product, farms it out with joblib, and memoises eac | [R4.9](when_a_result_is_real.ipynb#r4-9) |
| Navigation | Find the recipe you need | you have a task ('impute missing values', 'purged CV split', 'block bootstrap CI') and want the cell that does | [R1.4](the_volatility_surface.ipynb#r1-4) |
| Navigation | Understand the shape every notebook shares | you are about to open any other notebook and want to know where things are. All nine share the same skeleton,  | [R1.5](the_volatility_surface.ipynb#r1-5) |
| Provenance | Make every frame carry its own provenance | a number in a notebook needs to be traceable to a file. Attaching provenance to the frame itself means you can | [R1.7](the_volatility_surface.ipynb#r1-7) |
| Provenance | Fingerprint the exact inputs | a result may need to be reproduced after files, row order or dtypes change. The hash covers schema, values, in | [R9.4](competition_workbench.ipynb#r9-4) |
| Reporting | Open once, refit once, align by sample ID | the OOF decision and diagnostics are complete. | [R12.10](volatility_project_03_panel.ipynb#r12-10) |
| Reporting | Export the result so someone else can use it | the analysis is finished. Write tidy tables, the figures, and a manifest recording what produced them — togeth | [R4.21](when_a_result_is_real.ipynb#r4-21) |
| Reporting | Write down the result, including when it is negative | the end of any study. State the claim, the evidence, the caveat and what would change your mind — in that orde | [R4.22](when_a_result_is_real.ipynb#r4-22) |
| Reproducibility | Force the synthetic path to test reproducibility | you want to know whether a notebook depends on data a colleague will not have. Setting `RVLAB_FORCE_SYNTHETIC= | [R1.3](the_volatility_surface.ipynb#r1-3) |
| Reproducibility | Make a simulation reproducible | any Monte Carlo. A result you cannot reproduce is not a result. Pass an explicit `Generator` rather than relyi | [R2.1](what_roughness_claims.ipynb#r2-1) |
| Reproducibility | Record what a sweep actually ran | the sweep is finished and the result is going into a document. Store the grid, the data provenance, the librar | [R4.14](when_a_result_is_real.ipynb#r4-14) |
| Reproducibility | Write a manifest so the run can be reproduced | the moment a result goes into a document or a submission. Record the data provenance, the configuration, the l | [R5.36](the_cross_section.ipynb#r5-36) |
| Reproducibility | Save a reloadable evidence bundle | a model or submission leaves the notebook. Save folds, OOF predictions, metrics, feature order, selected confi | [R9.19](competition_workbench.ipynb#r9-19) |

<!-- END RECIPE INDEX -->

---

## Notebook conventions

Every notebook has the same skeleton, so the structure is worth learning once:

1. **The question** — why this notebook exists, then one heading per section it covers. No code.
2. **Setup** — a small import-first bootstrap plus the environment report; tagged `setup`.
3. **Parameters** — every knob for every section in one cell; tagged `parameters`, so
   [papermill](https://papermill.readthedocs.io) can override them.
4. **Where the data comes from** — the first section's primary data-context cell, tagged
   `swap-data`. Each later section has one of its own at its head, tagged `section-data`.
   Edit one to re-aim the section. File-I/O and reload-validation recipes may deliberately
   read additional artifacts because reading them is the technique being demonstrated.
5. **Recipes in this notebook** — the contents, grouped by section.
6. **Things to watch** — the mistakes this material actually invites, itemised as *Python and
   pandas* and *Analysis*. Read it before the recipes; it is what the recipes assume you know.
7. **Sections** — each a `## heading`, its data cell, then its recipes as
   `## R<n> · technique`, each with **Topic** / **Use when** / **Inputs** / **Needs** /
   **Gotchas** / **See also**.
8. **Verdict** — what was established, section by section, and what carries into the next
   notebook.

Reading order is not in the filenames — they carry no numbers. It lives in
[`.order`](.order), one filename per line, which is what `scripts/run_notebooks.sh` reads and
the only place the sequence is written down.

A recipe that genuinely builds on an earlier one declares it in a **Needs** line, and the
copy-paste test runs those prerequisites first — so a recipe can never depend on a variable
some *other* recipe happened to leave lying around. Each recipe is tested with its **own
section's** data cell in scope, not the last one in the file, which is what makes a recipe
portable out of a notebook that covers three topics.

Enforced by `scripts/lint_notebooks.py`: exactly one setup, parameters and swap-data cell;
stored execution for every code cell with no hidden exception; complete, anchored recipe
cards; concise cells; markdown between code blocks; semantic agreement between recipe labels
and link anchors; globally unique recipe IDs; and an exact, duplicate-free `.order`. README,
cross-notebook and same-document links all resolve, so a rename cannot rot the series silently.

`scripts/build_notebook_index.py` regenerates the index above from the notebooks themselves,
so it cannot drift. Each recipe's **Topic** is mapped to one of eleven families, which is how
the index is grouped; a topic with no family is a build error rather than a silent
`Other` bucket. `--check` fails if the index is stale.
