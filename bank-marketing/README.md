# Customer targeting under temporal distribution shift

A reproducible machine-learning study of bank marketing response: **nine model
configurations, three temporal backtests, calibration and threshold experiments,
and a local interactive demo** on 41,188 UCI records.

The main finding is methodological: a simpler logistic regression outperformed
XGBoost under the declared development selection rule, but performance varied
substantially across windows. The evidence supports further investigation,
**not deployment**. The original test outcomes were inspected before this study;
all new evaluations are explicitly exploratory.

![Development benchmark](results/figures/01_model_benchmark.png)

## Measured results

| Measure | Selected logistic regression |
|---|---:|
| Existing final-window ROC AUC | 0.643 |
| Average precision / evaluation prevalence | 0.510 / 0.385 |
| Lift at a 10% contact budget | 1.37× |
| Precision / recall at policy-selected threshold | 46.2% / 74.9% |
| Raw / selected-calibration log loss | 0.685 / 0.790 |

Calibration helped on the policy-validation partition but worsened final-window
probability accuracy. The decision rule remains frozen rather than being changed
after seeing evaluation outcomes. Original XGBoost results (ROC AUC 0.563,
1.18× lift) are preserved in `outputs/`; their training protocol differs from
this benchmark, so that comparison is not a controlled ablation.

Read [findings and experiment table](results/FINDINGS.md), the
[model card](docs/MODEL_CARD.md), and the [feature audit](docs/FEATURE_AUDIT.md).

## Reproduce

Use Python 3.13 for the verified dependency lock. On macOS, install OpenMP with
`brew install libomp` before importing XGBoost.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python download_data.py
python run_experiments.py
python -m unittest discover -s tests -v
python -m streamlit run app.py --server.address 127.0.0.1
```

The download uses Python's standard library and retrieves the original UCI
archive, selected full CSV, source dictionary and checksum metadata. The experiment
run regenerates result tables, figures, narrative findings, resume notes and a
saved inference bundle. Runtime depends on the machine. Unit tests run without
data; artifact and demo integration checks activate after running experiments.
Alternatively, use `make setup`, `make reproduce`, `make test`, and `make demo`.

Supported dependency ranges are in `requirements.txt`; exact verified versions
are in `requirements-lock.txt`.

## Evaluation protocol

- Preserve source order; evaluate rows 55–65%, 65–75%, and 75–85% of the dataset.
- Before each evaluation, use earlier rows for model fitting and split the
  intervening validation region into early-stopping, calibration, and policy sets.
- Fit preprocessing only on training rows. Exclude outcome and call duration.
- Compare a prevalence baseline, logistic regression, random forest, and six
  XGBoost variants: base, regularized, engineered, no economics, recent history,
  and recent history without economics. See [exact settings](docs/FEATURE_AUDIT.md).
- Select by mean AP divided by evaluation prevalence, breaking ties by log loss.
  This normalization does not remove differences in task difficulty.
- Fit sigmoid calibration on calibration rows; policy rows choose raw versus
  calibrated scores and select an F1 threshold. Evaluation rows never select them.
- Freeze the candidate before evaluating the existing last 15% of rows. This is
  an already-inspected evaluation population, not an untouched holdout.

The artifact bundle includes fitted preprocessing, classifier, selected calibration,
threshold, schema and exact partition boundaries. `research.infer` is shared by
the demo and integration tests. Score ties in contact-budget metrics use expected
random tie-breaking, avoiding arbitrary outcome-dependent ordering.

## Drift and business analysis

The original response rate rises from 5.6% in training to 38.5% in testing.
Descriptive occupation decomposition attributes about **1.9 percentage points**
to occupation mix and **31.0 points** to changing within-occupation response
rates. This accounting does not establish why rates changed. Separate category
decompositions must not be summed.

The study also includes numeric PSI, categorical total variation, unknown and
duplicate counts, ten consecutive row-block summaries, and segment diagnostics.
A small demo lets a reader change contact budget, assumed contact cost and
assumed subscription value, and score five sample dataset customers.
Scenario dollars are hypothetical response values, not actual profit or causal
campaign uplift. See `results/contact_budget_scenarios.csv` for exact assumptions.

## Project map

| Path | Contents |
|---|---|
| `config.json` | Candidate list, folds, seed and scenario assumptions |
| `research.py` | Feature controls, preprocessing, fitting, calibration, policies and inference |
| `run_experiments.py` | Benchmark, drift analysis, result exports and narrative generation |
| `results/` | Curated aggregate findings, metrics, manifest and PNG/SVG research figures |
| `docs/` | Model card, feature audit and suggested resume/interview wording |
| `app.py` | Local Streamlit portfolio demo |
| `tests/` | Leakage, temporal separation, ties, thresholds, artifact and demo checks |
| `outputs/research/` | Generated inference bundle and row-level predictions (Git-ignored) |
| `train.py`, `generate_visuals.py` | Original starter model and its six diagnostic charts |
| `data/raw/` | Original source files (Git-ignored) |

GitHub Actions workflows live in the repository's `.github/workflows/` directory:
automatic unit checks and a manually triggered full reproduction job. They have
been configured locally; no hosted CI run or GitHub publication is claimed.

## Limitations and next evidence

Source order is only a proxy for time: there are no exact dates or customer IDs,
so calendar groups may be split and repeat-customer overlap cannot be ruled out.
Economic indicator publication lags and revisions are unresolved. The data comes
from one institution in 2008–2010. No model consistently outperforms random ranking
across all development windows. Demographic segment tables do not establish fairness.
Untouched later data with stable customer IDs and as-of timestamps is required
for a stronger generalization claim; randomized campaign evidence is required
before claiming incremental business impact.

The six original SHAP and diagnostic charts explain the original XGBoost model,
not the selected logistic regression. They can be regenerated with
`python train.py` followed by `python generate_visuals.py`.

## Source and attribution

[UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing),
`bank-additional-full.csv`: 41,188 rows, 20 inputs and one target; semicolon-delimited.
License: **CC BY 4.0**. Dataset citation: Moro, S., Rita, P., & Cortez, P. (2014),
[DOI 10.24432/C5K306](https://doi.org/10.24432/C5K306).
Research paper: Moro, S., Cortez, P., & Rita, P. (2014),
*A Data-Driven Approach to Predict the Success of Bank Telemarketing*,
[DOI 10.1016/j.dss.2014.03.001](https://doi.org/10.1016/j.dss.2014.03.001).
