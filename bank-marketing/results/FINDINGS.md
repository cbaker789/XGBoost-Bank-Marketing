# Customer targeting under temporal distribution shift

## Decision and evidence

Development selection chose **logistic** using mean average precision
normalized by each evaluation window's prevalence. It was selected before the
final-window evaluation. All results remain exploratory: the full dataset and
original test outcomes were already inspected before this study. No new external
or prospective holdout is available. Model improvement is not guaranteed.

The selected candidate does not beat random targeting consistently across
development windows. The operational recommendation is **do not deploy** on
these results alone. This project demonstrates evaluation and diagnosis rather
than a proven marketing system.

## Temporal benchmark

Three successive source-order evaluation windows compare nine fixed candidates.
Each window has disjoint training, early-stopping, calibration, threshold-policy,
and evaluation rows. The validation region is divided into three equal parts.
Full and recent-history fits differ only as declared in the candidate settings.
Preprocessing is fitted on each training partition. Calibration uses sigmoid
scaling fitted on calibration rows; later policy rows choose raw versus calibrated
scores by log loss and then choose an F1 threshold. Final outcomes never choose
these options. All candidates use seed 42 and the same evaluation populations.
Development evaluation windows are nonoverlapping, but training histories overlap;
fold ranges are sensitivity evidence, not independent confidence intervals.

| Candidate | Mean AP / prevalence | Mean lift @ 10% | Mean ROC AUC | Mean log loss |
|---|---:|---:|---:|---:|
| logistic | 1.163 | 1.156 | 0.528 | 0.342 |
| xgb_base | 1.047 | 1.140 | 0.484 | 0.334 |
| xgb_regularized | 1.004 | 1.050 | 0.489 | 0.343 |
| prior | 1.000 | 1.000 | 0.500 | 0.341 |
| xgb_no_economics | 0.999 | 0.921 | 0.495 | 0.344 |
| random_forest | 0.975 | 0.989 | 0.484 | 0.338 |
| xgb_engineered | 0.948 | 0.898 | 0.472 | 0.343 |
| xgb_recent_no_economics | 0.932 | 0.757 | 0.459 | 0.336 |
| xgb_recent | 0.930 | 0.761 | 0.459 | 0.335 |

Normalizing AP provides a comparison relative to each window's no-skill reference;
it does not eliminate the effect of changing task difficulty. Exact per-window
scores and training sizes are in `development_metrics.csv`.

## Distribution shift investigation

The original train/test response-rate difference is 32.9% points.
A symmetric descriptive decomposition by occupation attributes
1.9% points to occupation mix and
31.0% points to response-rate changes within occupations;
0.0% points arise from unmatched categories.
The components sum to the rate difference. This accounting is not causal proof.
Separate decompositions by contact type, education, and other categories must
not be added together. `source_order_blocks.csv` records response and economic
indicators across ten consecutive row blocks; exact calendar dates are absent.
`feature_drift.csv` reports categorical total variation and numeric PSI with
training-quantile bins and 0.0001 probability smoothing, renormalized. Different
metric families have different scales; no universal drift alarm is asserted.

## Existing final window

- Selected model ROC AUC: **0.643**.
- Average precision: **0.510**; test prevalence: **38.5%**.
- Lift at the top 10%: **1.37×**.
- Log loss: **0.790**, versus training-prevalence baseline **1.097**.
- Raw / selected Brier score: **0.238 / 0.261**.
- Calibration choice: **sigmoid**.
- Raw / selected log loss: **0.685 / 0.790**.
  Calibration was selected using policy data. Its final-window change is
  evidence about transfer, not permission to reselect using these outcomes.
  Here the selected calibration increases final log loss.
  A validation improvement need not transfer through drift.
- Validation-selected F1 threshold: **0.093**;
  evaluation precision **46.2%**, recall **74.9%**.

Compare original and new results cautiously: the new protocol changes training
and validation boundaries. Development-window comparisons are the controlled
experiment. See `final_evaluation.json` for exact row ranges and both baselines.

## Contact budget scenarios

The highest-scoring 10% captures 13.7% of subscribers
with 1.37× lift. Tied scores use expected random tie-breaking.
The scenario table assumes $2/contact and $100/observed subscription, both
hypothetical. Net response value is observed selected subscriptions × assumed
value − contacts × assumed cost. It is not actual revenue, profit, or incremental
campaign impact: untreated outcomes and causal treatment effects are unavailable.
The demo allows these assumptions to change without changing observed results.

## Limits and next evidence

No exact dates or customer identifiers are available. Source-order splits can
cut calendar groups and cannot prevent repeat-customer overlap. Duplicate rows
are retained because anonymization prevents establishing identity. Economic
indicators may contain publication lags or revisions that this dataset cannot
resolve. Current contact channel, scheduled day/month, and campaign count are
assumed known at the planned-call decision moment; `duration` is always excluded.
The dataset covers one Portuguese institution in 2008–2010 and cannot establish
present-day performance. Segment diagnostics are descriptive and do not establish
fairness. Future work needs untouched later data with customer IDs and as-of
feature timestamps, followed by prospective or randomized campaign evaluation.
