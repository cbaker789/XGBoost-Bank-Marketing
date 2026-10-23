# Model card

**Purpose:** retrospective ranking of likely term-deposit subscribers among customers
scheduled for a marketing call. Educational research and portfolio demonstration.
**Not approved for:** automated financial eligibility, real customer decisions,
current-day deployment, or claims of incremental treatment impact.

**Source:** UCI Bank Marketing, bank-additional-full.csv, 41,188 records from one
Portuguese bank, May 2008–November 2010. License: CC BY 4.0.
Citation: Moro, S., Rita, P., & Cortez, P. (2014), DOI 10.24432/C5K306.
Publication: Moro, S., Cortez, P., & Rita, P. (2014). A Data-Driven Approach to
Predict the Success of Bank Telemarketing. Decision Support Systems, DOI
10.1016/j.dss.2014.03.001.

**Target:** observed subscription (`y=yes`). **Leakage controls:** drop duration
and target before transformation; fit encoders/scalers only on training rows;
fit calibration and select thresholds on separate later rows. Month/day/channel
and campaign count are assumed available for the scheduled call. Macroeconomic
features have unresolved as-of publication/revision risk.

**Selection:** nine fixed candidates in three expanding source-order backtests.
Candidates include a training-prevalence baseline, logistic regression, random
forest, and six XGBoost configurations. XGBoost settings test regularization,
recency, economic-feature removal, and explicit previous-contact engineering.
Ranking uses mean AP divided by each window's prevalence; log loss breaks ties.
Sigmoid calibration is fitted on calibration rows and retained only if it lowers
log loss on policy rows; F1 threshold is selected on policy rows.

**Evaluation status:** all evaluations are exploratory. The full source data and
original final test outcomes were inspected before this extension. No prospective,
external, or demonstrably untouched holdout exists. Exact date and customer ID
are absent. Training histories overlap across backtests; fold ranges are not
independent confidence intervals. Source-order splits may divide calendar groups.

**Metrics:** see results/FINDINGS.md and results/final_evaluation.json. AP and
precision depend on prevalence. ROC AUC alone does not describe business value.
Calibration is not guaranteed to transfer across changing periods. Contact-budget
lift is retrospective ranking lift, not causal uplift. Monetary values are assumed
scenario inputs and never measured campaign revenue.

**Data quality and segments:** unknown strings are explicit categories; pdays=999
means no previous contact. Exact duplicate rows remain because anonymized records
cannot establish customer identity. Occupation, education, age bands and contact
channel diagnostics show counts, rates, mean scores and recall. These descriptive
checks do not constitute a fairness assessment; sparse groups require caution.

**Explanation:** original SHAP figures explain the original saved XGBoost model,
not the newly selected benchmark winner. They are labeled separately in the
project. Feature contributions are associations in model predictions, not causes.

**Reproducibility:** config.json, requirements-lock.txt, fixed seeds, input hashes,
a run manifest, saved preprocessing/model/calibration bundle, and unit tests.
Only deserialize trusted locally generated joblib files. The local demo uses these
artifacts and original sample rows; it does not send data to an external service.

**Next evidence required:** later untouched observations, stable customer IDs,
as-of feature timestamps and an intervention/control design before claiming
incremental business benefit. A real operational system would also need privacy,
access controls, monitoring and human decision governance tailored to its use.
