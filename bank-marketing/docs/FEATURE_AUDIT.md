# Feature availability and experiment audit

Decision moment: a customer has been chosen for a planned telephone contact;
channel and schedule are assumed known. This differs from choosing which unknown
customer to contact before any campaign planning.

| Feature group | Fields | Treatment and unresolved issue |
|---|---|---|
| Outcome / call result | y, duration | Always excluded from prediction features. Duration is only known after the call. |
| Customer profile | age, job, marital, education, default, housing, loan | Assumed known before the planned call. Unknown is an explicit categorical value. Demographic associations require caution; no fairness certification is claimed. |
| Scheduled contact | contact, month, day_of_week | Assumed known for the planned call. These would be unavailable or decision-dependent in a different deployment workflow. |
| Campaign history | campaign, previous, pdays, poutcome | campaign includes the current contact; assumed computable as prior count + planned call. pdays=999 means never previously contacted, not 999 ordinary elapsed days. previous/poutcome describe earlier campaigns. |
| Economic context | emp.var.rate, cons.price.idx, cons.conf.idx, euribor3m, nr.employed | Preserved in applicable configurations and removed in ablation experiments. Historical publication lags and revisions cannot be resolved from this source. |

Engineered configuration adds a prior-contact indicator, replaces pdays=999 with
missing elapsed days (XGBoost handles missing values), and adds log campaign and
previous-contact counts. XGBoost numeric inputs pass through without scaling;
this avoids undefined scaler parameters when a historical fold contains only
missing prior-contact days. Logistic regression uses training-fitted standardization.

## Fixed candidate set

All random seeds are 42. All models use the same fold evaluation rows.

| Candidate | Change from reference |
|---|---|
| prior | Constant probability equal to training prevalence |
| logistic | L2 logistic regression, C=1, max_iter=1500 |
| random_forest | 150 trees, depth 12, minimum leaf 15 |
| xgb_base | Up to 500 trees, learning rate .04, depth 4, min child weight 1, lambda 1 |
| xgb_regularized | depth 3, min child weight 20, lambda 10 |
| xgb_engineered | regularized settings + explicit recency/history transformations |
| xgb_no_economics | regularized settings − five economic indicators |
| xgb_recent | regularized settings, most recent 25% of dataset rows before train boundary |
| xgb_recent_no_economics | recent-history settings − economic indicators |

XGBoost uses histogram trees, subsample .8, column sample .8, and log-loss early
stopping with patience 30. This is a small, documented experimental grid, not an
exhaustive hyperparameter optimization. Negative results are retained.
