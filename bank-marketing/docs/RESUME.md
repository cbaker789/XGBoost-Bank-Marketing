# Résumé and interview notes

Suggested title: **Customer Targeting Under Temporal Distribution Shift**

Defensible résumé bullet:

> Built a reproducible customer-targeting study on 41,188 banking records,
> comparing nine model configurations across three temporal backtests with
> leakage controls, probability calibration, contact-budget evaluation, and SHAP diagnostics.

Optional measured result (label as retrospective):

> Achieved 1.37× retrospective lift at a 10% contact budget
> on an already-inspected evaluation window; documented temporal drift and validation limitations.

Do not claim live deployment, actual revenue increases, a pristine test set,
causal uplift, guaranteed fairness, or production readiness.

Prepare to explain: why duration leaks the outcome; why shuffled splitting
answers a different question; how calibration differs from ranking; why AP
changes with prevalence; why source-order folds are imperfect; how a negative
result changes the business recommendation. Read results/FINDINGS.md before
using any metric in a résumé.
