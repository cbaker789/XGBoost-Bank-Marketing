# Local verification record

The complete 27-fit development benchmark plus selected-model and baseline final
evaluations ran successfully. Original model artifacts were preserved.

- 15 unit and integration checks passed with `python -m unittest discover -s tests -v`.
- Reloaded inference reproduced saved final probabilities and targets.
- Published ROC AUC, AP and log loss matched the saved prediction rows.
- Temporal partitions were disjoint and source ordered.
- Fitted logistic scaling means matched training rows only.
- Contact-budget metrics reconciled, including score ties and full-budget totals.
- Category response decompositions reconciled to the observed rate change.
- The demo loaded, scored samples, and updated a 20% contact-budget scenario.
- Desktop and 390-pixel mobile browser screenshots were inspected. Figures were
  visually checked for labels, marks and clipping. Screenshots are local QA output,
  not committed portfolio evidence.
- `pip check`, Python compilation and `git diff --check` completed successfully.

Configured GitHub Actions have not run on a hosted repository. An AppTest warning
about missing ScriptRunContext is expected in the local test runner; no app
exceptions occurred. No prospective holdout, production reliability, fairness,
causal uplift or real monetary benefit is established by these checks.
