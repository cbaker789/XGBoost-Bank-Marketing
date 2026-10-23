"""Run preconfigured temporal comparisons, drift analysis, and frozen final evaluation."""
import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
import subprocess

import joblib
import numpy as np
import pandas as pd
from research import ROOT, ECONOMIC, load_data, fit_candidate, budget_metrics

RESULTS = ROOT / 'results'
ARTIFACTS = ROOT / 'outputs/research'


def drift_analysis(frame):
    a, b = int(len(frame)*.7), int(len(frame)*.85)
    train, test = frame.iloc[:a], frame.iloc[b:]
    blocks = []
    for i, indices in enumerate(np.array_split(np.arange(len(frame)), 10)):
        block = frame.iloc[indices]
        row = {'source_order_decile': i+1, 'n': len(block), 'subscription_rate': float((block.y=='yes').mean())}
        row.update({col: float(block[col].mean()) for col in ECONOMIC})
        blocks.append(row)
    pd.DataFrame(blocks).to_csv(RESULTS/'source_order_blocks.csv', index=False)
    quality = {'rows': len(frame), 'columns': len(frame.columns), 'null_cells': int(frame.isna().sum().sum()),
               'exact_duplicate_rows': int(frame.duplicated().sum()),
               'duplicate_feature_rows': int(frame.drop(columns='y').duplicated().sum()),
               'unknown_counts': {c: int((frame[c]=='unknown').sum()) for c in frame.select_dtypes(include='object')},
               'duplicates_action': 'Retained: anonymization and missing identifiers prevent proof of repeated customers.',
               'target_counts': frame.y.value_counts().to_dict()}
    (RESULTS/'data_quality.json').write_text(json.dumps(quality, indent=2)+'\n')
    drift, groups, decompositions = [], [], []
    for col in frame.drop(columns=['y', 'duration']):
        if frame[col].dtype == 'object':
            keys = sorted(set(train[col]) | set(test[col]))
            p = train[col].value_counts(normalize=True).reindex(keys, fill_value=0)
            q = test[col].value_counts(normalize=True).reindex(keys, fill_value=0)
            drift.append({'feature': col, 'metric': 'total_variation', 'value': float(.5*np.abs(p-q).sum())})
            rates, weights = [], []
            for label, data in [('train', train), ('test', test)]:
                aggregate = data.assign(target=(data.y=='yes').astype(int)).groupby(col).target.agg(['mean','size'])
                rates.append(aggregate['mean'].reindex(keys))
                weights.append(aggregate['size'].reindex(keys, fill_value=0)/len(data))
                for category, row in aggregate.iterrows():
                    groups.append({'feature': col, 'category': category, 'partition': label, 'n': int(row['size']), 'subscription_rate': float(row['mean'])})
            common = rates[0].notna() & rates[1].notna()
            mix = float(((weights[1]-weights[0])*(rates[0]+rates[1])/2)[common].sum())
            within = float(((rates[1]-rates[0])*(weights[0]+weights[1])/2)[common].sum())
            difference = float((test.y=='yes').mean()-(train.y=='yes').mean())
            residual = difference-mix-within
            decompositions.append({'feature': col, 'rate_change': difference, 'mix_component': mix, 'within_group_component': within, 'unmatched_category_component': residual})
        else:
            edges = np.unique(np.r_[-np.inf, train[col].quantile(np.linspace(.1,.9,9)).values, np.inf])
            p = pd.cut(train[col], edges).value_counts(sort=False, normalize=True).to_numpy()
            q = pd.cut(test[col], edges).value_counts(sort=False, normalize=True).to_numpy()
            p, q = np.maximum(p, 1e-4), np.maximum(q, 1e-4)
            p, q = p/p.sum(), q/q.sum()
            drift.append({'feature': col, 'metric': 'population_stability_index', 'value': float(((q-p)*np.log(q/p)).sum())})
    pd.DataFrame(drift).to_csv(RESULTS/'feature_drift.csv', index=False)
    pd.DataFrame(groups).to_csv(RESULTS/'category_response_rates.csv', index=False)
    pd.DataFrame(decompositions).to_csv(RESULTS/'rate_decomposition.csv', index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=ROOT/'config.json')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    RESULTS.mkdir(exist_ok=True); ARTIFACTS.mkdir(parents=True, exist_ok=True)
    frame = load_data()
    drift_analysis(frame)
    rows = []
    for fold, window in enumerate(config['development_windows'], 1):
        for name in config['candidate_names']:
            print(f'Development window {fold}: {name}', flush=True)
            result, _, _, _, _ = fit_candidate(frame, name, window, config['seed'])
            rows.append({'fold': fold, **result})
    development = pd.DataFrame(rows)
    development.to_csv(RESULTS/'development_metrics.csv', index=False)
    leaderboard = development.groupby('name').agg(
        mean_ap_ratio=('ap_over_prevalence','mean'), min_ap_ratio=('ap_over_prevalence','min'),
        mean_lift_10pct=('lift_at_10pct','mean'), mean_roc_auc=('roc_auc','mean'),
        mean_log_loss=('log_loss','mean'), mean_brier=('brier','mean'),
        min_lift_10pct=('lift_at_10pct','min'), max_lift_10pct=('lift_at_10pct','max')
    ).sort_values(['mean_ap_ratio', 'mean_log_loss'], ascending=[False, True])
    leaderboard.to_csv(RESULTS/'leaderboard.csv')
    winner = leaderboard.index[0]
    # Persist selection before accessing the final-window evaluation outcomes.
    selection = {'winner': winner, 'selection_rule': config['selection_metric'],
                 'tie_breaker': 'lower mean log loss', 'config': config,
                 'evaluation_status': 'Exploratory: the source dataset and original test outcomes have already been inspected.'}
    (RESULTS/'selection.json').write_text(json.dumps(selection, indent=2)+'\n')
    print(f'Frozen development choice: {winner}. Evaluating existing final window.', flush=True)
    final, bundle, y, probability, raw = fit_candidate(frame, winner, config['final_window'], config['seed'])
    prior, _, _, _, _ = fit_candidate(frame, 'prior', config['final_window'], config['seed'])
    baseline = {'selected_model': final, 'training_prevalence_baseline': prior,
                'warning': selection['evaluation_status'], 'partitions': bundle['partitions']}
    (RESULTS/'final_evaluation.json').write_text(json.dumps(baseline, indent=2)+'\n')
    calibration_evidence = {key: final[key] for key in ['policy_raw_log_loss','policy_selected_log_loss','policy_n']}
    (RESULTS/'calibration_selection_evidence.json').write_text(json.dumps(calibration_evidence, indent=2)+'\n')
    joblib.dump(bundle, ARTIFACTS/'selected_bundle.joblib')
    start, end = bundle['partitions']['evaluation']
    predictions = pd.DataFrame({'source_row': np.arange(start,end), 'actual': y,
                                'raw_probability': raw, 'probability': probability})
    predictions.to_csv(ARTIFACTS/'final_predictions.csv', index=False)
    budgets = []
    for budget in config['budgets']:
        row = budget_metrics(y, probability, budget)
        cost, value = config['hypothetical_contact_cost'], config['hypothetical_subscription_value']
        row['hypothetical_net_response_value'] = row['expected_subscribers']*value-row['contacts']*cost
        row['random_expected_subscribers'] = row['contacts']*float(y.mean())
        row['hypothetical_gain_over_random'] = (row['expected_subscribers']-row['random_expected_subscribers'])*value
        budgets.append(row)
    pd.DataFrame(budgets).to_csv(RESULTS/'contact_budget_scenarios.csv', index=False)
    segments = []
    for col in ['age', 'job', 'education', 'contact']:
        values = pd.cut(frame.iloc[start:end][col], [0,30,45,60,np.inf], right=False).astype(str) if col=='age' else frame.iloc[start:end][col].astype(str)
        table = predictions.copy(); table['segment'] = values.to_numpy()
        for category, part in table.groupby('segment'):
            segments.append({'feature': col, 'segment': category, 'n': len(part),
                             'observed_rate': float(part.actual.mean()), 'mean_score': float(part.probability.mean()),
                             'recall': float(((part.probability>=bundle['threshold']) & (part.actual==1)).sum()/part.actual.sum()) if part.actual.sum() else None})
    pd.DataFrame(segments).to_csv(RESULTS/'segment_diagnostics.csv', index=False)
    manifest = {'run_at_utc': datetime.now(timezone.utc).isoformat(), 'python': platform.python_version(),
                'data_sha256': hashlib.sha256((ROOT/'data/raw/bank-additional-full.csv').read_bytes()).hexdigest(),
                'source': 'https://archive.ics.uci.edu/dataset/222/bank+marketing',
                'config': config, 'limitations': selection['evaluation_status'],
                'code_and_dependency_sha256': {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ['research.py','run_experiments.py','project_figures.py','config.json','requirements-lock.txt']}}
    (RESULTS/'run_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(leaderboard.to_string(), flush=True)
    print(json.dumps(baseline, indent=2))
    import project_figures
    project_figures.build()
    write_findings(leaderboard, development, baseline, budgets)


def write_findings(leaderboard, development, final, budgets):
    selected = final['selected_model']; base = final['training_prevalence_baseline']
    decomp = pd.read_csv(RESULTS/'rate_decomposition.csv')
    job = decomp[decomp.feature=='job'].iloc[0]
    calibration_change = 'increases' if selected['log_loss'] > selected['raw_log_loss'] else 'reduces' if selected['log_loss'] < selected['raw_log_loss'] else 'does not change'
    consistency = 'does not beat random targeting consistently' if leaderboard.loc[selected['name'], 'min_lift_10pct'] < 1 else 'meets or exceeds random targeting in each window'
    rows = '\n'.join(f"| {name} | {r.mean_ap_ratio:.3f} | {r.mean_lift_10pct:.3f} | {r.mean_roc_auc:.3f} | {r.mean_log_loss:.3f} |" for name,r in leaderboard.iterrows())
    text = f'''# Customer targeting under temporal distribution shift

## Decision and evidence

Development selection chose **{selected['name']}** using mean average precision
normalized by each evaluation window's prevalence. It was selected before the
final-window evaluation. All results remain exploratory: the full dataset and
original test outcomes were already inspected before this study. No new external
or prospective holdout is available. Model improvement is not guaranteed.

The selected candidate {consistency} across
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
{rows}

Normalizing AP provides a comparison relative to each window's no-skill reference;
it does not eliminate the effect of changing task difficulty. Exact per-window
scores and training sizes are in `development_metrics.csv`.

## Distribution shift investigation

The original train/test response-rate difference is {job.rate_change:.1%} points.
A symmetric descriptive decomposition by occupation attributes
{job.mix_component:.1%} points to occupation mix and
{job.within_group_component:.1%} points to response-rate changes within occupations;
{job.unmatched_category_component:.1%} points arise from unmatched categories.
The components sum to the rate difference. This accounting is not causal proof.
Separate decompositions by contact type, education, and other categories must
not be added together. `source_order_blocks.csv` records response and economic
indicators across ten consecutive row blocks; exact calendar dates are absent.
`feature_drift.csv` reports categorical total variation and numeric PSI with
training-quantile bins and 0.0001 probability smoothing, renormalized. Different
metric families have different scales; no universal drift alarm is asserted.

## Existing final window

- Selected model ROC AUC: **{selected['roc_auc']:.3f}**.
- Average precision: **{selected['average_precision']:.3f}**; test prevalence: **{selected['prevalence']:.1%}**.
- Lift at the top 10%: **{selected['lift_at_10pct']:.2f}×**.
- Log loss: **{selected['log_loss']:.3f}**, versus training-prevalence baseline **{base['log_loss']:.3f}**.
- Raw / selected Brier score: **{selected['raw_brier']:.3f} / {selected['brier']:.3f}**.
- Calibration choice: **{selected['calibration']}**.
- Raw / selected log loss: **{selected['raw_log_loss']:.3f} / {selected['log_loss']:.3f}**.
  Calibration was selected using policy data. Its final-window change is
  evidence about transfer, not permission to reselect using these outcomes.
  Here the selected calibration {calibration_change} final log loss.
  A validation improvement need not transfer through drift.
- Validation-selected F1 threshold: **{selected['threshold']:.3f}**;
  evaluation precision **{selected['precision']:.1%}**, recall **{selected['recall']:.1%}**.

Compare original and new results cautiously: the new protocol changes training
and validation boundaries. Development-window comparisons are the controlled
experiment. See `final_evaluation.json` for exact row ranges and both baselines.

## Contact budget scenarios

The highest-scoring 10% captures {budgets[1]['capture']:.1%} of subscribers
with {budgets[1]['lift']:.2f}× lift. Tied scores use expected random tie-breaking.
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
'''
    (RESULTS/'FINDINGS.md').write_text(text)
    (ROOT/'docs/RESUME.md').write_text(f'''# Résumé and interview notes

Suggested title: **Customer Targeting Under Temporal Distribution Shift**

Defensible résumé bullet:

> Built a reproducible customer-targeting study on 41,188 banking records,
> comparing nine model configurations across three temporal backtests with
> leakage controls, probability calibration, contact-budget evaluation, and SHAP diagnostics.

Optional measured result (label as retrospective):

> Achieved {selected['lift_at_10pct']:.2f}× retrospective lift at a 10% contact budget
> on an already-inspected evaluation window; documented temporal drift and validation limitations.

Do not claim live deployment, actual revenue increases, a pristine test set,
causal uplift, guaranteed fairness, or production readiness.

Prepare to explain: why duration leaks the outcome; why shuffled splitting
answers a different question; how calibration differs from ranking; why AP
changes with prevalence; why source-order folds are imperfect; how a negative
result changes the business recommendation. Read results/FINDINGS.md before
using any metric in a résumé.
''')


if __name__ == '__main__':
    main()
