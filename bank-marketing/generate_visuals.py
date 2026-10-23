"""Recreate six diagnostic charts from the saved model, without retraining."""
from pathlib import Path
import hashlib
import json
import math

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from sklearn.metrics import average_precision_score, precision_recall_curve, confusion_matrix, brier_score_loss
from xgboost import XGBClassifier, DMatrix

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs'
FIG = OUT / 'figures'
BLUE, GOLD, INK, GREY = '#27659B', '#C78B26', '#253342', '#76818C'
FILES = []
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.titlesize': 17,
    'axes.labelsize': 11, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': '#BAC2CA', 'text.color': INK, 'axes.labelcolor': INK,
    'xtick.color': INK, 'ytick.color': INK, 'grid.color': '#E6EAEE',
    'figure.facecolor': 'white', 'axes.facecolor': 'white', 'svg.fonttype': 'none',
})


def canvas(title, subtitle, figsize=(10, 6.7)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(left=.12, right=.95, top=.79, bottom=.22)
    fig.text(.08, .95, 'BANK MARKETING  /  XGBOOST DIAGNOSTICS', size=9, color=GREY)
    fig.text(.08, .89, title, size=18, weight='bold')
    fig.text(.08, .84, subtitle, size=10, color=GREY)
    ax.set_axisbelow(True)
    ax.grid(axis='y')
    return fig, ax


def save(fig, stem, note):
    fig.text(.08, .085, note, size=9, color=GREY, va='bottom', linespacing=1.6)
    fig.text(.08, .035, 'Source: UCI bank-additional-full.csv + saved project artifacts | duration excluded', size=8, color=GREY)
    for ext in ('png', 'svg'):
        fig.savefig(FIG / f'{stem}.{ext}', dpi=180, facecolor='white')
    FILES.append(stem)
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(ROOT / 'data/raw/bank-additional-full.csv', sep=';')
    metrics = json.loads((OUT / 'metrics.json').read_text())
    saved = pd.read_csv(OUT / 'test_predictions.csv')
    a, b = int(len(frame) * .7), int(len(frame) * .85)
    X = frame.drop(columns=['y', 'duration'])
    y_all = frame.y.map({'no': 0, 'yes': 1})
    y = y_all.iloc[b:].to_numpy()
    np.testing.assert_array_equal(saved.source_row_zero_based, X.iloc[b:].index)
    np.testing.assert_array_equal(saved.actual, y)
    encoder = joblib.load(OUT / 'preprocessor.joblib')
    model = XGBClassifier()
    model.load_model(OUT / 'model.ubj')
    encoded = encoder.transform(X.iloc[b:])
    p = model.predict_proba(encoded)[:, 1]
    np.testing.assert_allclose(p, saved.probability_yes, rtol=1e-5, atol=1e-7)
    n, positives, prevalence = len(y), int(y.sum()), float(y.mean())

    # 1. Observed rates, preserving the chronological partition order.
    split_data = pd.DataFrame([
        {'split': name, 'n': len(part), 'subscribers': int(part.sum()), 'subscription_rate': float(part.mean())}
        for name, part in [('Train', y_all.iloc[:a]), ('Validation', y_all.iloc[a:b]), ('Test', y_all.iloc[b:])]
    ])
    np.testing.assert_allclose(split_data.subscription_rate, list(metrics['positive_rates'].values()))
    split_data.to_csv(FIG / 'subscription_rates.csv', index=False)
    fig, ax = canvas('Subscription rate by chronological partition', 'All 41,188 records | partitions follow the source file order')
    ax.bar(split_data.split, split_data.subscription_rate, color=[BLUE, BLUE, GOLD], width=.55)
    for i, row in split_data.iterrows():
        ax.text(i, row.subscription_rate+.013, f'{row.subscription_rate:.1%}', ha='center', weight='bold', size=15)
    ax.set_xticks(range(3), [f'{row.split}\nn = {row.n:,}' for row in split_data.itertuples()])
    ax.set_ylim(0, .48)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_ylabel('Customers who subscribed')
    save(fig, '01_subscription_rates', 'Observed response rates differ substantially between partitions.\nThese are sequential groups, not equally spaced calendar periods.')

    # 2. Average precision, rather than a trapezoidal PR area.
    precision, recall, thresholds = precision_recall_curve(y, p)
    ap = float(average_precision_score(y, p))
    np.testing.assert_allclose(ap, metrics['xgboost_test']['average_precision'])
    pd.DataFrame({'recall': recall, 'precision': precision, 'threshold': np.r_[thresholds, np.nan]}).to_csv(FIG / 'precision_recall.csv', index=False)
    fig, ax = canvas('Precision–recall curve', f'Chronological test set | n = {n:,} | {positives:,} subscribers')
    ax.step(recall, precision, where='post', color=BLUE, lw=2, label=f'XGBoost: average precision {ap:.3f}')
    ax.axhline(prevalence, color=INK, ls='--', lw=1.3, label=f'No-skill reference: {prevalence:.1%}')
    ax.set(xlim=(0, 1), ylim=(0, 1.04), xlabel='Recall: share of subscribers found', ylabel='Precision: share of selected customers who subscribe')
    ax.xaxis.set_major_formatter(PercentFormatter(1)); ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(loc='upper right', frameon=False, fontsize=10)
    save(fig, '02_precision_recall', 'Higher precision means fewer wasted contacts; higher recall means fewer missed subscribers.\nThe reference uses test prevalence; this plot does not select a deployment threshold.')

    # 3. Cumulative lift: descending scores, averaged across tied score cutoffs.
    grouped = pd.DataFrame({'score': p, 'actual': y}).groupby('score', sort=True).actual.agg(['sum', 'count']).iloc[::-1]
    cum_n = grouped['count'].cumsum().to_numpy()
    cum_y = grouped['sum'].cumsum().to_numpy()
    # Linear interpolation inside score ties is expected performance for random tie-breaking.
    fractions = np.arange(1, 101) / 100
    contacted = fractions * n
    captured = np.interp(contacted, np.r_[0, cum_n], np.r_[0, cum_y])
    lift = (captured / contacted) / prevalence
    top_capture = float(captured[9] / positives)
    top_lift = float(lift[9])
    pd.DataFrame({'contact_fraction': fractions, 'expected_subscribers_captured': captured, 'share_subscribers_captured': captured / positives, 'cumulative_lift': lift}).to_csv(FIG / 'cumulative_lift.csv', index=False)
    fig, ax = canvas('Cumulative lift by contact budget', 'Rank customers by predicted probability | chronological test set')
    ax.plot(fractions, lift, color=BLUE, lw=2, label='XGBoost ranking')
    ax.axhline(1, color=INK, ls='--', lw=1.3, label='Random selection: 1.0×')
    ax.scatter(.1, top_lift, color=GOLD, s=60, zorder=5)
    ax.annotate(f'Top 10%: {top_lift:.2f}× lift\nCaptures {top_capture:.1%} of subscribers', xy=(.1, top_lift), xytext=(.38, max(1.45, top_lift+.18)), arrowprops={'arrowstyle': '-', 'color': GREY}, fontsize=11)
    ax.set(xlim=(0, 1), ylim=(0, max(2, float(lift.max())*1.16)), xlabel='Share of customers contacted, highest scores first', ylabel='Cumulative lift versus random selection')
    ax.xaxis.set_major_formatter(PercentFormatter(1)); ax.legend(loc='lower right', frameon=False)
    save(fig, '03_cumulative_lift', 'Lift = subscription rate among selected customers / subscription rate in the test set.\nScore ties use expected random tie-breaking. Retrospective ranking is not causal campaign lift.')

    # 4. Confusion matrix at the existing threshold, without tuning on test outcomes.
    threshold = float(metrics['threshold'])
    cm = confusion_matrix(y, p >= threshold, labels=[0, 1])
    np.testing.assert_array_equal(cm, metrics['xgboost_test']['confusion_matrix_labels_0_1'])
    pd.DataFrame(cm, index=['actual_no', 'actual_yes'], columns=['predicted_no', 'predicted_yes']).to_csv(FIG / 'confusion_matrix.csv')
    fig, ax = canvas('Confusion matrix at the current threshold', f'Predict subscription when probability ≥ {threshold:.2f} | n = {n:,}')
    fig.subplots_adjust(left=.24)
    ax.grid(False)
    ax.imshow(cm, cmap=LinearSegmentedColormap.from_list('count', ['#F0F4F7', BLUE]), vmin=0, vmax=cm.max(), aspect='auto')
    labels = [['True negatives', 'False positives'], ['False negatives', 'True positives']]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{cm[i,j]:,}\n{labels[i][j]}', ha='center', va='center', fontsize=14, color='white' if cm[i,j] > cm.max()*.55 else INK)
    ax.set_xticks([0, 1], ['No subscription', 'Subscription']); ax.set_yticks([0, 1], ['No subscription', 'Subscription'])
    ax.set(xlabel='Predicted outcome', ylabel='Actual outcome')
    save(fig, '04_confusion_matrix', 'At 0.50 the current model predicts no subscribers and has zero recall.\nAny future threshold choice should use validation data, then a fresh final holdout.')

    # 5. Native exact TreeSHAP, restricted to the same trees as predict_proba.
    rng = np.random.default_rng(42)
    sample_idx = np.sort(rng.choice(n, min(1000, n), replace=False))
    sample = encoded[sample_idx]
    dmatrix = DMatrix(sample)
    active = model.get_booster()[:model.best_iteration + 1]
    contribution = active.predict(dmatrix, pred_contribs=True)
    margin = active.predict(dmatrix, output_margin=True)
    np.testing.assert_allclose(contribution.sum(axis=1), margin, atol=2e-5)
    np.testing.assert_allclose(1/(1+np.exp(-margin)), p[sample_idx], atol=1e-6)
    shap_values = contribution[:, :-1]
    names = np.array([s.replace('categorical__', '').replace('remainder__', '') for s in encoder.get_feature_names_out()])
    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(-mean_abs, kind='stable')[:15]
    pd.DataFrame({'feature': names, 'mean_absolute_shap_log_odds': mean_abs}).sort_values('mean_absolute_shap_log_odds', ascending=False).to_csv(FIG / 'shap_importance.csv', index=False)
    pd.DataFrame(shap_values, columns=names, index=X.iloc[b:].index[sample_idx]).rename_axis('source_row_zero_based').to_csv(FIG / 'shap_values_sample.csv')
    dense = sample.toarray() if hasattr(sample, 'toarray') else np.asarray(sample)
    fig, ax = canvas('SHAP feature summary', f'Exact TreeSHAP | reproducible sample of {len(sample_idx):,} test customers | top 15 encoded features', figsize=(11, 8.8))
    fig.subplots_adjust(left=.28, right=.88, bottom=.21, top=.81)
    cmap = LinearSegmentedColormap.from_list('feature_value', [BLUE, '#C43C79'])
    for row, col in enumerate(order):
        values = dense[:, col]
        low, high = float(np.nanmin(values)), float(np.nanmax(values))
        normalized = (values-low)/(high-low) if high > low else np.full(len(values), .5)
        ax.scatter(shap_values[:, col], row+rng.uniform(-.23, .23, len(sample_idx)), c=normalized, cmap=cmap, vmin=0, vmax=1, s=9, alpha=.6, edgecolors='none', rasterized=True)
    ax.set_yticks(range(len(order)), names[order]); ax.invert_yaxis()
    ax.axvline(0, color=INK, lw=1)
    ax.set_xlabel('SHAP contribution to subscription log-odds (higher → higher score)')
    fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap=cmap), ax=ax, pad=.03, fraction=.025, ticks=[0, 1]).ax.set_yticklabels(['Low', 'High'])
    save(fig, '05_shap_summary', 'Rows rank features by mean absolute SHAP value. Colors show within-feature minimum to maximum.\nFor one-hot features, low = absent and high = present. Contributions explain the model, not causality.')

    # 6. Equal-frequency score bins with volume and prediction distribution.
    bins = pd.qcut(pd.Series(p), q=10, duplicates='drop')
    cal = pd.DataFrame({'bin': bins, 'probability': p, 'actual': y}).groupby('bin', observed=True).agg(mean_prediction=('probability', 'mean'), observed_rate=('actual', 'mean'), n=('actual', 'size'), positives=('actual', 'sum')).reset_index()
    cal.to_csv(FIG / 'calibration_bins.csv', index=False)
    brier = float(brier_score_loss(y, p))
    fig, ax = canvas('Calibration: predicted versus observed response', f'Test n = {n:,} | equal-frequency bins (up to 10) | Brier score = {brier:.3f}')
    ax.plot([0, 1], [0, 1], color=INK, ls='--', lw=1.3, label='Perfect calibration')
    ax.scatter(cal.mean_prediction, cal.observed_rate, color=BLUE, s=45, label='Observed rate per score bin', zorder=4)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel='Mean predicted probability in bin', ylabel='Observed subscription rate in bin')
    ax.xaxis.set_major_formatter(PercentFormatter(1)); ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(loc='lower right', frameon=False, fontsize=10)
    save(fig, '06_calibration', f'Each point summarizes {cal.n.min():,}–{cal.n.max():,} customers; exact bin counts are saved in calibration_bins.csv.\nPoints above the diagonal indicate underprediction. Descriptive rates; no confidence intervals.')

    summary = {
        'test_n': n, 'test_subscribers': positives, 'average_precision': ap,
        'top_10_percent_lift': top_lift, 'top_10_percent_subscriber_capture': top_capture,
        'brier_score': brier, 'probability_range': [float(p.min()), float(p.max())],
        'shap_sample_n': len(sample_idx), 'shap_random_seed': 42,
        'shap_trees': model.best_iteration + 1,
        'checks': ['Saved prediction row identities and targets match raw data', 'Reloaded model probabilities match saved predictions', 'Rates, AP and confusion matrix match metrics.json', 'SHAP additivity and reconstructed probabilities pass'],
        'input_sha256': {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in [ROOT/'data/raw/bank-additional-full.csv', OUT/'model.ubj', OUT/'preprocessor.joblib', OUT/'test_predictions.csv']},
    }
    (FIG / 'visuals_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    # Preview sheet only; full-resolution originals remain the primary charts.
    thumbs = []
    for stem in FILES:
        with Image.open(FIG / f'{stem}.png') as image:
            tile = image.convert('RGB'); tile.thumbnail((900, 750))
            thumbs.append(tile)
    sheet = Image.new('RGB', (1800, 2250), 'white')
    for i, tile in enumerate(thumbs):
        sheet.paste(tile, ((i%2)*900, (i//2)*750))
    sheet.save(FIG / '00_overview.png')
    (FIG / 'README.md').write_text(f'''# Bank Marketing model visuals

Six figures are available as PNG images and editable SVG vectors. `00_overview.png`
is a preview sheet; use the numbered originals for reading or sharing.

1. Subscription rates: observed rates and sample counts in chronological partitions.
2. Precision–recall: test average precision {ap:.3f}; prevalence reference {prevalence:.1%}.
3. Cumulative lift: highest-scoring 10% captures {top_capture:.1%} of test subscribers
   ({top_lift:.2f}× random-selection lift). Score ties use expected random tie-breaking.
4. Confusion matrix: existing 0.50 threshold; no subscribers predicted.
5. SHAP summary: exact TreeSHAP on 1,000 test customers, seed 42, using the
   {model.best_iteration + 1} trees used by the saved early-stopped model. Values are log-odds.
   Encoded features are ranked by mean absolute SHAP value. Colors are scaled
   separately within each feature. One-hot values indicate absence/presence.
6. Calibration: up to ten equal-frequency bins; the note shows the sample-count range.
   Exact sample counts are available in calibration_bins.csv.
   Test Brier score {brier:.3f} (lower is better). No uncertainty intervals are estimated.

## Reproduce

From the bank-marketing directory:

```sh
.venv/bin/python generate_visuals.py
```

Install project requirements first if needed. The script reads saved artifacts;
it does not refit the model or select a new decision threshold. CSV files contain
chart values; `visuals_summary.json` records input checksums and verification results.
Prediction probabilities were checked against saved test predictions. SHAP
contributions were checked for additivity and reconstructed probability agreement.

## Interpretation

Response rates change substantially across sequential partitions. The current
model has weak discrimination and underpredicts test response rates. At 0.50 it
misses all subscribers. Lift describes retrospective ranking, not causal impact
of contacting a customer. These inspected test outcomes are diagnostic; use
validation data and a fresh final holdout for future model or threshold decisions.
No customer identifiers or exact dates are available for stronger split guarantees.

Source: [UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing),
local `bank-additional-full.csv`, and the project's saved model and predictions.
SHAP uses XGBoost's native `pred_contribs` API; see the
[XGBoost API reference](https://xgboost.readthedocs.io/en/stable/python/python_api.html).
''')
    print(json.dumps(summary, indent=2))
    print(f'Saved all six charts (PNG + SVG), underlying CSVs, and overview to {FIG}')


if __name__ == '__main__':
    main()
