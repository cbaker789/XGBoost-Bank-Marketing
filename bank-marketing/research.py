"""Shared, testable temporal evaluation and inference primitives."""
from pathlib import Path
import json
import math

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (average_precision_score, roc_auc_score, log_loss,
                            brier_score_loss, precision_score, recall_score, f1_score)
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent
ECONOMIC = ['emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
LEAKAGE = ['duration', 'y']


def load_data():
    frame = pd.read_csv(ROOT / 'data/raw/bank-additional-full.csv', sep=';')
    if len(frame) != 41188 or len(frame.columns) != 21 or frame.isna().any().any():
        raise ValueError('Unexpected source shape or null values.')
    if set(frame.y) != {'no', 'yes'}:
        raise ValueError('Unexpected target values.')
    return frame


def features(frame, name):
    x = frame.drop(columns=[c for c in LEAKAGE if c in frame]).copy()
    if 'no_economics' in name:
        x = x.drop(columns=ECONOMIC)
    if name == 'xgb_engineered':
        x['previously_contacted'] = (x.pdays != 999).astype(int)
        x['days_since_previous_contact'] = x.pdays.replace(999, np.nan)
        x = x.drop(columns='pdays')
        x['log_campaign'] = np.log1p(x.campaign)
        x['log_previous'] = np.log1p(x.previous)
    if set(LEAKAGE).intersection(x.columns):
        raise AssertionError('Outcome or duration entered feature matrix.')
    return x


def temporal_parts(n, window):
    """Train / stopping / calibration / policy / evaluation; no shared rows."""
    train_end, eval_start, eval_end = [int(n * v) for v in window]
    step = (eval_start - train_end) // 3
    cuts = [0, train_end, train_end + step, train_end + 2*step, eval_start, eval_end]
    if any(b <= a for a, b in zip(cuts, cuts[1:])) or cuts[-1] > n:
        raise ValueError('Window must contain five nonempty ordered partitions.')
    return {name: slice(a, b) for name, a, b in zip(
        ['train', 'stopping', 'calibration', 'policy', 'evaluation'], cuts, cuts[1:])}


def budget_metrics(y, p, fraction):
    """Exact integer contact budget; expected response for randomly broken score ties."""
    y, p = np.asarray(y), np.asarray(p)
    if not 0 < fraction <= 1 or len(y) == 0 or not np.isfinite(p).all():
        raise ValueError('Invalid budget or probabilities.')
    k = max(1, math.ceil(len(y) * fraction))
    cutoff = np.partition(p, len(p)-k)[len(p)-k]
    above, tied = p > cutoff, p == cutoff
    captured = float(y[above].sum() + (k-above.sum()) * y[tied].mean())
    base = float(y.mean())
    return {'budget': fraction, 'contacts': k, 'expected_subscribers': captured,
            'precision': captured/k, 'capture': captured/y.sum() if y.sum() else 0.0,
            'lift': (captured/k)/base if base else None}


def score(y, p, threshold=.5):
    pred = np.asarray(p) >= threshold
    rate = float(np.mean(y))
    ap = float(average_precision_score(y, p))
    return {'n': len(y), 'prevalence': rate, 'roc_auc': float(roc_auc_score(y, p)),
            'average_precision': ap, 'ap_over_prevalence': ap/rate,
            'log_loss': float(log_loss(y, p, labels=[0, 1])),
            'brier': float(brier_score_loss(y, p)), 'threshold': float(threshold),
            'precision': float(precision_score(y, pred, zero_division=0)),
            'recall': float(recall_score(y, pred, zero_division=0)),
            'f1': float(f1_score(y, pred, zero_division=0)),
            'lift_at_10pct': budget_metrics(y, p, .10)['lift']}


def tune_threshold(y, p):
    """Policy-partition F1 selection only; never pass evaluation outcomes here."""
    options = np.unique(np.r_[0, np.quantile(p, np.linspace(0, 1, 101)), 1])
    scores = [f1_score(y, np.asarray(p) >= t, zero_division=0) for t in options]
    return float(options[int(np.argmax(scores))])


def apply_calibration(calibrator, p):
    if calibrator is None:
        return np.asarray(p)
    z = np.log(np.clip(p, 1e-7, 1-1e-7) / np.clip(1-p, 1e-7, 1))
    return calibrator.predict_proba(z.reshape(-1, 1))[:, 1]


def infer(bundle, frame):
    x = features(frame, bundle['name'])
    expected = bundle['input_columns']
    if set(x.columns) != set(expected):
        raise ValueError('Input columns do not match the trained feature schema.')
    raw = bundle['model'].predict_proba(bundle['preprocessor'].transform(x[expected]))[:, 1]
    return apply_calibration(bundle['calibrator'], raw)


def fit_candidate(frame, name, window, seed=42):
    parts = temporal_parts(len(frame), window)
    if 'recent' in name:
        end = parts['train'].stop
        parts['train'] = slice(max(0, end-int(len(frame)*.25)), end)
    x = features(frame, name)
    y = frame.y.map({'no': 0, 'yes': 1})
    numeric = x.select_dtypes(exclude='object').columns.tolist()
    categorical = x.select_dtypes(include='object').columns.tolist()
    # Logistic/RF do not use engineered missing numeric values in this candidate set.
    preprocessing = ColumnTransformer([
        ('categorical', OneHotEncoder(handle_unknown='ignore'), categorical),
        ('numeric', 'passthrough' if name.startswith('xgb') else StandardScaler(), numeric),
    ])
    train = preprocessing.fit_transform(x.iloc[parts['train']])
    matrices = {key: preprocessing.transform(x.iloc[part]) for key, part in parts.items() if key != 'train'}
    if name == 'prior':
        model = DummyClassifier(strategy='prior')
    elif name == 'logistic':
        model = LogisticRegression(C=1, max_iter=1500, random_state=seed)
    elif name == 'random_forest':
        model = RandomForestClassifier(n_estimators=150, max_depth=12, min_samples_leaf=15, n_jobs=4, random_state=seed)
    else:
        regular = name != 'xgb_base'
        model = XGBClassifier(n_estimators=500, learning_rate=.04,
            max_depth=3 if regular else 4, min_child_weight=20 if regular else 1,
            reg_lambda=10 if regular else 1, subsample=.8, colsample_bytree=.8,
            tree_method='hist', early_stopping_rounds=30, eval_metric='logloss',
            n_jobs=4, random_state=seed)
    kwargs = {'eval_set': [(matrices['stopping'], y.iloc[parts['stopping']])], 'verbose': False} if name.startswith('xgb') else {}
    model.fit(train, y.iloc[parts['train']], **kwargs)
    raw = {key: model.predict_proba(value)[:, 1] for key, value in matrices.items()}
    yc = y.iloc[parts['calibration']]
    pc = np.clip(raw['calibration'], 1e-7, 1-1e-7)
    calibrator = None
    if yc.nunique() == 2 and np.ptp(pc) > 1e-8:
        fitted = LogisticRegression(C=1, max_iter=1000, random_state=seed).fit(np.log(pc/(1-pc)).reshape(-1, 1), yc)
        if fitted.coef_[0, 0] >= 0:
            calibrator = fitted
    policy_y = y.iloc[parts['policy']]
    calibrated_policy = apply_calibration(calibrator, raw['policy'])
    # Calibration method selected on later policy data, not the evaluation partition.
    use_calibration = calibrator is not None and log_loss(policy_y, calibrated_policy, labels=[0,1]) < log_loss(policy_y, raw['policy'], labels=[0,1])
    if not use_calibration:
        calibrator = None
    threshold = tune_threshold(policy_y, apply_calibration(calibrator, raw['policy']))
    eval_y = y.iloc[parts['evaluation']].to_numpy()
    eval_p = apply_calibration(calibrator, raw['evaluation'])
    result = score(eval_y, eval_p, threshold)
    result.update({'name': name, 'calibration': 'sigmoid' if calibrator is not None else 'raw',
                   'raw_log_loss': float(log_loss(eval_y, raw['evaluation'], labels=[0,1])),
                   'raw_brier': float(brier_score_loss(eval_y, raw['evaluation'])),
                   'policy_raw_log_loss': float(log_loss(policy_y, raw['policy'], labels=[0,1])),
                   'policy_selected_log_loss': float(log_loss(policy_y, apply_calibration(calibrator, raw['policy']), labels=[0,1])),
                   'policy_n': len(policy_y),
                   'train_n': len(y.iloc[parts['train']]),
                   'trees': int(model.best_iteration+1) if name.startswith('xgb') else None})
    bundle = {'name': name, 'model': model, 'preprocessor': preprocessing,
              'calibrator': calibrator, 'threshold': threshold, 'input_columns': list(x.columns),
              'partitions': {key: [part.start, part.stop] for key, part in parts.items()}}
    return result, bundle, eval_y, eval_p, raw['evaluation']
