"""Train a reproducible starting model with an untouched final test partition."""
from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score, log_loss,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent


def evaluate(y, probability):
    predicted = probability >= 0.5
    return {
        "roc_auc": float(roc_auc_score(y, probability)),
        "average_precision": float(average_precision_score(y, probability)),
        "log_loss": float(log_loss(y, probability)),
        "precision_at_0.5": float(precision_score(y, predicted, zero_division=0)),
        "recall_at_0.5": float(recall_score(y, predicted, zero_division=0)),
        "f1_at_0.5": float(f1_score(y, predicted, zero_division=0)),
        "confusion_matrix_labels_0_1": confusion_matrix(y, predicted).tolist(),
    }


def main():
    path = ROOT / "data" / "raw" / "bank-additional-full.csv"
    if not path.exists():
        raise SystemExit("Dataset missing. Run: python download_data.py")
    frame = pd.read_csv(path, sep=";")
    if len(frame) != 41188 or set(frame["y"].unique()) != {"yes", "no"}:
        raise ValueError("Unexpected dataset; expected the full bank-additional version.")
    # Duration is only known after the call, so it cannot support pre-call prediction.
    X = frame.drop(columns=["y", "duration"])
    y = frame["y"].map({"no": 0, "yes": 1})
    # UCI documents this file as ordered by date. Preserve its order.
    train_end, valid_end = int(len(frame) * 0.70), int(len(frame) * 0.85)
    X_train, X_valid, X_test = X.iloc[:train_end], X.iloc[train_end:valid_end], X.iloc[valid_end:]
    y_train, y_valid, y_test = y.iloc[:train_end], y.iloc[train_end:valid_end], y.iloc[valid_end:]
    encoder = ColumnTransformer(
        [("categorical", OneHotEncoder(handle_unknown="ignore"), X.select_dtypes(include="object").columns.tolist())],
        remainder="passthrough",
    )
    train = encoder.fit_transform(X_train)
    valid, test = encoder.transform(X_valid), encoder.transform(X_test)
    model = XGBClassifier(
        n_estimators=1000, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, tree_method="hist",
        eval_metric="logloss", early_stopping_rounds=40,
        random_state=42, n_jobs=4,
    )
    model.fit(train, y_train, eval_set=[(valid, y_valid)], verbose=False)
    baseline = DummyClassifier(strategy="prior").fit(train, y_train)
    probability = model.predict_proba(test)[:, 1]
    metrics = {
        "split": "chronological 70% train / 15% validation / 15% test",
        "split_counts": {"train": len(y_train), "validation": len(y_valid), "test": len(y_test)},
        "positive_rates": {"train": float(y_train.mean()), "validation": float(y_valid.mean()), "test": float(y_test.mean())},
        "excluded_features": ["duration"],
        "best_iteration_zero_based": int(model.best_iteration),
        "threshold": 0.5,
        "xgboost_test": evaluate(y_test, probability),
        "baseline_test": evaluate(y_test, baseline.predict_proba(test)[:, 1]),
    }
    output = ROOT / "outputs"
    output.mkdir(exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    model.save_model(output / "model.ubj")
    joblib.dump(encoder, output / "preprocessor.joblib")
    pd.DataFrame({"source_row_zero_based": X_test.index, "actual": y_test, "probability_yes": probability}).to_csv(output / "test_predictions.csv", index=False)
    pd.DataFrame({"feature": encoder.get_feature_names_out(), "importance": model.feature_importances_}).sort_values("importance", ascending=False).to_csv(output / "feature_importance.csv", index=False)
    print(json.dumps(metrics, indent=2))
    print(f"Saved model, preprocessing, metrics, predictions, and feature importances to {output}")


if __name__ == "__main__":
    main()
