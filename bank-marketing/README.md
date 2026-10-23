# Bank Marketing — XGBoost starter project

Predict whether a customer subscribes to a term deposit (`y`: yes/no).

## Dataset

Source: [UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing).
This project uses **bank-additional-full.csv**: 41,188 records, 20 input
features, and the target. UCI documents the rows as ordered by date, from
May 2008 through November 2010. The file uses semicolon delimiters.
The dataset is provided under CC BY 4.0.

Citation: Moro, S., Rita, P., & Cortez, P. (2014). Bank Marketing.
UCI Machine Learning Repository. https://doi.org/10.24432/C5K306

## Run

From this folder, using Python 3.10 or newer:

On macOS, XGBoost also requires the OpenMP runtime: `brew install libomp`.
It was installed during the initial project setup.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python download_data.py
python train.py
```

The dataset is already downloaded during initial setup. Re-running the
download script retrieves the original archive and refreshes the local copy.
If `requirements-lock.txt` is present, install it instead of `requirements.txt`
to reproduce the versions used in the verified initial run.

## Files

- `data/raw/`: original archive, selected CSV, original data dictionary, and source/checksum metadata.
- `download_data.py`: reproducible download using only Python's standard library.
- `train.py`: one-hot encoding, XGBoost classifier, and constant-probability baseline.
- `outputs/`: saved model, fitted preprocessor, metrics, predictions, and feature importance.
- `requirements.txt`: supported dependency ranges.

## Modeling choices

- Exclude `duration`: call length is unavailable before a call and leaks information about its outcome.
- Preserve the source row order for a 70%/15%/15% train/validation/test split.
  Preprocessing is fitted on training rows only; validation controls early stopping.
- Report ROC AUC, average precision, log loss, and precision/recall/F1 at
  a fixed 0.5 threshold. Compare against a training-prevalence baseline.
- Keep categorical `unknown` values as explicit categories. The value `999`
  in `pdays` means no prior contact; see the original dictionary.

This is a learning baseline. Target rates can shift substantially between
time periods; inspect the rates recorded in `metrics.json`. The file has no
customer identifier for enforcing customer separation, and no exact date
column for precise time-based backtests. Other contact-related fields must
be reviewed for availability at the intended prediction moment. Feature
importance measures model usage, not causal effects. Choose any future
decision threshold on validation data before evaluating a fresh holdout.

Raw data, generated outputs, and the virtual environment are ignored by Git.

## Verified initial run

The full download and training workflow completed successfully. The initial
chronological test ROC AUC was **0.563**, versus **0.500** for the constant
baseline. Positive rates changed from 5.6% in training to 10.7% in validation
and 38.5% in testing. At the default 0.5 threshold the model predicted no
positive cases (zero recall). These results expose substantial temporal
distribution shift; this starter model needs further work before practical
use. The test partition has now been inspected, so keep a new holdout for
any future final evaluation after substantial model development.
