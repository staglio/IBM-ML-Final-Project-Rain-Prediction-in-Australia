# Rainfall Prediction Classifier

A machine learning pipeline that predicts daily rainfall in the Melbourne area using the Australian weather dataset (`weatherAUS`), sourced from the Australian Bureau of Meteorology.

## Overview

The project builds and tunes a classifier to predict whether it rains on a given day. Rather than predicting `RainTomorrow`, the target is reframed as **`RainToday`** so that all historical measurements (up to the previous evening) are available at prediction time, making the prediction more realistic.

## Data

- **Source:** `weatherAUS_2.csv` (~145k daily observations across Australia).
- **Filtering:** rows with missing values are dropped, and the analysis is restricted to three nearby stations — `Melbourne`, `MelbourneAirport`, and `Watsonia` (~7,500 observations) — to keep weather patterns regionally consistent.
- **Feature engineering:** a `Season` feature is derived from `Date`, which is then dropped.

## Pipeline

1. **Preprocessing** (`ColumnTransformer`)
   - Numeric features → `StandardScaler`
   - Categorical features → `OneHotEncoder`
2. **Model:** `RandomForestClassifier` with `class_weight='balanced'` to handle class imbalance (~76% "No rain").
3. **Tuning:** `GridSearchCV` over `n_estimators`, `max_depth`, and `min_samples_split`, with 5-fold `StratifiedKFold` cross-validation.

## Evaluation

- Best parameters and cross-validation score
- Test-set accuracy
- Classification report and confusion matrix
- Feature importances from the best estimator

## Requirements

```
pandas
matplotlib
scikit-learn
```

## Usage

Place `weatherAUS_2.csv` in the project directory and run the notebook:

```bash
jupyter notebook Rainfall_Prediction.ipynb
```
