import json
import yaml
import joblib
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.base import clone
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.inspection import permutation_importance

def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def build_preprocessor(num_features, cat_features):
    return ColumnTransformer(
        transformers=[
            ('num', 'passthrough', num_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)
        ]
    )

def build_model(config):
    return RandomForestRegressor(
        n_estimators=config['training']['n_estimators'],
        max_depth=config['training']['max_depth'],
        n_jobs=-1,
        random_state=config['training']['random_state']
    )


def score_metrics(y_true, predictions):
    return {
        'MAE': round(float(mean_absolute_error(y_true, predictions)), 4),
        'RMSE': round(float(mean_squared_error(y_true, predictions) ** 0.5), 4),
        'R2': round(float(r2_score(y_true, predictions)), 4),
    }

def train():
    config = load_config()

    Path(config['model']['reports_path']).parent.mkdir(parents=True, exist_ok=True)
    Path(config['model']['save_path']).parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(config['data']['raw_path'])

    num_features = config['model']['features']['numeric']
    cat_features = config['model']['features']['categorical']
    target = config['model']['target']

    df = df.sort_values('timestamp')
    X = df[num_features + cat_features]
    y = df[target]

    split_idx = int(len(df) * (1 - config['training']['test_size']))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    preprocessor = build_preprocessor(num_features, cat_features)

    X_train_prep = preprocessor.fit_transform(X_train)
    X_test_prep = preprocessor.transform(X_test)

    model = build_model(config)

    model.fit(X_train_prep, y_train)

    preds = model.predict(X_test_prep)

    baseline_by_hour = df.iloc[:split_idx].groupby('hour')[target].mean()
    baseline_preds = X_test['hour'].map(baseline_by_hour).fillna(y_train.mean())
    comparison = {
        'Mean by hour baseline': score_metrics(y_test, baseline_preds),
        'Random Forest': score_metrics(y_test, preds),
    }

    time_split = TimeSeriesSplit(n_splits=5)
    cv_scores = []
    for train_indices, validation_indices in time_split.split(X):
        fold_preprocessor = build_preprocessor(num_features, cat_features)
        fold_train = fold_preprocessor.fit_transform(X.iloc[train_indices])
        fold_validation = fold_preprocessor.transform(X.iloc[validation_indices])
        fold_model = clone(build_model(config))
        fold_model.fit(fold_train, y.iloc[train_indices])
        fold_predictions = fold_model.predict(fold_validation)
        cv_scores.append({
            **score_metrics(y.iloc[validation_indices], fold_predictions),
        })

    importance = permutation_importance(
        model, X_test_prep, y_test, n_repeats=5,
        random_state=config['training']['random_state'], scoring='neg_mean_absolute_error'
    )
    feature_importance = dict(sorted(
        zip(preprocessor.get_feature_names_out(), importance.importances_mean),
        key=lambda item: item[1], reverse=True
    )[:15])
    missing_values = df.isna().sum().to_dict()
    numeric_summary = df.select_dtypes(include='number').describe().to_dict()

    metrics = {
        "MAE": comparison['Random Forest']['MAE'],
        "RMSE": comparison['Random Forest']['RMSE'],
        "R2": comparison['Random Forest']['R2'],
        "target": target,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "train_end": str(df.iloc[split_idx - 1]['timestamp']),
        "test_start": str(df.iloc[split_idx]['timestamp']),
        "eda": {'missing_values': missing_values, 'numeric_summary': numeric_summary},
        "permutation_importance": feature_importance,
        "time_series_cv": cv_scores,
        "model_comparison": comparison,
    }

    # Save the metrics report.
    with open(config['model']['reports_path'], 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4)

    # Save the trained model.
    joblib.dump({'preprocessor': preprocessor, 'model': model}, config['model']['save_path'])

    print(f"Metrics: {metrics}")
    print("Model and metrics successfully saved!")

if __name__ == "__main__":
    train()