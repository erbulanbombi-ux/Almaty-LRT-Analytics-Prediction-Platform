# Almaty LRT/BRT Analytics

An educational machine learning project for predicting public transport delay using synthetic data. It demonstrates a complete workflow: problem definition, data generation, exploratory analysis, time-based validation, baseline comparison, interpretability, and an API.

> The dataset is synthetic. The results are not an official forecast of Almaty's transport system.

## Problem

**Target variable:** `delay_minutes`, the transport delay on a route segment measured in minutes.

This is a **regression** problem because the target is a continuous numeric value, not a class such as "late" or "on time". The model uses information available at prediction time to estimate the delay of the next observation.

## Data

`data_generator.py` creates 30 days of observations at 15-minute intervals for three synthetic route corridors. The dataset contains:

- hour and day of week;
- peak-hour, weekend, and holiday indicators;
- weather (`clear`, `rain`, `snow`);
- passenger density;
- turning conflicts and lane isolation;
- corridor elevation slope;
- delay at the previous and second-previous observation of the same corridor.

The target is generated from a realistic educational relationship between these factors plus random noise. EDA information is saved in `reports/metrics.json`, including missing-value counts and numerical summaries.

## Method

1. The data is sorted by `timestamp`.
2. The first two observations of each corridor are removed because both lag features cannot be calculated honestly for them.
3. The first 80% of time is used for training and the final 20% for testing. A random split is avoided because it could expose future patterns during training.
4. The baseline predicts the mean delay for each hour, calculated using training data only.
5. The main model is a `RandomForestRegressor` with a fixed `random_state=42`.
6. One-hot encoding for categorical features is fitted only on the training set.
7. `TimeSeriesSplit` is also used to check stability across several historical validation windows.

### Data leakage check

Lag features are calculated with `groupby('corridor_id').shift()`. For a row at time `t`, they use only the observed delay of the same corridor at `t-1` and `t-2`. Rows without a valid history are removed instead of being filled with the mean of the whole dataset. The hourly baseline is also calculated from training data only. Therefore, future test information is not used during training or feature filling.

## Results

The latest reproducible run produced these results on the future test period:

| Model | MAE, min | RMSE, min | R² |
|---|---:|---:|---:|
| Mean by hour baseline | 2.1609 | 2.7532 | 0.5157 |
| Random Forest | 0.9793 | 1.2706 | 0.8968 |

- **MAE:** the model is wrong by about 0.98 minutes on average.
- **RMSE:** penalizes large errors more strongly; its error scale is about 1.27 minutes.
- **R²:** the proportion of target variation explained by the model; 0.8968 means approximately 89.7% on this synthetic test set.

`reports/metrics.json` also contains five time-series CV folds and permutation importance. In the current run, `passenger_density`, `turning_conflicts`, and weather were among the most useful features. This is not a causal conclusion: feature importance measures predictive usefulness, not causation.

## Reproduce

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python data_generator.py
python train.py
python -m pytest -q
```

To start the local API:

```powershell
uvicorn app:app --reload
```

Interactive API documentation is available at `http://127.0.0.1:8000/docs`.

## Structure

```text
├── data/                  # synthetic CSV
├── models/                # saved Random Forest artifact
├── notebooks/             # EDA notebook
├── reports/               # metrics, EDA, and feature importance
├── src/                   # package boundary for future reusable modules
├── app.py                 # FastAPI endpoints
├── data_generator.py      # reproducible synthetic data generator
├── train.py               # temporal split, models, and reports
├── predict.py             # local prediction example
├── visualize.py           # plots and feature importance visualization
├── config.yaml            # paths, features, and hyperparameters
├── requirements.txt       # dependencies
└── tests/                 # pipeline and API smoke tests
```

## Limitations

- The data is synthetic and follows a predefined formula rather than real GPS or GTFS measurements.
- One month of observations is not enough to represent seasonality, major events, or long-term changes.
- The hourly baseline is intentionally simple and does not account for corridor or weather.
- Random Forest captures associations, not causation.
- Real-world deployment would require licensed data, data-quality validation, drift monitoring, and prediction intervals.

## Portfolio note

The main strength of this project is not using a fashionable model. It is the measurable comparison on future dates: Random Forest substantially reduces MAE compared with a simple hourly rule, while the synthetic-data limitations are stated explicitly.
