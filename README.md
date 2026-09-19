#  Almaty LRT Analytics & Delay Prediction Platform

[![CI](https://github.com/erbulanbombi-ux/Almaty-LRT-Analytics-Prediction-Platform/actions/workflows/ci.yml/badge.svg)](https://github.com/erbulanbombi-ux/Almaty-LRT-Analytics-Prediction-Platform/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)

![Almaty LRT Concept](assets/lrt-concept.png)

A data science and machine learning project about urban transit in Almaty, Kazakhstan. It combines field observations of problem intersections with a delay-prediction model and a small web visualization, to explore how a dedicated Light Rail Transit (LRT) corridor could improve travel time and street safety.

**Tech stack:** Python, FastAPI, Pandas, NumPy, scikit-learn, XGBoost, Pydantic, Docker, Three.js

> **Data note:** the model is trained on a **synthetic dataset** produced by `data_generator.py`. The relationships in it are based on field observations and assumptions, not on official transit logs. Metrics below describe how well the model learns this dataset and should not be read as real-world accuracy.

---

## Table of contents

1. [Project status](#project-status)
2. [Phase 1: Field observations](#phase-1-field-observations)
3. [Phase 2: Station and corridor concept](#phase-2-station-and-corridor-concept)
4. [Design trade-offs](#design-trade-offs)
5. [Repository structure](#repository-structure)
6. [Model and results](#model-and-results)
7. [REST API](#rest-api)
8. [Quick start](#quick-start)
9. [Docker](#docker)
10. [Testing and CI](#testing-and-ci)
11. [Limitations and roadmap](#limitations-and-roadmap)
12. [Contributing](#contributing)
13. [License](#license)

---

## Project status

- Time-aware evaluation with `TimeSeriesSplit` and a `Ridge` baseline versus a gradient boosting model are implemented in `train.py`.
- Exploratory analysis is in [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb).
- A FastAPI service exposes prediction and health-check endpoints (`app.py`).
- Tests run with `pytest` locally and through GitHub Actions.
- A minimal [`Dockerfile`](Dockerfile) is included.
- A Three.js web visualization is available in `index.html`, `script.js` and `style.css`.

---

## Phase 1: Field observations

Real conflict points at unmanaged Almaty intersections were photographed and used to decide which features the model should include.

| Unregulated turning conflict | Pedestrian and vehicle conflict zone |
| :---: | :---: |
| ![Intersection 1](assets/real-intersection-1.jpg) | ![Intersection 2](assets/real-intersection-2.jpg) |

### Observed problems

- **Unregulated left turns:** vehicles crossing uncontrolled corridors force oncoming traffic to brake.
- **Micro-mobility interference:** mopeds and bicycles share lanes with cars without barriers, causing sudden speed drops.
- **Crosswalks right after turns:** uncontrolled zebra crossings behind turning points create bottlenecks at rush hour.
- **Non-adaptive signals:** fixed-cycle lights on major streets (Tole Bi, Timiryazev) lengthen stops at peak hours.

> The delay figures used in the model (for example 1.5–3 min bottlenecks) are working assumptions from observation, not measured statistics.

---

## Phase 2: Station and corridor concept

The concept combines an isolated rail corridor with closed, turnstile-controlled stations.

- **Closed stations with contactless fare payment**, inspired by the Istanbul Metrobüs / İstanbulkart approach and compatible in principle with the Onay! card used in Kazakhstan.
- **Fast level boarding** through several doors, aiming to cut dwell time from about 40 s to under 10 s.
- **Dedicated corridor** separated by physical barriers, targeting operating speeds up to 70 km/h.
- **Transit Signal Priority (TSP)**, a proposed concept: signals give priority to approaching LRT vehicles while holding turning cars. TSP is not implemented in this repository.

---

## Design trade-offs

| Challenge | Proposed approach |
| --- | --- |
| **Road width vs. closed stations.** Wide platforms on narrow streets remove 1–2 car lanes. | Staggered stops on opposite sides of intersections and narrow (~1.5 m) enclosed modules. |
| **Turning conflicts at intersections.** Cars crossing a high-speed dedicated lane risk collisions. | Signal control that halts turning vehicles while the LRT passes; the model helps estimate where the risk is highest. |
| **Station placement and land acquisition.** Dense areas need relocated infrastructure; typical spacing is 600–800 m. | Site selection based on demand and cost. This is a planned analysis and is not part of the current code. |

---

## Repository structure

```
.
├── .github/workflows/     # CI configuration
├── assets/                # Images used in the README
├── cache/                 # Local cache (should be git-ignored)
├── data/                  # Datasets
├── models/                # Trained model artifacts
├── notebooks/             # EDA notebooks
├── reports/               # Generated reports and figures
├── tests/                 # pytest test suite
├── app.py                 # FastAPI server
├── config.yaml            # Model, data and API configuration
├── data_generator.py      # Synthetic dataset generator
├── predict.py             # Inference utilities
├── train.py               # Training and evaluation pipeline
├── visualize.py           # Plotting helpers
├── index.html             # Web visualization (Three.js)
├── script.js
├── style.css
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Model and results

**Target:** delay in minutes (`delay_actual`).

**Input features**

| Feature | Type | Range | Description |
| --- | --- | --- | --- |
| `elevation_slope_deg` | float | 0.0–5.0 | Street grade |
| `lane_isolation_score` | float | 0.0–1.0 | Degree of lane segregation |
| `turning_conflicts` | int | 0–10 | Number of unregulated turn points |
| `passenger_density` | float | 0.0–10.0 | Passengers relative to vehicle capacity |
| `delay_lag_15m` | float | 0.0–5.0 | Delay 15 minutes earlier |
| `delay_lag_30m` | float | 0.0–5.0 | Delay 30 minutes earlier |
| `corridor_id` | string | | Corridor identifier |
| `weather_impact` | string | | Weather category, e.g. `clear` |
| `is_peak_hour` | int | 0 or 1 | Rush-hour flag |

### Feature importance

| Feature | Importance |
| --- | --- |
| Turning conflicts | 0.285 |
| Lane isolation score | 0.198 |
| Elevation slope | 0.156 |
| Delay lag (30 min) | 0.142 |
| Passenger density | 0.114 |
| Delay lag (15 min) | 0.105 |

Main takeaways:

- Turning conflicts and lane isolation together carry about **48%** of the importance, so road design matters more than any other group of features.
- Elevation slope is the third strongest feature, reflecting Almaty's north–south gradient.
- The two delay lags together account for about **25%**, showing that delays persist over short periods.
- Passenger density has a smaller but measurable effect.

### Metrics

Reported on a held-out test set from the synthetic dataset:

| Metric | Value |
| --- | --- |
| R² | 0.8844 |
| MAE | 0.4616 min (~28 s) |
| MSE | 0.4671 |
| RMSE | 0.683 min (~41 s) |

**Caveats:** the data is synthetic, and lag features can leak information if a random split is used. For a fair estimate use the time-ordered evaluation (`TimeSeriesSplit`) and treat these numbers as an upper bound.

---

## REST API

Start the server (see [Quick start](#quick-start)); interactive docs are at `http://localhost:8000/docs`.

### `POST /predict`

Returns the expected delay for one corridor segment.

```json
{
  "elevation_slope_deg": 2.5,
  "lane_isolation_score": 0.8,
  "turning_conflicts": 3,
  "passenger_density": 4.5,
  "delay_lag_15m": 1.2,
  "delay_lag_30m": 0.8,
  "corridor_id": "LRT-1",
  "weather_impact": "clear",
  "is_peak_hour": 1
}
```

Response `200 OK`:

```json
{
  "predicted_delay_minutes": 1.85
}
```

Example with cURL:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "elevation_slope_deg": 2.5,
    "lane_isolation_score": 0.8,
    "turning_conflicts": 3,
    "passenger_density": 4.5,
    "delay_lag_15m": 1.2,
    "delay_lag_30m": 0.8,
    "corridor_id": "LRT-1",
    "weather_impact": "clear",
    "is_peak_hour": 1
  }'
```

Example with Python:

```python
import requests

payload = {
    "elevation_slope_deg": 2.5,
    "lane_isolation_score": 0.8,
    "turning_conflicts": 3,
    "passenger_density": 4.5,
    "delay_lag_15m": 1.2,
    "delay_lag_30m": 0.8,
    "corridor_id": "LRT-1",
    "weather_impact": "clear",
    "is_peak_hour": 1,
}
print(requests.post("http://localhost:8000/predict", json=payload).json())
```

### `GET /health`

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### `POST /batch-predict`

Accepts a list of the same objects as `/predict` and returns one prediction per item.

### Errors

- **422 Unprocessable Entity:** a field is missing or has the wrong type (standard FastAPI validation).
- **500 Internal Server Error:** model inference failed.

---

## Quick start

**Requirements:** Python 3.11 (recommended) and Git.

```bash
git clone https://github.com/erbulanbombi-ux/Almaty-LRT-Analytics-Prediction-Platform.git
cd Almaty-LRT-Analytics-Prediction-Platform

python -m venv venv
# Windows:      venv\Scripts\activate
# macOS/Linux:  source venv/bin/activate

pip install -r requirements.txt
```

Generate data, train, and predict:

```bash
python data_generator.py      # create the synthetic dataset
python train.py               # train and evaluate (settings in config.yaml)
python predict.py --help      # see inference options
```

Run the API:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Model, data paths and hyperparameters are configured in `config.yaml`.

---

## Docker

```bash
docker build -t almaty-lrt .
docker run -p 8000:8000 almaty-lrt
```

---

## Testing and CI

```bash
pytest -q
```

GitHub Actions runs the same tests on every push to `main` (see `.github/workflows/`). The badge at the top of this file shows the current status.

---

## Limitations and roadmap

**Current limitations**

- Training data is synthetic, so the model has not been validated against real Almaty transit logs.
- Delay is modeled per corridor segment; network-wide effects are not captured.
- TSP, station placement optimization and GIS analysis are concepts, not implemented features.

**Planned**

- Replace or supplement the synthetic data with real observations (GPS traces, timetables, counts).
- Report metrics only from time-ordered validation.
- Add a GIS-based station placement analysis.
- Prototype a simple TSP simulation.

---

## Contributing

1. Fork the repository.
2. Create a branch: `git checkout -b feature/my-feature`.
3. Commit and push your changes.
4. Open a Pull Request.

Please follow PEP 8 and run `pytest -q` before submitting.

---

## License

Released under the MIT License. See the [LICENSE](LICENSE) file.

---

## Contact

Project lead: **BOMBI** ([@erbulanbombi-ux](https://github.com/erbulanbombi-ux)). Questions and suggestions are welcome via [GitHub Issues](https://github.com/erbulanbombi-ux/Almaty-LRT-Analytics-Prediction-Platform/issues).

## Acknowledgments

- Istanbul Metrobüs for the closed-station design inspiration.
- The scikit-learn, XGBoost and FastAPI open-source communities.
