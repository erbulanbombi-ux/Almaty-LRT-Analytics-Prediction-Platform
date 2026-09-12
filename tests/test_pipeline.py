import json
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

import app
import train
from route_planner import dijkstra, simulate


ROOT = Path(__file__).resolve().parents[1]


def test_training_writes_time_series_metrics():
    metrics = json.loads((ROOT / "reports" / "metrics.json").read_text(encoding="utf-8"))
    assert len(metrics["time_series_cv"]) == 5
    assert "Mean by hour baseline" in metrics["model_comparison"]
    assert "Random Forest" in metrics["model_comparison"]
    assert metrics["model_comparison"]["Random Forest"]["MAE"] < metrics["model_comparison"]["Mean by hour baseline"]["MAE"]


def test_training_columns_match_config():
    config = train.load_config(ROOT / "config.yaml")
    data = pd.read_csv(ROOT / config["data"]["raw_path"])
    columns = config["model"]["features"]["numeric"] + config["model"]["features"]["categorical"]
    assert set(columns + [config["model"]["target"]]).issubset(data.columns)


def test_api_health_and_prediction():
    client = TestClient(app.app)
    assert client.get("/health").json() == {"status": "healthy", "model_loaded": True}
    response = client.post(
        "/predict",
        json={
            "elevation_slope_deg": 2.5,
            "lane_isolation_score": 0.8,
            "turning_conflicts": 3,
            "passenger_density": 45.0,
            "delay_lag_15m": 1.2,
            "delay_lag_30m": 0.8,
            "corridor_id": "LRT-1",
            "weather_impact": "clear",
            "hour": 8,
            "day_of_week": 1,
            "is_weekend": 0,
            "is_holiday": 0,
            "is_peak_hour": 1,
        },
    )
    assert response.status_code == 200
    assert response.json()["predicted_delay_minutes"] >= 0


def test_dijkstra_returns_shortest_route():
    result = dijkstra("LRT-1", "LRT-6")
    assert result == {
        "stations": ["LRT-1", "LRT-3", "LRT-5", "LRT-6"],
        "distance_km": 12.5,
    }


def test_simulation_responds_to_frequency():
    low_frequency = simulate(50, 50, 6)
    high_frequency = simulate(50, 50, 30)
    assert high_frequency["delay_minutes"] < low_frequency["delay_minutes"]
    assert high_frequency["travel_time_minutes"] < low_frequency["travel_time_minutes"]
