from pathlib import Path

import joblib
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from route_planner import dijkstra, simulate


ROOT = Path(__file__).resolve().parent
with (ROOT / 'config.yaml').open(encoding='utf-8') as file:
    CONFIG = yaml.safe_load(file)
SAVED = joblib.load(ROOT / CONFIG['model']['save_path'])
app = FastAPI(title='Almaty LRT/BRT Analytics API')


class PredictionInput(BaseModel):
    elevation_slope_deg: float = 2.5
    lane_isolation_score: float = Field(0.8, ge=0, le=1)
    turning_conflicts: int = Field(3, ge=0)
    passenger_density: float = Field(45.0, ge=0)
    hour: int = Field(8, ge=0, le=23)
    day_of_week: int = Field(1, ge=0, le=6)
    is_weekend: int = Field(0, ge=0, le=1)
    is_holiday: int = Field(0, ge=0, le=1)
    is_peak_hour: int = Field(1, ge=0, le=1)
    delay_lag_15m: float = Field(1.2, ge=0)
    delay_lag_30m: float = Field(0.8, ge=0)
    corridor_id: str = 'tole-bi-01'
    weather_impact: str = 'clear'


def predict_one(payload: PredictionInput) -> float:
    features = pd.DataFrame([payload.model_dump()])
    prediction = SAVED['model'].predict(SAVED['preprocessor'].transform(features))[0]
    return max(0.0, float(prediction))


@app.get('/health')
def health():
    return {'status': 'healthy', 'model_loaded': True}


@app.post('/predict')
def predict(payload: PredictionInput):
    return {'predicted_delay_minutes': round(predict_one(payload), 2)}


@app.post('/batch-predict')
def batch_predict(payloads: list[PredictionInput]):
    return {
        'results': [predict(payload) for payload in payloads],
        'processed_count': len(payloads),
    }


@app.post('/simulate')
def simulation(payload: dict):
    return simulate(
        int(payload.get('traffic', 50)),
        int(payload.get('passenger_demand', 50)),
        int(payload.get('frequency', 15)),
    )


@app.post('/route')
def route(payload: dict):
    try:
        return dijkstra(payload['start'], payload['end'])
    except (KeyError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
