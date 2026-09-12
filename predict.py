import yaml
import joblib
import pandas as pd

def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def predict():
    config = load_config()
    
    saved_objects = joblib.load(config['model']['save_path'])
    preprocessor = saved_objects['preprocessor']
    model = saved_objects['model']
    
    sample_data = pd.DataFrame([{
        "elevation_slope_deg": 2.5,
        "lane_isolation_score": 0.8,
        "turning_conflicts": 3,
        "passenger_density": 45.0,
        "hour": 8,
        "day_of_week": 1,
        "is_weekend": 0,
        "is_holiday": 0,
        "delay_lag_15m": 1.2,
        "delay_lag_30m": 0.8,
        "corridor_id": "tole-bi-01",
        "weather_impact": "clear",
        "is_peak_hour": 1
    }])
    
    prepared_data = preprocessor.transform(sample_data)
    delay_prediction = model.predict(prepared_data)[0]
    
    print(f"Predicted delay: {delay_prediction:.2f} min.")

if __name__ == "__main__":
    predict()