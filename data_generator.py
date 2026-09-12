from pathlib import Path

import numpy as np
import pandas as pd


RANDOM_STATE = 42


def generate_dataset(output_path='data/lrt_data.csv'):
    rng = np.random.default_rng(RANDOM_STATE)
    timestamps = pd.date_range(start='2026-08-01', periods=2880, freq='15min')
    corridors = ['tole-bi-01', 'abylai-khan-01', 'momyshuly-01']
    holidays = pd.to_datetime(['2026-08-30'])
    rows = [
        {'timestamp': timestamp, 'corridor_id': corridor}
        for timestamp in timestamps
        for corridor in corridors
    ]
    df = pd.DataFrame(rows)
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_holiday'] = df['timestamp'].dt.normalize().isin(holidays).astype(int)
    df['is_peak_hour'] = df['hour'].isin([7, 8, 9, 17, 18, 19]).astype(int)

    elevation_by_corridor = {
        'tole-bi-01': 2.3,
        'abylai-khan-01': 1.2,
        'momyshuly-01': 3.6,
    }
    df['elevation_slope_deg'] = (
        df['corridor_id'].map(elevation_by_corridor)
        + rng.normal(0, 0.1, len(df))
    )
    df['lane_isolation_score'] = rng.uniform(0.6, 1.0, len(df))
    df['turning_conflicts'] = rng.integers(1, 8, len(df))
    df['passenger_density'] = (
        20 + 65 * df['is_peak_hour'] + rng.normal(0, 8, len(df))
    ).clip(10, 100)
    df['weather_impact'] = rng.choice(
        ['clear', 'rain', 'snow'], len(df), p=[0.7, 0.2, 0.1]
    )
    weather_numeric = df['weather_impact'].map({'clear': 0, 'rain': 1, 'snow': 2})
    trend = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds() / 86400
    df['delay_minutes'] = (
        3.0
        + 4.5 * df['is_peak_hour']
        + 1.5 * df['is_weekend']
        + 1.5 * df['is_holiday']
        + 0.6 * df['elevation_slope_deg']
        - 4.0 * df['lane_isolation_score']
        + 0.7 * df['turning_conflicts']
        + 0.035 * df['passenger_density']
        + 2.2 * weather_numeric
        + 0.03 * trend
        + rng.normal(0, 0.8, len(df))
    ).clip(lower=0).round(2)

    df = df.sort_values(['corridor_id', 'timestamp']).reset_index(drop=True)
    df['delay_lag_15m'] = df.groupby('corridor_id')['delay_minutes'].shift(1)
    df['delay_lag_30m'] = df.groupby('corridor_id')['delay_minutes'].shift(2)
    df = df.dropna(subset=['delay_lag_15m', 'delay_lag_30m'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    return df


if __name__ == '__main__':
    generated = generate_dataset()
    print(f'Generated {len(generated)} rows -> data/lrt_data.csv')