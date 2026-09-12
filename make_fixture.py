import pandas as pd
import numpy as np

# 1. Using actual station IDs from your stations_sydney.csv file
station_ids = ["560001", "560002", "560003", "560004"] 

# 2. Generate 200 hours of timestamps
timestamps = pd.date_range(start="2026-01-01 00:00:00", periods=200, freq="h")

# 3. Create a DataFrame holding all combinations
data = []
for ts in timestamps:
    for sid in station_ids:
        # Simulate realistic rainfall: mostly 0.0, occasionally random rain
        rain = 0.0 if np.random.rand() > 0.2 else np.round(np.random.uniform(0.1, 15.0), 1)
        data.append({"timestamp": ts, "station_id": sid, "rainfall_mm": rain})

df = pd.DataFrame(data)

# 4. Simulate missing data (broken gauges) by randomly dropping 10% of the rows
drop_indices = np.random.choice(df.index, size=int(len(df) * 0.1), replace=False)
df_dropped = df.drop(drop_indices).reset_index(drop=True)

# 5. Save to the test directory
output_path = "tests/data/graphs/gauge_hourly.csv"
df_dropped.to_csv(output_path, index=False)
print(f"✅ Fixture saved to {output_path} with {len(df_dropped)} rows.")