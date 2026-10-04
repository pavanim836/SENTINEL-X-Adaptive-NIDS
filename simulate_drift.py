
import pandas as pd

# Load the cleaned dataset
df = pd.read_csv("cleaned_ddos.csv")

# Select the features used by our model
features = [
    "Destination Port",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Packet Length Mean"
]

# Keep only the required columns
df = df[features + ["Label"]].copy()

# Remove invalid values
df = df.replace([float("inf"), float("-inf")], pd.NA)
df = df.dropna()

# Split the data into two periods
split = len(df) // 2
period1 = df.iloc[:split].copy()
period2 = df.iloc[split:].copy()

# Simulate a change in traffic in period 2
period2["Flow Duration"] *= 1.5
period2["Total Fwd Packets"] *= 1.5
period2["Total Backward Packets"] *= 1.5

# Combine the periods
stream = pd.concat([period1, period2], ignore_index=True)

stream.to_csv("drift_stream.csv", index=False)

print("Drift simulation completed!")
print("Period 1:", len(period1), "rows")
print("Period 2:", len(period2), "rows")
print("Saved as drift_stream.csv")
