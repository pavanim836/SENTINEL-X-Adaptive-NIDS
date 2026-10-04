
import pandas as pd
from river import preprocessing, linear_model, optim

# Load the dataset
df = pd.read_csv("cleaned_ddos.csv")

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

df = df[features + ["Label"]]
df = df.replace([float("inf"), float("-inf")], pd.NA)
df = df.dropna()

# Create the model
scaler = preprocessing.StandardScaler()
classifier = linear_model.LogisticRegression(
    optimizer=optim.SGD(0.01)
)
model = scaler | classifier

# Train on the first 50,000 records
for _, row in df.iloc[:50000].iterrows():
    x = row[features].to_dict()
    y = row["Label"] == "DDoS"
    model.learn_one(x, y)

# Select one record to explain
row = df.iloc[50000]
x = row[features].to_dict()

# Scale the record using the trained scaler
scaled_x = scaler.transform_one(x)

# Predict before learning from this record
prediction = model.predict_one(x)
probability = model.predict_proba_one(x).get(True, 0)

print("\nNETWORK INTRUSION ALERT")
print("-----------------------")
print("Prediction:", "DDoS" if prediction else "BENIGN")
print(f"Attack probability: {probability:.2%}")

# Calculate contributions using scaled values
weights = classifier.weights
contributions = []

for feature in features:
    weight = weights.get(feature, 0)
    contribution = weight * scaled_x.get(feature, 0)
    contributions.append((feature, contribution))

# Sort by absolute contribution
contributions.sort(key=lambda item: abs(item[1]), reverse=True)

print("\nFEATURE CONTRIBUTIONS")
print("---------------------")

for feature, value in contributions:
    direction = "toward DDoS" if value > 0 else "toward BENIGN"
    print(f"{feature}: {value:.4f} ({direction})")

# Display the intercept
print(f"\nModel intercept: {classifier.intercept:.4f}")

# Update the model after prediction
model.learn_one(x, row["Label"] == "DDoS")
