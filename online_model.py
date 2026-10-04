import pandas as pd
from river import compose
from river import preprocessing
from river import linear_model
from river import optim
from river import metrics

# Load cleaned dataset
df = pd.read_csv("cleaned_ddos.csv")
df.columns = df.columns.str.strip()

# Select a few numerical features
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

# Create online learning model
model = compose.Pipeline(
    preprocessing.StandardScaler(),
    linear_model.LogisticRegression(
        optimizer=optim.SGD(0.01)
    )
)

# Evaluation metrics
accuracy = metrics.Accuracy()
f1 = metrics.F1()

# Process records one by one
for i, row in df.iterrows():
    x = {feature: float(row[feature])
         for feature in features}

    y = row["Label"] == "DDoS"

    # Predict before learning
    prediction = model.predict_one(x)

    if prediction is not None:
        accuracy.update(y, prediction)
        f1.update(y, prediction)

    # Learn from the correct label
    model.learn_one(x, y)

    if (i + 1) % 20000 == 0:
        print("Processed:", i + 1)
        print("Accuracy:", accuracy.get())
        print("F1-score:", f1.get())
        print()

print("Final accuracy:", accuracy.get())
print("Final F1-score:", f1.get())
