import pandas as pd
from river import compose
from river import preprocessing
from river import linear_model
from river import optim
from river import metrics

df = pd.read_csv("cleaned_ddos.csv")
df.columns = df.columns.str.strip()

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

# Remove any remaining infinite or missing values
import numpy as np
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(subset=features + ["Label"], inplace=True)

# Use the first 60% for initial training
split = int(len(df) * 0.6)
train = df.iloc[:split]
test = df.iloc[split:]

def create_model():
    return compose.Pipeline(
        preprocessing.StandardScaler(),
        linear_model.LogisticRegression(
            optimizer=optim.SGD(0.01)
        )
    )

static_model = create_model()
online_model = create_model()

# Train both models on identical initial data
for row in train.to_dict("records"):
    x = {f: float(row[f]) for f in features}
    y = row["Label"] == "DDoS"

    static_model.learn_one(x, y)
    online_model.learn_one(x, y)

# Evaluate only on the remaining 40%
static_accuracy = metrics.Accuracy()
online_accuracy = metrics.Accuracy()

static_f1 = metrics.F1()
online_f1 = metrics.F1()

for row in test.to_dict("records"):
    x = {f: float(row[f]) for f in features}
    y = row["Label"] == "DDoS"

    # Static model: predict, never update
    pred_static = static_model.predict_one(x)
    static_accuracy.update(y, pred_static)
    static_f1.update(y, pred_static)

    # Online model: predict, then update
    pred_online = online_model.predict_one(x)
    online_accuracy.update(y, pred_online)
    online_f1.update(y, pred_online)

    online_model.learn_one(x, y)

print("Test records:", len(test))
print("\nSTATIC MODEL")
print("Accuracy:", static_accuracy.get())
print("F1-score:", static_f1.get())

print("\nONLINE MODEL")
print("Accuracy:", online_accuracy.get())
print("F1-score:", online_f1.get())