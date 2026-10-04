
import pandas as pd
from river import preprocessing, linear_model, optim
from sklearn.metrics import accuracy_score, f1_score

# Load the simulated drift dataset
df = pd.read_csv("drift_stream.csv")

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
X = df[features].to_dict(orient="records")
y = (df["Label"] == "DDoS").tolist()

# Split into training and testing periods
split = len(df) // 2
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

def create_model():
    return (
        preprocessing.StandardScaler()
        | linear_model.LogisticRegression(
            optimizer=optim.SGD(0.01)
        )
    )

# Create three models
static_model = create_model()
immediate_model = create_model()
delayed_model = create_model()

# Train all models on the first period
for x, label in zip(X_train, y_train):
    static_model.learn_one(x, label)
    immediate_model.learn_one(x, label)
    delayed_model.learn_one(x, label)

# Store predictions and labels
static_preds = []
immediate_preds = []
delayed_preds = []

# Store records waiting for labels
pending = []
DELAY = 100

# Process the second period
for i, (x, label) in enumerate(zip(X_test, y_test)):

    # Make predictions before learning
    static_pred = static_model.predict_one(x)
    immediate_pred = immediate_model.predict_one(x)
    delayed_pred = delayed_model.predict_one(x)

    static_preds.append(bool(static_pred))
    immediate_preds.append(bool(immediate_pred))
    delayed_preds.append(bool(delayed_pred))

    # Immediate model learns after prediction
    immediate_model.learn_one(x, label)

    # Add the record to the delayed model's queue
    pending.append((x, label))

    # Learn only after the specified delay
    if len(pending) > DELAY:
        old_x, old_label = pending.pop(0)
        delayed_model.learn_one(old_x, old_label)

# Learn from remaining records after the test stream ends.
# These updates do not affect the predictions already recorded.
for x, label in pending:
    delayed_model.learn_one(x, label)

# Display results
def evaluate(name, predictions):
    print(f"\n{name}")
    print("Accuracy:", accuracy_score(y_test, predictions))
    print("F1-score:", f1_score(y_test, predictions))

evaluate("STATIC MODEL", static_preds)
evaluate("IMMEDIATE ONLINE MODEL", immediate_preds)
evaluate("DELAYED ONLINE MODEL", delayed_preds)
