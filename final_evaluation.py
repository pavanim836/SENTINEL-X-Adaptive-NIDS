import pandas as pd
from river import preprocessing, linear_model, optim, drift
from sklearn.metrics import accuracy_score, f1_score

# -----------------------------
# LOAD DATA
# -----------------------------

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

df = df.replace(
    [float("inf"), float("-inf")],
    float("nan")
)

df = df.dropna(
    subset=features + ["Label"]
).reset_index(drop=True)


# -----------------------------
# MODEL CREATION
# -----------------------------

def create_model():
    return (
        preprocessing.StandardScaler()
        | linear_model.LogisticRegression(
            optimizer=optim.SGD(0.01)
        )
    )


# -----------------------------
# TRAINING PERIOD
# -----------------------------

TRAIN_SIZE = 50000

train = df.iloc[:TRAIN_SIZE]
test = df.iloc[TRAIN_SIZE:]


# -----------------------------
# CREATE MODELS
# -----------------------------

static_model = create_model()
online_model = create_model()
adaptive_model = create_model()


# -----------------------------
# INITIAL TRAINING
# -----------------------------

print("Training initial models...")

for _, row in train.iterrows():

    x = row[features].to_dict()
    y = row["Label"] == "DDoS"

    static_model.learn_one(x, y)
    online_model.learn_one(x, y)
    adaptive_model.learn_one(x, y)


# -----------------------------
# DRIFT DETECTOR
# -----------------------------

detector = drift.ADWIN()

RETRAIN_WINDOW = 5000

recent_records = []


# -----------------------------
# RESULTS
# -----------------------------

static_predictions = []
online_predictions = []
adaptive_predictions = []

true_labels = []

drift_count = 0
retraining_count = 0


# -----------------------------
# TEST STREAM
# -----------------------------

print("Testing models...")

for index, (_, row) in enumerate(test.iterrows()):

    x = row[features].to_dict()

    actual = row["Label"] == "DDoS"


    # --------------------------------
    # STATIC MODEL
    # --------------------------------

    static_prediction = static_model.predict_one(x)


    # --------------------------------
    # ONLINE MODEL
    # --------------------------------

    online_prediction = online_model.predict_one(x)


    # --------------------------------
    # ADAPTIVE MODEL
    # --------------------------------

    adaptive_prediction = adaptive_model.predict_one(x)


    # Save predictions

    static_predictions.append(
        bool(static_prediction)
    )

    online_predictions.append(
        bool(online_prediction)
    )

    adaptive_predictions.append(
        bool(adaptive_prediction)
    )

    true_labels.append(actual)


    # --------------------------------
    # ADAPTIVE ERROR MONITORING
    # --------------------------------

    error = int(
        bool(adaptive_prediction) != actual
    )

    detector.update(error)


    # Store recent labeled data

    recent_records.append(
        (x, actual)
    )

    if len(recent_records) > RETRAIN_WINDOW:

        recent_records.pop(0)


    # --------------------------------
    # DRIFT DETECTED
    # --------------------------------

    if detector.drift_detected:

        drift_count += 1

        print(
            f"Drift detected at test record "
            f"{index}"
        )


        # Retrain adaptive model

        adaptive_model = create_model()

        for recent_x, recent_y in recent_records:

            adaptive_model.learn_one(
                recent_x,
                recent_y
            )

        retraining_count += 1

        print(
            "Adaptive model retrained."
        )


    # --------------------------------
    # ONLINE UPDATES
    # --------------------------------

    online_model.learn_one(
        x,
        actual
    )

    adaptive_model.learn_one(
        x,
        actual
    )


# -----------------------------
# FINAL RESULTS
# -----------------------------

print("\n==============================")
print("FINAL MODEL COMPARISON")
print("==============================")


static_accuracy = accuracy_score(
    true_labels,
    static_predictions
)

static_f1 = f1_score(
    true_labels,
    static_predictions
)


online_accuracy = accuracy_score(
    true_labels,
    online_predictions
)

online_f1 = f1_score(
    true_labels,
    online_predictions
)


adaptive_accuracy = accuracy_score(
    true_labels,
    adaptive_predictions
)

adaptive_f1 = f1_score(
    true_labels,
    adaptive_predictions
)


print("\nSTATIC MODEL")

print(
    f"Accuracy: {static_accuracy:.4f}"
)

print(
    f"F1-score: {static_f1:.4f}"
)


print("\nONLINE MODEL")

print(
    f"Accuracy: {online_accuracy:.4f}"
)

print(
    f"F1-score: {online_f1:.4f}"
)


print("\nADAPTIVE MODEL")

print(
    f"Accuracy: {adaptive_accuracy:.4f}"
)

print(
    f"F1-score: {adaptive_f1:.4f}"
)


print("\n==============================")

print(
    f"Drift events: {drift_count}"
)

print(
    f"Retraining events: {retraining_count}"
)

print("==============================")