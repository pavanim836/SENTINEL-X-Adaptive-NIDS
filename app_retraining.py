import streamlit as st
import pandas as pd
import textwrap

from river import preprocessing, linear_model, optim, drift
from sklearn.metrics import f1_score


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="SENTINEL-X | Adaptive NIDS",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# HTML HELPER
# IMPORTANT: USE st.html(), NOT st.markdown(), FOR HTML CARDS
# ============================================================

def html(content):
    st.html(textwrap.dedent(content))


# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {
    --bg: #f4f6f3;
    --card: #ffffff;
    --ink: #17211b;
    --muted: #7c877f;
    --line: #e3e8e3;
    --accent: #b7e35b;
    --accent-dark: #719b22;
    --green: #4ca56a;
    --blue: #5d8dff;
    --red: #ed6868;
    --orange: #e9a34a;
}

html, body, [class*="css"] {
    font-family: "DM Sans", sans-serif;
}

.stApp {
    background: var(--bg);
    color: var(--ink);
}

.block-container {
    max-width: 1500px;
    padding: 28px 42px 50px 42px;
}

header {
    background: transparent !important;
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}


/* ============================================================
   SIDEBAR
   ============================================================ */

section[data-testid="stSidebar"] {
    background: #17221b;
}

section[data-testid="stSidebar"] * {
    color: #edf3eb !important;
}

section[data-testid="stSidebar"] .stButton > button {
    background: #26352b;
    border: 1px solid #3b4b40;
    color: white !important;
    border-radius: 10px;
}

section[data-testid="stSidebar"] .stButton > button:hover {
    border-color: #b7e35b;
    background: #304036;
}


/* ============================================================
   CARDS
   ============================================================ */

.card {
    background: white;
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 22px;
    box-shadow: 0 7px 25px rgba(35, 48, 38, 0.045);
}

.card-title {
    color: #667169;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    margin-bottom: 15px;
}

.metric-card {
    background: white;
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 20px;
    min-height: 115px;
    box-shadow: 0 7px 25px rgba(35, 48, 38, 0.045);
}

.metric-title {
    color: #748078;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1px;
}

.metric-value {
    font-family: "Space Grotesk", sans-serif;
    font-size: 30px;
    font-weight: 700;
    margin-top: 10px;
}

.metric-note {
    color: #9aa39c;
    font-size: 9px;
    margin-top: 6px;
}


/* ============================================================
   THREAT
   ============================================================ */

.threat-card {
    background: #17221b;
    color: white;
    border-radius: 20px;
    padding: 26px;
    min-height: 260px;
}

.threat-label {
    color: #9ba79e;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.1px;
    text-transform: uppercase;
}

.threat-value {
    font-family: "Space Grotesk", sans-serif;
    font-size: 42px;
    font-weight: 700;
    margin-top: 20px;
}

.threat-safe {
    color: #b7e35b;
}

.threat-danger {
    color: #ff8585;
}

.risk-track {
    height: 8px;
    background: #35433a;
    border-radius: 20px;
    overflow: hidden;
    margin-top: 15px;
}

.risk-fill {
    height: 100%;
    border-radius: 20px;
}


/* ============================================================
   FEATURE BARS
   ============================================================ */

.feature-row {
    margin-bottom: 14px;
}

.feature-head {
    display: flex;
    justify-content: space-between;
    margin-bottom: 6px;
}

.feature-name {
    color: #68736b;
    font-size: 11px;
}

.feature-number {
    font-family: monospace;
    font-size: 11px;
}

.feature-track {
    height: 7px;
    background: #edf0ed;
    border-radius: 10px;
    overflow: hidden;
}

.feature-positive {
    height: 100%;
    background: #ed7777;
    border-radius: 10px;
}

.feature-negative {
    height: 100%;
    background: #6f9aff;
    border-radius: 10px;
}


/* ============================================================
   NETWORK INFO
   ============================================================ */

.info-box {
    background: #f7f9f6;
    border-radius: 12px;
    padding: 13px;
    margin-bottom: 9px;
}

.info-label {
    color: #89938b;
    font-size: 9px;
    text-transform: uppercase;
    letter-spacing: .8px;
}

.info-value {
    color: #202a23;
    font-size: 15px;
    font-weight: 700;
    margin-top: 4px;
}


/* ============================================================
   PIPELINE
   ============================================================ */

.pipeline {
    background: white;
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 22px;
}

.pipeline-step {
    background: #f7f9f6;
    border: 1px solid #e7ece7;
    border-radius: 14px;
    text-align: center;
    padding: 17px 7px;
}

.pipeline-icon {
    font-size: 23px;
}

.pipeline-name {
    color: #5d685f;
    font-size: 10px;
    font-weight: 700;
    margin-top: 7px;
}


/* ============================================================
   STATUS
   ============================================================ */

.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    background: #eaf5d9;
    color: #52791b;
    padding: 9px 14px;
    border-radius: 30px;
    font-size: 11px;
    font-weight: 700;
}

.status-dot {
    width: 8px;
    height: 8px;
    background: #6fbd45;
    border-radius: 50%;
}


/* ============================================================
   BUTTONS
   ============================================================ */

.stButton > button {
    border-radius: 10px;
    border: 1px solid #d9e0d9;
    background: white;
    color: #29332c;
    font-weight: 700;
    min-height: 40px;
}

.stButton > button:hover {
    border-color: #91b34d;
    color: #668b25;
}


/* ============================================================
   FOOTER
   ============================================================ */

.footer {
    text-align: center;
    color: #9aa39c;
    font-size: 10px;
    letter-spacing: .7px;
    margin-top: 35px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# PROJECT CONFIG
# ============================================================

FEATURES = [
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

TRAIN_SIZE = 50000
RETRAIN_WINDOW = 5000


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    data = pd.read_csv("drift_stream.csv")

    data = data.replace(
        [float("inf"), float("-inf")],
        float("nan")
    )

    data = data.dropna(
        subset=FEATURES + ["Label"]
    )

    return data.reset_index(drop=True)


df = load_data()


# ============================================================
# CREATE MODEL
# ============================================================

def create_model():

    return (
        preprocessing.StandardScaler()
        |
        linear_model.LogisticRegression(
            optimizer=optim.SGD(0.01)
        )
    )


# ============================================================
# INITIALIZE MODEL
# ============================================================

if "model" not in st.session_state:

    with st.spinner("Initializing SENTINEL-X detection engine..."):

        model = create_model()

        for _, row in df.iloc[:TRAIN_SIZE].iterrows():

            x = row[FEATURES].to_dict()

            y = row["Label"] == "DDoS"

            model.learn_one(x, y)

        st.session_state.model = model

        st.session_state.detector = drift.ADWIN()

        st.session_state.next_index = TRAIN_SIZE

        st.session_state.processed = 0

        st.session_state.correct = 0

        st.session_state.drift_count = 0

        st.session_state.retraining_count = 0

        st.session_state.predictions = []

        st.session_state.actual_labels = []

        st.session_state.recent_records = []

        st.session_state.history = []

        st.session_state.last_result = None


# ============================================================
# RETRAINING
# ============================================================

def retrain_model():

    records = st.session_state.recent_records

    if len(records) < 100:
        return False

    new_model = create_model()

    training_records = records[-RETRAIN_WINDOW:]

    for x, label in training_records:

        new_model.learn_one(x, label)

    st.session_state.model = new_model

    st.session_state.retraining_count += 1

    return True


# ============================================================
# PROCESS RECORDS
# ============================================================

def process_records(number):

    for _ in range(number):

        model = st.session_state.model

        index = st.session_state.next_index

        if index >= len(df):
            break

        row = df.iloc[index]

        x = row[FEATURES].to_dict()

        actual = row["Label"] == "DDoS"

        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        prediction = model.predict_one(x)

        probability = model.predict_proba_one(x).get(
            True,
            0
        )

        prediction = bool(prediction)

        # ----------------------------------------------------
        # ERROR
        # ----------------------------------------------------

        error = int(prediction != actual)

        # ----------------------------------------------------
        # ADWIN
        # ----------------------------------------------------

        st.session_state.detector.update(error)

        drift_detected = (
            st.session_state.detector.drift_detected
        )

        # ----------------------------------------------------
        # STORE RECENT LABELED TRAFFIC
        # ----------------------------------------------------

        st.session_state.recent_records.append(
            (x, actual)
        )

        if len(st.session_state.recent_records) > RETRAIN_WINDOW:

            st.session_state.recent_records.pop(0)

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        st.session_state.predictions.append(
            prediction
        )

        st.session_state.actual_labels.append(
            actual
        )

        st.session_state.processed += 1

        st.session_state.correct += (
            1 - error
        )

        # ----------------------------------------------------
        # RETRAIN IF DRIFT DETECTED
        # ----------------------------------------------------

        retrained = False

        if drift_detected:

            st.session_state.drift_count += 1

            retrained = retrain_model()

        # Refresh model after possible retraining
        model = st.session_state.model

        # ----------------------------------------------------
        # ONLINE LEARNING
        # ----------------------------------------------------

        model.learn_one(
            x,
            actual
        )

        # ----------------------------------------------------
        # FEATURE CONTRIBUTIONS
        # ----------------------------------------------------

        scaler = model["StandardScaler"]

        classifier = model["LogisticRegression"]

        scaled_x = scaler.transform_one(x)

        contributions = []

        for feature in FEATURES:

            weight = classifier.weights.get(
                feature,
                0
            )

            contribution = (
                weight *
                scaled_x.get(feature, 0)
            )

            contributions.append(
                {
                    "Feature": feature,
                    "Contribution": contribution,
                    "Direction":
                        "Toward DDoS"
                        if contribution > 0
                        else "Toward BENIGN"
                }
            )

        contributions.sort(
            key=lambda item: abs(
                item["Contribution"]
            ),
            reverse=True
        )

        # ----------------------------------------------------
        # SAVE RESULT
        # ----------------------------------------------------

        st.session_state.last_result = {

            "index": index,

            "prediction": prediction,

            "probability": probability,

            "actual": actual,

            "drift": drift_detected,

            "retrained": retrained,

            "contributions": contributions
        }

        # ----------------------------------------------------
        # HISTORY
        # ----------------------------------------------------

        st.session_state.history.append(

            {
                "Record": index,

                "Prediction":
                    "DDoS"
                    if prediction
                    else "BENIGN",

                "Actual":
                    "DDoS"
                    if actual
                    else "BENIGN",

                "Probability": probability,

                "Error": error,

                "Drift": drift_detected,

                "Retrained": retrained
            }
        )

        st.session_state.next_index += 1


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    html("""
    <div style="
        font-size:22px;
        font-weight:700;
        margin-bottom:4px;
    ">
        🛡️ SENTINEL-X
    </div>

    <div style="
        color:#9ca89f;
        font-size:12px;
    ">
        Adaptive Network Intelligence
    </div>
    """)

    st.divider()

    st.markdown("### STREAM CONTROL")

    st.write(
        f"Next record: **{st.session_state.next_index:,}**"
    )

    st.write(
        f"Retraining window: **{RETRAIN_WINDOW:,}**"
    )

    st.divider()

    if st.button(
        "Process next record",
        use_container_width=True
    ):

        process_records(1)

        st.rerun()

    if st.button(
        "Process next 1,000",
        use_container_width=True
    ):

        process_records(1000)

        st.rerun()

    if st.button(
        "Process next 10,000",
        use_container_width=True
    ):

        process_records(10000)

        st.rerun()

    st.divider()

    if st.button(
        "Reset experiment",
        use_container_width=True
    ):

        for key in [
            "model",
            "detector",
            "next_index",
            "processed",
            "correct",
            "drift_count",
            "retraining_count",
            "predictions",
            "actual_labels",
            "recent_records",
            "history",
            "last_result"
        ]:

            st.session_state.pop(
                key,
                None
            )

        st.rerun()

    st.divider()

    st.markdown("### SYSTEM")

    st.success(
        "Online learning active"
    )

    st.success(
        "ADWIN monitoring active"
    )

    st.success(
        "Automatic retraining enabled"
    )


# ============================================================
# METRICS
# ============================================================

processed = st.session_state.processed

if processed > 0:

    accuracy = (
        st.session_state.correct /
        processed
    )

    f1 = f1_score(
        st.session_state.actual_labels,
        st.session_state.predictions
    )

else:

    accuracy = 0

    f1 = 0


# ============================================================
# HEADER
# ============================================================

h1, h2 = st.columns(
    [4, 1]
)

with h1:

    html("""
    <div style="
        font-family:'Space Grotesk',sans-serif;
        font-size:30px;
        font-weight:700;
        letter-spacing:-1px;
    ">
        SENTINEL<span style="color:#719b22;">-X</span>
    </div>

    <div style="
        color:#7c877f;
        font-size:11px;
        font-weight:600;
        letter-spacing:1.5px;
        margin-top:5px;
    ">
        ADAPTIVE NETWORK INTRUSION INTELLIGENCE
    </div>
    """)

with h2:

    html("""
    <div style="
        text-align:right;
        padding-top:5px;
    ">
        <span class="status-pill">
            <span class="status-dot"></span>
            SYSTEM OPERATIONAL
        </span>
    </div>
    """)


st.markdown("<br>", unsafe_allow_html=True)


# ============================================================
# KPI CARDS
# ============================================================

k1, k2, k3, k4, k5 = st.columns(5)

kpis = [

    (
        "TRAFFIC PROCESSED",
        f"{processed:,}",
        "#719b22"
    ),

    (
        "ACCURACY",
        f"{accuracy:.2%}",
        "#4ca56a"
    ),

    (
        "F1 SCORE",
        f"{f1:.2%}",
        "#4ca56a"
    ),

    (
        "DRIFT EVENTS",
        str(st.session_state.drift_count),
        "#e9a34a"
    ),

    (
        "RETRAININGS",
        str(st.session_state.retraining_count),
        "#5d8dff"
    )
]

for column, data in zip(
    [k1, k2, k3, k4, k5],
    kpis
):

    title, value, color = data

    with column:

        html(f"""
        <div class="metric-card">

            <div class="metric-title">
                {title}
            </div>

            <div class="metric-value"
                 style="color:{color};">
                {value}
            </div>

            <div class="metric-note">
                Adaptive detection engine
            </div>

        </div>
        """)


# ============================================================
# MAIN SECTION
# ============================================================

st.markdown("<br>", unsafe_allow_html=True)

left, right = st.columns(
    [0.9, 1.25]
)


# ============================================================
# THREAT ASSESSMENT
# ============================================================

with left:

    result = st.session_state.last_result

    if result is None:

        prediction_text = "WAITING"

        probability = 0

        threat_class = "threat-safe"

        risk_color = "#b7e35b"

    else:

        prediction_text = (
            "DDoS"
            if result["prediction"]
            else "BENIGN"
        )

        probability = (
            result["probability"] * 100
        )

        threat_class = (
            "threat-danger"
            if result["prediction"]
            else "threat-safe"
        )

        if probability >= 70:

            risk_color = "#ed6868"

        elif probability >= 30:

            risk_color = "#e9a34a"

        else:

            risk_color = "#b7e35b"


    html(f"""
    <div class="threat-card">

        <div class="threat-label">
            LATEST THREAT ASSESSMENT
        </div>

        <div class="threat-value {threat_class}">
            {prediction_text}
        </div>

        <div style="
            color:#a5b0a8;
            font-size:12px;
            margin-top:5px;
        ">
            DDoS probability
        </div>

        <div style="
            font-size:25px;
            font-weight:700;
            margin-top:4px;
        ">
            {probability:.2f}%
        </div>

        <div class="risk-track">

            <div class="risk-fill"
                 style="
                    width:{probability}%;
                    background:{risk_color};
                 ">
            </div>

        </div>

        <div style="
            display:flex;
            justify-content:space-between;
            color:#87938b;
            font-size:10px;
            margin-top:8px;
        ">

            <span>LOW RISK</span>

            <span>HIGH RISK</span>

        </div>

    </div>
    """)


# ============================================================
# CURRENT NETWORK EVENT
# ============================================================

with right:

    if result is None:

        html("""
        <div class="card" style="min-height:260px;">

            <div class="card-title">
                CURRENT NETWORK EVENT
            </div>

            <div style="
                color:#7c877f;
                font-size:13px;
                padding-top:35px;
                text-align:center;
            ">
                Process a record from the sidebar to inspect live model output.
            </div>

        </div>
        """)

    else:

        record_value = f'{result["index"]:,}'
        prediction_value = "DDoS" if result["prediction"] else "BENIGN"
        actual_value = "DDoS" if result["actual"] else "BENIGN"

        if result["prediction"] == result["actual"]:
            status_text = "Prediction matched the actual label."
            status_bg = "#d8f5df"
            status_color = "#2f9e5b"
        else:
            status_text = "Prediction did not match the actual label."
            status_bg = "#fde0e0"
            status_color = "#d95353"

        if result["drift"]:
            status_text = "Concept drift detected by ADWIN."
            status_bg = "#fff0d6"
            status_color = "#c77b19"
        elif result["retrained"]:
            status_text = "Model automatically retrained."
            status_bg = "#e2edff"
            status_color = "#4d78d8"

        html(f"""
        <div class="card" style="min-height:260px;">

            <div class="card-title">
                CURRENT NETWORK EVENT
            </div>

            <div style="
                display:grid;
                grid-template-columns:1fr 1fr 1fr;
                gap:14px;
                margin-top:4px;
            ">

                <div style="
                    background:#f7f9f6;
                    border-radius:12px;
                    padding:15px;
                ">
                    <div style="
                        color:#89938b;
                        font-size:10px;
                        text-transform:uppercase;
                        letter-spacing:.8px;
                    ">
                        Record
                    </div>
                    <div style="
                        color:#202a23;
                        font-family:'Space Grotesk',sans-serif;
                        font-size:24px;
                        font-weight:700;
                        margin-top:5px;
                    ">
                        {record_value}
                    </div>
                </div>

                <div style="
                    background:#f7f9f6;
                    border-radius:12px;
                    padding:15px;
                ">
                    <div style="
                        color:#89938b;
                        font-size:10px;
                        text-transform:uppercase;
                        letter-spacing:.8px;
                    ">
                        Prediction
                    </div>
                    <div style="
                        color:{'#ed6868' if result["prediction"] else '#719b22'};
                        font-family:'Space Grotesk',sans-serif;
                        font-size:24px;
                        font-weight:700;
                        margin-top:5px;
                    ">
                        {prediction_value}
                    </div>
                </div>

                <div style="
                    background:#f7f9f6;
                    border-radius:12px;
                    padding:15px;
                ">
                    <div style="
                        color:#89938b;
                        font-size:10px;
                        text-transform:uppercase;
                        letter-spacing:.8px;
                    ">
                        Actual
                    </div>
                    <div style="
                        color:#202a23;
                        font-family:'Space Grotesk',sans-serif;
                        font-size:24px;
                        font-weight:700;
                        margin-top:5px;
                    ">
                        {actual_value}
                    </div>
                </div>

            </div>

            <div style="
                background:{status_bg};
                color:{status_color};
                border-radius:11px;
                padding:13px 16px;
                margin-top:18px;
                font-size:12px;
                font-weight:600;
            ">
                {status_text}
            </div>

        </div>
        """)


# ============================================================
# TRAFFIC ACTIVITY
# ============================================================

st.markdown("<br>", unsafe_allow_html=True)

activity, engine = st.columns(
    [1.45, 1]
)

with activity:

    html("""
    <div class="card">

        <div class="card-title">
            TRAFFIC ACTIVITY
        </div>

    </div>
    """)

    history_df = pd.DataFrame(
        st.session_state.history
    )

    if not history_df.empty:

        chart_df = history_df[
            ["Record", "Probability"]
        ].copy()

        chart_df["Probability"] = (
            chart_df["Probability"] * 100
        )

        chart_df = chart_df.set_index(
            "Record"
        )

        chart_df.columns = [
            "DDoS probability (%)"
        ]

        st.line_chart(
            chart_df,
            height=250
        )

    else:

        st.info(
            "The traffic chart will appear after processing records."
        )


# ============================================================
# ADAPTIVE ENGINE
# ============================================================

with engine:

    html("""
    <div class="card">

        <div class="card-title">
            ADAPTIVE ENGINE
        </div>

    </div>
    """)

    st.markdown(
        "**Online learning**  \n"
        "🟢 ACTIVE"
    )

    st.markdown(
        "**ADWIN detector**  \n"
        "🟢 MONITORING"
    )

    st.markdown(
        f"**Recent traffic buffer**  \n"
        "🔵 "
        f"{len(st.session_state.recent_records):,}"
    )

    st.markdown(
        f"**Initial training**  \n"
        "🟢 "
        f"{TRAIN_SIZE:,} records"
    )

    st.markdown(
        f"**Retraining window**  \n"
        "🟠 "
        f"{RETRAIN_WINDOW:,} records"
    )


# ============================================================
# AI EXPLANATION
# ============================================================

st.markdown("<br>", unsafe_allow_html=True)

explain, network = st.columns(
    [1.35, 1]
)

with explain:

    html("""
    <div class="card">

        <div class="card-title">
            AI EXPLANATION — FEATURE CONTRIBUTIONS
        </div>

    </div>
    """)

    if result is None:

        st.info(
            "Feature contributions will appear after processing a record."
        )

    else:

        contributions = result[
            "contributions"
        ]

        maximum = max(
            abs(item["Contribution"])
            for item in contributions
        ) or 1

        for item in contributions:

            value = item[
                "Contribution"
            ]

            width = (
                abs(value) /
                maximum *
                100
            )

            if value >= 0:

                bar_class = (
                    "feature-positive"
                )

                number_color = "#d95d5d"

            else:

                bar_class = (
                    "feature-negative"
                )

                number_color = "#5d8dff"

            html(f"""
            <div class="feature-row">

                <div class="feature-head">

                    <span class="feature-name">
                        {item["Feature"]}
                    </span>

                    <span class="feature-number"
                          style="
                            color:{number_color};
                          ">
                        {value:+.3f}
                    </span>

                </div>

                <div class="feature-track">

                    <div class="{bar_class}"
                         style="
                            width:{width}%;
                         ">
                    </div>

                </div>

            </div>
            """)

        st.caption(
            "Red pushes the prediction toward DDoS; "
            "blue pushes it toward BENIGN."
        )


# ============================================================
# NETWORK PROFILE
# ============================================================

with network:

    html("""
    <div class="card">

        <div class="card-title">
            NETWORK PROFILE
        </div>

    </div>
    """)

    if result is None:

        st.info(
            "Process a record to inspect its traffic features."
        )

    else:

        row = df.iloc[
            result["index"]
        ]

        display_features = [

            (
                "Destination Port",
                row["Destination Port"]
            ),

            (
                "Flow Duration",
                row["Flow Duration"]
            ),

            (
                "Forward Packets",
                row["Total Fwd Packets"]
            ),

            (
                "Backward Packets",
                row["Total Backward Packets"]
            ),

            (
                "Forward Length",
                row["Total Length of Fwd Packets"]
            ),

            (
                "Backward Length",
                row["Total Length of Bwd Packets"]
            ),

            (
                "Flow Bytes/s",
                row["Flow Bytes/s"]
            ),

            (
                "Flow Packets/s",
                row["Flow Packets/s"]
            )
        ]

        for label, value in display_features:

            try:

                if isinstance(
                    value,
                    float
                ):

                    formatted = (
                        f"{value:,.2f}"
                    )

                else:

                    formatted = (
                        f"{value:,}"
                    )

            except Exception:

                formatted = str(value)

            html(f"""
            <div class="info-box">

                <div class="info-label">
                    {label}
                </div>

                <div class="info-value">
                    {formatted}
                </div>

            </div>
            """)


# ============================================================
# ADAPTIVE PIPELINE
# ============================================================

st.markdown("<br>", unsafe_allow_html=True)

html("""
<div class="pipeline">

    <div class="card-title">
        ADAPTIVE ML PIPELINE
    </div>

</div>
""")

p1, p2, p3, p4 = st.columns(4)

pipeline = [

    (
        p1,
        "📡",
        "NETWORK TRAFFIC"
    ),

    (
        p2,
        "🧠",
        "ONLINE PREDICTION"
    ),

    (
        p3,
        "📊",
        "ADWIN MONITORING"
    ),

    (
        p4,
        "♻️",
        "MODEL RETRAINING"
    )
]

for column, icon, name in pipeline:

    with column:

        html(f"""
        <div class="pipeline-step">

            <div class="pipeline-icon">
                {icon}
            </div>

            <div class="pipeline-name">
                {name}
            </div>

        </div>
        """)


st.caption(
    "Traffic → Prediction → Error monitoring → "
    "ADWIN drift detection → Model adaptation"
)


# ============================================================
# RECENT DETECTIONS
# ============================================================

st.markdown("<br>", unsafe_allow_html=True)

html("""
<div class="card">

    <div class="card-title">
        RECENT DETECTIONS
    </div>

</div>
""")

if not history_df.empty:

    recent = history_df.tail(
        8
    ).copy()

    recent = recent[
        [
            "Record",
            "Prediction",
            "Actual",
            "Probability",
            "Error",
            "Drift",
            "Retrained"
        ]
    ]

    recent["Probability"] = (
        recent["Probability"]
        .map(
            lambda x:
            f"{x:.2%}"
        )
    )

    recent["Error"] = (
        recent["Error"]
        .map(
            lambda x:
            "Yes" if x else "No"
        )
    )

    recent["Drift"] = (
        recent["Drift"]
        .map(
            lambda x:
            "Detected" if x else "—"
        )
    )

    recent["Retrained"] = (
        recent["Retrained"]
        .map(
            lambda x:
            "Yes" if x else "—"
        )
    )

    st.dataframe(
        recent.iloc[::-1],
        use_container_width=True,
        hide_index=True
    )

else:

    st.info(
        "Recent predictions will appear here."
    )


# ============================================================
# FOOTER
# ============================================================

html("""
<div class="footer">

    SENTINEL-X • ONLINE LEARNING • ADWIN DRIFT DETECTION
    • AUTOMATIC MODEL ADAPTATION

    <br><br>

    CICIDS2017 NETWORK INTRUSION DETECTION
    RESEARCH PROTOTYPE

</div>
""")