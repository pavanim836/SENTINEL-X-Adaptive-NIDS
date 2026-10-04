# SENTINEL-X – Adaptive Network Intrusion Detection System

## Team
**DriftGuard**

## Overview

SENTINEL-X is an adaptive Network Intrusion Detection System (NIDS) designed to detect DDoS attacks in network traffic and adapt when traffic patterns change.

Traditional machine learning models are usually trained once and may lose performance when network traffic changes over time. SENTINEL-X addresses this problem using online learning, concept drift detection, and automatic model retraining.

## Key Features

- Online machine learning for continuous traffic classification
- DDoS and BENIGN traffic classification
- ADWIN-based concept drift detection
- Automatic model retraining after drift detection
- Recent traffic buffer for adaptive retraining
- DDoS probability estimation
- Feature contribution-based explanation
- Streamlit dashboard for monitoring
- Performance monitoring using Accuracy and F1-score

## System Architecture

```text
CICIDS2017 Network Traffic
          ↓
   Data Preprocessing
          ↓
 Online Logistic Regression
          ↓
      Prediction
          ↓
   ADWIN Monitoring
          ↓
   Drift Detection
          ↓
 Automatic Retraining
          ↓
    Updated Model
          ↓
   Streamlit Dashboard

Technologies Used
Python
Pandas
Scikit-learn
River
Streamlit
CICIDS2017 Dataset
Dataset

The project uses the Friday afternoon DDoS traffic from the CICIDS2017 dataset.

The dataset contains two classes:

BENIGN
DDoS

The project uses selected network-flow features such as:

Destination Port
Flow Duration
Total Forward Packets
Total Backward Packets
Flow Bytes/s
Flow Packets/s
Packet Length Mean
Machine Learning Model

SENTINEL-X uses Logistic Regression from the River library for online learning.

The model follows a sequential process:

Receive a network-flow record.
Predict whether it is BENIGN or DDoS.
Receive the actual label.
Update the model using the new labeled record.
Monitor prediction errors for concept drift.
Concept Drift Detection

SENTINEL-X uses the ADWIN (Adaptive Windowing) algorithm to monitor changes in prediction-error behavior.

When a significant change is detected:

A drift event is recorded.
Recent labeled traffic is collected.
The model is retrained using the recent traffic.
The updated model continues processing new traffic.
Results
Static Model vs Online Model
Model	Accuracy	F1-score
Static Model	70.87%	76.28%
Online Model	98.01%	97.89%
Performance Under Simulated Drift
Model	Accuracy	F1-score
Static Model	88.01%	89.71%
Immediate Online Model	98.17%	98.27%
Delayed Online Model	96.89%	97.06%

The drift experiment uses controlled changes in network-flow features to simulate changing traffic conditions.

Dashboard

The Streamlit dashboard provides:

Traffic processed
Accuracy
F1-score
DDoS probability
Current prediction
Actual label
Drift events
Retraining events
Feature contributions
Recent detections
Adaptive pipeline status