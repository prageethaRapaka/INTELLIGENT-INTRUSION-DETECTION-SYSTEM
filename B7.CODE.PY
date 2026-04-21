# intelligent_intrusion_detection.py
# Design and Development of an Intelligent Intrusion Detection System for Network Security

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier


def generate_network_data(samples=5000):
    np.random.seed(42)
    duration = np.random.exponential(2.0, samples)
    src_bytes = np.random.normal(400, 100, samples)
    dst_bytes = np.random.normal(600, 150, samples)
    packet_rate = np.random.poisson(12, samples)
    login_attempts = np.random.binomial(6, 0.12, samples)

    threat_score = (
        0.4 * login_attempts +
        0.3 * duration +
        0.002 * src_bytes +
        np.random.normal(0, 1, samples)
    )
    
    label = (threat_score > np.percentile(threat_score, 65)).astype(int)
    df = pd.DataFrame({
        "duration": duration,
        "src_bytes": src_bytes,
        "dst_bytes": dst_bytes,
        "packet_rate": packet_rate,
        "login_attempts": login_attempts,
        "label": label
    })

    return df



def create_and_save_dataset():

    df = generate_network_data()
    df.to_csv("intelligent_network_data.csv", index=False)

    print("Synthetic network dataset saved.")


def load_and_inspect_data():

    df = pd.read_csv("intelligent_network_data.csv")

    print("\nDataset Head:")
    print(df.head())

    print("\nDataset Info:")
    print(df.info())

    print("\nDataset Description:")
    print(df.describe())

    return df


def feature_engineering(df):

    df = df.copy()

    df["byte_ratio"] = df["src_bytes"] / (df["dst_bytes"] + 1)

    df["packets_per_second"] = df["packet_rate"] / (df["duration"] + 1)

    df["activity_score"] = (
        0.5 * df["login_attempts"] +
        0.3 * df["packet_rate"] +
        0.001 * df["src_bytes"]
    )

    df["traffic_intensity"] = (
        df["src_bytes"] + df["dst_bytes"]
    ) / (df["duration"] + 1)

    return df



def preprocess_data(df):

    X = df.drop("label", axis=1)
    y = df["label"]

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    joblib.dump(scaler, "scaler.pkl")

    print("Scaler saved as scaler.pkl")

    return X_scaled, y


def train_model(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    class_weights = {0: 1, 1: 4}

    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=15,
        class_weight=class_weights,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    joblib.dump(model, "intelligent_ids_model.pkl")

    print("Improved Intelligent IDS model trained and saved")

    return model, X_test, y_test


def evaluate_model(model, X_test, y_test):

    preds = model.predict(X_test)

    print("\nIDS Accuracy:", accuracy_score(y_test, preds))

    print("\nClassification Report:\n", classification_report(y_test, preds))

    cm = confusion_matrix(y_test, preds)

    plt.figure(figsize=(5, 4))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Greys")

    plt.title("Intelligent IDS Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.tight_layout()

    plt.savefig("intelligent_ids_confusion_matrix.png")

    plt.show()

def detect_intrusion(sample_input):

    scaler = joblib.load("scaler.pkl")
    model = joblib.load("intelligent_ids_model.pkl")

    feature_names = [
        "duration",
        "src_bytes",
        "dst_bytes",
        "packet_rate",
        "login_attempts",
        "byte_ratio",
        "packets_per_second",
        "activity_score",
        "traffic_intensity"
    ]

    sample_df = pd.DataFrame([sample_input], columns=feature_names)

    sample_scaled = scaler.transform(sample_df)

    prediction = model.predict(sample_scaled)

    return prediction[0]



def main():

    if not os.path.exists("intelligent_network_data.csv"):
        create_and_save_dataset()

    df = load_and_inspect_data()

    df = feature_engineering(df)

    X, y = preprocess_data(df)

    model, X_test, y_test = train_model(X, y)

    evaluate_model(model, X_test, y_test)

    sample = X_test[0]

    decision = detect_intrusion(sample)

    print("\nReal-Time Intrusion Detection Output (1=Attack, 0=Normal):", decision)


if __name__ == "__main__":
    main()