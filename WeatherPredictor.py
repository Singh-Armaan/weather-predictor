# weather_predictor_api.py
# Train a "Rain Tomorrow?" model on weatherAUS.csv and predict for real cities via OpenWeatherMap.

import os, sys, joblib, requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from typing import Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

DATA_PATH = Path("weatherAUS.csv")
MODEL_PATH = Path("rain_model.joblib")

# --------------------------- 1) DATA LOADING --------------------------------
def load_data(csv_path: Path) -> pd.DataFrame:
    """Read the Kaggle CSV and keep a compact set of useful columns."""
    if not csv_path.exists():
        print(f"Missing {csv_path}. Put weatherAUS.csv in this folder.")
        sys.exit(1)

    df = pd.read_csv(csv_path)

    # Choose high-signal columns (+ two engineered features we'll add later)
    cols = [
        "MinTemp","MaxTemp","Rainfall",
        "Evaporation","Sunshine",
        "WindGustSpeed","WindSpeed9am","WindSpeed3pm",
        "Humidity9am","Humidity3pm",
        "Pressure9am","Pressure3pm",
        "Cloud9am","Cloud3pm",
        "Temp9am","Temp3pm",
        "RainToday","RainTomorrow"
    ]
    df = df[cols].copy()

    # Map Yes/No -> 1/0 for classification
    yn = {"Yes": 1, "No": 0}
    df["RainToday"] = df["RainToday"].map(yn)
    df["RainTomorrow"] = df["RainTomorrow"].map(yn)

    # Drop rows with no label (target)
    df = df.dropna(subset=["RainTomorrow"]).reset_index(drop=True)

    # Simple engineered features
    df["TempRange"] = df["MaxTemp"] - df["MinTemp"]               # spread
    df["PressureDelta"] = df["Pressure3pm"] - df["Pressure9am"]   # falling pressure -> rain
    return df

# --------------------------- 2) SPLIT & PREPROCESS --------------------------
def make_train_test(df: pd.DataFrame):
    """Split data and set up preprocessing (impute missing, scale numeric)."""
    y = df["RainTomorrow"].astype(int)
    X = df.drop(columns=["RainTomorrow"])

    # We'll use all columns except the target
    feature_names = list(X.columns)

    # Preprocessor: median imputation for missing values + standard scaling
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), feature_names)
        ],
        remainder="drop"
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X[feature_names], y, test_size=0.2, random_state=42, stratify=y
    )
    return pre, X_train, X_test, y_train, y_test, feature_names

# --------------------------- 3) MODEL PIPELINE ------------------------------
def build_model(preprocessor):
    """Logistic Regression is fast, works well here, and is interpretable."""
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    pipe = Pipeline([("pre", preprocessor), ("clf", clf)])
    return pipe

# --------------------------- 4) TRAIN & EVAL --------------------------------
def train_and_evaluate():
    """Train on Kaggle data, evaluate, save model pipeline to disk."""
    df = load_data(DATA_PATH)
    pre, X_train, X_test, y_train, y_test, feats = make_train_test(df)
    model = build_model(pre)
    model.fit(X_train, y_train)

    # Predict
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    print("\n=== Classification Report ===")
    print(classification_report(y_test, pred, target_names=["No Rain", "Rain"]))

    try:
        auc = roc_auc_score(y_test, proba)
        print(f"ROC AUC: {auc:.3f}")
    except Exception:
        pass

    # Confusion matrix
    cm = confusion_matrix(y_test, pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Rain","Rain"],
                yticklabels=["No Rain","Rain"])
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.title("Rain Tomorrow — Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    print("[saved] confusion_matrix.png")
    plt.show()

    # Save the whole pipeline (preprocessor + model) + feature list
    joblib.dump({"model": model, "features": feats}, MODEL_PATH)
    print(f"[saved] {MODEL_PATH}")

# --------------------------- 5) API FETCH -----------------------------------
def get_api_key() -> str:
    key = os.getenv("OPENWEATHER_API_KEY", "")
    if not key:
        print("Set OPENWEATHER_API_KEY in your environment.")
        sys.exit(1)
    return key

def fetch_weather_from_api(city: str, units: str = "metric") -> Dict[str, Any]:
    """
    Call OpenWeatherMap 'current weather' endpoint for a city.
    units='metric' -> Celsius, m/s; 'imperial' -> Fahrenheit, mph.
    """
    key = get_api_key()
    url = "http://api.openweathermap.org/data/2.5/weather"
    params = {"q": city, "appid": key, "units": units}
    r = requests.get(url, params=params, timeout=15)
    if r.status_code != 200:
        raise RuntimeError(f"API error: {r.status_code} {r.text}")
    return r.json()

def api_to_feature_row(api_json: Dict[str, Any]) -> Dict[str, float]:
    """
    Map API fields to the features our model expects.
    Some fields are not available from the API; we leave them as NaN so the
    model's imputer fills them with training medians.
    """
    main = api_json.get("main", {})
    wind = api_json.get("wind", {})
    clouds = api_json.get("clouds", {})
    rain = api_json.get("rain", {})  # may have '1h' or '3h'

    # Helper getters with defaults
    def g(obj, key, default=np.nan):
        return obj.get(key, default)

    # Map to training features
    row = {
        "MinTemp": g(main, "temp_min"),                  # °C (metric) or °F (imperial)
        "MaxTemp": g(main, "temp_max"),
        "Rainfall": g(rain, "1h", g(rain, "3h", 0.0)),  # not strictly "yesterday", but proxy
        "Evaporation": np.nan,                           # not provided by API
        "Sunshine": np.nan,                              # not provided by API
        "WindGustSpeed": g(wind, "gust", g(wind, "speed", np.nan)),
        "WindSpeed9am": g(wind, "speed"),
        "WindSpeed3pm": g(wind, "speed"),
        "Humidity9am": g(main, "humidity"),
        "Humidity3pm": g(main, "humidity"),
        "Pressure9am": g(main, "pressure"),
        "Pressure3pm": g(main, "pressure"),
        "Cloud9am": g(clouds, "all", np.nan) / 10.0,    # API gives 0..100%; dataset uses 0..10 oktas
        "Cloud3pm": g(clouds, "all", np.nan) / 10.0,
        "Temp9am": g(main, "temp"),
        "Temp3pm": g(main, "temp"),
        # If API reports any rain amount now, treat as rainy today (1), else 0
        "RainToday": 1 if (("1h" in rain and rain["1h"] > 0) or ("3h" in rain and rain["3h"] > 0)) else 0,
    }

    # Engineered features (OK if they become NaN; imputer handles it)
    row["TempRange"] = (row["MaxTemp"] - row["MinTemp"]) if (pd.notna(row["MaxTemp"]) and pd.notna(row["MinTemp"])) else np.nan
    row["PressureDelta"] = (row["Pressure3pm"] - row["Pressure9am"]) if (pd.notna(row["Pressure3pm"]) and pd.notna(row["Pressure9am"])) else np.nan
    return row

# --------------------------- 6) PREDICT -------------------------------------
def predict_for_city(city: str, units: str = "metric"):
    """Load the trained model, fetch live weather for a city, and predict."""
    if not MODEL_PATH.exists():
        print("Model not found. Train first:\n  python weather_predictor_api.py train")
        sys.exit(1)

    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    features = bundle["features"]

    api_json = fetch_weather_from_api(city, units=units)
    row = api_to_feature_row(api_json)

    # Build a one-row DataFrame in the same feature order used in training
    X_one = pd.DataFrame([row], columns=features)

    # Predict probability of rain tomorrow
    p = model.predict_proba(X_one)[0, 1]
    label = "Rain" if p >= 0.5 else "No Rain"

    # Units note (for display only)
    units_note = "°C, m/s" if units == "metric" else "°F, mph"
    print(f"\nCity: {city}  (units: {units_note})")
    print(f"Prediction: {label}  |  Probability of rain = {p:.1%}")

# --------------------------- 7) CLI -----------------------------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Rain Tomorrow predictor with live API input.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Train and evaluate the model on weatherAUS.csv")

    p_pred = sub.add_parser("predict", help="Predict for a real city via OpenWeatherMap API")
    p_pred.add_argument("--city", default="Dallas", help="City name (e.g., 'Dallas' or 'Sydney,AU')")
    p_pred.add_argument("--units", choices=["metric","imperial"], default="metric", help="Units for API (metric=°C/m/s, imperial=°F/mph)")

    args = ap.parse_args()

    if args.cmd == "train":
        train_and_evaluate()
    else:
        predict_for_city(args.city, units=args.units)
