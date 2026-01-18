from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}})

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model_bittensor.keras")
SCALER_X_PATH = os.path.join(BASE_DIR, "scaler_X_bittensor.pkl")
SCALER_Y_PATH = os.path.join(BASE_DIR, "scaler_y_bittensor.pkl")
DATASET_PATH = os.path.join(BASE_DIR, "bittensor.xlsx")

WINDOW = 10
USD_TO_IDR = 16909.10

FEATURES = [
    'priceOpen','priceHigh','priceLow','volume',
    'hour_sin','hour_cos','day_sin','day_cos','month_sin','month_cos'
]

model = load_model(MODEL_PATH)
scaler_X = joblib.load(SCALER_X_PATH)
scaler_y = joblib.load(SCALER_Y_PATH)

df = pd.read_excel(DATASET_PATH)
df['timeOpen'] = pd.to_datetime(df['timeOpen'])
df = df.sort_values('timeOpen')

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data:
        return jsonify({"error": "JSON tidak diterima"}), 400

    tanggal = data.get("tanggal")
    jam = data.get("jam")

    if not tanggal or jam is None:
        return jsonify({"error": "Tanggal atau jam kosong"}), 400

    jam = int(jam)
    cutoff = pd.to_datetime(f"{tanggal} {jam}:00:00")

    df_cut = df[df['timeOpen'] < cutoff].tail(WINDOW)

    if len(df_cut) < WINDOW:
        return jsonify({"error": "Data historis tidak cukup"}), 400

    hour = jam
    day = cutoff.weekday()
    month = cutoff.month

    X = df_cut[['priceOpen','priceHigh','priceLow','volume']].copy()
    X['hour_sin'] = np.sin(2*np.pi*hour/24)
    X['hour_cos'] = np.cos(2*np.pi*hour/24)
    X['day_sin'] = np.sin(2*np.pi*day/7)
    X['day_cos'] = np.cos(2*np.pi*day/7)
    X['month_sin'] = np.sin(2*np.pi*month/12)
    X['month_cos'] = np.cos(2*np.pi*month/12)

    X = scaler_X.transform(X.values)
    X = X.reshape(1, WINDOW, len(FEATURES))

    pred_scaled = model.predict(X)
    price_usd = scaler_y.inverse_transform(pred_scaled)[0][0]
    price_idr = price_usd * USD_TO_IDR

    return jsonify({
        "usd": round(float(price_usd), 2),
        "idr": round(float(price_idr), 0)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
