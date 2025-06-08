from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import pandas as pd
import numpy as np
import time
import joblib
import os

app = FastAPI(title="Heart Disease Prediction API")

# -------------------- METRIK PROMETHEUS --------------------
PREDICTIONS_TOTAL = Counter("heart_model_predictions_total", "Total predictions made.")
PREDICTION_ERRORS_TOTAL = Counter(
    "heart_model_prediction_errors_total", "Prediction errors."
)
PREDICTION_LATENCY_SECONDS = Histogram(
    "heart_model_prediction_latency_seconds", "Prediction latency in seconds."
)
LAST_INPUT_FEATURES_COUNT = Gauge(
    "heart_model_last_input_features_count", "Number of features in last input."
)
PREDICTION_SCORE_DISTRIBUTION = Histogram(
    "heart_model_prediction_score_distribution", "Prediction score distribution."
)
PREDICT_ENDPOINT_REQUESTS_TOTAL = Counter(
    "heart_model_predict_requests_total", "Total /predict requests."
)
INVALID_INPUT_STRUCTURE_TOTAL = Counter(
    "heart_model_invalid_input_total", "Invalid input count."
)
ACTIVE_MODEL_VERSION = Gauge(
    "heart_model_active_version", "Version of active model", ["model_version_label"]
)
INPUT_FEATURE_VALUE_DISTRIBUTION = Histogram(
    "heart_model_input_value_distribution",
    "Distribution of input values",
    ["feature_name"],
)
PREDICTIONS_BY_CLASS_TOTAL = Counter(
    "heart_model_predictions_by_class", "Predictions per class", ["predicted_class"]
)

# -------------------- KONFIGURASI --------------------
MODEL_PATH = "/app/model.pkl"
MODEL_VERSION_LABEL = "1.0.0"

EXPECTED_FEATURE_NAMES = [
    "Age",
    "RestingBP",
    "Cholesterol",
    "FastingBS",
    "MaxHR",
    "Oldpeak",
    "Sex_M",
    "ChestPainType_ATA",
    "ChestPainType_NAP",
    "ChestPainType_TA",
    "RestingECG_Normal",
    "RestingECG_ST",
    "ExerciseAngina_Y",
    "ST_Slope_Flat",
    "ST_Slope_Up",
]

print(f"✅ Fitur yang diharapkan model: {EXPECTED_FEATURE_NAMES}")

model = None
try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ File model tidak ditemukan di {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    ACTIVE_MODEL_VERSION.labels(model_version_label=MODEL_VERSION_LABEL).set(1)
    print(f"✅ Model versi {MODEL_VERSION_LABEL} berhasil dimuat.")
except Exception as e:
    print(f"❌ Gagal memuat model: {e}")
    ACTIVE_MODEL_VERSION.labels(model_version_label="load_failed").set(0)

# -------------------- PROMETHEUS METRICS SERVER --------------------
try:
    start_http_server(8000)
    print("✅ Prometheus metrics aktif di :8000/metrics")
except Exception as e:
    print(f"❌ Gagal start Prometheus: {e}")


# -------------------- ENDPOINT PREDIKSI --------------------
@app.post("/predict")
async def predict(request: Request):
    PREDICT_ENDPOINT_REQUESTS_TOTAL.inc()

    if model is None:
        PREDICTION_ERRORS_TOTAL.inc()
        return JSONResponse(status_code=500, content={"error": "Model tidak tersedia."})

    try:
        data_json = await request.json()
        if not isinstance(data_json, dict):
            INVALID_INPUT_STRUCTURE_TOTAL.inc()
            return JSONResponse(
                status_code=400, content={"error": "Input harus berupa JSON object."}
            )

        missing = [f for f in EXPECTED_FEATURE_NAMES if f not in data_json]
        if missing:
            INVALID_INPUT_STRUCTURE_TOTAL.inc()
            return JSONResponse(
                status_code=400, content={"error": f"Fitur hilang: {missing}"}
            )

        input_data = {key: data_json[key] for key in EXPECTED_FEATURE_NAMES}
        df = pd.DataFrame([input_data])
        LAST_INPUT_FEATURES_COUNT.set(len(df.columns))

        for f in ["Age", "Oldpeak", "ST_Slope_Up"]:
            try:
                INPUT_FEATURE_VALUE_DISTRIBUTION.labels(feature_name=f).observe(
                    float(df[f].iloc[0])
                )
            except Exception:
                pass

        start = time.time()
        pred = model.predict(df)
        prob = model.predict_proba(df) if hasattr(model, "predict_proba") else None
        latency = time.time() - start

        PREDICTION_LATENCY_SECONDS.observe(latency)
        PREDICTIONS_TOTAL.inc()

        pred_class = str(pred[0])
        PREDICTIONS_BY_CLASS_TOTAL.labels(predicted_class=pred_class).inc()

        result = {"prediction": pred_class, "latency_seconds": latency}

        if prob is not None:
            result["probabilities"] = prob[0].tolist()
            result["predicted_score"] = (
                float(prob[0][1]) if len(prob[0]) > 1 else float(prob[0][0])
            )
            PREDICTION_SCORE_DISTRIBUTION.observe(result["predicted_score"])

        return JSONResponse(content=result)

    except Exception as e:
        PREDICTION_ERRORS_TOTAL.inc()
        INVALID_INPUT_STRUCTURE_TOTAL.inc()
        return JSONResponse(status_code=400, content={"error": str(e)})
