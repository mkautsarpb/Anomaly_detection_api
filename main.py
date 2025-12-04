# main.py
import os
import joblib
import requests
import shutil
from typing import Optional, Dict, Any
from fastapi import FastAPI, Header, HTTPException, UploadFile, File
from pydantic import BaseModel
from fastapi.responses import JSONResponse

# -------------------------
# ENV / CONFIG
# -------------------------
GITHUB_REPO = os.getenv("GITHUB_REPO")                # e.g. "username/model-registry"
GITHUB_BRANCH = os.getenv("GITHUB_BRANCH", "main")
FASTAPI_API_KEY = os.getenv("FASTAPI_API_KEY")        # e.g. "secret123"

# Temporary places to save downloaded/uploaded models
TMP_DIR = "/tmp/anomaly_api"
os.makedirs(TMP_DIR, exist_ok=True)
TMP_MODEL_PATH = os.path.join(TMP_DIR, "model.joblib")
TMP_METADATA_PATH = os.path.join(TMP_DIR, "metadata.json")

# FEATURES order must match training pipeline
FEATURES = [
    "air_temperature_k",
    "process_temperature_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
    "hours_since_last_maintenance",
]

# -------------------------
# APP + STATE
# -------------------------
app = FastAPI(title="Anomaly Detection API")
_current_model: Optional[Dict[str, Any]] = None   # will hold bundle with scaler, isolation_forest, one_class_svm, thresholds, etc.
_current_metadata: Optional[Dict[str, Any]] = None

# -------------------------
# AUTH
# -------------------------
def check_auth(x_api_key: Optional[str]):
    """Raise 401 if API key is configured and doesn't match."""
    if FASTAPI_API_KEY:
        if not x_api_key:
            raise HTTPException(status_code=401, detail="Missing x-api-key header")
        if x_api_key != FASTAPI_API_KEY:
            raise HTTPException(status_code=401, detail="Unauthorized")


# -------------------------
# SCHEMAS
# -------------------------
class PredictRequest(BaseModel):
    air_temperature_k: float
    process_temperature_k: float
    rotational_speed_rpm: float
    torque_nm: float
    tool_wear_min: float
    hours_since_last_maintenance: float


# -------------------------
# UTIL: classify by thresholds
# -------------------------
def classify_iso_score(score: float, t_low: Optional[float], t_med: Optional[float], t_high: Optional[float]) -> str:
    """Return a human readable anomaly level based on iso score thresholds.
    The same logic as in your DAG: higher score -> more normal.
    """
    if t_low is None or t_med is None or t_high is None:
        # fallback simple rule
        if score > 0:
            return "Normal"
        elif score > -0.5:
            return "Low Risk"
        elif score > -1.0:
            return "Medium Risk"
        else:
            return "High Risk"

    if score > t_low:
        return "Normal"
    elif score > t_med:
        return "Low Risk"
    elif score > t_high:
        return "Medium Risk"
    else:
        return "High Risk"


# -------------------------
# MODEL LOADING HELPERS
# -------------------------
def _load_joblib_file(path: str):
    """Load a joblib file and return its content."""
    try:
        return joblib.load(path)
    except Exception as e:
        raise RuntimeError(f"Failed to load joblib file {path}: {e}")


def _normalize_bundle(obj: Any) -> Dict[str, Any]:
    """
    Accept several bundle formats:
    - dict with keys 'scaler', 'isolation_forest', 'one_class_svm', 'lowered_threshold', 'anomaly_level_thresholds'
    - a single sklearn model (in which case we cannot run full ensemble — raise)
    Returns normalized dict with expected keys or raises.
    """
    if isinstance(obj, dict):
        # expected bundle
        bundle = {}
        bundle["scaler"] = obj.get("scaler")
        bundle["isolation_forest"] = obj.get("isolation_forest")
        bundle["one_class_svm"] = obj.get("one_class_svm")
        # lowered_threshold might be stored as {"lowered_threshold": value} in DAG; normalize
        if "lowered_threshold" in obj:
            val = obj["lowered_threshold"]
            # if it's a dict container
            if isinstance(val, dict) and "lowered_threshold" in val:
                bundle["lowered_threshold"] = val["lowered_threshold"]
            else:
                bundle["lowered_threshold"] = val
        else:
            bundle["lowered_threshold"] = None

        # anomaly_level_thresholds may be dict {"t_low":..}
        if "anomaly_level_thresholds" in obj:
            alt = obj["anomaly_level_thresholds"]
            if isinstance(alt, dict):
                bundle["anomaly_level_thresholds"] = {
                    "t_low": alt.get("t_low"),
                    "t_med": alt.get("t_med"),
                    "t_high": alt.get("t_high"),
                }
            else:
                bundle["anomaly_level_thresholds"] = None
        else:
            bundle["anomaly_level_thresholds"] = None

        return bundle

    # If obj looks like a single sklearn model (e.g., IsolationForest), we cannot run ensemble
    raise RuntimeError("Loaded joblib is not a recognized model bundle. Expect dict bundle with scaler/isolation_forest/one_class_svm.")


def _set_current_model(bundle: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None):
    global _current_model, _current_metadata
    _current_model = bundle
    _current_metadata = metadata


# -------------------------
# ENDPOINTS
# -------------------------
@app.get("/")
def root():
    return {"status": "ok", "message": "Anomaly Detection API is running"}


@app.get("/api/ml/anomaly/model_info")
def model_info(x_api_key: Optional[str] = Header(None)):
    """Return basic info about currently loaded model bundle."""
    check_auth(x_api_key)
    if _current_model is None:
        return JSONResponse(status_code=204, content={"detail": "No model loaded"})
    info = {
        "has_scaler": _current_model.get("scaler") is not None,
        "has_isolation_forest": _current_model.get("isolation_forest") is not None,
        "has_one_class_svm": _current_model.get("one_class_svm") is not None,
        "lowered_threshold": _current_model.get("lowered_threshold"),
        "anomaly_level_thresholds": _current_model.get("anomaly_level_thresholds"),
        "metadata": _current_metadata,
    }
    return info


@app.get("/api/ml/anomaly/load_latest_model")
def load_latest_model(x_api_key: Optional[str] = Header(None)):
    """
    Load latest model.joblib from GitHub raw (repo must have model_upload/model.joblib).
    Expects GITHUB_REPO env var set (username/repo).
    """
    check_auth(x_api_key)

    if not GITHUB_REPO:
        raise HTTPException(500, "GITHUB_REPO not configured in environment")

    base_url = f"https://raw.githubusercontent.com/{GITHUB_REPO}/{GITHUB_BRANCH}/model_upload"

    # fetch metadata.json
    murl = f"{base_url}/metadata.json"
    r = requests.get(murl, timeout=30)
    if r.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Failed to fetch metadata.json from {murl}: {r.status_code}")

    metadata = r.json()

    # fetch model.joblib
    model_url = f"{base_url}/model.joblib"
    r2 = requests.get(model_url, timeout=60)
    if r2.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Failed to fetch model.joblib from {model_url}: {r2.status_code}")

    # save to temp
    with open(TMP_MODEL_PATH, "wb") as f:
        f.write(r2.content)

    # load joblib
    loaded = None
    try:
        loaded = _load_joblib_file(TMP_MODEL_PATH)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    try:
        bundle = _normalize_bundle(loaded)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model format error: {e}")

    _set_current_model(bundle, metadata)
    return {"status": "success", "message": "Model loaded successfully", "model_version": metadata.get("version")}


@app.post("/api/ml/anomaly/upload_model")
async def upload_model(model_file: UploadFile = File(...), metadata_file: UploadFile = File(...), x_api_key: Optional[str] = Header(None)):
    """
    Accept a multipart upload (model_file=joblib, metadata_file=json).
    We store locally under /tmp; we do not push to GitHub here (Airflow should handle pushing to model-registry).
    """
    check_auth(x_api_key)

    # save model
    with open(TMP_MODEL_PATH, "wb") as f:
        content = await model_file.read()
        f.write(content)

    # save metadata
    with open(TMP_METADATA_PATH, "wb") as f:
        f.write(await metadata_file.read())

    # load model and set as current if possible
    try:
        loaded = _load_joblib_file(TMP_MODEL_PATH)
        bundle = _normalize_bundle(loaded)
        md = None
        try:
            import json as _json
            with open(TMP_METADATA_PATH, "r") as md_f:
                md = _json.load(md_f)
        except Exception:
            md = None
        _set_current_model(bundle, md)
    except Exception as e:
        # still accept upload but report that model couldn't be used as runtime bundle
        return JSONResponse(status_code=202, content={"status": "uploaded", "warning": f"model uploaded but couldn't be loaded as bundle: {e}"})

    return {"status": "uploaded_and_loaded", "model_version": _current_metadata.get("version") if _current_metadata else None}


@app.post("/api/ml/anomaly/predict")
def predict(payload: PredictRequest, x_api_key: Optional[str] = Header(None)):
    """
    Predict endpoint using the same FEATURES order as training.
    Accepts the 6 features (floats) and returns iso_score, svm_pred, anomaly flag, and anomaly_level.
    """
    check_auth(x_api_key)

    if _current_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Call /api/ml/anomaly/load_latest_model first.")

    # build feature array in the exact order used for training
    arr = [
        payload.air_temperature_k,
        payload.process_temperature_k,
        payload.rotational_speed_rpm,
        payload.torque_nm,
        payload.tool_wear_min,
        payload.hours_since_last_maintenance,
    ]

    import numpy as np
    X = np.array(arr).reshape(1, -1)

    scaler = _current_model.get("scaler")
    iso = _current_model.get("isolation_forest")
    svm = _current_model.get("one_class_svm")
    lowered_threshold = _current_model.get("lowered_threshold")
    alt = _current_model.get("anomaly_level_thresholds") or {}
    t_low = alt.get("t_low")
    t_med = alt.get("t_med")
    t_high = alt.get("t_high")

    if scaler is None or iso is None or svm is None:
        raise HTTPException(status_code=500, detail="Loaded model bundle is incomplete (missing scaler / iso / svm)")

    # scale
    X_scaled = scaler.transform(X)

    # isolation forest
    try:
        iso_pred = iso.predict(X_scaled)           # returns array of -1 or 1
        iso_score_arr = iso.decision_function(X_scaled)  # higher = more normal
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"IsolationForest predict error: {e}")

    # one-class svm
    try:
        svm_pred_arr = svm.predict(X_scaled)       # -1 anomaly, 1 normal
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OneClassSVM predict error: {e}")

    iso_pred_flag = 1 if iso_pred[0] == -1 else 0
    svm_pred_flag = 1 if svm_pred_arr[0] == -1 else 0
    iso_score = float(iso_score_arr[0])

    # ensemble rule similar to DAG:
    # Anomaly if ISO predicts -1 OR SVM predicts -1 OR ISO score < lowered_threshold (if threshold available)
    anomaly = 1 if (iso_pred_flag == 1 or svm_pred_flag == 1) else 0
    if lowered_threshold is not None:
        try:
            # lowered_threshold might be a dict or simple value
            if isinstance(lowered_threshold, dict):
                # DAG saved lowered_threshold as {"lowered_threshold": value}
                lv = lowered_threshold.get("lowered_threshold", None)
            else:
                lv = lowered_threshold
            if lv is not None and iso_score < float(lv):
                anomaly = 1
        except Exception:
            # ignore threshold parsing errors (do not fail predict)
            pass

    # compute anomaly level
    anomaly_level = classify_iso_score(iso_score, t_low, t_med, t_high)

    resp = {
        "iso_score": iso_score,
        "iso_pred_flag": int(iso_pred_flag),
        "svm_pred_flag": int(svm_pred_flag),
        "anomaly": int(anomaly),
        "anomaly_level": anomaly_level,
    }
    return resp
