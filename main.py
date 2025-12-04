import os
import joblib
import requests
from fastapi import FastAPI, Header, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import shutil

# ============================
# ENV VARIABLES
# ============================
GITHUB_REPO = os.getenv("GITHUB_REPO")
GITHUB_BRANCH = os.getenv("GITHUB_BRANCH", "main")
FASTAPI_API_KEY = os.getenv("FASTAPI_API_KEY")

# ============================
# APP
# ============================
app = FastAPI(title="Anomaly Detection API")
current_model = None
metadata = None


# ============================
# AUTH
# ============================
def check_auth(key: str):
    if FASTAPI_API_KEY and key != FASTAPI_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


# ============================
# ---------- SCHEMAS ----------
# ============================
class PredictRequest(BaseModel):
    features: list


# ============================
# ---------- ENDPOINTS ----------
# ============================

@app.get("/api/ml/anomaly/load_latest_model")
def load_latest_model(x_api_key: str = Header(None)):
    check_auth(x_api_key)

    global current_model, metadata

    if not GITHUB_REPO:
        raise HTTPException(500, "GITHUB_REPO not set in Railway variables")

    base_url = f"https://raw.githubusercontent.com/{GITHUB_REPO}/{GITHUB_BRANCH}/model_upload"

    # Fetch metadata.json
    r = requests.get(f"{base_url}/metadata.json")
    if r.status_code != 200:
        raise HTTPException(500, "Failed to fetch metadata.json from GitHub")

    metadata = r.json()

    # Fetch model file
    r2 = requests.get(f"{base_url}/model.joblib")
    if r2.status_code != 200:
        raise HTTPException(500, "Failed to fetch model.joblib from GitHub")

    # Save temporarily
    tmp_path = "/tmp/latest_model.joblib"
    with open(tmp_path, "wb") as f:
        f.write(r2.content)

    # Load model
    current_model = joblib.load(tmp_path)

    return {
        "status": "success",
        "message": "Model loaded successfully",
        "model_version": metadata.get("version")
    }


@app.post("/api/ml/anomaly/upload_model")
async def upload_model(
    model_file: UploadFile = File(...),
    metadata_file: UploadFile = File(...),
    x_api_key: str = Header(None)
):
    check_auth(x_api_key)

    upload_dir = "/tmp/model_received"
    if os.path.exists(upload_dir):
        shutil.rmtree(upload_dir)
    os.makedirs(upload_dir, exist_ok=True)

    # Save model file
    model_path = os.path.join(upload_dir, "model.joblib")
    with open(model_path, "wb") as f:
        f.write(await model_file.read())

    # Save metadata file
    metadata_path = os.path.join(upload_dir, "metadata.json")
    with open(metadata_path, "wb") as f:
        f.write(await metadata_file.read())

    return {
        "status": "success",
        "message": "Model received by API (not saved to GitHub — GitHub upload handled by Airflow)"
    }


@app.post("/api/ml/anomaly/predict")
def predict(req: PredictRequest, x_api_key: str = Header(None)):
    check_auth(x_api_key)

    global current_model

    if current_model is None:
        raise HTTPException(503, "Model not loaded. Call /api/ml/anomaly/load_latest_model first.")

    import numpy as np

    arr = np.array(req.features).reshape(1, -1)
    pred = current_model.predict(arr)

    return {"prediction": pred.tolist()}


@app.get("/")
def root():
    return {"status": "ok", "message": "Anomaly Detection API running"}
