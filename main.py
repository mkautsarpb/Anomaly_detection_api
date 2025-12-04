import os, joblib, requests
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

GITHUB_REPO = os.getenv("GITHUB_REPO")
GITHUB_BRANCH = os.getenv("GITHUB_BRANCH", "main")
FASTAPI_API_KEY = os.getenv("FASTAPI_API_KEY")

app = FastAPI()
current_model = None
metadata = None

class PredictRequest(BaseModel):
    features: list

def auth(key):
    if FASTAPI_API_KEY and key != FASTAPI_API_KEY:
        raise HTTPException(401, "Unauthorized")

@app.get("/load_latest_model")
def load_latest_model(x_api_key: str = Header(None)):
    auth(x_api_key)
    global current_model, metadata
    base_url = f"https://raw.githubusercontent.com/{GITHUB_REPO}/{GITHUB_BRANCH}/model_upload"
    r = requests.get(f"{base_url}/metadata.json")
    if r.status_code != 200:
        raise HTTPException(500, "Failed to fetch metadata.json")
    metadata = r.json()
    r2 = requests.get(f"{base_url}/model.joblib")
    if r2.status_code != 200:
        raise HTTPException(500, "Failed to fetch model.joblib")
    tmp = "/tmp/latest_model.joblib"
    with open(tmp, "wb") as f:
        f.write(r2.content)
    current_model = joblib.load(tmp)
    return {"status": "loaded", "model_version": metadata.get("version")}

@app.post("/predict")
def predict(req: PredictRequest, x_api_key: str = Header(None)):
    auth(x_api_key)
    if current_model is None:
        raise HTTPException(503, "Model not loaded. Call /load_latest_model first.")
    import numpy as np
    arr = np.array(req.features).reshape(1, -1)
    pred = current_model.predict(arr)
    return {"prediction": pred.tolist()}
