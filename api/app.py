from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from .inference import InferenceEngine, HeatmapResult

app = FastAPI(title="OncoVision API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

engine = InferenceEngine()  

class PredictOut(BaseModel):
    prob_cancer: float
    heatmap_png_b64: Optional[str]  
    message: Optional[str] = None

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/predict", response_model=PredictOut)
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    result: HeatmapResult = engine.predict_with_heatmap(image_bytes)
    return PredictOut(prob_cancer=result.prob, heatmap_png_b64=result.overlay_png_b64, message=result.msg)
