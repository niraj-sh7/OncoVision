# OncoVision: Histopathology AI Assistant

End-to-end demo that classifies **histology patches** as suspicious/benign and returns an **explainability heatmap (Grad-CAM)**.

## Features
- **FastAPI** backend: `/predict` for image upload → JSON with probability + heatmap.
- **PyTorch** model: Transfer-learning with **ResNet18** (ImageNet) for binary classification.
- **Grad-CAM**: Visual overlay to highlight regions driving the prediction.
- **Streamlit** demo UI: drag & drop an image, see probability + heatmap instantly.

> ⚠️ Scope: patch-level classification (e.g., 224×224 H&E tiles). Not a clinical tool.

---

## Quickstart

### 1) Python env
```bash
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
```

### 2) Start the API
```bash
uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
# Swagger: http://localhost:8000/docs
```

### 3) Run the demo UI
```bash
streamlit run ui/streamlit_app.py
```

### 4) (Optional) Fine-tune the model
- Drop a small folder dataset like:
```
data/
  train/
    cancer/
    normal/
  val/
    cancer/
    normal/
```
- Then run:
```bash
python training/train_resnet18.py --data_dir data --epochs 2 --batch_size 32
# Outputs weights: artifacts/resnet18_histopath.pt
```

### API
`POST /predict` — multipart/form-data with key `file` (image).  
Returns JSON with `prob_cancer`, and base64-encoded `heatmap_png` (overlay).

---

## Notes
- Default uses ImageNet features + randomly initialized final layer; for demo, we simulate OK accuracy. 
- If you fine-tune, save weights to `artifacts/resnet18_histopath.pt` and the API will auto-load them if present.
- Works on CPU. CUDA will be used automatically if available.
