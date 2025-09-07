import os, torch
import torch.nn as nn
from torchvision.models import resnet18  

WEIGHTS_PATH = os.environ.get("ONCOVISION_WEIGHTS", "artifacts/resnet18_histopath.pt")

def get_resnet18_binary():
    model = resnet18(weights=None) 
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, 1)
    if os.path.exists(WEIGHTS_PATH):
        try:
            state = torch.load(WEIGHTS_PATH, map_location="cpu")
            model.load_state_dict(state, strict=False)
            print(f"[OncoVision] Loaded weights: {WEIGHTS_PATH}")
        except Exception as e:
            print("[OncoVision] Warning: could not load weights ->", e)
    return model
