# api/inference.py
import io
import os
import base64
from dataclasses import dataclass
from typing import Optional

from PIL import Image
import torch
import torchvision.transforms as T

from .model import get_resnet18_binary
from .viz import grad_cam_overlay


@dataclass
class HeatmapResult:
    prob: float                     # P(cancer)
    overlay_png_b64: Optional[str]  # Grad-CAM overlay (base64 PNG) or None
    msg: Optional[str] = None       # Info/warning message


class InferenceEngine:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.preprocess = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
        # If your trained positive class is "normal", set this env var to invert at inference.
        #   Windows (current session):  $env:ONCOVISION_PROB_IS_NORMAL = "1"
        #   Windows (persist):          setx ONCOVISION_PROB_IS_NORMAL 1
        #   Unix:                       export ONCOVISION_PROB_IS_NORMAL=1
        self.prob_is_normal = os.getenv("ONCOVISION_PROB_IS_NORMAL", "0") == "1"

    def _load(self):
        if self.model is None:
            self.model = get_resnet18_binary().to(self.device).eval()

    @torch.inference_mode()
    def _prob_only(self, x: torch.Tensor) -> float:
        """
        Returns P(cancer) as a float in [0,1].
        If the model was trained with '1 = normal', we invert at runtime.
        """
        logits = self.model(x)                 # shape [1, 1]
        p = torch.sigmoid(logits)[0, 0].item() # scalar float
        return 1.0 - p if self.prob_is_normal else p

    def predict_with_heatmap(self, image_bytes: bytes) -> HeatmapResult:
        self._load()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # --- Always compute probability without grads ---
        x_no_grad = self.preprocess(image).unsqueeze(0).to(self.device)
        prob_cancer = self._prob_only(x_no_grad)

        # --- Try to compute a Grad-CAM overlay with grads enabled ---
        overlay_b64 = None
        msg = None
        try:
            x_cam = self.preprocess(image).unsqueeze(0).to(self.device)
            self.model.zero_grad(set_to_none=True)

            # Use the pre-sigmoid logit for Grad-CAM target.
            out = self.model(x_cam)          # [1, 1]
            score = out[0, 0]                # scalar logit

            # If model's positive class is "normal", negate the logit so CAM highlights cancer evidence.
            if self.prob_is_normal:
                score = -score

            score.backward()  # compute grads for CAM

            overlay = grad_cam_overlay(self.model, x_cam, original=image)
            buf = io.BytesIO()
            overlay.save(buf, format="PNG")
            overlay_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        except Exception as e:
            msg = f"Heatmap unavailable: {e}"

        return HeatmapResult(prob=prob_cancer, overlay_png_b64=overlay_b64, msg=msg)
