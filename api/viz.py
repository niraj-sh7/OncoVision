import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2

# Simple Grad-CAM for last conv layer
def _find_last_conv_module(model):
    last = None
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            last = m
    return last

def grad_cam_overlay(model, x, original: Image.Image):
    model.eval()
    target_layer = _find_last_conv_module(model)
    feats = []
    grads = []

    def fwd_hook(_, __, output): feats.append(output.detach())
    def bwd_hook(_, grad_in, grad_out): grads.append(grad_out[0].detach())

    fh = target_layer.register_forward_hook(fwd_hook)
    bh = target_layer.register_full_backward_hook(bwd_hook)

    out = model(x)  # forward has been done already usually
    # assuming backward was called outside; but ensure here:
    if not grads:
        out.sum().backward()

    fh.remove(); bh.remove()

    fmap = feats[-1][0]            # (C,H,W)
    grad = grads[-1][0]            # (C,H,W)
    weights = grad.mean(dim=(1,2)) # (C,)

    cam = (weights[:, None, None] * fmap).sum(0)  # (H,W)
    cam = F.relu(cam)
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    cam = cam.cpu().numpy()

    # resize to original and colorize
    orig = np.array(original)
    H, W = orig.shape[:2]
    cam_resized = cv2.resize(cam, (W, H))
    heat = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(orig, 0.6, heat, 0.4, 0)
    return Image.fromarray(overlay)
