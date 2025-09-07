# ui/streamlit_app.py
import os, io, json, base64, math, zipfile, subprocess, re, inspect
from pathlib import Path
from typing import Optional, List, Tuple

import requests
import streamlit as st
from PIL import Image

# =========================
# Small helpers
# =========================

def _img_kwargs_for_streamlit() -> dict:
    """
    Streamlit changed `st.image` arg name across versions.
    Use whatever the runtime supports to avoid
    'unexpected keyword argument use_container_width'.
    """
    try:
        sig = inspect.signature(st.image)
        if "use_container_width" in sig.parameters:
            return dict(use_container_width=True)
        if "use_column_width" in sig.parameters:
            return dict(use_column_width=True)
    except Exception:
        pass
    return {}

IMG_KW = _img_kwargs_for_streamlit()

def _toast(msg: str, icon: str = "ℹ️"):
    try:
        st.toast(msg, icon=icon)  # newer versions
    except Exception:
        st.info(msg)  # fallback


# =========================
# Kaggle creds bootstrap (supports Streamlit Secrets)
# =========================
def _write_kaggle_json_from_secrets() -> bool:
    """
    If secrets.toml has:
    [KAGGLE]
    username="..."
    key="..."
    write ~/.kaggle/kaggle.json (0600) and return True.
    """
    try:
        if "KAGGLE" not in st.secrets:
            return False
        creds = st.secrets["KAGGLE"]
        if not isinstance(creds, dict):
            return False
        username = creds.get("username")
        key = creds.get("key")
        if not (username and key):
            return False
        kdir = Path.home() / ".kaggle"
        kdir.mkdir(parents=True, exist_ok=True)
        f = kdir / "kaggle.json"
        f.write_text(json.dumps({"username": username, "key": key}))
        try:
            os.chmod(f, 0o600)
        except Exception:
            pass
        # also set env for the kaggle CLI, just in case
        os.environ["KAGGLE_USERNAME"] = str(username)
        os.environ["KAGGLE_KEY"] = str(key)
        return True
    except Exception:
        return False

# legacy bootstrap (supports older secrets schema)
def _write_legacy_kaggle_from_flat_secrets() -> bool:
    ok = False
    try:
        if "KAGGLE_USERNAME" in st.secrets and "KAGGLE_KEY" in st.secrets:
            os.environ["KAGGLE_USERNAME"] = st.secrets["KAGGLE_USERNAME"]
            os.environ["KAGGLE_KEY"] = st.secrets["KAGGLE_KEY"]
            ok = True
        elif "kaggle_json" in st.secrets:
            kdir = Path.home() / ".kaggle"
            kdir.mkdir(parents=True, exist_ok=True)
            (kdir / "kaggle.json").write_text(st.secrets["kaggle_json"])
            os.chmod(kdir / "kaggle.json", 0o600)
            ok = True
    except Exception:
        pass
    return ok

def have_kaggle_creds() -> bool:
    if _write_kaggle_json_from_secrets():
        return True
    if _write_legacy_kaggle_from_flat_secrets():
        return True
    # local dev fallback
    return (Path.home() / ".kaggle" / "kaggle.json").exists()


# =========================
# Page setup
# =========================
st.set_page_config(page_title="OncoVision: Your Histopathology Assistant", layout="wide")
st.markdown(
    """
    <style>
      .badge {display:inline-block;padding:2px 8px;border-radius:6px;font-size:0.8rem;background:#ffeee6;color:#b24500;border:1px solid #ffc7ad;}
      .metric-box {background: #0f172a0a; border: 1px solid #e5e7eb; padding: 10px 12px; border-radius: 12px;}
      .footer-note {color:#475569;font-size:0.85rem;}
      .stButton>button {border-radius:10px;padding:0.5rem 1rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# Config / Secrets
# =========================
API_DEFAULT = os.getenv("API_URL", "http://localhost:8000")
API_URL = st.secrets.get("API_URL", API_DEFAULT)

INLINE_ENGINE = False
try:
    from api.inference import InferenceEngine as _Engine  # project engine (optional)
except Exception:
    _Engine = None
    INLINE_ENGINE = True

if os.getenv("ONCOVISION_FORCE_INLINE") == "1":
    INLINE_ENGINE = True

# =========================
# Sidebar
# =========================
st.sidebar.header("Settings")
run_mode = st.sidebar.radio("Run mode", ["Remote API", "Local (in-app)"], index=1, key="run_mode")
API_URL = st.sidebar.text_input(
    "API URL",
    value=API_URL,
    help="Your FastAPI base URL (e.g., https://oncovision-api.onrender.com)",
    disabled=(run_mode == "Local (in-app)"),
    key="api_url",
)
weights_path = st.sidebar.text_input(
    "Local weights path",
    value="artifacts/resnet18_histopath.pt",
    help="Only used in Local mode",
    disabled=(run_mode == "Remote API"),
    key="weights_path",
)
threshold = st.sidebar.slider("Decision threshold (Cancer)", 0.00, 1.00, 0.50, 0.01, key="thresh")
st.sidebar.caption("Prediction ≥ threshold → label = Cancer (else Benign)")
st.sidebar.markdown('<span class="badge">Research demo</span>', unsafe_allow_html=True)

# =========================
# Header
# =========================
colh1, colh2 = st.columns([0.75, 0.25])
with colh1:
    st.title("OncoVision — Histopathology AI Assistant")
    st.write("Upload a histology patch to get **cancer probability** and a **Grad-CAM heatmap** highlighting influential regions.")

def api_ok(url: str) -> bool:
    try:
        r = requests.get(f"{url}/health", timeout=3)
        return bool(r.ok)
    except Exception:
        return False

with colh2:
    if run_mode == "Remote API":
        ok_api = api_ok(API_URL)
        st.markdown(f"**API:** {'🟢 Live' if ok_api else '🔴 Offline'}")
        st.caption(API_URL)
    else:
        ok_api = True
        st.markdown("**Local engine:** 🟢 Enabled")
        st.caption(Path(weights_path).as_posix())

st.divider()

# =========================
# Kaggle sample gallery
# =========================
KAGGLE_DATASET = "andrewmvd/lung-and-colon-cancer-histopathological-images"
KAGGLE_CACHE = Path("data/kaggle_lc25000")
KAGGLE_ZIP = KAGGLE_CACHE / "lc25000.zip"

def download_kaggle_dataset():
    KAGGLE_CACHE.mkdir(parents=True, exist_ok=True)

    # If images already exist, skip
    if any(KAGGLE_CACHE.rglob("*.png")) or any(KAGGLE_CACHE.rglob("*.jpg")):
        return

    if not KAGGLE_ZIP.exists():
        cmd = ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET, "-p", str(KAGGLE_CACHE)]
        subprocess.run(cmd, check=True)
        zips = list(KAGGLE_CACHE.glob("*.zip"))
        if zips:
            zips[0].rename(KAGGLE_ZIP)

    # unzip top-level
    with zipfile.ZipFile(KAGGLE_ZIP, "r") as zf:
        zf.extractall(KAGGLE_CACHE)

    # unzip nested zips (there are sub-archives)
    for inner_zip in list(KAGGLE_CACHE.rglob("*.zip")):
        try:
            with zipfile.ZipFile(inner_zip, "r") as zf:
                target_dir = inner_zip.parent / inner_zip.stem
                target_dir.mkdir(parents=True, exist_ok=True)
                zf.extractall(target_dir)
            inner_zip.unlink()
        except Exception as e:
            print(f"Skipping {inner_zip}: {e}")

POS_TOKENS = {
    "adenocarcinoma", "carcinoma", "malignant", "cancer",
    "aca", "lung_aca", "colon_aca",
    "scc", "lung_scc", "squamous"
}
NEG_TOKENS = {"benign", "normal", "lung_n", "colon_n", "_n"}

def infer_class_from_path(p: Path) -> str:
    text = str(p).lower().replace("\\", "/")
    for tok in POS_TOKENS:
        if tok in text:
            return "1_cancer"
    for tok in NEG_TOKENS:
        if tok in text:
            return "0_normal"
    if re.search(r"/(lung|colon)_(aca|scc)/", text):
        return "1_cancer"
    if re.search(r"/(lung|colon)_n/", text):
        return "0_normal"
    return "unknown"

def build_kaggle_index() -> List[Tuple[str, Path]]:
    exts = {".png", ".jpg", ".jpeg"}
    items: List[Tuple[str, Path]] = []
    if not KAGGLE_CACHE.exists():
        return items
    for p in KAGGLE_CACHE.rglob("*"):
        if p.suffix.lower() in exts:
            items.append((infer_class_from_path(p), p))
    return items

with st.expander("Try sample images from Kaggle (LC25000)", expanded=False):
    if not have_kaggle_creds():
        st.warning(
            "Kaggle credentials not found. Add them in Streamlit Secrets as:\n"
            "[KAGGLE]\nusername=\"...\"\nkey=\"...\"\nThen restart the app."
        )
    else:
        col_dl, col_stats = st.columns([1, 1])
        with col_dl:
            if st.button("Download / Refresh LC25000", key="btn_kaggle_download"):
                with st.spinner("Downloading from Kaggle… first time may take a few minutes"):
                    try:
                        download_kaggle_dataset()
                        st.success("Dataset ready.")
                        st.session_state.pop("kaggle_index", None)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Download failed: {e}")

        if "kaggle_index" not in st.session_state:
            st.session_state["kaggle_index"] = build_kaggle_index()
        idx: List[Tuple[str, Path]] = st.session_state["kaggle_index"]

        if not idx:
            st.info("No images indexed yet. Click “Download / Refresh LC25000”.")
        else:
            LABEL_MAP = {"0_normal": "Benign / Normal", "1_cancer": "Cancer", "unknown": "Unknown"}
            COLOR_MAP = {"0_normal": "#0ea5e9", "1_cancer": "#ef4444", "unknown": "#64748b"}

            with col_stats:
                counts = {}
                for c, _ in idx:
                    counts[c] = counts.get(c, 0) + 1
                st.caption("Indexed images:")
                st.write(", ".join([f"{LABEL_MAP.get(c, c)}: {counts[c]}" for c in sorted(counts)]))

            filter_options = ["All"] + [LABEL_MAP.get(c, c) for c in sorted(set(c for c, _ in idx))]
            sel = st.selectbox("Class filter", filter_options, index=0, key="kaggle_filter")
            sel_code = None if sel == "All" else next((k for k, v in LABEL_MAP.items() if v == sel), sel)

            items = idx if sel_code is None else [(c, p) for (c, p) in idx if c == sel_code]
            items = sorted(items, key=lambda t: t[1].name.lower())

            if not items:
                st.warning("No images match this filter. Try a different class.")
            else:
                page_size = 12
                pages = max(1, math.ceil(len(items) / page_size))
                page = st.number_input("Page", min_value=1, max_value=pages, value=1, step=1, key="kaggle_page")
                page_items = items[(page - 1) * page_size : page * page_size]

                grid = st.columns(4)
                for i, (cls, p) in enumerate(page_items):
                    with grid[i % 4]:
                        try:
                            img = Image.open(p).convert("RGB")
                            badge = f"""
                                <div style="display:inline-block;padding:2px 8px;border-radius:8px;
                                            font-size:0.8rem;margin-bottom:6px;background:{COLOR_MAP.get(cls,'#e5e7eb')};
                                            color:white;">
                                    {LABEL_MAP.get(cls, cls)}
                                </div>
                            """
                            st.markdown(badge, unsafe_allow_html=True)
                            st.image(img, caption=p.name, **IMG_KW)
                            if st.button(f"Use {p.name}", key=f"kgl_{p.name}_{i}"):
                                buf = io.BytesIO(); img.save(buf, format="PNG")
                                st.session_state["image_bytes"] = buf.getvalue()
                                st.session_state["image_src"] = f"{LABEL_MAP.get(cls,cls)} • {p.name}"
                                _toast(f"Loaded {p.name}", "✅")
                                st.rerun()
                        except Exception as e:
                            st.write(f"Image unreadable: {e}")

st.divider()

# =========================
# Upload (single source of truth + reset)
# =========================
def _on_upload_change():
    """Callback ensures the new upload replaces previous image."""
    file = st.session_state.get("upload_widget")
    if file is not None:
        st.session_state["image_bytes"] = file.read()
        st.session_state["image_src"] = f"uploaded • {file.name}"
        # clear any previous results if you cache them
        st.session_state.pop("last_prediction", None)

uploaded = st.file_uploader(
    "Upload image",
    type=["png", "jpg", "jpeg"],
    key="upload_widget",          # explicit key prevents DuplicateWidgetID
    on_change=_on_upload_change,  # keep session state in sync
)

# Clear button for convenience
if st.button("Clear image", key="btn_clear_image"):
    for k in ("image_bytes", "image_src", "upload_widget", "last_prediction"):
        st.session_state.pop(k, None)
    st.rerun()

# Resolve current image bytes from state
image_bytes: Optional[bytes] = st.session_state.get("image_bytes")

# =========================
# Local engine loader (robust)
# =========================
def get_local_engine(weights_path: str):
    """
    Return an InferenceEngine instance; try project engine first,
    then a minimal inline fallback.
    """
    global INLINE_ENGINE

    if not INLINE_ENGINE and _Engine is not None:
        for attempt in ("kw", "pos", "noarg"):
            try:
                if attempt == "kw":
                    return _Engine(weights_path=weights_path)
                if attempt == "pos":
                    return _Engine(weights_path)  # type: ignore
                eng = _Engine()
                if hasattr(eng, "load_weights"):
                    try:
                        eng.load_weights(weights_path)  # type: ignore
                    except Exception:
                        pass
                return eng
            except TypeError:
                continue
            except Exception:
                break
        INLINE_ENGINE = True  # fall through

    # ---- INLINE FALLBACK ----
    import io as _io, base64 as _b64
    from dataclasses import dataclass
    import numpy as _np
    import torch as _torch
    import torchvision.transforms as _T
    import torch.nn as _nn
    from torchvision import models as _models
    from matplotlib import colormaps as _colormaps

    @dataclass
    class HeatmapResult:
        prob: float
        overlay_png_b64: Optional[str]
        msg: Optional[str] = None

    def _get_resnet18_binary():
        m = _models.resnet18(weights=_models.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = _nn.Linear(m.fc.in_features, 1)
        return m

    def _grad_cam_overlay(model: _nn.Module, x: _torch.Tensor, original: Image.Image) -> Image.Image:
        last_conv = None
        for m in model.modules():
            if isinstance(m, _nn.Conv2d):
                last_conv = m
        if last_conv is None:
            return original

        feats: List[_torch.Tensor] = []
        grads: List[_torch.Tensor] = []

        def fwd_hook(_, __, out): feats.append(out)
        def bwd_hook(_, __, gout): grads.append(gout[0])

        h1 = last_conv.register_forward_hook(fwd_hook)
        h2 = last_conv.register_full_backward_hook(bwd_hook)
        try:
            out = model(x)
            score = _torch.sigmoid(out)[0]
            score.backward()

            A = feats[-1].detach()
            G = grads[-1].detach()
            w = G.mean(dim=(2,3), keepdim=True)
            cam = _torch.relu((A * w).sum(dim=1, keepdim=True))
            cam = cam / (cam.max() + 1e-6)
            cam = _torch.nn.functional.interpolate(
                cam, size=(original.height, original.width),
                mode="bilinear", align_corners=False
            )[0,0].cpu().numpy()

            cmap = _colormaps["jet"]
            color = cmap(cam)[:, :, :3]
            base = _np.asarray(original.convert("RGB")) / 255.0
            overlay = (0.55 * base + 0.45 * color)
            overlay = (overlay.clip(0,1) * 255).astype("uint8")
            return Image.fromarray(overlay)
        finally:
            h1.remove(); h2.remove()

    class _InlineEngine:
        def __init__(self, weights_path: str = "artifacts/resnet18_histopath.pt"):
            self.weights_path = weights_path
            self.device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
            self.model = None
            self.pre = _T.Compose([
                _T.Resize((224, 224)),
                _T.ToTensor(),
                _T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
            ])

        def _load(self):
            if self.model is None:
                self.model = _get_resnet18_binary()
                state = _torch.load(self.weights_path, map_location=self.device)
                self.model.load_state_dict(state, strict=False)
                self.model = self.model.to(self.device).eval()

        @_torch.inference_mode()
        def _prob_only(self, x):
            logits = self.model(x)
            return float(_torch.sigmoid(logits)[0].item())

        def predict_with_heatmap(self, image_bytes: bytes) -> "HeatmapResult":
            self._load()
            image = Image.open(_io.BytesIO(image_bytes)).convert("RGB")

            x0 = self.pre(image).unsqueeze(0).to(self.device)
            prob = self._prob_only(x0)

            overlay_b64, msg = None, None
            try:
                x = self.pre(image).unsqueeze(0).to(self.device)
                x.requires_grad_(True)
                self.model.zero_grad(set_to_none=True)
                out = self.model(x)
                score = _torch.sigmoid(out)[0]
                score.backward()

                overlay = _grad_cam_overlay(self.model, x, original=image)
                buf = _io.BytesIO(); overlay.save(buf, format="PNG")
                overlay_b64 = _b64.b64encode(buf.getvalue()).decode("utf-8")
            except Exception as e:
                msg = f"Heatmap unavailable: {e}"

            return HeatmapResult(prob=prob, overlay_png_b64=overlay_b64, msg=msg)

    return _InlineEngine(weights_path=weights_path)

@st.cache_resource(show_spinner=False)
def _get_engine_cached(path: str):
    return get_local_engine(path)

# =========================
# Main two-column layout
# =========================
left, right = st.columns([0.55, 0.45])

# Input preview
with left:
    st.subheader("Input")
    if image_bytes:
        try:
            img_preview = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            st.image(img_preview, **IMG_KW)
            if st.session_state.get("image_src"):
                st.caption(f"Source: {st.session_state['image_src']}")
        except Exception as e:
            st.error(f"Could not read image: {e}")
    else:
        st.info("Upload an image or choose a Kaggle sample above to begin.")

# Prediction panel
with right:
    st.subheader("Prediction")
    analyze_disabled = (not image_bytes) or (not ok_api)
    if st.button("Analyze", type="primary", disabled=analyze_disabled, use_container_width=True, key="btn_analyze"):
        if not image_bytes:
            st.error("Please select or upload an image first.")
        else:
            try:
                if run_mode == "Remote API":
                    if not api_ok(API_URL):
                        st.error("API is offline. Check your API URL in the sidebar.")
                    else:
                        files = {"file": ("input.png", image_bytes, "image/png")}
                        r = requests.post(f"{API_URL}/predict", files=files, timeout=60)
                        if not r.ok:
                            st.error(f"API error: {r.status_code} — {r.text[:300]}")
                        else:
                            data = r.json()
                            prob = float(data.get("prob", 0.0))
                            label = "Cancer" if prob >= threshold else "Benign"

                            m1, m2 = st.columns(2)
                            with m1: st.metric("Cancer probability", f"{prob:.3f}", help=f"Decision threshold = {threshold:.2f}")
                            with m2: st.metric("Predicted label", label)

                            b64 = data.get("overlay_png_b64")
                            if b64:
                                st.image(Image.open(io.BytesIO(base64.b64decode(b64))), caption="Heatmap Overlay", **IMG_KW)
                            else:
                                st.info("Heatmap unavailable for this image.")
                else:
                    engine = _get_engine_cached(weights_path)
                    res = engine.predict_with_heatmap(image_bytes)
                    prob = float(res.prob)
                    label = "Cancer" if prob >= threshold else "Benign"

                    m1, m2 = st.columns(2)
                    with m1: st.metric("Cancer probability", f"{prob:.3f}", help=f"Decision threshold = {threshold:.2f}")
                    with m2: st.metric("Predicted label", label)

                    if getattr(res, "overlay_png_b64", None):
                        st.image(Image.open(io.BytesIO(base64.b64decode(res.overlay_png_b64))), caption="Heatmap Overlay", **IMG_KW)
                    elif getattr(res, "msg", None):
                        st.info(res.msg)
                    else:
                        st.info("Heatmap unavailable for this image.")
            except Exception as e:
                _toast(f"Request failed: {e}", "⚠️")

st.divider()

# =========================
# Metrics / Plots (optional artifacts)
# =========================
st.subheader("Model Evaluation")
artifacts = Path("artifacts")
summary_path = artifacts / "metrics_summary.json"
cm_path = artifacts / "confusion_matrix.png"
roc_path = artifacts / "roc_curve.png"
pr_path  = artifacts / "pr_curve.png"

if summary_path.exists():
    try:
        summary = json.loads(summary_path.read_text())
        acc = summary.get("acc"); prec = summary.get("precision")
        rec = summary.get("recall"); roc_auc = summary.get("roc_auc"); pr_auc = summary.get("pr_auc")
        cols = st.columns(5)
        cols[0].markdown(f'<div class="metric-box"><b>Accuracy</b><br/>{acc:.3f}</div>', unsafe_allow_html=True)
        cols[1].markdown(f'<div class="metric-box"><b>Precision</b><br/>{prec:.3f}</div>', unsafe_allow_html=True)
        cols[2].markdown(f'<div class="metric-box"><b>Recall</b><br/>{rec:.3f}</div>', unsafe_allow_html=True)
        cols[3].markdown(f'<div class="metric-box"><b>ROC AUC</b><br/>{roc_auc:.3f}</div>', unsafe_allow_html=True)
        cols[4].markdown(f'<div class="metric-box"><b>PR AUC</b><br/>{pr_auc:.3f}</div>', unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Could not read metrics_summary.json: {e}")
else:
    st.caption("No metrics_summary.json found. Run the evaluation script to generate one.")

pcols = st.columns(3)
if cm_path.exists():
    pcols[0].image(str(cm_path), caption="Confusion Matrix", **IMG_KW)
if roc_path.exists():
    pcols[1].image(str(roc_path), caption="ROC Curve", **IMG_KW)
if pr_path.exists():
    pcols[2].image(str(pr_path), caption="Precision–Recall Curve", **IMG_KW)

st.markdown('<div class="footer-note">Note: Trained on LC25000 (lung & colon histopathology). Dataset is relatively clean; real-world performance may vary.</div>', unsafe_allow_html=True)
